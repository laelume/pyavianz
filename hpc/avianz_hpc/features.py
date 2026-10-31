"""Spectrogram features shared by dataset building and inference.

Both ``build_dataset`` and ``infer`` go through ``SpectrogramExtractor`` and
``InputPreparer``, so a model always sees the same preprocessing at training and
inference time. The extraction reproduces model-testing's ``SpectrogramProcessor``
(per-clip RMS normalisation, AviaNZ spectrogram, rot90) using the AviaNZ
Spectrogram class from this repository.

``FeatureSpec`` is written into ``labels.json`` by the dataset builder, copied
into ``<model>_model_config.json`` by the trainer, and read back by inference.
"""

import contextlib
from dataclasses import asdict, dataclass, fields
import io
from typing import Optional
import warnings

import numpy as np

from avianz_hpc._avianz import Spectrogram
from avianz_hpc.core import config


@dataclass
class FeatureSpec:
    sample_rate: int = config.DEFAULT_SAMPLE_RATE
    window_seconds: float = config.DEFAULT_WINDOW_SECONDS
    hop_seconds: float = config.DEFAULT_HOP_SECONDS
    freq_bins: int = config.DEFAULT_FREQ_BINS
    time_bins: int = config.DEFAULT_TIME_BINS
    sg_type: str = "Standard"           # Standard | Reassigned | Multi-tapered
    window_type: str = "Hamming"        # Hann | Hamming | Blackman | BlackmanHarris ...
    sg_scale: str = "Mel Frequency"     # Mel Frequency | Bark Frequency | Linear
    mean_normalise: bool = False
    equal_loudness: bool = False
    # How training clips were cut: "window" (fixed-length windows) or "segment"
    # (one clip per annotated segment, as in model-testing). clip_seconds is the
    # window length in "window" mode and the default inference window otherwise.
    clip_mode: str = "window"
    clip_seconds: Optional[float] = None

    def __post_init__(self):
        if self.clip_seconds is None:
            self.clip_seconds = round(self.time_bins * self.hop_seconds, 3)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_model_config(cls, model_config):
        """Recover the spec from a trainer model config.

        Configs written by this package carry ``feature_spec``. Configs from the
        original model-testing trainer only carry the module defaults, which do
        NOT reflect ``--spec-type``/``--sg-scale`` used when building the data,
        so callers must supply the right overrides for those checkpoints.
        """
        if "feature_spec" in model_config:
            return cls.from_dict(model_config["feature_spec"])
        warnings.warn(
            "Model config has no feature_spec (trained outside avianz_hpc). Falling back to "
            "its spectrogram_params; pass --sg-type/--window-type/--sg-scale if the training "
            "data was built with different settings (model-testing build scripts used "
            "Standard/Hamming/Mel Frequency)."
        )
        sp = model_config.get("spectrogram_params", {})
        return cls(
            sample_rate=model_config.get("sample_rate", config.DEFAULT_SAMPLE_RATE),
            window_seconds=model_config.get("window_seconds", config.DEFAULT_WINDOW_SECONDS),
            hop_seconds=model_config.get("hop_seconds", config.DEFAULT_HOP_SECONDS),
            freq_bins=model_config.get("freq_bins", config.DEFAULT_FREQ_BINS),
            time_bins=model_config.get("time_bins", config.DEFAULT_TIME_BINS),
            sg_type=sp.get("sgType", "Reassigned"),
            window_type=sp.get("windowType", "Hamming"),
            sg_scale=sp.get("sgScale", "Mel Frequency"),
            mean_normalise=sp.get("mean_normalise", False),
            equal_loudness=sp.get("equal_loudness", False),
            clip_mode="segment",
        )


def downsample_freq_bins(sg, nfilters):
    """Average adjacent linear-frequency bins down to nfilters bins.

    Port of model-testing's Spectrogram.downsample_freq_bins, which the AviaNZ
    Spectrogram does not have. sg has shape (frames, bins).
    """
    current_bins = sg.shape[1]
    if current_bins <= nfilters:
        return sg
    pooling = np.zeros((current_bins, nfilters))
    per = current_bins / nfilters
    for m in range(nfilters):
        start = int(m * per)
        end = current_bins if m == nfilters - 1 else int((m + 1) * per)
        if end > start:
            pooling[start:end, m] = 1.0 / (end - start)
    return np.dot(sg, pooling)


class SpectrogramExtractor:
    """Load recordings and turn audio slices into raw (linear power) spectrograms."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.window_width = int(spec.window_seconds * spec.sample_rate)
        self.incr = int(spec.hop_seconds * spec.sample_rate)
        self.sp = Spectrogram(window_width=self.window_width, incr=self.incr)

    def load(self, path):
        """Read a whole recording (mono, float) resampled to spec.sample_rate."""
        self.sp.readSoundFile(path, silent=True)
        if self.sp.audio_data.sample_rate != self.spec.sample_rate:
            self.sp.resample(self.spec.sample_rate)
        return np.asarray(self.sp.audio_data.data, dtype=np.float64)

    def extract(self, audio):
        """Spectrogram of an audio slice, shape (freq_bins, frames), float32.

        Returns None for silent/degenerate input or NaNs.
        """
        if len(audio) < 2:
            return None
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio / rms * 0.1
        self.sp.audio_data.data = audio
        # AviaNZ's convertToMel prints on every call; keep job logs readable.
        with contextlib.redirect_stdout(io.StringIO()):
            sg = self.sp.spectrogram(
                window_width=self.window_width,
                incr=self.incr,
                window=self.spec.window_type,
                sgType=self.spec.sg_type,
                sgScale=self.spec.sg_scale,
                nfilters=self.spec.freq_bins,
                mean_normalise=self.spec.mean_normalise,
                equal_loudness=self.spec.equal_loudness,
            )
        if sg is None or np.size(sg) == 0:
            return None
        if self.spec.sg_scale == "Linear":
            sg = downsample_freq_bins(sg, self.spec.freq_bins)
        sg = np.rot90(sg)
        if np.isnan(sg).any():
            return None
        return np.ascontiguousarray(sg, dtype=np.float32)


def window_starts(duration, clip_seconds, hop_seconds):
    """Start times of fixed-length windows covering [0, duration).

    Windows are always full length: instead of a short tail window, a final window
    ending exactly at `duration` is added. Recordings shorter than one window get
    a single window starting at 0.
    """
    if duration <= clip_seconds:
        return [0.0]
    last = duration - clip_seconds
    starts = list(np.arange(0.0, last + 1e-9, hop_seconds))
    if last - starts[-1] > 1e-6:
        starts.append(last)
    return [round(float(s), 6) for s in starts]


def trim_to_length(sg, target_time_bins):
    """Keep the highest-energy run of target_time_bins columns (model-testing behaviour)."""
    if sg.shape[1] <= target_time_bins:
        return sg
    energy = sg.sum(axis=0)
    window = np.convolve(energy, np.ones(target_time_bins), mode="valid")
    start = int(np.argmax(window))
    return sg[:, start:start + target_time_bins]


class InputPreparer:
    """Turn raw spectrograms into model input tensors exactly as validation does."""

    def __init__(self, spec: FeatureSpec, spec_transform="Log", bg_subtract=False,
                 median_filter=False, use_deltas=False):
        from avianz_hpc.data.data_utils import SpectrogramDataset

        # An empty eval-mode dataset is only used for its preprocessing methods.
        self._ds = SpectrogramDataset(
            [], np.zeros((1, 1), dtype=np.float32), spec.freq_bins, spec.time_bins,
            config.DEFAULT_CHANNELS, cropping_mode="center", noise_ratio=0.0,
            spec_transform=spec_transform, training=False, bg_subtract=bg_subtract,
            median_filter=median_filter, use_temporal_roll=False, use_deltas=use_deltas,
        )

    @classmethod
    def from_model_config(cls, spec, model_config):
        return cls(
            spec,
            spec_transform=model_config.get("spec_transform", config.DEFAULT_SPEC_TRANSFORM),
            bg_subtract=model_config.get("bg_subtract", False),
            median_filter=model_config.get("median_filter", False),
            use_deltas=model_config.get("use_deltas", False),
        )

    def __call__(self, sg):
        return self._ds.eval_transform(sg)
