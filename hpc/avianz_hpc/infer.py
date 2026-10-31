"""Run a trained detector and/or classifier over raw recordings (one shard).

For every recording in this shard's slice of the manifest:
  1. cut fixed-length windows (default: the training clip length, 50% overlap),
  2. compute spectrograms with the same AviaNZ code and settings used to build
     the training data (FeatureSpec in the model config),
  3. run the detector; run the classifier on windows whose detector score is
     >= --detector-threshold (all windows when there is no detector),
  4. write  <out>/predictions/<relpath>.csv   per-window scores
            <out>/annotations/<relpath>.data  AviaNZ annotations (merged windows)

<relpath>.data mirrors AviaNZ's sidecar naming (rec.wav -> rec.wav.data), so
copying the annotations tree over the audio tree lets you review detections in
the AviaNZ GUI, and confirmed labels feed straight back into build-dataset.

A recording whose .data already exists is skipped, so a timed-out or
preempted array task can simply be resubmitted.
"""

import argparse
import glob
import json
import os
import time
import zlib

import numpy as np
import torch

from avianz_hpc import annotations, manifest
from avianz_hpc.features import (FeatureSpec, InputPreparer, SpectrogramExtractor,
                                 window_starts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class LoadedModel:
    """A trained model plus everything needed to feed it."""

    def __init__(self, path, device, checkpoint=None, spec_overrides=None):
        config_path, ckpt_path = self._locate(path, checkpoint)
        with open(config_path) as f:
            self.config = json.load(f)
        self.spec = FeatureSpec.from_model_config(self.config)
        if spec_overrides and "feature_spec" not in self.config:
            for k, v in spec_overrides.items():
                if v is not None:
                    setattr(self.spec, k, v)
        self.class_names = list(self.config["class_names"])
        self.kbird_prior = self.config.get("kbird_prior", 0.0)
        self.name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        self.model = self._build(self.config).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = device
        print(f"Loaded {self.config.get('model_type')} '{self.name}' from {ckpt_path}: "
              f"{len(self.class_names)} classes, spec={self.spec}")

    @staticmethod
    def _locate(path, checkpoint):
        if os.path.isdir(path):
            configs = sorted(glob.glob(os.path.join(path, "*_model_config.json")))
            if len(configs) != 1:
                raise FileNotFoundError(f"Expected one *_model_config.json in {path}, found {configs}")
            config_path = configs[0]
        else:
            config_path = path
        if checkpoint is None:
            prefix = config_path[: -len("_config.json")]
            best, final = prefix + "_best.pt", prefix + ".pt"
            checkpoint = best if os.path.exists(best) else final
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return config_path, checkpoint

    @staticmethod
    def _build(cfg):
        from avianz_hpc.core.models import AST, RegNetModel

        n = cfg["num_classes"]
        if cfg.get("model_type", "AST").lower() == "regnet":
            return RegNetModel(
                n, pretrained_path=None,
                model_name=cfg.get("model_name", "regnety_008"),
                use_cnn_adapter=cfg.get("use_cnn_adapter", False),
                use_sed_head=cfg.get("use_sed_head", False),
                use_gated_head=cfg.get("use_gated_head", False),
                in_chans=cfg.get("in_chans", 1),
            )
        input_size = (cfg["freq_bins"], cfg["time_bins"])
        model = AST(n, input_size=input_size, dropout=0.0,
                    use_reconstruction=cfg.get("use_reconstruction", False),
                    use_adapters=cfg.get("use_adapters", False),
                    per_chunk_norm=cfg.get("per_chunk_norm", False),
                    use_cnn_adapter=cfg.get("use_cnn_adapter", False),
                    use_sed_head=cfg.get("use_sed_head", False))
        model.interpolate_pos_embed(input_size)
        return model

    @torch.no_grad()
    def predict(self, x, batch_size):
        """Probabilities for a (N, C, H, W) tensor, matching Trainer._get_probs."""
        out = []
        for i in range(0, len(x), batch_size):
            logits = self.model(x[i:i + batch_size].to(self.device, non_blocking=True))
            if isinstance(logits, tuple):
                logits = logits[0]
            p = torch.sigmoid(logits)
            if self.kbird_prior > 0:
                total = p.sum(dim=1, keepdim=True).clamp(min=1e-7)
                p = (p / (total / self.kbird_prior).clamp(min=1.0)).clamp(0.0, 1.0)
            out.append(p.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)), np.float32)


# ---------------------------------------------------------------------------
# Feature extraction (runs in DataLoader worker processes)
# ---------------------------------------------------------------------------

class RecordingWindows(torch.utils.data.Dataset):
    """One item per recording: its window times and model inputs per model."""

    def __init__(self, entries, models, window_seconds, window_hop):
        self.entries = entries
        # (spec, model_config) per model; extractors/preparers are built lazily
        # inside each worker process.
        self.model_specs = [(m.spec, m.config) for m in models]
        self.window_seconds = window_seconds
        self.window_hop = window_hop
        self._tools = None

    def _setup(self):
        tools, by_spec = [], {}
        for spec, cfg in self.model_specs:
            key = json.dumps(spec.to_dict(), sort_keys=True)
            if key not in by_spec:
                by_spec[key] = SpectrogramExtractor(spec)
            tools.append((key, by_spec[key], InputPreparer.from_model_config(spec, cfg)))
        self._tools = tools

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self._tools is None:
            self._setup()
        root, rel = self.entries[idx]
        path = os.path.join(root, rel)
        # bg_subtract resamples outliers randomly; seed per recording for reproducibility
        np.random.seed(zlib.crc32(rel.encode()) & 0xFFFFFFFF)
        try:
            audio_cache, inputs, starts, duration = {}, [], None, None
            for key, extractor, prepare in self._tools:
                if key not in audio_cache:
                    audio_cache[key] = extractor.load(path)
                audio = audio_cache[key]
                sr = extractor.spec.sample_rate
                if starts is None:
                    duration = len(audio) / sr
                    starts = window_starts(duration, self.window_seconds, self.window_hop)
                xs = []
                for s in starts:
                    sg = extractor.extract(audio[int(s * sr):int((s + self.window_seconds) * sr)])
                    if sg is None:  # silent/degenerate slice: zeros = padding value
                        sg = np.zeros((extractor.spec.freq_bins, 1), np.float32)
                    xs.append(prepare(sg))
                inputs.append(torch.stack(xs))
            return {"rel": rel, "starts": starts, "duration": duration, "inputs": inputs, "error": None}
        except Exception as e:
            return {"rel": rel, "starts": None, "duration": None, "inputs": None, "error": repr(e)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def merge_windows(starts, window_seconds, probs, threshold):
    """Merge consecutive/overlapping windows at or above threshold into segments."""
    segs = []
    for s, p in zip(starts, probs):
        if p < threshold:
            continue
        e = s + window_seconds
        if segs and s <= segs[-1][1] + 1e-6:
            segs[-1][1] = max(segs[-1][1], e)
            segs[-1][2] = max(segs[-1][2], p)
        else:
            segs.append([s, e, p])
    return segs


def write_csv(path, starts, window_seconds, duration, columns):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = list(columns)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(",".join(["start", "end"] + [f'"{n}"' for n in names]) + "\n")
        for i, s in enumerate(starts):
            vals = [columns[n][i] for n in names]
            f.write(",".join([f"{s:.3f}", f"{min(s + window_seconds, duration):.3f}"]
                             + ["" if np.isnan(v) else f"{v:.4f}" for v in vals]) + "\n")
    os.replace(path + ".tmp", path)


def parse_thresholds(text, default):
    if not text:
        return {}, default
    if os.path.isfile(text):
        with open(text) as f:
            per = json.load(f)
    else:
        per = json.loads(text)
    return per, per.pop("default", default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    shard, num_shards = manifest.resolve_shard(args)
    if args.detector is None and args.classifier is None:
        raise SystemExit("Give --classifier and/or --detector")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    print(f"Device: {device}")

    overrides = {"sg_type": args.sg_type, "window_type": args.window_type, "sg_scale": args.sg_scale}
    detector = LoadedModel(args.detector, device, spec_overrides=overrides) if args.detector else None
    classifier = LoadedModel(args.classifier, device, spec_overrides=overrides) if args.classifier else None
    models = [m for m in (detector, classifier) if m is not None]

    window_seconds = args.window_seconds or models[-1].spec.clip_seconds
    window_hop = args.window_hop or window_seconds / 2
    for m in models:
        if window_seconds > m.spec.time_bins * m.spec.hop_seconds + 1e-6:
            raise SystemExit(f"--window-seconds {window_seconds} is longer than {m.name}'s input "
                             f"({m.spec.time_bins} bins x {m.spec.hop_seconds}s); it would be cropped")
    cls_thr, cls_default = parse_thresholds(args.thresholds, args.threshold)
    filter_name = args.filter_name or (classifier or detector).name

    entries = manifest.select(manifest.read(args.manifest), shard, num_shards)
    pred_root = os.path.join(args.output, "predictions")
    ann_root = os.path.join(args.output, "annotations")
    todo = [e for e in entries
            if args.overwrite or not os.path.exists(os.path.join(ann_root, e[1] + ".data"))]
    print(f"Shard {shard}/{num_shards}: {len(entries)} recordings, {len(entries) - len(todo)} already done; "
          f"windows {window_seconds}s every {window_hop}s")

    loader = torch.utils.data.DataLoader(
        RecordingWindows(todo, models, window_seconds, window_hop),
        batch_size=None, shuffle=False, num_workers=args.num_workers,
        persistent_workers=False, prefetch_factor=2 if args.num_workers else None,
    )

    os.makedirs(args.output, exist_ok=True)
    failed_log = os.path.join(args.output, f"failed_shard{shard:04d}.txt")
    n_done, n_failed, t0 = 0, 0, time.time()
    for item in loader:
        rel = item["rel"]
        if item["error"]:
            n_failed += 1
            print(f"  ! {rel}: {item['error']}")
            with open(failed_log, "a") as f:
                f.write(f"{rel}\t{item['error']}\n")
            continue
        starts, duration, inputs = item["starts"], item["duration"], item["inputs"]
        columns, detections = {}, []

        gate = np.ones(len(starts), dtype=bool)
        if detector is not None:
            det = detector.predict(inputs[0], args.batch_size)
            gate = det.max(axis=1) >= args.detector_threshold
            for j, name in enumerate(detector.class_names):
                columns[f"detector:{name}"] = det[:, j]
                if classifier is None:
                    for s, e, p in merge_windows(starts, window_seconds, det[:, j], args.detector_threshold):
                        detections.append((s, min(e, duration), name, p))

        if classifier is not None:
            probs = np.full((len(starts), len(classifier.class_names)), np.nan, np.float32)
            if gate.any():
                probs[gate] = classifier.predict(inputs[-1][torch.from_numpy(gate)], args.batch_size)
            for j, name in enumerate(classifier.class_names):
                columns[name] = probs[:, j]
                thr = cls_thr.get(name, cls_default)
                p = np.nan_to_num(probs[:, j], nan=0.0)
                for s, e, pm in merge_windows(starts, window_seconds, p, thr):
                    detections.append((s, min(e, duration), name, pm))

        write_csv(os.path.join(pred_root, rel + ".csv"), starts, window_seconds, duration, columns)
        annotations.write_detections(os.path.join(ann_root, rel + ".data"), detections, duration, filter_name)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(todo):
            rate = n_done / (time.time() - t0)
            print(f"  {n_done}/{len(todo)} recordings ({rate * 60:.1f}/min), {n_failed} failed")

    print(f"Shard {shard} finished: {n_done} processed, {n_failed} failed"
          + (f" (see {failed_log})" if n_failed else ""))


def main(argv=None):
    p = argparse.ArgumentParser(prog="avianz_hpc infer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="Manifest TSV from `avianz_hpc manifest`")
    p.add_argument("output", help="Output directory (shared by all shards)")
    p.add_argument("--classifier", help="Trained classifier: training output dir or *_model_config.json")
    p.add_argument("--detector", help="Trained detector: training output dir or *_model_config.json")
    p.add_argument("--detector-threshold", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5, help="Default per-species classifier threshold")
    p.add_argument("--thresholds", default=None,
                   help='Per-species thresholds: JSON file or string, e.g. \'{"Morepork": 0.7, "default": 0.5}\'')
    p.add_argument("--window-seconds", type=float, default=None,
                   help="Analysis window (default: the model's training clip length)")
    p.add_argument("--window-hop", type=float, default=None, help="Window step (default: half a window)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)) - 1,
                   help="Feature extraction processes (default: $SLURM_CPUS_PER_TASK - 1)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--filter-name", default=None,
                   help="Value of the label 'filter' field in .data output (default: model folder name)")
    p.add_argument("--overwrite", action="store_true", help="Reprocess recordings that already have output")
    g = p.add_argument_group("legacy models (configs without feature_spec, e.g. trained in model-testing)")
    g.add_argument("--sg-type", default=None)
    g.add_argument("--window-type", default=None)
    g.add_argument("--sg-scale", default=None)
    manifest.shard_args(p)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
