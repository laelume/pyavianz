"""End-to-end checks on synthetic recordings (no GPU, no timm needed).

    python hpc/tests/test_pipeline.py        # or: pytest hpc/tests

Builds a tiny dataset from generated audio + AviaNZ annotations, checks the
training and inference input pipelines agree exactly, then runs `infer` with a
stand-in network in place of RegNet. Scratch output goes to hpc/tests/_tmp.
"""

import json
import os
import shutil
import sys
import zlib

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from avianz_hpc import annotations, build_dataset, infer, manifest, merge_dataset, merge_predictions  # noqa: E402
from avianz_hpc._avianz import Segment, SegmentList  # noqa: E402
from avianz_hpc.data.data_utils import SpectrogramDataset  # noqa: E402
from avianz_hpc.features import FeatureSpec, InputPreparer, window_starts  # noqa: E402

TMP = os.path.join(HERE, "_tmp")
SR = 16000  # deliberately != FeatureSpec sample rate, so resampling is exercised
SMALL = ["--freq-bins", "64", "--time-bins", "256", "--clip-seconds", "2.0"]


def make_recording(path, calls, duration=12.0, reviewed=True):
    """Noise plus tone bursts; calls = [(start, end, species, freq)]."""
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    t = np.arange(int(duration * SR)) / SR
    audio = 0.01 * rng.standard_normal(len(t))
    segs = SegmentList()
    segs.metadata = {"Operator": "Tester", "Reviewer": "Tester" if reviewed else "", "Duration": duration}
    for start, end, species, freq in calls:
        m = (t >= start) & (t < end)
        audio[m] += 0.3 * np.sin(2 * np.pi * freq * t[m])
        label = {"species": species, "certainty": 100 if reviewed else 50}
        if not reviewed:
            label.update(filter="SomeFilter", calltype="c1")
        segs.addSegment(Segment(start, end, 0, 0, [label]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.astype(np.float32), SR)
    segs.saveJSON(path + ".data")


def setup_audio():
    shutil.rmtree(TMP, ignore_errors=True)
    audio = os.path.join(TMP, "audio")
    for i in range(4):
        make_recording(os.path.join(audio, f"site{i % 2}", f"rec{i}.wav"),
                       [(1.0, 3.0, "Morepork", 800), (6.5, 7.5, "Kiwi (Nth Is Brown)", 2500)])
    make_recording(os.path.join(audio, "site0", "unreviewed.wav"), [(2.0, 4.0, "Morepork", 800)], reviewed=False)
    make_recording(os.path.join(audio, "site1", "short.wav"), [(0.2, 0.8, "Morepork", 800)], duration=1.5)
    return audio


def test_window_starts():
    assert window_starts(1.0, 2.0, 1.0) == [0.0]
    assert window_starts(5.0, 2.0, 2.0) == [0.0, 2.0, 3.0]
    assert window_starts(6.0, 2.0, 2.0) == [0.0, 2.0, 4.0]


def test_label_window():
    class A:
        task, detector_label, min_certainty, include_auto = "classifier", "bird", 50, False
        label_map, label_map_columns, label_remap, species = None, "from,to", None, None
    lab = build_dataset.Labeller(A)
    segs = [Segment(1.0, 3.0, 0, 0, [{"species": "Morepork", "certainty": 100}]),
            Segment(5.0, 5.2, 0, 0, [{"species": "Don't Know", "certainty": 0}])]
    assert build_dataset.label_window(segs, lab, 0, 2, 0.5) == ({"Morepork"}, True)
    assert build_dataset.label_window(segs, lab, 2.8, 4.8, 0.5) == (set(), False)   # 0.2 s overlap: ambiguous
    assert build_dataset.label_window(segs, lab, 4.5, 6.5, 0.5) == (set(), False)   # Don't Know
    assert build_dataset.label_window(segs, lab, 7, 9, 0.5) == (set(), True)        # background


def test_build_merge_and_parity():
    audio = setup_audio()
    man = os.path.join(TMP, "manifest.tsv")
    manifest.main([audio, "-o", man])
    ds = os.path.join(TMP, "dataset")
    for shard in range(2):
        build_dataset.main([man, ds, "--shard", str(shard), "--num-shards", "2"] + SMALL)
    merge_dataset.main([ds, "--test-fraction", "0.25"])

    with open(os.path.join(ds, "labels.json")) as f:
        labels = json.load(f)
    assert labels["categories"] == ["Kiwi (Nth Is Brown)", "Morepork"], labels["categories"]
    assert labels["feature_spec"]["freq_bins"] == 64
    sources = {e["source_file"] for e in labels["files"]}
    assert not any("unreviewed" in s for s in sources), "unreviewed auto labels must be skipped"
    assert labels["background_count"] > 0
    with open(os.path.join(ds, "train", "labels.json")) as f:
        train = {e["source_file"] for e in json.load(f)["files"]}
    with open(os.path.join(ds, "test", "labels.json")) as f:
        test = {e["source_file"] for e in json.load(f)["files"]}
    assert train and test and not train & test, "split must be grouped by recording"

    # Training (SpectrogramDataset, eval mode) and inference (InputPreparer) inputs must match.
    spec = FeatureSpec.from_dict(labels["feature_spec"])
    files = [os.path.join(ds, "data", e["filename"]) for e in labels["files"][:5]]
    y = np.zeros((len(files), 2), np.float32)
    for kw in ({}, {"bg_subtract": True}, {"spec_transform": "PCEN", "use_deltas": True}):
        ds_eval = SpectrogramDataset(files, y, spec.freq_bins, spec.time_bins, 1, cropping_mode="center",
                                     noise_ratio=0.0, training=False, use_temporal_roll=False,
                                     **{"spec_transform": "Log", **kw})
        prep = InputPreparer(spec, **{"spec_transform": "Log", **kw})
        for i, f in enumerate(files):
            np.random.seed(i)
            a = ds_eval[i][0]
            np.random.seed(i)
            b = prep(np.load(f))
            assert a.shape == b.shape == ((3 if kw.get("use_deltas") else 1), 64, 256), a.shape
            assert torch.equal(a, b), f"pipelines differ for {kw}"


def test_detector_segment_mode():
    man = os.path.join(TMP, "manifest.tsv")
    ds = os.path.join(TMP, "dataset_det")
    build_dataset.main([man, ds, "--task", "detector", "--mode", "segment", "--species", "Morepork",
                        "--shard", "0", "--num-shards", "1"] + SMALL)
    merge_dataset.main([ds, "--test-fraction", "0"])
    with open(os.path.join(ds, "labels.json")) as f:
        labels = json.load(f)
    assert labels["categories"] == ["bird"] and labels["task"] == "detector"
    positives = [e for e in labels["files"] if e["class_names"]]
    # one clip per verified Morepork segment (4 reviewed recordings + short.wav)
    assert len(positives) == 5, len(positives)
    assert all(e["end_time"] - e["start_time"] <= 2.0 + 1e-6 for e in positives)


class Stand_in(torch.nn.Module):
    """Mean energy in two frequency bands -> 2 logits. Replaces RegNet in tests."""

    def __init__(self, n):
        super().__init__()
        self.lin = torch.nn.Linear(2, n)

    def forward(self, x):
        h = x.shape[2]
        return self.lin(torch.stack([x[:, :, : h // 2].mean((1, 2, 3)), x[:, :, h // 2:].mean((1, 2, 3))], 1))


def test_infer_and_merge():
    ds = os.path.join(TMP, "dataset")
    with open(os.path.join(ds, "labels.json")) as f:
        labels = json.load(f)
    model_dir = os.path.join(TMP, "models", "toy_cls")
    os.makedirs(model_dir, exist_ok=True)
    cfg = {"model_type": "RegNet", "num_classes": 2, "class_names": labels["categories"],
           "feature_spec": labels["feature_spec"], "freq_bins": 64, "time_bins": 256,
           "spec_transform": "Log", "bg_subtract": False}
    with open(os.path.join(model_dir, "regnet_model_config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(0)
    torch.save(Stand_in(2).state_dict(), os.path.join(model_dir, "regnet_model_best.pt"))

    orig = infer.LoadedModel._build
    infer.LoadedModel._build = staticmethod(lambda c: Stand_in(c["num_classes"]))
    try:
        out = os.path.join(TMP, "inference")
        man = os.path.join(TMP, "manifest.tsv")
        args = [man, out, "--classifier", model_dir, "--threshold", "0.0", "--num-workers", "0", "--device", "cpu"]
        infer.main(args + ["--shard", "0", "--num-shards", "1"])
        n_rec = len(manifest.read(man))
        data_files = [p for p, _ in merge_predictions.iter_files(os.path.join(out, "annotations"), ".data")]
        assert len(data_files) == n_rec
        segs = annotations.load_segments(data_files[0])
        assert segs and segs[0].labels[0]["certainty"] == 50 and "prob" in segs[0].labels[0]
        # resubmission skips finished recordings
        mtime = os.path.getmtime(data_files[0])
        infer.main(args + ["--shard", "0", "--num-shards", "1"])
        assert os.path.getmtime(data_files[0]) == mtime
        merge_predictions.main([out, "--windows"])
        assert os.path.getsize(os.path.join(out, "detections.csv")) > 0
        # infer output is reviewable AviaNZ data: build-dataset must ignore it until confirmed
        ann = os.path.join(out, "annotations")
        rel = os.path.relpath(data_files[0], ann)[: -len(".data")]
        assert annotations.find_annotation("/elsewhere/" + rel, ann, rel) == data_files[0]
        assert not any(annotations.is_verified(lab) for seg in segs for lab in seg.labels)
    finally:
        infer.LoadedModel._build = orig


if __name__ == "__main__":
    test_window_starts()
    test_label_window()
    test_build_merge_and_parity()
    test_detector_segment_mode()
    test_infer_and_merge()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nALL TESTS PASSED")
