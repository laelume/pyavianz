"""Build training spectrograms from AviaNZ-annotated recordings (one shard).

Run one shard per Slurm array task, then ``merge-dataset`` once. Each shard
writes ``<out>/data/s<shard>_<n>.npy`` plus ``<out>/shards/shard_<shard>.json``,
and a finished shard is skipped on resubmission unless --overwrite is given.

Tasks
  classifier  multi-label species targets (one output per species)
  detector    every kept species collapsed into one label (default "bird"); use
              --species to make a detector for particular target species

Clip modes
  window   fixed windows over each annotated recording (default). A window is
           positive for a species when a verified segment overlaps it by at least
           --min-overlap seconds, and background when it overlaps no annotation at
           all. Partial overlaps are skipped as ambiguous. This matches how
           ``infer`` scans recordings.
  segment  one clip per verified segment, trimmed to the highest-energy
           time_bins (the model-testing behaviour), plus background windows.

Only human-verified labels are used by default (certainty 100, or manual
labels); see annotations.is_verified. Use --include-auto to also train on
unreviewed detector output, as model-testing did with certainty >= 50.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np

from avianz_hpc import annotations, manifest
from avianz_hpc.features import (FeatureSpec, SpectrogramExtractor, trim_to_length,
                                 window_starts)


def feature_args(p):
    d = FeatureSpec()
    g = p.add_argument_group("spectrogram features (stored in labels.json and the model config)")
    g.add_argument("--sample-rate", type=int, default=d.sample_rate)
    g.add_argument("--window-seconds", type=float, default=d.window_seconds, help="STFT window length")
    g.add_argument("--hop-seconds", type=float, default=d.hop_seconds, help="STFT hop")
    g.add_argument("--freq-bins", type=int, default=d.freq_bins)
    g.add_argument("--time-bins", type=int, default=d.time_bins)
    g.add_argument("--sg-type", default=d.sg_type, choices=["Standard", "Reassigned", "Multi-tapered"])
    g.add_argument("--window-type", default=d.window_type)
    g.add_argument("--sg-scale", default=d.sg_scale, choices=["Mel Frequency", "Bark Frequency", "Linear"])


def parse_remap(text):
    remap = {}
    for pair in filter(None, (text or "").split(",")):
        src, dst = pair.split(":", 1)
        remap[src.strip()] = dst.strip()
    return remap


def load_label_map(path, columns):
    src_col, dst_col = columns.split(",")
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src, dst = (row.get(src_col) or "").strip(), (row.get(dst_col) or "").strip()
            if src and dst:
                mapping[src] = dst
    return mapping


class Labeller:
    """Maps raw AviaNZ species names to training targets for the chosen task."""

    def __init__(self, args):
        self.task = args.task
        self.detector_label = args.detector_label
        self.min_certainty = args.min_certainty
        self.include_auto = args.include_auto
        self.mapping = load_label_map(args.label_map, args.label_map_columns) if args.label_map else {}
        self.mapping.update(parse_remap(args.label_remap))
        self.species = set(s.strip() for s in args.species.split(",")) if args.species else None

    def targets(self, segment):
        """Training labels this segment contributes (after mapping/filtering)."""
        out = set()
        for lab in segment.labels:
            if self.include_auto:
                ok = lab["species"] != annotations.DONT_KNOW and lab["certainty"] >= self.min_certainty
            else:
                ok = annotations.is_verified(lab, self.min_certainty)
            if not ok:
                continue
            name = self.mapping.get(lab["species"], lab["species"])
            if self.species is not None and name not in self.species:
                continue
            out.add(self.detector_label if self.task == "detector" else name)
        return out

    def has_unusable_label(self, segment):
        """True if the segment holds a sound we can't train on (unverified or
        Don't Know), so overlapping windows can't be used as background."""
        for lab in segment.labels:
            if lab["species"] == annotations.DONT_KNOW:
                return True
            if not self.include_auto and not annotations.is_verified(lab, self.min_certainty):
                return True
        return False


def label_window(segs, labeller, ws, we, min_overlap):
    """Return (labels, usable) for window [ws, we)."""
    labels, usable = set(), True
    for seg in segs:
        ov = min(we, seg.end_time) - max(ws, seg.start_time)
        if ov <= 0:
            continue
        need = min(min_overlap, max(seg.end_time - seg.start_time, 1e-3))
        tg = labeller.targets(seg)
        if tg and ov >= need - 1e-6:
            labels |= tg
        elif tg or labeller.has_unusable_label(seg):
            usable = False
        # verified segments of species filtered out by --species are fine as
        # negatives for the kept classes, so they don't make the window unusable
    return labels, usable


def build_shard(args):
    shard, num_shards = manifest.resolve_shard(args)
    shard_dir = os.path.join(args.output, "shards")
    data_dir = os.path.join(args.output, "data")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    shard_file = os.path.join(shard_dir, f"shard_{shard:04d}.json")
    if os.path.exists(shard_file) and not args.overwrite:
        print(f"Shard {shard} already built ({shard_file}); use --overwrite to rebuild")
        return

    spec = FeatureSpec(
        sample_rate=args.sample_rate, window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds, freq_bins=args.freq_bins, time_bins=args.time_bins,
        sg_type=args.sg_type, window_type=args.window_type, sg_scale=args.sg_scale,
        clip_mode=args.mode, clip_seconds=args.clip_seconds,
    )
    max_clip = spec.time_bins * spec.hop_seconds
    if spec.clip_seconds > max_clip + 1e-6:
        raise ValueError(f"--clip-seconds {spec.clip_seconds} exceeds time_bins*hop = {max_clip:.2f}s")
    clip_hop = args.clip_hop or spec.clip_seconds

    labeller = Labeller(args)
    extractor = SpectrogramExtractor(spec)
    rng = random.Random(f"{args.seed}-{shard}")

    entries = manifest.select(manifest.read(args.manifest), shard, num_shards)
    print(f"Shard {shard}/{num_shards}: {len(entries)} recordings, task={args.task}, mode={args.mode}")

    files, n_files_used, n_missing, n_failed = [], 0, 0, 0
    t0 = time.time()
    for i, (root, rel) in enumerate(entries):
        audio_path = os.path.join(root, rel)
        data_path = annotations.find_annotation(audio_path, args.annotation_root, rel)
        if data_path is None:
            n_missing += 1
            continue
        try:
            segs = annotations.load_segments(data_path)
            # Recordings without a single usable label are skipped entirely:
            # their silence may just mean nobody reviewed them.
            if not any(labeller.targets(s) for s in segs):
                continue
            audio = extractor.load(audio_path)
        except Exception as e:  # corrupt audio/annotation: log and move on
            print(f"  ! {rel}: {e}")
            n_failed += 1
            continue

        sr = spec.sample_rate
        duration = len(audio) / sr
        clips = []  # (start, end, labels, trim)

        if args.mode == "segment":
            for seg in segs:
                tg = labeller.targets(seg)
                if tg:
                    s, e = max(0.0, seg.start_time), min(duration, seg.end_time)
                    if e - s > spec.window_seconds:
                        clips.append((s, e, tg, True))

        for ws in window_starts(duration, spec.clip_seconds, clip_hop):
            we = min(ws + spec.clip_seconds, duration)
            labels, usable = label_window(segs, labeller, ws, we, args.min_overlap)
            if not usable:
                continue
            if labels and args.mode == "window":
                clips.append((ws, we, labels, False))
            elif not labels and args.background == "annotated-files":
                if rng.random() < args.background_fraction:
                    clips.append((ws, we, set(), False))

        for s, e, labels, trim in clips:
            sg = extractor.extract(audio[int(s * sr):int(e * sr)])
            if sg is None:
                n_failed += 1
                continue
            if trim:
                sg = trim_to_length(sg, spec.time_bins)
            name = f"s{shard:04d}_{len(files):07d}.npy"
            np.save(os.path.join(data_dir, name), sg)
            files.append({
                "filename": name,
                "class_names": sorted(labels),
                "source_root": root,
                "source_file": rel,
                "start_time": round(s, 3),
                "end_time": round(e, 3),
                "row_id": f"{rel}_{s:.2f}",
            })
        n_files_used += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(entries)} recordings, {len(files)} clips, {time.time() - t0:.0f}s")

    payload = {
        "feature_spec": spec.to_dict(),
        "task": args.task,
        "build_args": {k: v for k, v in vars(args).items() if k not in ("shard", "num_shards")},
        "stats": {"recordings_used": n_files_used, "no_annotation": n_missing, "failed": n_failed},
        "files": files,
    }
    tmp = shard_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, shard_file)
    n_bg = sum(1 for e in files if not e["class_names"])
    print(f"Shard {shard} done: {len(files)} clips ({n_bg} background) from {n_files_used} recordings; "
          f"{n_missing} without annotations, {n_failed} failures")


def main(argv=None):
    p = argparse.ArgumentParser(prog="avianz_hpc build-dataset", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="Manifest TSV from `avianz_hpc manifest`")
    p.add_argument("output", help="Dataset directory (shared by all shards)")
    p.add_argument("--annotation-root", default=None,
                   help="Where .data files live if not beside the audio (see annotations.annotation_candidates)")
    p.add_argument("--task", choices=["classifier", "detector"], default="classifier")
    p.add_argument("--detector-label", default="bird", help="Label used for --task detector")
    p.add_argument("--mode", choices=["window", "segment"], default="window")
    p.add_argument("--clip-seconds", type=float, default=None,
                   help="Window length (default: time_bins * hop = 10.24 s)")
    p.add_argument("--clip-hop", type=float, default=None, help="Window step (default: --clip-seconds)")
    p.add_argument("--min-overlap", type=float, default=0.5,
                   help="Seconds a call must overlap a window to label it (capped at the call length)")
    p.add_argument("--background", choices=["annotated-files", "none"], default="annotated-files",
                   help="Take unannotated windows from recordings that have verified labels as negatives")
    p.add_argument("--background-fraction", type=float, default=1.0,
                   help="Keep this fraction of background windows (merge-dataset can also cap them)")
    p.add_argument("--species", default=None,
                   help="Comma-separated names to keep (after mapping); others become negatives")
    p.add_argument("--label-map", default=None, help="CSV mapping AviaNZ names to training labels")
    p.add_argument("--label-map-columns", default="from,to",
                   help="Source,target column names in --label-map (e.g. CommonName,eBird)")
    p.add_argument("--label-remap", default=None, help='Inline renames, e.g. "Tui:Tui/Bellbird,Bellbird:Tui/Bellbird"')
    p.add_argument("--min-certainty", type=float, default=50)
    p.add_argument("--include-auto", action="store_true",
                   help="Also use unreviewed automatic labels with certainty >= --min-certainty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    feature_args(p)
    manifest.shard_args(p)
    build_shard(p.parse_args(argv))


if __name__ == "__main__":
    main()
