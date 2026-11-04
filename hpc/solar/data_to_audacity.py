#!/usr/bin/env python3
"""Make Audacity label-track copies of AviaNZ .data annotation files.

Reads every .data file under INPUT (a folder, searched recursively, or a single
file) and writes an Audacity label file for each into a separate output tree,
mirroring the folder structure:

  INPUT/site1/rec.wav.data  ->  OUTPUT/site1/rec.txt

The .data files and audio are never modified. In Audacity, open the recording,
then File > Import > Labels... and choose the matching .txt.

Label format (tab-separated, times in seconds from the start of the recording):
  start  end  Species | calltype | certainty
  \\     low_hz  high_hz        (only when the annotation has a frequency range)
A segment with several labels becomes one Audacity label per species.

Usage:
  python data_to_audacity.py D:\\nibk_test                       # -> D:\\nibk_test_audacity
  python data_to_audacity.py results/ -o labels/ --species "Kiwi (Nth Is Brown)"
  python data_to_audacity.py results/ --no-freq --min-certainty 90
"""

import argparse
import json
import os
import sys


def read_segments(path):
    """Segments [start, end, freq_low, freq_high, labels] from an AviaNZ .data file."""
    with open(path, encoding="utf-8") as f:
        annots = json.load(f)
    segs = []
    for i, a in enumerate(annots):
        # metadata: a dict (current format) or a list starting with -1 (old format)
        if isinstance(a, dict) or (i == 0 and isinstance(a, list) and a and a[0] == -1):
            continue
        if isinstance(a, list) and len(a) == 5:
            labels = a[4]
            if isinstance(labels, str):  # very old files stored a bare species name
                labels = [{"species": labels, "certainty": 100}]
            segs.append((float(a[0]), float(a[1]), float(a[2]), float(a[3]), labels or []))
    return sorted(segs, key=lambda s: s[0])


def label_text(lab):
    parts = [str(lab.get("species", "?"))]
    if lab.get("calltype"):
        parts.append(str(lab["calltype"]))
    if "certainty" in lab:
        parts.append(f"{lab['certainty']:g}")
    return " | ".join(parts)


def convert(data_path, out_path, args):
    """Write one Audacity label file; returns the number of labels written."""
    lines = []
    for start, end, flow, fhigh, labels in read_segments(data_path):
        for lab in labels:
            if not isinstance(lab, dict):
                continue
            if args.species and lab.get("species") not in args.species:
                continue
            if lab.get("certainty", 0) < args.min_certainty:
                continue
            lines.append(f"{start:.6f}\t{end:.6f}\t{label_text(lab)}")
            # Audacity spectral selection line; AviaNZ uses 0,0 for full-band segments
            if not args.no_freq and fhigh > 0 and fhigh > flow:
                lines.append(f"\\\t{flow:.6f}\t{fhigh:.6f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return sum(1 for l in lines if not l.startswith("\\"))


def output_name(data_name):
    """rec.wav.data -> rec.txt ; rec.data -> rec.txt"""
    stem = data_name[: -len(".data")]
    root, ext = os.path.splitext(stem)
    return (root if ext.lower() in (".wav", ".flac", ".bmp") else stem) + ".txt"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Folder of .data files (searched recursively) or a single .data file")
    p.add_argument("-o", "--output", default=None, help="Output folder (default: <input>_audacity)")
    p.add_argument("--species", action="append", default=None,
                   help="Only export this species (repeat for several)")
    p.add_argument("--min-certainty", type=float, default=0, help="Skip labels below this certainty")
    p.add_argument("--no-freq", action="store_true", help="Do not write frequency ranges")
    p.add_argument("--skip-annotations-dir", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip the 'annotations' record folder inside solar_survey results (default: on)")
    args = p.parse_args(argv)

    src = os.path.abspath(args.input)
    if os.path.isfile(src):
        jobs = [(src, os.path.join(args.output or os.path.dirname(src), output_name(os.path.basename(src))))]
    elif os.path.isdir(src):
        out_root = os.path.abspath(args.output or src.rstrip("\\/") + "_audacity")
        if os.path.commonpath([out_root, src]) == src:
            sys.exit("Output folder must be outside the input folder")
        jobs = []
        for root, dirs, files in os.walk(src):
            dirs.sort()
            if args.skip_annotations_dir and root == src and "annotations" in dirs:
                dirs.remove("annotations")
            for name in sorted(files):
                if name.endswith(".data"):
                    rel = os.path.relpath(root, src)
                    jobs.append((os.path.join(root, name), os.path.join(out_root, rel, output_name(name))))
    else:
        sys.exit(f"Not found: {src}")

    n_files = n_labels = n_failed = 0
    for data_path, out_path in jobs:
        try:
            n_labels += convert(data_path, out_path, args)
            n_files += 1
        except (ValueError, OSError) as e:
            n_failed += 1
            print(f"  ! {data_path}: {e}")
    where = os.path.dirname(jobs[0][1]) if len(jobs) == 1 else (out_root if jobs else "-")
    print(f"Wrote {n_files} Audacity label files ({n_labels} labels) to {where}"
          + (f"; {n_failed} failed" if n_failed else ""))


if __name__ == "__main__":
    main()
