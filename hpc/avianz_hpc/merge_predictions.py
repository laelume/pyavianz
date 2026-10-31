"""Collect per-recording inference outputs into tables.

Reads <out>/annotations/**/*.data written by `infer` and writes
  <out>/detections.csv      one row per detected segment
  <out>/summary.csv         per species: recordings with detections, segments, total seconds
and, with --windows, <out>/windows.csv concatenating every per-window score CSV.
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def iter_files(root, suffix):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(suffix):
                full = os.path.join(dirpath, name)
                yield full, os.path.relpath(full, root)[: -len(suffix)]


def main(argv=None):
    p = argparse.ArgumentParser(prog="avianz_hpc merge-predictions", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="Directory given to `infer`")
    p.add_argument("--windows", action="store_true", help="Also concatenate per-window scores")
    args = p.parse_args(argv)

    ann_root = os.path.join(args.output, "annotations")
    n_rec = 0
    per_species = defaultdict(lambda: {"recordings": set(), "segments": 0, "seconds": 0.0})
    with open(os.path.join(args.output, "detections.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recording", "start", "end", "species", "prob", "filter"])
        for path, rel in iter_files(ann_root, ".data"):
            n_rec += 1
            with open(path) as fh:
                annots = json.load(fh)
            for seg in annots[1:]:
                start, end, _, _, labels = seg
                for lab in labels:
                    w.writerow([rel, start, end, lab["species"], lab.get("prob", ""), lab.get("filter", "")])
                    s = per_species[lab["species"]]
                    s["recordings"].add(rel)
                    s["segments"] += 1
                    s["seconds"] += end - start

    with open(os.path.join(args.output, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "recordings_with_detections", "recordings_total", "segments", "seconds"])
        for sp in sorted(per_species):
            s = per_species[sp]
            w.writerow([sp, len(s["recordings"]), n_rec, s["segments"], round(s["seconds"], 1)])

    if args.windows:
        out_path = os.path.join(args.output, "windows.csv")
        header = None
        with open(out_path, "w", encoding="utf-8") as out:
            for path, rel in iter_files(os.path.join(args.output, "predictions"), ".csv"):
                with open(path, encoding="utf-8") as fh:
                    first = fh.readline().rstrip("\n")
                    if header is None:
                        header = first
                        out.write("recording," + header + "\n")
                    elif first != header:
                        raise ValueError(f"{path} has different columns; were shards run with different models?")
                    for line in fh:
                        out.write(f'"{rel}",{line}')

    print(f"{n_rec} recordings, {sum(s['segments'] for s in per_species.values())} detected segments")
    for sp in sorted(per_species):
        print(f"  {sp:40s} {len(per_species[sp]['recordings'])} recordings, {per_species[sp]['segments']} segments")


if __name__ == "__main__":
    main()
