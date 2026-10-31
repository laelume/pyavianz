"""Recording manifests and Slurm array sharding.

A manifest is a TSV of ``root<TAB>relpath`` lines. Keeping the root separate lets
outputs mirror the input tree (``<out>/<relpath>``) even when several roots are
scanned. Every array task reads the same manifest and takes a strided slice, so
long and short recordings spread evenly across tasks.
"""

import argparse
import os

AUDIO_EXTS = (".wav", ".flac")


def scan(roots, exts=AUDIO_EXTS, require_annotation=False):
    entries = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Audio root not found: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if not name.lower().endswith(exts):
                    continue
                full = os.path.join(dirpath, name)
                if require_annotation and not os.path.isfile(full + ".data"):
                    continue
                entries.append((root, os.path.relpath(full, root)))
    return entries


def write(entries, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for root, rel in entries:
            f.write(f"{root}\t{rel}\n")


def read(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                root, rel = line.split("\t", 1)
                entries.append((root, rel))
    return entries


def shard_args(parser):
    """Add --shard/--num-shards, defaulting to the Slurm array environment."""
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (default: $SLURM_ARRAY_TASK_ID, else 0)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Number of shards (default: $SLURM_ARRAY_TASK_COUNT, else 1)")


def resolve_shard(args):
    shard = args.shard if args.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num = args.num_shards if args.num_shards is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if not 0 <= shard < num:
        raise ValueError(f"shard {shard} out of range for {num} shards (array indices must start at 0)")
    return shard, num


def select(entries, shard, num_shards):
    return entries[shard::num_shards]


def main(argv=None):
    p = argparse.ArgumentParser(prog="avianz_hpc manifest",
                                description="List audio files under one or more roots into a manifest TSV.")
    p.add_argument("roots", nargs="+", help="Directories to scan recursively")
    p.add_argument("-o", "--output", required=True, help="Manifest file to write")
    p.add_argument("--annotated-only", action="store_true",
                   help="Only list recordings with a sidecar <audio>.data file")
    args = p.parse_args(argv)

    entries = scan(args.roots, require_annotation=args.annotated_only)
    write(entries, args.output)
    print(f"Wrote {len(entries)} recordings to {args.output}")


if __name__ == "__main__":
    main()
