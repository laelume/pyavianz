"""Merge build-dataset shards into labels.json and a grouped train/test split.

Writes:
  <dataset>/labels.json         every clip (categories, feature_spec, counts)
  <dataset>/train/labels.json   training clips   (train/data -> ../data)
  <dataset>/test/labels.json    held-out clips   (test/data  -> ../data)

The split is grouped by recording (or by directory, e.g. one site/recorder per
folder), so windows of the same recording never land on both sides. Point
`train` at <dataset>/train and pass <dataset>/test as --test-folder. Note the
trainer still makes its own random 80/20 validation split inside train/, which
is only used for early stopping and model selection.
"""

import argparse
import glob
import json
import os
import random
import shutil
from collections import Counter, defaultdict


def load_shards(dataset):
    paths = sorted(glob.glob(os.path.join(dataset, "shards", "shard_*.json")))
    if not paths:
        raise FileNotFoundError(f"No shard files in {dataset}/shards")
    files, spec, task = [], None, None
    for p in paths:
        with open(p) as f:
            shard = json.load(f)
        if spec is None:
            spec, task = shard["feature_spec"], shard["task"]
        elif shard["feature_spec"] != spec or shard["task"] != task:
            raise ValueError(f"{p} was built with different settings from the other shards")
        files.extend(shard["files"])
    return files, spec, task, len(paths)


def cap_per_class(files, max_per_class, max_background, rng):
    """Keep at most max_per_class clips per class and max_background negatives."""
    order = files[:]
    rng.shuffle(order)
    counts, kept, n_bg = Counter(), [], 0
    for e in order:
        if not e["class_names"]:
            if max_background is None or n_bg < max_background:
                kept.append(e)
                n_bg += 1
        elif max_per_class is None or any(counts[c] < max_per_class for c in e["class_names"]):
            kept.append(e)
            counts.update(e["class_names"])
    return kept


def group_split(files, group_by, test_fraction, rng):
    groups = defaultdict(list)
    for e in files:
        key = e["source_file"] if group_by == "file" else os.path.dirname(e["source_file"])
        groups[(e.get("source_root", ""), key)].append(e)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_test = int(round(len(keys) * test_fraction))
    if test_fraction > 0 and n_test == 0 and len(keys) > 1:
        n_test = 1
    test_keys = set(keys[:n_test])
    train = [e for k in keys if k not in test_keys for e in groups[k]]
    test = [e for k in keys if k in test_keys for e in groups[k]]
    return train, test


def write_labels(folder, files, categories, spec, task, name):
    os.makedirs(folder, exist_ok=True)
    counts = Counter(c for e in files for c in e["class_names"])
    payload = {
        "dataset": name,
        "task": task,
        "feature_spec": spec,
        "categories": categories,
        "num_classes": len(categories),
        "species_counts": dict(sorted(counts.items())),
        "background_count": sum(1 for e in files if not e["class_names"]),
        "files": files,
    }
    with open(os.path.join(folder, "labels.json"), "w") as f:
        json.dump(payload, f, indent=1)
    return counts


def link_data(split_dir):
    """Give a split folder a data/ that points at the shared ../data."""
    link = os.path.join(split_dir, "data")
    if os.path.lexists(link):
        return
    try:
        os.symlink(os.path.join("..", "data"), link, target_is_directory=True)
    except OSError:
        # e.g. Windows without symlink rights: fall back to copying the split's files
        shutil.copytree(os.path.join(os.path.dirname(split_dir), "data"), link)


def main(argv=None):
    p = argparse.ArgumentParser(prog="avianz_hpc merge-dataset", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Dataset directory given to build-dataset")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="Fraction of recordings (or directories) held out for testing; 0 = no split")
    p.add_argument("--group-by", choices=["file", "dir"], default="file")
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--max-background", type=int, default=None)
    p.add_argument("--min-examples", type=int, default=1,
                   help="Drop classes with fewer clips than this (they become negatives)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    files, spec, task, n_shards = load_shards(args.dataset)
    print(f"Loaded {len(files)} clips from {n_shards} shards (task={task})")

    counts = Counter(c for e in files for c in e["class_names"])
    rare = {c for c, n in counts.items() if n < args.min_examples}
    if rare:
        print(f"Dropping {len(rare)} classes with < {args.min_examples} clips: {sorted(rare)}")
        for e in files:
            e["class_names"] = [c for c in e["class_names"] if c not in rare]
    categories = sorted(c for c in counts if c not in rare)

    files = cap_per_class(files, args.max_per_class, args.max_background, rng)
    files.sort(key=lambda e: e["filename"])

    write_labels(args.dataset, files, categories, spec, task, "all")
    train, test = group_split(files, args.group_by, args.test_fraction, rng)
    for split, split_files in (("train", train), ("test", test)):
        if not split_files:
            continue
        folder = os.path.join(args.dataset, split)
        split_counts = write_labels(folder, split_files, categories, spec, task, split)
        link_data(folder)
        n_bg = sum(1 for e in split_files if not e["class_names"])
        print(f"{split}: {len(split_files)} clips ({n_bg} background)")
        for c in categories:
            print(f"    {c:40s} {split_counts.get(c, 0)}")


if __name__ == "__main__":
    main()
