"""Command-line entry point: python -m avianz_hpc <command> [args]"""

import importlib
import sys

COMMANDS = {
    "manifest": ("avianz_hpc.manifest", "list recordings into a manifest TSV"),
    "build-dataset": ("avianz_hpc.build_dataset", "annotated recordings -> spectrogram clips (one shard)"),
    "merge-dataset": ("avianz_hpc.merge_dataset", "combine shards into labels.json + train/test split"),
    "train": ("avianz_hpc.train", "train a detector or classifier (GPU)"),
    "infer": ("avianz_hpc.infer", "run detector/classifier over recordings (one shard)"),
    "merge-predictions": ("avianz_hpc.merge_predictions", "collect inference outputs into tables"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m avianz_hpc <command> [args]\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:20s} {desc}")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)
    module = importlib.import_module(COMMANDS[sys.argv[1]][0])
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()
