# avianz_hpc — AviaNZ detectors and classifiers on Rāpoi

Train your own bird **detector** and species **classifier** from AviaNZ
annotations, then run them over large recording collections as Slurm array jobs
on Rāpoi (VUW's HPC). Detections come back as AviaNZ `.data` files, so they can
be reviewed in the AviaNZ GUI, and confirmed labels become new training data.

The training code is a port of the `model-testing` research pipeline (RegNetY /
AST multi-label models). Spectrograms and annotation I/O use the AviaNZ code in
this repository (`avianz/src/core/spectrogram.py`, `annotation.py`), so features
match what AviaNZ shows. See [CHANGES.md](CHANGES.md) for what changed in the port and why.

```
annotated recordings (.wav + .data)              new recordings
        │ manifest                                     │ manifest
        ▼                                              ▼
 build-dataset  (CPU array, one shard/task)      infer  (GPU array, one shard/task)
        │                                          │  detector ─► classifier on windows
        ▼                                          │             that pass the detector
 merge-dataset  → labels.json, train/, test/       ▼
        │         (split grouped by recording)   predictions/<rec>.csv  per-window scores
        ▼                                        annotations/<rec>.data AviaNZ segments
 train  (1 GPU) → <model>/regnet_model_best.pt     │
                  + regnet_model_config.json       ▼
                  (records the feature settings)  merge-predictions → detections.csv, summary.csv
                                                   │
                         review in AviaNZ GUI ◄────┘   confirmed labels (certainty 100)
                         └──────────► back into build-dataset
```

## Detector vs classifier

Both are the same multi-label model trained on different label sets:

| | `--task detector` | `--task classifier` |
|---|---|---|
| Labels | every kept species → one label (`bird`, or `--detector-label`) | one output per species |
| Use | cheap first pass: "is there a call here?" | which species are calling |
| Narrow it | `--species "Morepork"` → a Morepork detector | `--species "A,B,C"` → only these outputs |

At inference, the classifier runs only on windows where the detector's score is
at least `--detector-threshold`. Either model can also be used on its own.
(model-testing's `--gated-head` is a single-model alternative: a presence gate
and species head trained jointly; it is still available in `train`.)

## One-time setup on Rāpoi

```bash
ssh <user>@raapoi.vuw.ac.nz                      # VPN/wired network needed off campus
git clone <this repo> ~/avianz && cd ~/avianz    # code lives in home (50 GB, backed up)
cp hpc/config/project.env.example hpc/config/project.env
nano hpc/config/project.env                      # paths, modules, shard counts

# Build the venv on a compute node, not the login node
srun --partition=quicktest --cpus-per-task=4 --mem=8G --time=1:00:00 --pty bash
module spider Python          # pick a Python >= 3.10 module and put it in MODULES
bash hpc/slurm/setup_env.sh
exit
```

Then put the real BirdCLEF weights at `$AVIANZ_HPC_PRETRAINED` (see
[pretrained/README.md](pretrained/README.md); the copy in model-testing is a
Git LFS pointer). Copy audio to scratch with `rsync` from an interactive job,
not the login node. If you use `--model-type ast`, also uncomment
`transformers` in `requirements-hpc.txt` and run
`python -c "from transformers import ASTModel; ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')"`
once with network access so the checkpoint is cached under `$HF_HOME`.

## Train

```bash
# Classifier on two species
bash hpc/slurm/submit_train.sh kiwi_ruru classifier --species "Morepork,Kiwi (Nth Is Brown)"

# Generic bird detector; train args after "--"
bash hpc/slurm/submit_train.sh bird_det detector --background-fraction 0.5 -- --epochs 30 --patience 10

# Dataset already built: retrain with other settings
SKIP_BUILD=1 bash hpc/slurm/submit_train.sh kiwi_ruru classifier -- --bg-subtract --kbird-prior 2.0
```

This submits `manifest → build-dataset (array) → merge-dataset → train`, chained
with `afterok`. Outputs go to `$WORK/datasets/<name>` and `$WORK/models/<name>`,
and logs to `$WORK/logs`. Pass merge options as `MERGE_ARGS="--max-per-class 5000 --group-by dir"`
(`--group-by dir` keeps whole sites/recorders out of the training set when each
folder is one site).

**What becomes training data** (`build-dataset`):

* Only recordings with at least one usable label. Silence in a recording nobody
  reviewed is not evidence of absence.
* Only *verified* labels: certainty 100 (confirmed in review) or manual labels
  (`filter` absent or `"M"`). Raw recogniser output, including this package's
  own detections (certainty 50), is ignored until someone confirms it. Use
  `--include-auto` to reproduce model-testing's behaviour (any label with
  certainty ≥ 50).
* `--mode window` (default): fixed windows (`--clip-seconds`, default 10.24 s)
  over each recording. A window is positive for a species if a verified call
  overlaps it by ≥ `--min-overlap` s; background if it touches no annotation;
  skipped if the overlap is partial, or the window touches "Don't Know" or an
  unverified label. This matches how `infer` scans recordings.
* `--mode segment`: one clip per verified segment, trimmed to the
  highest-energy 10.24 s. This is how model-testing built its AviaNZ data.
* Annotations are found beside the audio (`rec.wav.data`), or under
  `ANNOTATION_ROOT` in any layout AviaNZ writes (the batch `-asd` tree, a mirrored
  tree, or a flat folder of `<stem>.data`).
* Labels: `--label-remap "Tui:Tui/Bellbird,Bellbird:Tui/Bellbird"` or
  `--label-map file.csv --label-map-columns from,to`. For eBird codes, use
  `--label-map hpc/data/DOC_bird_naming_map.csv --label-map-columns CommonName,eBird`.

`train` accepts every model-testing training option (`python -m avianz_hpc train -h`).
It defaults to RegNetY-008 and reads the input size from the dataset. It
evaluates on the held-out `test/` split and writes metrics and plots into the model folder.

## Run inference

```bash
bash hpc/slurm/submit_infer.sh survey_2025 \
    --detector   /nfs/scratch/$USER/avianz-hpc/models/bird_det \
    --classifier /nfs/scratch/$USER/avianz-hpc/models/kiwi_ruru \
    --detector-threshold 0.3 --thresholds '{"Morepork": 0.6, "default": 0.5}'
```

This scans `INFER_AUDIO_ROOTS` into a manifest, runs `INFER_SHARDS` array tasks
(at most `MAX_PARALLEL` at once; the cluster has 6 A100s), then
`merge-predictions`. Results in `$WORK/inference/survey_2025/`:

| Path | Contents |
|---|---|
| `annotations/<relpath>.data` | AviaNZ annotations; label `{"species", "certainty": 50, "filter": <model>, "calltype": "nn", "prob"}` |
| `predictions/<relpath>.csv` | per-window scores: `start,end,detector:bird,<species...>` (blank = classifier skipped by detector) |
| `detections.csv`, `summary.csv` | all segments; per-species recording/segment counts |
| `failed_shard*.txt` | unreadable recordings |

Windows default to the model's training clip length with 50% overlap
(`--window-seconds`, `--window-hop`). Consecutive windows above threshold merge
into one segment. A timed-out or failed shard can be resubmitted as-is, because
recordings that already have a `.data` file are skipped (`--overwrite` reprocesses them).
`INFER_DEVICE=cpu` runs on the `parallel` partition instead.

**Review loop:** copy `annotations/` over the audio tree (the names match
AviaNZ's `rec.wav.data` sidecars), or rsync both to your machine, and review in
AviaNZ. Confirm or correct the labels, then point `build-dataset` at the reviewed
files (`ANNOTATION_ROOT` = the annotations folder works too).

## Commands

All commands run as `python -m avianz_hpc <command> -h` once `hpc/slurm/env.sh`
is sourced (or with `hpc/` on `PYTHONPATH`).

| Command | Where | What |
|---|---|---|
| `manifest ROOT... -o M.tsv` | quicktest | list `.wav/.flac` recordings |
| `build-dataset M.tsv DS --shard i --num-shards n` | CPU array | spectrogram clips + shard JSON |
| `merge-dataset DS` | quicktest | `labels.json`, grouped `train/` + `test/` |
| `train DS/train MODEL --test-folder DS/test` | GPU | model + `*_model_config.json` |
| `infer M.tsv OUT --detector D --classifier C` | GPU/CPU array | per-recording CSV + `.data` |
| `merge-predictions OUT` | quicktest | `detections.csv`, `summary.csv` |

## Using model-testing checkpoints

`infer --classifier <model-testing output dir>` works, but configs written by
the original trainer record the **default** spectrogram type (`Reassigned`), not
the one the data was built with. The model-testing build scripts used
`Standard / Hamming / Mel Frequency`, so pass
`--sg-type Standard --window-type Hamming --sg-scale "Mel Frequency"`. Those
models were trained on segment clips, which may not transfer well to sliding windows.

## Layout

```
hpc/
  avianz_hpc/            Python package (python -m avianz_hpc ...)
    _avianz.py           puts the repo's AviaNZ code on sys.path (no GUI/TensorFlow deps)
    features.py          FeatureSpec, AviaNZ spectrogram extraction, model input prep
    annotations.py       find/read/write AviaNZ .data files
    manifest.py          recording lists + Slurm array sharding
    build_dataset.py, merge_dataset.py, train.py, infer.py, merge_predictions.py
    core/ data/ evaluation/   training code ported from model-testing/src
  slurm/                 env.sh, setup_env.sh, *.sbatch, submit_train.sh, submit_infer.sh
  config/project.env.example
  data/DOC_bird_naming_map.csv   optional AviaNZ/common-name → eBird mapping
  pretrained/            where BirdCLEF weights go (git-ignored)
  tests/test_pipeline.py synthetic end-to-end test (python hpc/tests/test_pipeline.py)
```

## Not yet verified on Rāpoi

The pipeline was tested locally on synthetic audio (CPU, with a stand-in network
because `timm` was not installed). Before relying on it, check:

* the module names in `MODULES` (`module spider Python`); Python ≥ 3.10 is needed for numpy 2.1;
* a short `train` run on the `gpu` partition (RegNet + `timm` was not run locally);
* throughput: spectrogram extraction is CPU-bound, so tune `--cpus-per-task` in
  `infer.sbatch` against GPU utilisation (`nvidia-smi` on the node via `srun --jobid`);
* per-user GPU/array limits (not stated in the Rāpoi docs).

Known limitation: outputs are keyed by path relative to each audio root, so two
roots containing the same relative path (e.g. both have `site1/rec.wav`) would
overwrite each other. Use one root per run, or roots with distinct subfolders.
