# Changes: model-testing → hpc/avianz_hpc

Record of everything added or changed when porting `model-testing/model-testing`
into `hpc/` (2026-10-31). Nothing under `model-testing/` or `avianz/` was
modified. Outside `hpc/`, the only edits are three `.gitignore` lines and an
entry in the root `Changelog.md`.

## Files copied from model-testing (vendored)

| hpc/avianz_hpc/ | from model-testing/model-testing/ | Edits |
|---|---|---|
| `core/config.py` | `src/core/config.py` | imports only |
| `core/trainer_config.py` | `src/core/trainer_config.py` | imports only |
| `core/utils.py` | `src/core/utils.py` | imports only |
| `core/models.py` | `src/core/models.py` | imports; `transformers` imported lazily inside `AST` |
| `core/model_trainer.py` | `src/core/model_trainer.py` | imports; GPU pick fix; model config contents (below) |
| `data/data_utils.py` | `src/data/data_utils.py` | imports; eval transform refactor; class names (below) |
| `data/normalizer.py` | `src/data/normalizer.py` | imports only |
| `evaluation/evaluation_utils.py` | `src/evaluation/evaluation_utils.py` | imports only |
| `evaluation/attention_viz.py` | `src/evaluation/attention_viz.py` | imports only |
| `train.py` | `train.py` | see below |
| `hpc/data/DOC_bird_naming_map.csv` | `data/DOC_bird_naming_map.csv` | none |

"imports" means `from src.X` → `from avianz_hpc.X`. The following were **not**
ported: `src/data/spectrogram.py`, `audio_loader.py`, `audio_data.py`,
`wavio.py`, `dpss.py`, and `ext/`; the AviaNZ originals in `avianz/src` are used
instead. Also not ported: `spectrogram_utils.py` (replaced by `features.py`);
`dataset_builder.py`, `data_pipeline.py`, and `src/experiments/*` (replaced by
`build_dataset.py`/`merge_dataset.py`); `src/evaluation/predict.py` (replaced by
`infer.py`); `base_trainer.py` (only used by DANN, which `Trainer` does not use);
and the BirdNET/Kaytoo comparison scripts and figure scripts.

## Behaviour changes to vendored code

**`core/model_trainer.py`**
- `_save_model` now also writes `feature_spec` (the spectrogram settings read
  from the training `labels.json`) and overwrites `spectrogram_params`,
  `sample_rate`, `window_seconds` and `hop_seconds` with those real values.
  *Why:* the original wrote `config.get_model_config()` defaults
  (`sgType: Reassigned`) even when the data was built with `--spec-type Standard`,
  so the config could not be trusted for inference.
- `_save_model` also writes `use_sed_head`, `in_chans`, `use_deltas`,
  `kbird_prior`, `use_adapters`, `per_chunk_norm`, `categories`.
  *Why:* these are needed to rebuild the model and reproduce its probabilities;
  without them, SED-head, delta-channel and adapter models could not be loaded.
- New `_dataset_feature_spec()` helper (reads and caches `labels.json`).
- When `CUDA_VISIBLE_DEVICES` is unset and `pick_free_gpu()` returns `None`, the
  variable is left unset. *Why:* the original set it to the string `"None"`,
  which hides every GPU. (Under Slurm `--gres` it is always set, so this path is rare.)

**`data/data_utils.py`**
- `SpectrogramDataset.__getitem__`: the transform/normalise block moved into
  `_transform_and_normalize()` (same code, same order).
- New `SpectrogramDataset.eval_transform(sg)`: the augmentation-free path of
  `__getitem__` for in-memory spectrograms, used by inference.
  `tests/test_pipeline.py` asserts it gives tensors identical to `__getitem__`
  in eval mode, for Log, bg-subtract, and PCEN+deltas.
- `DataLoader._get_class_names` returns the categories unchanged unless a mapping
  path is passed explicitly. *Why:* it used to read a **cwd-relative**
  `data/DOC_bird_naming_map.csv`, so class names depended on where the job was
  started, and AviaNZ species names had `_` replaced with spaces. Inference writes
  class names into `.data` files, so they must equal the training labels.

**`core/models.py`** — `from transformers import ASTModel` moved inside
`AST.__init__`, so RegNet-only environments don't need `transformers`.

**`train.py`** (now `python -m avianz_hpc train`)
- `--model-type` defaults to `regnet` (was `ast`). Every model-testing pipeline
  script passed `--model-type regnet`.
- Pretrained lookup: `--pretrained`, then `$AVIANZ_HPC_PRETRAINED`, then
  `hpc/pretrained/model_fold0.pth`. Files under 1 MB are treated as Git LFS
  pointers: an explicit `--pretrained` errors, and the auto-lookup warns and trains
  from scratch. *Why:* `BirdClefModels/model_fold0.pth` in model-testing is a
  133-byte LFS pointer, which the original passed to `torch.load`.
- `freq_bins`/`time_bins` are taken from the dataset's `feature_spec`.
- `main(argv=None)` so it can be called from the CLI dispatcher.

## New code

- `_avianz.py` — adds the repo root and `avianz/` to `sys.path` (AviaNZ uses
  both `avianz.src...` and bare `src...` imports) and re-exports AviaNZ's
  `Spectrogram`, `Segment`, `SegmentList`. Avoids installing the AviaNZ package
  and its PyQt6/TensorFlow dependencies on the cluster.
- `features.py` — `FeatureSpec`; `SpectrogramExtractor`, which reproduces
  model-testing's `SpectrogramProcessor` (per-clip RMS normalisation to 0.1,
  AviaNZ spectrogram, rot90, linear-scale bin downsampling ported from
  model-testing's spectrogram) and suppresses AviaNZ's per-call `mel` print;
  `InputPreparer`; `window_starts`; `trim_to_length` (same result as
  model-testing's `trim_spectrogram_to_length`, vectorised).
- `annotations.py` — finds `.data` files in AviaNZ's layouts; `is_verified`
  (certainty 100, or manual labels); writes detections as AviaNZ annotations
  (certainty 50, `calltype: "nn"`, `prob`).
- `manifest.py` — manifest TSV and strided array sharding.
- `build_dataset.py` / `merge_dataset.py` — sharded dataset building, with
  detector/classifier tasks, window/segment clip modes, verified-label filtering,
  label mapping, per-class caps, and a recording- or directory-grouped train/test split.
- `infer.py` — sharded, resumable sliding-window inference with detector →
  classifier gating, per-window CSVs and AviaNZ `.data` output. Loads
  model-testing configs too, with `--sg-type`/`--window-type`/`--sg-scale` overrides.
- `merge_predictions.py`, `__main__.py` (CLI dispatcher).
- `slurm/` — `env.sh`, `setup_env.sh`, six `.sbatch` job scripts,
  `submit_train.sh`, `submit_infer.sh`; `config/project.env.example`;
  `requirements-hpc.txt`; `tests/test_pipeline.py`; `README.md`; `pretrained/README.md`.

## Differences from model-testing results to be aware of

- **Stereo files:** AviaNZ's loader keeps the left channel, while model-testing's
  copy took the per-sample maximum across channels. This only matters for stereo recordings.
- **Resampling:** recordings are resampled once per file and then sliced (the
  same in build and inference). model-testing resampled each segment separately,
  so clip edges differ very slightly.
- **Default labels:** names stay as AviaNZ species names (no eBird mapping), and
  unreviewed automatic labels are excluded unless `--include-auto` is given.
- **Validation leakage:** the trainer's internal 80/20 validation split is
  still random across clips, so neighbouring windows of one recording can fall
  on both sides. It is used only for early stopping. The held-out `test/` split
  from `merge-dataset` is grouped by recording and is the number to report.

## Repository files outside hpc/

- `.gitignore`: added `hpc/config/project.env`, `hpc/tests/_tmp/`, `hpc/pretrained/*.pth`.
- `Changelog.md`: entry under Unreleased → Added.

## Later additions

- `hpc/solar/` (`solar_survey.py`, `solar_survey.sbatch`, `README.md`): runs an
  AviaNZ recogniser from Rāpoi over folders on the VUW acoustic drives (SoLAR).
  It copies one folder at a time to node-local `/tmp` with `smbclient`, runs AviaNZ
  batch mode (`-c -b ... -asd`), and copies the entire recordings with detections,
  with their annotations, to scratch. Separate from `avianz_hpc`: it uses
  AviaNZ's own recognisers and needs no trained models. Tested locally with
  `--local-source` and the real Kiwi (Nth Is Brown) recogniser; smbclient use on
  Rāpoi is untested.
  - Read-only guard: only `ls`, `du`, `cd`, `recurse`, `prompt`, `mask`, `lcd` and `mget`
    may be sent to SoLAR (`READ_ONLY_SMB_COMMANDS`); anything else is refused before connecting.
  - Progress: folder sizes are measured up front (`sizes.json`), AviaNZ output is
    streamed for per-file progress, and `status.json` is kept up to date;
    `solar_survey.py status [--watch]` shows a progress bar with an ETA.
- `hpc/recogniser/train_recogniser.py` (+ `README.md`): command-line (no GUI) port
  of the "Build recogniser" wizard's training. It uses the same core calls
  (`SegmentList.exportGT`, `WaveletSegment.waveletSegment_trainChp` /
  `waveletSegment_train`) and the same default parameters. Call-type clustering is
  replaced by `--calltype`, and clicking the ROC curve by a rule (default: nearest
  the top-left; or `--max-fpr`, `--threshold`). It works on a copy of the
  training data, refuses to overwrite an existing recogniser, and writes a
  training log with every setting. Tested locally on a stand-in set: training runs,
  and the output loads and detects in AviaNZ's batch processor.
  - `data_to_audacity.py`: copies AviaNZ `.data` annotations to Audacity label
    files (`start end label`, plus spectral-selection lines for frequency boxes)
    in a separate folder tree; the originals are not modified. Standard library only.
- `avianz/src/core/batch_processor.py`, `avianz/AviaNZ.py` (AviaNZ core, not part
  of `hpc/`): batch mode never writes `.data` next to the source audio. See the
  root `Changelog.md` (Changed). The old code paths are kept as commented-out
  `OPTION (disabled)` lines.
