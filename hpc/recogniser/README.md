# Train an AviaNZ recogniser without the GUI

`train_recogniser.py` trains an AviaNZ recogniser: the wavelet or changepoint
detector stored as `Filters/<name>.txt`. It does what the GUI's *Build
recogniser* wizard does, by calling the same core AviaNZ functions. The wizard's
manual choices (call-type clustering, editing parameters, clicking the ROC
curve) become command-line options. Your annotations are never modified.

## 1. Prepare training data

A folder (subfolders allowed) of recordings, each with its AviaNZ annotation
beside it as `<recording>.wav.data`:

* **Annotate every target call in each file.** Unannotated time counts as "no
  call" when scoring thresholds.
* Only **human-verified** labels are used: certainty 100 (confirmed) or manual
  labels. Unreviewed recogniser output is ignored unless you pass `--include-auto`.
* **Call type:** with `--calltype Male`, only labels with `"calltype": "Male"` are
  used; add `--include-untyped` to also use labels that have no call type. Other
  calls of the species (e.g. Female) are treated as "not a male call".
* The minimum call length defaults to your **shortest** annotation. If you
  annotated whole calling bouts rather than single calls, set `--minlen`.
* Keep some recordings (ideally from different sites) **out** of training, to
  test the recogniser afterwards.

## 2. Train

On your PC (PowerShell, `avianz` conda environment). A small set takes about a
minute:
```powershell
D:\miniconda-windows\envs\avianz\python.exe D:\avianz\hpc\recogniser\train_recogniser.py D:\kiwi_train `
    --species "Kiwi (Nth Is Brown)" --calltype Male --include-untyped `
    --name "Kiwi (Nth Is Brown) Male v1" -o D:\recognisers
```
On Rāpoi (`source ~/avianz_env.sh` first), do the same with Linux paths. Use an
interactive `srun` session for small sets, or `sbatch` on `parallel` for large ones.

The run prints the ROC table: detection rate (TPR) and false-positive rate (FPR)
for each threshold tried. It then picks one, by default the point **closest to
the top-left** (the trade-off you'd usually click in the wizard). Alternatives:

* `--max-fpr 0.05`: highest detection rate with at most 5% false positives;
* `--threshold 0.27`: a specific threshold from the table.

Rerun with a different choice and `--overwrite`; training on a small set is quick.

Outputs in `-o`:

| File | Contents |
|---|---|
| `<name>.txt` | the recogniser (AviaNZ filter format, changepoint method by default) |
| `<name>_ROC.json` | the ROC data |
| `<name>_training_log.json` | the exact command, all options, derived parameters, the chosen point, the training files, and a checksum of the `.txt` |
| `work_<name>/` | working copy of the recordings plus `-GT.txt` presence files (safe to delete) |

The TPR/FPR are measured on the **training** data, so they're optimistic. Test
on held-out recordings before trusting them.

## 3. Use it

Copy `<name>.txt` into AviaNZ's Filters folder, then use the name with `-r`:

* PC: `%APPDATA%\AviaNZ\Filters\`
* Rāpoi: `~/.avianz/Filters/` (created on first AviaNZ run), and for the SoLAR survey:
  `sbatch ~/avianz/hpc/solar/solar_survey.sbatch ~/folders.txt -r "Kiwi (Nth Is Brown) Male v1"`

Without a verifier network, every detection gets certainty 50. To add one,
train it on top of this recogniser with AviaNZ's own command-line training:
```bash
python AviaNZ.py -c -t -d <manually annotated folder> -e <reviewed folder> -r "Kiwi (Nth Is Brown) Male v1" -x <width s>
```
That run rewrites the recogniser `.txt` in the Filters folder to point at the
new network, so keep a copy of the original.

## Notes

* `--method chp` (changepoint, AviaNZ's current default) is used unless you pass
  `--method wv`. The `wv` path follows the wizard but is untested here.
* Tested locally on a small stand-in set made from the bundled sample recording.
  It checked that training runs, the output loads as an AviaNZ recogniser, and
  AviaNZ's batch processor detects with it. Real accuracy depends on your data.
