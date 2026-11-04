# SoLAR survey: run an AviaNZ recogniser over the acoustic drives

`solar_survey.py` runs an AviaNZ recogniser (default **Kiwi (Nth Is Brown)**)
over folders on the VUW acoustic drives (`//vuwstocoissrin1.vuw.ac.nz/ECS_acoustic_0N`)
from Rāpoi. It keeps the **entire** recordings that have detections, plus
their annotations, on Rāpoi scratch.

SoLAR is not mounted on Rāpoi, so it works one folder at a time:
`smbclient` copy → AviaNZ batch run → keep recordings with detections →
delete the staged copy → record the folder in `done.txt`. Nothing is written
to SoLAR. The script only sends read commands to the drives (`ls`, `du`, `cd`,
`mget`, ...), and refuses any other `smbclient` command, such as `put`, `mput`,
`mkdir` or `del`, before connecting (`READ_ONLY_SMB_COMMANDS` in `solar_survey.py`). Resubmitting the job skips finished folders and retries failed ones.

## One-time setup

Do steps 1–7 of the Rāpoi setup first: code under `~/avianz`, and the
`avianz-cli` venv, with `~/avianz_env.sh` and the compiled `ext/` modules.
Then copy `hpc/solar/` to `~/avianz/hpc/solar/`.

A batch job can't type a password, so `smbclient` reads one from a file:

```bash
cat > ~/.solar_auth <<'EOF'
username = <user>@staff.vuw.ac.nz
password = <your VUW password>
EOF
chmod 600 ~/.solar_auth        # the script refuses to run if others can read it
```

This file holds your password in plain text in your home folder; delete it
when you're done. The Rāpoi docs only show an interactive password prompt for
`smbclient`, so check that this is acceptable to Digital Solutions.

## Build the folder list

One SoLAR folder per line; the first part is the share. Each folder is copied
whole (recursively) to node-local `/tmp`, so list folders of tens to hundreds
of GB, not whole shares:

```bash
srun --partition=quicktest --time=1:00:00 --pty bash
source ~/avianz_env.sh
python ~/avianz/hpc/solar/solar_survey.py --auth-file ~/.solar_auth list ECS_acoustic_02 > ~/folders.txt
python ~/avianz/hpc/solar/solar_survey.py --auth-file ~/.solar_auth list ECS_acoustic_02/Remutaka20 >> ~/folders.txt
nano ~/folders.txt             # remove folders you don't want; '#' comments out a line
```

Before downloading, the script checks each folder's size with `smbclient du`
and skips any folder bigger than 90% of the free staging space; split those
into subfolders.

## Run

```bash
sbatch ~/avianz/hpc/solar/solar_survey.sbatch ~/folders.txt
squeue -u $USER
tail -f ~/solar-survey_<jobid>.out
```

### Watching progress

At the start the job measures every listed folder (cached in `sizes.json`), so
progress and the estimated finish time are based on the amount of audio, not
the number of folders. From the login node:

```bash
python ~/avianz/hpc/solar/solar_survey.py status            # one snapshot
python ~/avianz/hpc/solar/solar_survey.py status --watch    # refresh every 60 s; Ctrl+C to stop
```
```
[################------------------------]  41.3%   elapsed 3h05m   ETA 4h24m (Sun 02 Nov 03:12)
folders: 2 done, 0 failed, of 5 this run (+0 done in earlier runs)
recordings: 386 processed, 12 kept
now: [3/5] ECS_acoustic_02/Remutaka20/site3: detecting, file 57/193
updated 1m ago   job 123456
```
The job output also prints a `progress` line every 5 minutes. Downloads give no
progress updates, so a large folder can show "downloading" for a while. If
`updated` gets older than 30 minutes, check `squeue -u $USER`. The ETA assumes
the remaining audio processes at the average rate so far, so it settles after
the first folder.

### Options

Options go after the list, e.g. `--min-certainty 90` (keep only stronger
detections), `--wind` (AviaNZ wind filter), `-r "<other recogniser>"`.
The job runs on the `parallel` partition (CPU, up to 10 days), one folder after
another; the Rāpoi docs ask for few concurrent `smbclient` sessions. Resubmit
with the same command to continue.

## Results

`/nfs/scratch/$USER/solar_survey/Kiwi_Nth_Is_Brown/`:

| Path | Contents |
|---|---|
| `<share>/<folder>/<recording>.wav` | full recordings with ≥1 detection (copies) |
| `<share>/<folder>/<recording>.wav.data` | their annotations, next to the **copy** so AviaNZ can open and train on them |
| `annotations/<share>/<folder>/...` | annotations for every processed recording, including those with no detections |
| `summary.csv` | per recording: detections, max certainty, kept or not |
| `done.txt`, `failed.txt` | finished folders; failed folders with the reason |
| `logs/` | smbclient and AviaNZ output per folder |

## Audacity label copies

`data_to_audacity.py` writes an Audacity label file for every `.data` file into a
separate folder tree. `rec.wav.data` becomes `rec.txt`, and the originals are not
modified. In Audacity, open the recording, then *File → Import → Labels…*.

```bash
python ~/avianz/hpc/solar/data_to_audacity.py /nfs/scratch/$USER/solar_survey/Kiwi_Nth_Is_Brown
#  -> /nfs/scratch/$USER/solar_survey/Kiwi_Nth_Is_Brown_audacity/<share>/<folder>/<recording>.txt
```
Each label reads `Species | calltype | certainty`. If the annotation has a
frequency range, it's added as Audacity's spectral selection; full-band
segments get none. Options: `-o DIR`, `--species "Kiwi (Nth Is Brown)"`,
`--min-certainty 90`, `--no-freq`. The `annotations/` record folder in the
results is skipped by default, because the kept recordings already have their
annotations beside them.

## Not yet verified on Rāpoi

Tested locally with `--local-source`, which substitutes a local folder for
SoLAR, and the real AviaNZ recogniser. Still to check on Rāpoi:

* `smbclient` is installed on compute nodes and can reach `vuwstocoissrin1`
  from them. If not, run the script in an interactive session instead of `sbatch`;
* the authentication file approach (`-A`);
* folder names containing unusual characters (e.g. `Joe_MoDone?`, as shown on
  the Linux mount) may need checking with `list`.

The recogniser also fired on the little spotted kiwi and generic kiwi test
recordings, so expect other kiwi species among the kept files; review them before
using them as training data.
