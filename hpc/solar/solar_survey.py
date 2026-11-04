#!/usr/bin/env python3
"""Run an AviaNZ recogniser over folders on the VUW acoustic drives (SoLAR) from Rāpoi.

SoLAR is not mounted on Rāpoi, so each folder is processed in turn:

  1. copy the folder's audio (recursively) from SoLAR to a local staging area with smbclient
  2. run AviaNZ batch mode on it (default recogniser: "Kiwi (Nth Is Brown)")
  3. for every recording with at least one detection, copy the ENTIRE recording
     and its annotation to the results folder on Rāpoi scratch
  4. delete the staged copy, append the folder to done.txt, move on

Nothing is ever written to SoLAR. Re-running the same command skips folders
listed in done.txt, so a job that hits its time limit can simply be resubmitted.

Results (default /nfs/scratch/$USER/solar_survey/<recogniser>/):
  <share>/<folder path>/<recording>.wav        full recordings with detections
  <share>/<folder path>/<recording>.wav.data   their AviaNZ annotations (beside the
                                               COPY, so AviaNZ can open/train on them)
  annotations/<share>/<folder path>/...        annotations for every processed recording,
                                               including those with no detections
  summary.csv      one row per processed recording
  done.txt         folders completed;  failed.txt   folders that failed (with reason)
  logs/            smbclient and AviaNZ output per folder
  status.json      live progress (fraction of audio done, ETA); read it with `status`
  sizes.json       folder sizes measured at the start, used for the ETA

Folder list (one per line, '#' comments allowed), first component is the share:
  ECS_acoustic_02/Remutaka20
  ECS_acoustic_02/Kakapo Recordings Andrew Digby/2019

Commands:
  python solar_survey.py list ECS_acoustic_02/Remutaka20   # list subfolders (to build a list)
  python solar_survey.py files ECS_acoustic_01 files.csv    # every file, recursively, to a CSV
  python solar_survey.py run folders.txt                    # process every folder in the list
  python solar_survey.py status --watch                     # progress bar + ETA of the current run
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

HOST = "vuwstocoissrin1.vuw.ac.nz"
AUDIO_EXTS = (".wav", ".flac")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVIANZ_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "avianz"))


def log(msg):
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# SoLAR access
# ---------------------------------------------------------------------------

def split_spec(spec):
    """'ECS_acoustic_02/a/b' -> ('ECS_acoustic_02', 'a/b')"""
    parts = spec.strip().strip("/").split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def check_auth_file(path):
    if not os.path.isfile(path):
        sys.exit(f"Authentication file not found: {path}")
    if os.name == "posix" and os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f"{path} is readable by others; run: chmod 600 {path}")


def smb_quote(path):
    if '"' in path or ";" in path:
        raise ValueError(f"Unsupported character in SoLAR path: {path}")
    return f'"{path}"'


# SoLAR is read-only for this script: only these smbclient commands may be sent.
# lcd changes the LOCAL (Rāpoi) directory; mget copies FROM SoLAR TO Rāpoi.
# Anything that writes to the share (put, mput, mkdir, del, rm, rename, ...) is refused.
READ_ONLY_SMB_COMMANDS = {"prompt", "recurse", "lcd", "cd", "mask", "mget", "ls", "du"}


def smbclient(share, commands, args, log_file=None, strict=True):
    """Run smbclient commands against a share; raise on failure (unless strict=False)."""
    # Safety check: refuse any command that could modify SoLAR, before connecting
    for c in commands:
        verb = c.split(None, 1)[0].lower()
        if verb not in READ_ONLY_SMB_COMMANDS:
            raise ValueError(f"Refusing smbclient command '{c}': only read-only commands are allowed")
    cmd = ["smbclient", f"//{args.host}/{share}"]
    if args.auth_file:
        cmd += ["-A", args.auth_file]
    else:
        cmd += ["-U", args.user]  # prompts for the password (interactive use only)
    cmd += ["-c", "; ".join(commands)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"$ smbclient //{args.host}/{share} -c '{'; '.join(commands)}'\n{proc.stdout}\n")
    errors = re.findall(r"NT_STATUS_[A-Z_]+", proc.stdout)
    if not strict:
        # tolerate unreadable subfolders only; a failed logon or missing share still raises
        errors = [e for e in errors if e not in ("NT_STATUS_ACCESS_DENIED", "NT_STATUS_NO_SUCH_FILE")]
        for line in proc.stdout.splitlines():
            if "NT_STATUS_" in line and not errors:
                print(f"warning: {line.strip()}", file=sys.stderr)
    if errors or (strict and proc.returncode != 0):
        raise RuntimeError(f"smbclient failed ({', '.join(sorted(set(errors))) or proc.returncode})")
    return proc.stdout


LS_LINE = re.compile(r"^  (.+?)\s+([A-Z]*)\s+(\d+)\s+\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d \d{4}$")


def remote_subfolders(spec, args):
    share, path = split_spec(spec)
    out = smbclient(share, [f"cd {smb_quote(path)}" if path else "cd /", "ls"], args)
    names = []
    for line in out.splitlines():
        m = LS_LINE.match(line)
        if m and "D" in m.group(2) and m.group(1) not in (".", ".."):
            names.append(m.group(1))
    return sorted(names)


def remote_files(spec, args):
    """Every file under a SoLAR folder, recursively, as (path from the share root, bytes)."""
    share, path = split_spec(spec)
    out = smbclient(share, [f"cd {smb_quote(path)}" if path else "cd /", "recurse ON", "ls"],
                    args, strict=False)
    files = []
    folder = path  # the first block of a recursive ls is the starting folder, with no header
    for line in out.splitlines():
        if line.startswith("\\"):  # header of each subfolder, e.g. \Remutaka20\Site 1
            folder = line.strip().strip("\\").replace("\\", "/")
            continue
        m = LS_LINE.match(line)
        if m and "D" not in m.group(2):
            files.append((f"{folder}/{m.group(1)}" if folder else m.group(1), int(m.group(3))))
    return sorted(files)


def remote_size(spec, args):
    """Total bytes under a remote folder via smbclient 'du', or None if unknown."""
    share, path = split_spec(spec)
    try:
        out = smbclient(share, [f"cd {smb_quote(path)}", "recurse ON", "du"], args)
    except RuntimeError:
        return None
    m = re.search(r"Total number of bytes:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def fetch_folder(spec, dest, args, log_file):
    """Copy the audio files of a remote folder (recursively) into dest."""
    os.makedirs(dest, exist_ok=True)
    if args.local_source:  # testing, or if SoLAR is ever mounted on Rāpoi
        share, path = split_spec(spec)
        src = os.path.join(args.local_source, share, path)
        if not os.path.isdir(src):
            raise RuntimeError(f"local source folder not found: {src}")
        for root, _, files in os.walk(src):
            for name in files:
                if name.lower().endswith(AUDIO_EXTS):
                    out_dir = os.path.join(dest, os.path.relpath(root, src))
                    os.makedirs(out_dir, exist_ok=True)
                    shutil.copy2(os.path.join(root, name), out_dir)
        return
    share, path = split_spec(spec)
    cmds = ["prompt OFF", "recurse ON", f"lcd {smb_quote(dest)}"]
    if path:
        cmds.append(f"cd {smb_quote(path)}")
    for ext in AUDIO_EXTS:
        cmds += [f"mask *{ext}", "mget *"]
    smbclient(share, cmds, args, log_file)


# ---------------------------------------------------------------------------
# AviaNZ
# ---------------------------------------------------------------------------

# AviaNZ prints this line before each file in batch mode (batch_processor.process_file_loop)
AVIANZ_PROGRESS = re.compile(r"\*\*\* Processing file (\d+) / (\d+)")


def run_avianz(audio_dir, annot_dir, args, log_file, on_file=None):
    """Run AviaNZ batch mode; on_file(files_done, files_total) is called as it progresses."""
    env = dict(os.environ)
    repo = os.path.dirname(args.avianz_dir)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo, args.avianz_dir, env.get("PYTHONPATH")]))
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [args.python, "AviaNZ.py", "-c", "-b", "-d", audio_dir, "-r", args.recogniser,
           "-asd", annot_dir, "--gpu" if args.gpu else "--no-gpu"]
    if args.wind:
        cmd.append("-w")
    ok = False
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        # Output is streamed line by line so per-file progress can be reported
        proc = subprocess.Popen(cmd, cwd=args.avianz_dir, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        # AviaNZ asks y/n questions (launch, resume); answer yes to all
        try:
            proc.stdin.write("y\n" * 1000)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        for line in proc.stdout:
            f.write(line)
            if "Analysis completed successfully" in line:
                ok = True
            m = AVIANZ_PROGRESS.search(line)
            if m and on_file:
                # the line is printed when file N starts, so N-1 files are done
                on_file(int(m.group(1)) - 1, int(m.group(2)))
        proc.wait()
    if proc.returncode != 0 or not ok:
        raise RuntimeError(f"AviaNZ batch failed (exit {proc.returncode}); see {log_file}")


def avianz_annotation_path(annot_dir, audio_path):
    """Where AviaNZ batch mode saves the annotation for audio_path
    (mirrors BatchProcessor.define_annotation_save_location)."""
    src_dir = os.path.abspath(os.path.dirname(audio_path))
    _, no_drive = os.path.splitdrive(src_dir)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(annot_dir, no_drive.lstrip(os.sep).lstrip("/"), stem + ".data")


def detections(data_path, species, min_certainty):
    """(number of matching segments, max certainty, duration) from a .data file."""
    with open(data_path, encoding="utf-8") as f:
        annots = json.load(f)
    meta = annots[0] if annots and isinstance(annots[0], dict) else {}
    n, best = 0, 0.0
    for seg in annots[1:] if meta else annots:
        if not isinstance(seg, list) or len(seg) != 5:
            continue
        for lab in seg[4] if isinstance(seg[4], list) else []:
            if isinstance(lab, dict) and lab.get("species") == species and lab.get("certainty", 0) >= min_certainty:
                n += 1
                best = max(best, float(lab.get("certainty", 0)))
    return n, best, meta.get("Duration", "")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]


def default_output(recogniser):
    scratch = os.environ.get("SCRATCH", f"/nfs/scratch/{os.environ.get('USER', 'user')}")
    return os.path.join(scratch, "solar_survey", safe_name(recogniser))


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def fmt_dur(seconds):
    if seconds is None:
        return "?"
    m = int(seconds // 60)
    d, h, m = m // 1440, (m // 60) % 24, m % 60
    return f"{d}d{h}h" if d else (f"{h}h{m:02d}m" if h else f"{m}m")


def folder_size(spec, args):
    """Bytes of audio in a folder (for progress estimates); None if unknown."""
    if args.local_source:
        share, path = split_spec(spec)
        root = os.path.join(args.local_source, share, path)
        return sum(os.path.getsize(os.path.join(r, n)) for r, _, fs in os.walk(root)
                   for n in fs if n.lower().endswith(AUDIO_EXTS)) or None
    return remote_size(spec, args)


class Progress:
    """Overall progress of a run, measured in bytes of audio, written to <out>/status.json.

    Folders count as done once processed (or failed); the current folder counts
    in proportion to the files AviaNZ has finished. The ETA assumes the rest of
    the audio processes at the average rate so far (downloads included).
    """
    LOG_EVERY = 300  # seconds between progress lines in the job output

    def __init__(self, out, todo, sizes, listed, done_before):
        self.path = os.path.join(out, "status.json")
        known = [s for s in sizes.values() if s]
        fallback = sum(known) / len(known) if known else 1.0
        self.sizes = {f: float(sizes.get(f) or fallback) for f in todo}
        self.t0 = time.time()
        self.done_bytes = 0.0
        self._last_log = 0.0
        self.s = {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "folders_listed": listed, "folders_done_before": done_before, "folders_todo": len(todo),
            "folders_finished": 0, "folders_failed": 0, "recordings": 0, "kept": 0,
            "current": None, "current_index": 0, "phase": "starting", "files_done": 0, "files_total": 0,
            "bytes_total": sum(self.sizes.values()),
        }
        self.write()

    def start_folder(self, index, spec):
        self.s.update(current=spec, current_index=index, phase="downloading", files_done=0, files_total=0)
        self.write(log_now=True)

    def set_phase(self, phase, files_total=None):
        self.s["phase"] = phase
        if files_total is not None:
            self.s.update(files_total=files_total, files_done=0)
        self.write()

    def file_progress(self, files_done, files_total):
        self.s.update(files_done=files_done, files_total=files_total)
        self.write()

    def finish_folder(self, spec, recordings=0, kept=0, failed=False):
        self.done_bytes += self.sizes.get(spec, 0.0)
        self.s["folders_failed" if failed else "folders_finished"] += 1
        self.s["recordings"] += recordings
        self.s["kept"] += kept
        self.s.update(current=None, phase="between folders", files_done=0, files_total=0)
        self.write(log_now=True)

    def finish(self):
        self.s.update(phase="finished", current=None)
        self.write(log_now=True)

    def fraction(self):
        cur = 0.0
        if self.s["current"] and self.s["files_total"]:
            cur = self.sizes.get(self.s["current"], 0.0) * self.s["files_done"] / self.s["files_total"]
        total = self.s["bytes_total"] or 1.0
        return 1.0 if self.s["phase"] == "finished" else min(1.0, (self.done_bytes + cur) / total)

    def write(self, log_now=False):
        frac, elapsed = self.fraction(), time.time() - self.t0
        eta = elapsed * (1 - frac) / frac if 0.001 < frac < 1 else (0 if frac >= 1 else None)
        self.s.update(updated=dt.datetime.now().isoformat(timespec="seconds"), fraction=round(frac, 4),
                      elapsed_s=round(elapsed), eta_s=None if eta is None else round(eta))
        write_json(self.path, self.s)
        if log_now or time.time() - self._last_log > self.LOG_EVERY:
            self._last_log = time.time()
            files = f", file {self.s['files_done']}/{self.s['files_total']}" if self.s["files_total"] else ""
            log(f"  progress {frac * 100:5.1f}% | elapsed {fmt_dur(elapsed)} | ETA {fmt_dur(eta)} | "
                f"{self.s['phase']}{files}")


def render_status(s, width=40):
    frac = s.get("fraction", 0.0)
    fill = int(width * frac)
    updated = dt.datetime.fromisoformat(s["updated"])
    age = (dt.datetime.now() - updated).total_seconds()
    eta = s.get("eta_s")
    eta_at = f" ({updated + dt.timedelta(seconds=eta):%a %d %b %H:%M})" if eta else ""
    lines = [
        f"[{'#' * fill}{'-' * (width - fill)}] {frac * 100:5.1f}%   "
        f"elapsed {fmt_dur(s.get('elapsed_s'))}   ETA {fmt_dur(eta)}{eta_at}",
        f"folders: {s['folders_finished']} done, {s['folders_failed']} failed, of {s['folders_todo']} this run"
        f" (+{s['folders_done_before']} done in earlier runs)",
        f"recordings: {s['recordings']} processed, {s['kept']} kept",
    ]
    if s.get("current"):
        files = f", file {s['files_done']}/{s['files_total']}" if s.get("files_total") else ""
        lines.append(f"now: [{s['current_index']}/{s['folders_todo']}] {s['current']}: {s['phase']}{files}")
    else:
        lines.append(f"now: {s['phase']}")
    note = ""
    if s["phase"] != "finished" and age > 1800:
        note = "  <- no update for a while: downloads are silent, or the job may have stopped (squeue -u $USER)"
    lines.append(f"updated {fmt_dur(age)} ago{note}" + (f"   job {s['job_id']}" if s.get("job_id") else ""))
    return lines


def read_done(path):
    return set(read_list(path)) if os.path.isfile(path) else set()


def process_folder(spec, idx, args, out, size, progress):
    share, path = split_spec(spec)
    tag = f"{idx:04d}_{safe_name(spec)}"[:150]
    stage = os.path.join(args.staging, tag)
    audio_dir, annot_dir = os.path.join(stage, "audio"), os.path.join(stage, "annotations")
    log_file = os.path.join(out, "logs", tag + ".log")
    shutil.rmtree(stage, ignore_errors=True)
    try:
        # size was measured up front (sizes.json); used here to check staging space
        if size is not None:
            free = shutil.disk_usage(args.staging).free
            log(f"  size {size / 1e9:.1f} GB, staging free {free / 1e9:.1f} GB")
            if size > 0.9 * free:
                raise RuntimeError(f"folder is {size / 1e9:.1f} GB but staging has {free / 1e9:.1f} GB free; "
                                   "list its subfolders separately")

        t0 = time.time()
        fetch_folder(spec, audio_dir, args, log_file)
        audio = sorted(os.path.join(r, n) for r, _, fs in os.walk(audio_dir) for n in fs
                       if n.lower().endswith(AUDIO_EXTS))
        log(f"  copied {len(audio)} recordings in {time.time() - t0:.0f}s")
        if not audio:
            return 0, 0

        t0 = time.time()
        progress.set_phase("detecting", files_total=len(audio))
        run_avianz(audio_dir, annot_dir, args, log_file, on_file=progress.file_progress)
        log(f"  AviaNZ finished in {time.time() - t0:.0f}s")
        progress.set_phase("saving results")

        kept = 0
        rows = []
        for a in audio:
            rel = os.path.relpath(a, audio_dir)
            data = avianz_annotation_path(annot_dir, a)
            if not os.path.isfile(data):
                rows.append([share, path, rel, "", "", "", "no annotation (skipped by AviaNZ)"])
                continue
            n, best, duration = detections(data, args.recogniser_species, args.min_certainty)
            # keep a record of every annotation, including empty ones
            rec_dir = os.path.join(out, "annotations", share, path, os.path.dirname(rel))
            os.makedirs(rec_dir, exist_ok=True)
            shutil.copy2(data, os.path.join(rec_dir, os.path.basename(rel) + ".data"))
            if n:
                dest = os.path.join(out, share, path, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(a, dest)                 # the entire recording
                shutil.copy2(data, dest + ".data")    # annotation beside the copy
                kept += 1
            rows.append([share, path, rel, duration, n, best, "kept" if n else ""])
        new = not os.path.isfile(os.path.join(out, "summary.csv"))
        with open(os.path.join(out, "summary.csv"), "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["share", "folder", "recording", "duration_s", "detections", "max_certainty", "status"])
            w.writerows(rows)
        return len(audio), kept
    finally:
        if not args.keep_staging:
            shutil.rmtree(stage, ignore_errors=True)


def cmd_run(args):
    if args.auth_file:
        check_auth_file(args.auth_file)
    if not args.local_source and shutil.which("smbclient") is None:
        sys.exit("smbclient not found on this node")
    if not os.path.isfile(os.path.join(args.avianz_dir, "AviaNZ.py")):
        sys.exit(f"AviaNZ.py not found in {args.avianz_dir}; set --avianz-dir")
    args.recogniser_species = args.species or args.recogniser

    out = args.output or default_output(args.recogniser)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    if args.staging is None:
        job = os.environ.get("SLURM_JOB_ID")
        base = f"/tmp/{os.environ.get('USER', 'user')}/{job}" if job else os.path.join(out, "_staging")
        args.staging = os.path.join(base, "stage")
    os.makedirs(args.staging, exist_ok=True)

    folders = read_list(args.folder_list)
    done_file, failed_file = os.path.join(out, "done.txt"), os.path.join(out, "failed.txt")
    done = read_done(done_file)
    todo = [f for f in folders if args.redo or f not in done]
    log(f"Recogniser: {args.recogniser} | results: {out} | staging: {args.staging}")
    log(f"{len(folders)} folders listed, {len(folders) - len(todo)} already done, {len(todo)} to process")

    # Measure every folder once (cached in sizes.json) so progress and ETA can
    # be estimated from the total amount of audio, not just the folder count
    sizes_file = os.path.join(out, "sizes.json")
    sizes = {}
    if os.path.isfile(sizes_file):
        with open(sizes_file, encoding="utf-8") as f:
            sizes = json.load(f)
    missing = [f for f in todo if f not in sizes]
    if missing:
        log(f"Measuring {len(missing)} folders for progress estimates...")
        for f in missing:
            sizes[f] = folder_size(f, args)
            write_json(sizes_file, sizes)
    known = [sizes[f] for f in todo if sizes.get(f)]
    log(f"Total to process: {sum(known) / 1e9:.1f} GB"
        + (f" ({len(todo) - len(known)} folders of unknown size)" if len(known) < len(todo) else ""))
    progress = Progress(out, todo, {f: sizes.get(f) for f in todo}, len(folders), len(folders) - len(todo))

    total_rec = total_kept = 0
    for i, spec in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {spec}")
        progress.start_folder(i, spec)
        try:
            n, kept = process_folder(spec, folders.index(spec), args, out, sizes.get(spec), progress)
        except Exception as e:
            log(f"  FAILED: {e}")
            with open(failed_file, "a", encoding="utf-8") as f:
                f.write(f"{spec}\t{dt.datetime.now():%Y-%m-%d %H:%M}\t{e}\n")
            progress.finish_folder(spec, failed=True)
            continue
        total_rec += n
        total_kept += kept
        log(f"  {kept} of {n} recordings had detections")
        with open(done_file, "a", encoding="utf-8") as f:
            f.write(spec + "\n")
        progress.finish_folder(spec, n, kept)
    progress.finish()
    log(f"Finished: {total_rec} recordings processed, {total_kept} kept. Results in {out}")


def cmd_status(args):
    path = os.path.join(args.output or default_output(args.recogniser), "status.json")
    while True:
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except FileNotFoundError:
            sys.exit(f"No status file at {path} (has a run started? is -r/-o the same as the run?)")
        except ValueError:  # caught mid-write; try again shortly
            time.sleep(1)
            continue
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")  # clear the terminal
        print("\n".join(render_status(s)), flush=True)
        if not args.watch or s.get("phase") == "finished":
            break
        time.sleep(args.watch)


def cmd_list(args):
    if args.auth_file:
        check_auth_file(args.auth_file)
    for name in remote_subfolders(args.folder, args):
        print(f"{args.folder.rstrip('/')}/{name}")


def cmd_files(args):
    if args.auth_file:
        check_auth_file(args.auth_file)
    share, _ = split_spec(args.folder)
    files = remote_files(args.folder, args)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "path", "bytes"])
        for path, size in files:
            full = f"{share}/{path}"
            w.writerow([full.replace("/", "_"), full, size])
    print(f"{len(files)} files written to {args.csv}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=HOST)
    p.add_argument("--auth-file", default=os.environ.get("SOLAR_AUTH_FILE"),
                   help="smbclient authentication file (chmod 600); needed for batch jobs. "
                        "Default: $SOLAR_AUTH_FILE")
    p.add_argument("--user", default=f"{os.environ.get('USER', '')}@staff.vuw.ac.nz",
                   help="SoLAR user when no --auth-file is given (password prompt)")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="print the subfolders of a SoLAR folder as list entries")
    pl.add_argument("folder", help="e.g. ECS_acoustic_02/Remutaka20")

    pf = sub.add_parser("files", help="write every file under a SoLAR folder (recursively) to a CSV")
    pf.add_argument("folder", help="e.g. ECS_acoustic_01 or ECS_acoustic_01/Remutaka20")
    pf.add_argument("csv", help="CSV to write: name (path joined with _), path, bytes")

    ps = sub.add_parser("status", help="show a progress bar and ETA for a run (safe on the login node)")
    ps.add_argument("-r", "--recogniser", default="Kiwi (Nth Is Brown)")
    ps.add_argument("-o", "--output", default=None, help="Results folder of the run (default as for run)")
    ps.add_argument("--watch", type=int, nargs="?", const=60, default=0, metavar="SECONDS",
                    help="Refresh every SECONDS (default 60) until the run finishes; Ctrl+C to stop")

    pr = sub.add_parser("run", help="process every folder in a folder list")
    pr.add_argument("folder_list")
    pr.add_argument("-r", "--recogniser", default="Kiwi (Nth Is Brown)")
    pr.add_argument("--species", default=None,
                    help="Species label to count as a detection (default: same as --recogniser)")
    pr.add_argument("--min-certainty", type=float, default=0,
                    help="Ignore detections below this certainty when deciding what to keep")
    pr.add_argument("-o", "--output", default=None,
                    help="Results folder (default: /nfs/scratch/$USER/solar_survey/<recogniser>)")
    pr.add_argument("--staging", default=None,
                    help="Where folders are copied for processing (default: node-local /tmp in a Slurm job)")
    pr.add_argument("--avianz-dir", default=DEFAULT_AVIANZ_DIR, help="Folder containing AviaNZ.py")
    pr.add_argument("--python", default=sys.executable, help="Python that runs AviaNZ")
    pr.add_argument("--wind", action="store_true", help="Apply AviaNZ's wind filter (-w)")
    pr.add_argument("--gpu", action="store_true", help="Let AviaNZ use a GPU for its NN (default: CPU)")
    pr.add_argument("--redo", action="store_true", help="Reprocess folders already in done.txt")
    pr.add_argument("--keep-staging", action="store_true", help="Do not delete staged audio (debugging)")
    pr.add_argument("--local-source", default=None,
                    help="Read <dir>/<share>/<path> from a local folder instead of SoLAR (testing)")
    args = p.parse_args(argv)
    {"run": cmd_run, "list": cmd_list, "files": cmd_files, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
