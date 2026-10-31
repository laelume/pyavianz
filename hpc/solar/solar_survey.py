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

Folder list (one per line, '#' comments allowed), first component is the share:
  ECS_acoustic_02/Remutaka20
  ECS_acoustic_02/Kakapo Recordings Andrew Digby/2019

Commands:
  python solar_survey.py list ECS_acoustic_02/Remutaka20   # list subfolders (to build a list)
  python solar_survey.py run folders.txt                    # process every folder in the list
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


def smbclient(share, commands, args, log_file=None):
    """Run smbclient commands against a share; raise on failure."""
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
    if proc.returncode != 0 or errors:
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

def run_avianz(audio_dir, annot_dir, args, log_file):
    env = dict(os.environ)
    repo = os.path.dirname(args.avianz_dir)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo, args.avianz_dir, env.get("PYTHONPATH")]))
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [args.python, "AviaNZ.py", "-c", "-b", "-d", audio_dir, "-r", args.recogniser,
           "-asd", annot_dir, "--gpu" if args.gpu else "--no-gpu"]
    if args.wind:
        cmd.append("-w")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        start = f.tell()
        # AviaNZ asks y/n questions (launch, resume); answer yes to all
        proc = subprocess.run(cmd, cwd=args.avianz_dir, env=env, input="y\n" * 1000,
                              stdout=f, stderr=subprocess.STDOUT, text=True)
    with open(log_file, "rb") as f:
        f.seek(start)
        ok = b"Analysis completed successfully" in f.read()
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


def read_done(path):
    return set(read_list(path)) if os.path.isfile(path) else set()


def process_folder(spec, idx, args, out):
    share, path = split_spec(spec)
    tag = f"{idx:04d}_{safe_name(spec)}"[:150]
    stage = os.path.join(args.staging, tag)
    audio_dir, annot_dir = os.path.join(stage, "audio"), os.path.join(stage, "annotations")
    log_file = os.path.join(out, "logs", tag + ".log")
    shutil.rmtree(stage, ignore_errors=True)
    try:
        if not args.local_source:
            size = remote_size(spec, args)
            free = shutil.disk_usage(args.staging).free
            if size is not None:
                log(f"  remote size {size / 1e9:.1f} GB, staging free {free / 1e9:.1f} GB")
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
        run_avianz(audio_dir, annot_dir, args, log_file)
        log(f"  AviaNZ finished in {time.time() - t0:.0f}s")

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

    out = args.output or os.path.join(os.environ.get("SCRATCH", f"/nfs/scratch/{os.environ.get('USER', 'user')}"),
                                      "solar_survey", safe_name(args.recogniser))
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

    total_rec = total_kept = 0
    for i, spec in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {spec}")
        try:
            n, kept = process_folder(spec, folders.index(spec), args, out)
        except Exception as e:
            log(f"  FAILED: {e}")
            with open(failed_file, "a", encoding="utf-8") as f:
                f.write(f"{spec}\t{dt.datetime.now():%Y-%m-%d %H:%M}\t{e}\n")
            continue
        total_rec += n
        total_kept += kept
        log(f"  {kept} of {n} recordings had detections")
        with open(done_file, "a", encoding="utf-8") as f:
            f.write(spec + "\n")
    log(f"Finished: {total_rec} recordings processed, {total_kept} kept. Results in {out}")


def cmd_list(args):
    if args.auth_file:
        check_auth_file(args.auth_file)
    for name in remote_subfolders(args.folder, args):
        print(f"{args.folder.rstrip('/')}/{name}")


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
    {"run": cmd_run, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()
