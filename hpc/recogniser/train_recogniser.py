#!/usr/bin/env python3
"""Train an AviaNZ recogniser (wavelet/changepoint detector) from the command line.

This is a no-GUI port of the training done by AviaNZ's "Build recogniser" wizard
(src/ui/dialogs_training/build_recognizer_wizard.py). It calls the same core
functions; the choices a person makes in the wizard are made by explicit options:

  wizard step                          here
  -----------------------------------  ---------------------------------------------
  pick training folder, species, fs    TRAIN_DIR, --species, --sample-rate
  clustering into call types           not done: one call type, chosen by --calltype
  per-call-type parameters (editable)  derived from the annotations as the wizard
                                       does; override with --freq-low/--freq-high/
                                       --minlen/--maxlen/--chp-window
  click a point on the ROC curve       default: nearest top-left; or --max-fpr X | --threshold T
  save recogniser                      OUT_DIR/<name>.txt (+ _ROC.json, training_log.json)

Your annotations are never modified: recordings are copied to a working folder,
where the 0/1 ground-truth files (<recording>-GT.txt) used by training are written.

Training data: a folder (searched recursively) of recordings with AviaNZ
annotations beside them (<recording>.wav.data). Only human-verified labels are
used (certainty 100, or manual labels) unless --include-auto is given. Every call
of the target type should be annotated in each file: unannotated time counts as
"not a call" when the ROC curve is computed.

Example (male NI brown kiwi, changepoint method):
  python train_recogniser.py /nfs/scratch/$USER/train/nibk \\
      --species "Kiwi (Nth Is Brown)" --calltype Male --include-untyped \\
      --name "Kiwi (Nth Is Brown) Male v1" -o /nfs/scratch/$USER/recognisers
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
AUDIO_EXTS = (".wav", ".flac")


def setup_avianz(repo):
    """Make the avianz package importable from the repository root (no pip install needed)."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    global np, sf, annotation, wavelet_segment
    import numpy as np
    import soundfile as sf
    from avianz.src.core import annotation, wavelet_segment


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def label_selected(lab, args):
    """Is this label a training example of the target call type?"""
    if lab.get("species") != args.species:
        return False
    if args.include_auto:
        ok = lab.get("certainty", 0) >= args.min_certainty
    else:
        # human-verified: confirmed in review (100) or entered manually
        ok = lab.get("certainty", 0) >= 100 or lab.get("filter", "M") == "M"
    if not ok:
        return False
    ct = lab.get("calltype")
    if args.calltype is None:
        return True
    return ct == args.calltype or (ct is None and args.include_untyped)


def scan_training_data(args):
    """Read annotations and keep only the target call type (nothing is written).

    Returns a list of (source path, relative path, SegmentList of target calls, info dict).
    """
    found = []
    for root, dirs, names in os.walk(args.train_dir):
        dirs.sort()
        for name in sorted(names):
            if not name.lower().endswith(AUDIO_EXTS) or name + ".data" not in names:
                continue
            src = os.path.join(root, name)
            segs = annotation.SegmentList()
            segs.parseJSON(src + ".data", silent=True)
            duration = sf.info(src).duration

            # keep only the target call type; other calls become "absent" in the GT
            chosen = annotation.SegmentList()
            chosen.metadata = dict(segs.metadata, Duration=duration)
            n_other = 0
            for seg in segs:
                labs = [l for l in seg.labels if label_selected(l, args)]
                if labs:
                    chosen.addSegment(annotation.Segment(seg.start_time, seg.end_time, seg.freq_low,
                                                         seg.freq_high, [dict(labs[0])]))
                elif any(l.get("species") == args.species for l in seg.labels):
                    n_other += 1
            if not chosen and not args.keep_empty:
                continue
            rel = os.path.relpath(src, args.train_dir)
            found.append((src, rel, chosen, {"file": rel, "duration_s": round(duration, 2),
                                             "calls": len(chosen), "other_calls_of_species": n_other}))
    return found


def prepare_workdir(found, resolution, args):
    """Copy the recordings to the work dir and export 0/1 GT files beside the copies."""
    if os.path.exists(args.work_dir) and os.listdir(args.work_dir):
        if not args.overwrite_work:
            sys.exit(f"Work folder {args.work_dir} is not empty; use --overwrite-work or another --work-dir")
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    for src, rel, chosen, _ in found:
        dst = os.path.join(args.work_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        # writes <dst without extension>-GT.txt beside the COPY
        chosen.exportGT(dst, args.species, resolution=resolution)


def derive_parameters(segments, args):
    """Call parameters from the annotations, as the wizard pre-fills them."""
    fs = args.sample_rate
    lens = [s.end_time - s.start_time for s in segments]
    lows = [s.freq_low for s in segments]
    highs = [s.freq_high for s in segments]
    p = {
        "minlen": round(min(lens), 2),
        "maxlen": round(max(lens), 2),
        "fLow": max(0, int(min(lows))),
        # full-band segments have freq_high 0 -> use Nyquist (wizard behaviour)
        "fHigh": int(min(fs / 2, max(highs))) if max(highs) > 0 else int(fs / 2),
    }
    # overrides
    for key, opt in (("minlen", "minlen"), ("maxlen", "maxlen"), ("fLow", "freq_low"), ("fHigh", "freq_high")):
        if getattr(args, opt) is not None:
            p[key] = getattr(args, opt)
    if args.method == "chp":
        # wizard default: 10% of the minimum call length, rounded UP to a multiple
        # of 32/fs so the window holds a whole number of wavelet coefficients
        minchpwin = 32 / fs
        win = args.chp_window if args.chp_window is not None else p["minlen"] / 10
        p["chpwin"] = math.ceil(round(win / minchpwin, 9)) * minchpwin
    else:
        p["avgslen"], p["maxgap"] = args.avgslen, args.maxgap
    return p


# ---------------------------------------------------------------------------
# Training and threshold choice
# ---------------------------------------------------------------------------

def train(params, args):
    spinfo = {"species": args.species, "method": args.method, "SampleRate": args.sample_rate, "Filters": []}
    if args.method == "chp":
        spinfo["Filters"] = [{"calltype": args.calltype or "Any", "TimeRange": [params["minlen"], params["maxlen"], 0.0, 0.0],
                              "FreqRange": [params["fLow"], params["fHigh"]]}]
        thr_list = np.geomspace(0.03, 10, num=9)          # as in the wizard
        m_list = None
        ws = wavelet_segment.WaveletSegment(spinfo)
        nodes, tp, fp, tn, fn = ws.waveletSegment_trainChp(args.work_dir, thr_list, maxlen=params["maxlen"],
                                                           window=params["chpwin"])
    else:
        spinfo["Filters"] = [{"calltype": args.calltype or "Any",
                              "TimeRange": [params["minlen"], params["maxlen"], params["avgslen"], params["maxgap"]],
                              "FreqRange": [params["fLow"], params["fHigh"]]}]
        thr_list = np.linspace(0.2, 1, num=50)             # as in the wizard
        m_list = np.linspace(params["avgslen"], params["avgslen"], num=1)
        ws = wavelet_segment.WaveletSegment(spinfo)
        nodes, tp, fp, tn, fn = ws.waveletSegment_train(args.work_dir, thr_list, m_list, d=False,
                                                        learnMode="recaa", window=1, inc=None)
    tp, fp, tn, fn = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (tp, fp, tn, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = tp / (tp + fn)
        fpr = 1 - tn / (fp + tn)
    return spinfo, thr_list, m_list, nodes, tpr, fpr


def choose_point(thr_list, tpr, fpr, args):
    """Replaces clicking on the ROC curve. Returns (M index, threshold index, reason)."""
    valid = ~(np.isnan(tpr) | np.isnan(fpr))
    if args.threshold is not None:
        j = int(np.argmin(np.abs(np.asarray(thr_list) - args.threshold)))
        return 0, j, f"threshold nearest {args.threshold}"
    if args.max_fpr is not None:
        ok = valid & (fpr <= args.max_fpr)
        if not ok.any():
            sys.exit(f"No threshold has FPR <= {args.max_fpr}; see the ROC table above")
        # highest TPR among allowed points; ties -> lowest FPR
        score = np.where(ok, tpr - 1e-9 * fpr, -np.inf)
        i, j = np.unravel_index(np.argmax(score), score.shape)
        return int(i), int(j), f"highest TPR with FPR <= {args.max_fpr}"
    # default: closest to the top-left corner (TPR 1, FPR 0), what clicking approximates
    dist = np.where(valid, (1 - tpr) ** 2 + fpr ** 2, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return int(i), int(j), "closest to top-left of ROC curve"


def to_py(x):
    """numpy -> plain Python for JSON."""
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train_dir", help="Folder of recordings with <recording>.wav.data annotations")
    p.add_argument("--species", required=True, help='Species label as in the annotations, e.g. "Kiwi (Nth Is Brown)"')
    p.add_argument("--calltype", default=None, help='Call type to train on, e.g. "Male" (default: all calls of the species)')
    p.add_argument("--include-untyped", action="store_true",
                   help="With --calltype, also use labels of the species that have no calltype")
    p.add_argument("--name", required=True, help='Recogniser name (file name), e.g. "Kiwi (Nth Is Brown) Male v1"')
    p.add_argument("-o", "--out-dir", required=True, help="Where to write <name>.txt and its logs")
    p.add_argument("--method", choices=["chp", "wv"], default="chp",
                   help="chp = changepoint (AviaNZ's current default), wv = older wavelet filter")
    p.add_argument("--sample-rate", type=int, default=16000, help="Analysis sample rate (default 16000)")
    p.add_argument("--work-dir", default=None, help="Working copy of the training data (default: <out-dir>/work_<name>)")
    p.add_argument("--overwrite-work", action="store_true", help="Delete and recreate a non-empty work folder")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing <name>.txt in --out-dir")
    p.add_argument("--keep-empty", action="store_true",
                   help="Also train on annotated recordings with no target calls (as negatives)")
    p.add_argument("--include-auto", action="store_true", help="Also use unreviewed automatic labels")
    p.add_argument("--min-certainty", type=float, default=50, help="With --include-auto: minimum certainty")
    g = p.add_argument_group("call parameters (default: derived from the annotations, as in the wizard)")
    g.add_argument("--freq-low", type=int, default=None)
    g.add_argument("--freq-high", type=int, default=None)
    g.add_argument("--minlen", type=float, default=None, help="Minimum call length (s)")
    g.add_argument("--maxlen", type=float, default=None, help="Maximum call length (s)")
    g.add_argument("--chp-window", type=float, default=None, help="Changepoint window (s); default minlen/10")
    g.add_argument("--avgslen", type=float, default=0.5, help="wv only: average syllable length (s)")
    g.add_argument("--maxgap", type=float, default=0.5, help="wv only: max gap between syllables (s)")
    s = p.add_argument_group("operating point (replaces clicking the ROC curve)")
    s.add_argument("--max-fpr", type=float, default=None, help="Highest TPR with FPR at most this (e.g. 0.05)")
    s.add_argument("--threshold", type=float, default=None, help="Use the tested threshold nearest this value")
    p.add_argument("--avianz-repo", default=REPO, help="Repository root containing avianz/")
    args = p.parse_args(argv)

    setup_avianz(args.avianz_repo)
    t_start = time.time()
    args.train_dir = os.path.abspath(args.train_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, args.name + ".txt")
    if os.path.exists(out_txt) and not args.overwrite:
        sys.exit(f"{out_txt} already exists; choose another --name or use --overwrite")
    args.work_dir = os.path.abspath(args.work_dir or os.path.join(args.out_dir, "work_" + args.name.replace(" ", "_")))

    found = scan_training_data(args)
    segments = [s for _, _, chosen, _ in found for s in chosen]
    files = [info for *_, info in found]
    if not segments:
        sys.exit("No training calls found: check --species / --calltype / --include-untyped and that labels are verified")
    params = derive_parameters(segments, args)
    # GT resolution must match the training method: chp uses its window, wv 1 s
    prepare_workdir(found, params["chpwin"] if args.method == "chp" else 1.0, args)

    print(f"\nTraining data: {len(segments)} calls in {len(files)} recordings "
          f"({sum(f['duration_s'] for f in files) / 60:.1f} min of audio)")
    print(f"Call parameters: {params}\n")

    spinfo, thr_list, m_list, nodes, tpr, fpr = train(params, args)

    print("\nROC (one row per tested threshold):")
    print("  idx  threshold      TPR      FPR")
    for j, thr in enumerate(thr_list):
        print(f"  {j:3d}  {thr:9.4f}  {tpr[0, j]:7.3f}  {fpr[0, j]:7.3f}")
    i, j, reason = choose_point(thr_list, tpr, fpr, args)
    best_nodes = [int(n) for n in nodes[i][j]]
    print(f"\nChosen: threshold {thr_list[j]:.4f} ({reason}); TPR {tpr[i, j]:.3f}, FPR {fpr[i, j]:.3f}; "
          f"nodes {best_nodes}")

    sub = spinfo["Filters"][0]
    if args.method == "chp":
        sub["WaveletParams"] = {"thr": float(thr_list[j]), "nodes": best_nodes, "win": params["chpwin"]}
    else:
        sub["WaveletParams"] = {"thr": float(thr_list[j]), "M": float(m_list[i]), "nodes": best_nodes}
    sub["TPR, FPR"] = [float(tpr[i, j]), float(fpr[i, j])]

    roc_name = args.name + "_ROC.json"
    with open(out_txt, "w", encoding="utf-8") as f:
        json.dump(spinfo, f)
    with open(os.path.join(args.out_dir, roc_name), "w", encoding="utf-8") as f:
        json.dump({sub["calltype"]: [to_py(tpr[0]), to_py(fpr[0]), to_py(nodes[0])],
                   "thr": to_py(thr_list)}, f)
    log = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "options": {k: v for k, v in vars(args).items()},
        "parameters": params,
        "chosen": {"threshold_index": j, "threshold": float(thr_list[j]), "reason": reason,
                   "TPR": float(tpr[i, j]), "FPR": float(fpr[i, j]), "nodes": best_nodes},
        "roc": {"threshold": to_py(thr_list), "TPR": to_py(tpr[0]), "FPR": to_py(fpr[0])},
        "training_files": files,
        "recogniser_sha256": sha256(out_txt),
        "runtime_s": round(time.time() - t_start),
    }
    with open(os.path.join(args.out_dir, args.name + "_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, default=str)

    print(f"\nWrote {out_txt}")
    print(f"      {os.path.join(args.out_dir, roc_name)}")
    print(f"      {os.path.join(args.out_dir, args.name + '_training_log.json')}")
    print("To use it with AviaNZ, copy the .txt into the AviaNZ Filters folder "
          "(~/.avianz/Filters on Linux, %APPDATA%\\AviaNZ\\Filters on Windows) and run with "
          f'-r "{args.name}". It has no verifier network yet (train one with AviaNZ.py -c -t).')


if __name__ == "__main__":
    main()
