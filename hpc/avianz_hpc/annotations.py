"""Reading and writing AviaNZ .data annotation files."""

import os

from avianz_hpc._avianz import Segment, SegmentList

DONT_KNOW = "Don't Know"


def annotation_candidates(audio_path, annotation_root=None, relpath=None):
    """Possible .data locations for a recording, most specific first.

    Covers the layouts AviaNZ produces:
      * sidecar next to the audio:              <audio>.wav.data
      * batch -asd/--annotation-save-dir tree:  <root>/<abs dir without drive>/<stem>.data
      * a tree mirroring the audio roots:       <root>/<relpath>.data, <root>/<rel stem>.data
      * a flat folder of exported annotations:  <root>/<stem>.data
    """
    cands = [audio_path + ".data"]
    if annotation_root:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        _, no_drive = os.path.splitdrive(os.path.abspath(os.path.dirname(audio_path)))
        cands.append(os.path.join(annotation_root, no_drive.lstrip("/\\"), stem + ".data"))
        if relpath:
            cands.append(os.path.join(annotation_root, relpath + ".data"))
            cands.append(os.path.join(annotation_root, os.path.splitext(relpath)[0] + ".data"))
        cands.append(os.path.join(annotation_root, stem + ".data"))
    return cands


def find_annotation(audio_path, annotation_root=None, relpath=None):
    for cand in annotation_candidates(audio_path, annotation_root, relpath):
        if os.path.isfile(cand):
            return cand
    return None


def load_segments(data_path):
    segs = SegmentList()
    segs.parseJSON(data_path, silent=True)
    return segs


def is_verified(label, min_certainty=50):
    """True for labels a person has checked: confirmed in review (certainty 100)
    or entered manually (filter "M" or no filter)."""
    if label["species"] == DONT_KNOW:
        return False
    if label["certainty"] >= 100:
        return True
    return label.get("filter", "M") == "M" and label["certainty"] >= min_certainty


def write_detections(out_path, detections, duration, filter_name):
    """Write detections as an AviaNZ .data file for review in the AviaNZ GUI.

    detections: iterable of (start, end, species, prob). Labels get certainty 50
    (AviaNZ's "unconfirmed" yellow), so they are excluded from training data by
    build_dataset until a reviewer confirms them (certainty 100).
    """
    segs = SegmentList()
    segs.metadata = {
        "Operator": "Auto",
        "Reviewer": "",
        "Duration": duration,
        "noiseLevel": None,
        "noiseTypes": [],
    }
    for start, end, species, prob in sorted(detections):
        segs.addSegment(Segment(start, end, 0, 0, [{
            "species": species,
            "certainty": 50,
            "filter": filter_name,
            "calltype": "nn",
            "prob": round(float(prob), 4),
        }]))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    segs.saveJSON(tmp)
    os.replace(tmp, out_path)
