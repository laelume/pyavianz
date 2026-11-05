"""Bridge to the AviaNZ package that lives in this repository.

AviaNZ is the ``avianz`` package at ``<repo>/avianz`` and imports itself by absolute
paths (``avianz.src.core...``), so only the repository root must be on ``sys.path``.
Doing it here instead of ``pip install`` means the HPC environment only needs the AviaNZ
*runtime* dependencies (numpy, scipy, pyfftw, resampy, soundfile, ...) and not
the GUI stack (PyQt6, pyqtgraph) or TensorFlow that ``pip install avianz`` pulls in.

Override the repository location with the AVIANZ_REPO environment variable.
"""

import os
import sys

REPO_ROOT = os.path.abspath(
    os.environ.get("AVIANZ_REPO", os.path.join(os.path.dirname(__file__), "..", ".."))
)
_AVIANZ_PKG = os.path.join(REPO_ROOT, "avianz")

if not os.path.isfile(os.path.join(_AVIANZ_PKG, "AviaNZ.py")):
    raise ImportError(
        f"AviaNZ source not found at {_AVIANZ_PKG}. Set AVIANZ_REPO to the repository root."
    )

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avianz.src.core.spectrogram import Spectrogram  # noqa: E402
from avianz.src.core.annotation import Segment, SegmentList  # noqa: E402

__all__ = ["REPO_ROOT", "Spectrogram", "Segment", "SegmentList"]
