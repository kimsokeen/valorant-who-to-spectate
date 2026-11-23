"""Broadcast POV data helpers that need no OpenCV.

Kept separate from pov.py (which reads images with OpenCV) because loading
OpenCV and LightGBM in the same program can crash on Windows: both ship their
own copy of the OpenMP library. Scripts that train with LightGBM import this.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path

FILL_S = 3.0


def load_pov(pov_csv: str | Path) -> list[tuple[float, str | None]]:
    with open(pov_csv, newline="", encoding="utf-8") as f:
        return [(float(r["time_s"]), r["player"] or None) for r in csv.DictReader(f)]


def remove_flicker(pov: list[tuple[float, str | None]]) -> list[tuple[float, str | None]]:
    """Observers don't cut away for one second and back: a single reading that
    differs from both neighbours, while the neighbours agree, is a misread."""
    known = [(t, p) for t, p in pov if p]
    out = list(known)
    for i in range(1, len(known) - 1):
        a, b, c = known[i - 1][1], known[i][1], known[i + 1][1]
        if a == c != b:
            out[i] = (known[i][0], a)
    return out


def pov_at(pov: list[tuple[float, str | None]], times: list[float]) -> dict[float, str | None]:
    """The broadcast's POV at each time: the latest reading at or before it,
    carried over gaps of up to FILL_S seconds, after removing one-second flicker."""
    known = remove_flicker(pov)
    kt = [t for t, _ in known]
    out = {}
    for t in times:
        i = bisect.bisect_right(kt, t) - 1
        out[t] = known[i][1] if i >= 0 and t - kt[i] <= FILL_S else None
    return out
