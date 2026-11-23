"""Who the real broadcast was showing: the POV name label ("LOUD luk xo").

The label sits above the ability bar during live play. We OCR it (once per
second, the rate pov_name crops are saved at) and match the text to the map's
10 players. During the buy phase the expanded player cards can cover the label,
so some seconds read as nothing; the POV rarely changes that fast, so a missing
reading is filled with the last one for up to FILL_S seconds.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path

import cv2

from .killfeed import match_player, white_text
from .roster import ocr_line

FILL_S = 3.0


def read_pov(map_dir: str | Path, players: list[str]) -> list[tuple[int, float, str | None]]:
    """[(sample, time_s, player or None)] for every pov_name crop of a map."""
    d = Path(map_dir)
    times = {}
    with open(d / "index.csv") as f:
        for r in csv.DictReader(f):
            times[int(r["sample"])] = float(r["time_s"])
    out = []
    for p in sorted((d / "pov_name").iterdir()):
        s = int(p.stem)
        img = cv2.imread(str(p))
        name, _ = match_player(ocr_line(white_text(img)), players)
        out.append((s, times[s], name))
    return out


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
