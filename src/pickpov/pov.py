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


from .pov_data import FILL_S, load_pov, pov_at, remove_flicker  # noqa: E402,F401  (re-exported)
