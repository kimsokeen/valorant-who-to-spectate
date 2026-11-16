"""Find rounds and their winners from the score digits in the HUD.

We never read the digits. A team's score looks exactly the same until that
team wins a round, so comparing the score area at the start of consecutive
gameplay stretches tells us:

    neither score changed -> the stretches are one round (split by a pause)
    one score changed     -> one round ended, won by that side
    both scores changed   -> two rounds ended (they ran together)

Measured on the Champions 2026 overlay: an unchanged score differs by exactly
0% of pixels, a changed one by 2.3% or more (8 -> 9 is the smallest change).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .crops import with_margin
from .regions import Region

SCORE_BOX = {"left": (800, 8, 856, 58), "right": (1066, 8, 1124, 58)}   # 1080p frame pixels
STRETCH_GAP_S = 5.0     # a gap this long between gameplay samples starts a new stretch
CHANGED_PCT = 1.0       # % of score pixels that must change by > PIXEL_DELTA
PIXEL_DELTA = 60


def _stretches(times: dict[int, float]) -> list[list[int]]:
    order = sorted(times)
    out = [[order[0]]] if order else []
    for a, b in zip(order, order[1:]):
        if times[b] - times[a] > STRETCH_GAP_S:
            out.append([])
        out[-1].append(b)
    return out


def detect_rounds(map_dir: str | Path, regions: dict[str, Region]) -> dict:
    """Assign every gameplay sample to a round and find who won each round.

    Returns {"round_of": {sample: round}, "rounds": n, "winners": {round: side},
    "spans": [[start_s, end_s], ...], "left_wins": n, "right_wins": n}.
    The last round's winner isn't visible (no stretch follows it); see
    finish_last_round().
    """
    d = Path(map_dir)
    m = json.loads((d / "manifest.json").read_text())
    fw, fh = m["video_info"]["width"], m["video_info"]["height"]
    s = fw / 1920
    hx, hy, _, _ = with_margin(regions["hud_top"], m["margin"]).pixels(fw, fh)
    times = {}
    with open(d / "index.csv") as f:
        for r in csv.DictReader(f):
            times[int(r["sample"])] = float(r["time_s"])

    def score_area(sample: int, side: str) -> np.ndarray:
        img = cv2.imread(str(d / "hud_top" / f"{sample:06d}.jpg"))
        x0, y0, x1, y1 = (round(v * s) for v in SCORE_BOX[side])
        return cv2.cvtColor(img[y0 - hy:y1 - hy, x0 - hx:x1 - hx], cv2.COLOR_BGR2GRAY).astype(np.float32)

    stretches = _stretches(times)
    reps = []
    for st in stretches:
        t0 = times[st[0]]
        # a few samples early in the stretch (buy phase: score settled, nothing moving)
        picks = [x for x in st if t0 + 3 <= times[x] <= t0 + 15][::8][:3] or st[:1]
        reps.append({side: np.median([score_area(x, side) for x in picks], axis=0) for side in SCORE_BOX})

    round_of, winners = {}, {}
    rnd = 0
    spans = []
    for i, st in enumerate(stretches):
        if i > 0:
            changed = [side for side in SCORE_BOX
                       if (np.abs(reps[i][side] - reps[i - 1][side]) > PIXEL_DELTA).mean() * 100 > CHANGED_PCT]
            if len(changed) == 1:
                winners[rnd] = changed[0]
                rnd += 1
            elif len(changed) == 2:          # two rounds ended: the earlier stretch held both
                rnd += 2
        if len(spans) == rnd + 1:
            spans[rnd][1] = times[st[-1]]
        else:
            while len(spans) < rnd:
                spans.append([times[st[0]], times[st[0]]])   # a round we never saw
            spans.append([times[st[0]], times[st[-1]]])
        for x in st:
            round_of[x] = rnd
    return {"round_of": round_of, "rounds": rnd + 1, "winners": winners, "spans": spans,
            "left_wins": sum(w == "left" for w in winners.values()),
            "right_wins": sum(w == "right" for w in winners.values())}


def finish_last_round(info: dict, final_score: tuple[int, int] | None) -> None:
    """Fill in the last round's winner from the final score (higher, lower).

    The map winner always wins the last round, so the side one short of the
    winning score is the one that took it.
    """
    if not final_score:
        return
    hi, lo = max(final_score), min(final_score)
    last = info["rounds"] - 1
    if last in info["winners"]:
        return
    if info["left_wins"] == hi - 1 and info["right_wins"] == lo:
        info["winners"][last] = "left"; info["left_wins"] += 1
    elif info["right_wins"] == hi - 1 and info["left_wins"] == lo:
        info["winners"][last] = "right"; info["right_wins"] += 1
