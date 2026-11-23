"""Score a picker against what actually happened: the kills.

For every kill with a known killer we look at the pick made LEAD seconds
before the kill appeared in the kill feed:

    caught the killer : the pick was the killer        (the main goal)
    caught the fight  : the pick was killer or victim   (partial success)

A random pick among the located alive players gives the chance level, so a
picker has to beat that to be worth anything.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Callable


def load_kills(kills_csv: str | Path) -> list[dict]:
    with open(kills_csv, newline="", encoding="utf-8") as f:
        return [dict(r, time_s=float(r["time_s"]), round=int(r["round"])) for r in csv.DictReader(f)]


def score_picks(frames: list, kills: list[dict], picks: dict[float, str | None],
                lead: float = 2.5, max_gap: float = 0.6, skip_unknown: bool = False) -> dict:
    """frames: [(t, round, frame)] (load_frames); picks: {t: player}.

    Kills without a killer, or without a frame within max_gap of (kill - lead)
    in the same round, are skipped and counted. With skip_unknown, kills where
    the pick is unknown (None) are skipped too: used for the real broadcast,
    whose POV label is sometimes unreadable, so it isn't blamed for our misreads.
    """
    times = [t for t, _, _ in frames]
    n = killer = fight = 0
    chance_killer = chance_fight = 0.0
    skipped = 0
    for k in kills:
        if not k["killer"]:
            skipped += 1
            continue
        target = k["time_s"] - lead
        i = bisect.bisect_left(times, target)
        best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - target), default=None)
        if best is None or abs(times[best] - target) > max_gap or frames[best][1] != k["round"]:
            skipped += 1
            continue
        t, _, frame = frames[best]
        if not frame:
            skipped += 1
            continue
        pick = picks.get(t)
        if skip_unknown and pick is None:
            skipped += 1
            continue
        n += 1
        killer += pick == k["killer"]
        fight += pick in (k["killer"], k["victim"])
        m = len(frame)
        chance_killer += (k["killer"] in frame) / m
        chance_fight += ((k["killer"] in frame) + (k["victim"] in frame)) / m
    return {"kills_scored": n, "skipped": skipped,
            "caught_killer": killer / n if n else 0.0, "caught_fight": fight / n if n else 0.0,
            "chance_killer": chance_killer / n if n else 0.0, "chance_fight": chance_fight / n if n else 0.0}


def run_picker(frames: list, picker: Callable) -> dict[float, str | None]:
    return {t: picker(frame) for t, _, frame in frames}
