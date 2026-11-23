"""Pickers: given where everyone is right now, who should the camera follow?

A frame is {player: (x, y, side)} for the alive players we located.
Every picker returns one player name (or None if there is nobody to pick).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

Frame = dict[str, tuple[float, float, str]]


def load_frames(positions_csv: str | Path) -> list[tuple[float, int, Frame]]:
    """[(time_s, round, frame)] from data/positions/<map>.csv, in time order."""
    frames: dict[float, tuple[int, Frame]] = {}
    with open(positions_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = float(r["time_s"])
            rnd, fr = frames.setdefault(t, (int(r["round"]), {}))
            fr[r["player"]] = (float(r["x"]), float(r["y"]), r["side"])
    return [(t, rnd, fr) for t, (rnd, fr) in sorted(frames.items())]


def nearest_enemy_distance(frame: Frame) -> dict[str, float]:
    """Each player's distance (minimap px) to the closest located enemy."""
    out = {}
    for p, (x, y, side) in frame.items():
        d = [math.hypot(x - ex, y - ey) for q, (ex, ey, es) in frame.items() if es != side]
        if d:
            out[p] = min(d)
    return out


def pick_nearest_enemy(frame: Frame) -> str | None:
    """Rule 1: the player closest to any enemy. Fights start where teams meet."""
    d = nearest_enemy_distance(frame)
    return min(d, key=d.get) if d else None


def enemies_within(frame: Frame, player: str, radius: float = 60.0) -> int:
    x, y, side = frame[player]
    return sum(1 for q, (qx, qy, qs) in frame.items() if qs != side and math.hypot(qx - x, qy - y) <= radius)


def pick_most_contested(frame: Frame, radius: float = 60.0) -> str | None:
    """Rule 2: the player with the most enemies within `radius` px; ties go to
    whoever is closest to an enemy. Best simple rule so far: on the killer 22.7%
    of the time 2.5 s before a kill, on the fight 48.4% (random: 14.6% / 29.0%)."""
    d = nearest_enemy_distance(frame)
    if not d:
        return None
    return max(d, key=lambda p: (enemies_within(frame, p, radius), -d[p]))
