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


# ---------- pickers that know about walls (need pickpov.walls.Summit + door states) ----------

def pick_most_visible(frame: Frame, summit, closed: tuple, radius: float = 150.0) -> str | None:
    """Rule 3: the player who can SEE the most enemies within `radius` px (walls and
    closed doors block sight); ties go to whoever has the closest visible enemy,
    then the closest enemy at all."""
    best, key = None, None
    for p, (x, y, side) in frame.items():
        vis, near_vis, near_any = 0, 1e9, 1e9
        for q, (qx, qy, qs) in frame.items():
            if qs == side:
                continue
            d = math.hypot(qx - x, qy - y)
            near_any = min(near_any, d)
            if d <= radius and summit.line_of_sight((x, y), (qx, qy), closed):
                vis += 1
                near_vis = min(near_vis, d)
        k = (vis, -near_vis, -near_any)
        if near_any < 1e9 and (key is None or k > key):
            best, key = p, k
    return best


def pick_nearest_walking(frame: Frame, summit, closed: tuple) -> str | None:
    """Rule 4: the player with the shortest walking path (around walls and closed
    doors) to any enemy."""
    best, bd = None, float("inf")
    for p, (x, y, side) in frame.items():
        for q, (qx, qy, qs) in frame.items():
            if qs != side:
                d = summit.walking_distance((x, y), (qx, qy), closed)
                if d < bd:
                    best, bd = p, d
    return best


def pick_contested_walking(frame: Frame, summit, closed: tuple, radius: float = 90.0) -> str | None:
    """Rule 5 (best so far): the player with the most enemies within `radius` px
    of WALKING distance (around walls and closed doors); ties go to the shortest
    walk to an enemy. Over 18 maps, 2.5 s before each kill: killer 24.0%, fight
    50.9% (straight-line version 21.8% / 46.7%; broadcast 27.7% / 55.2%).
    The radius was chosen on these same maps, so it is slightly flattered."""
    best, key = None, None
    for p, (x, y, side) in frame.items():
        ds = [summit.walking_distance((x, y), (qx, qy), closed)
              for q, (qx, qy, qs) in frame.items() if qs != side]
        if not ds:
            continue
        k = (sum(d <= radius for d in ds), -min(ds))
        if key is None or k > key:
            best, key = p, k
    return best
