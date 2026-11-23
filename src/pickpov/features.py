"""Per-player features at one moment, for the learned picker.

For a frame (who is where), the previous frames (to see movement) and the
door states, every located alive player gets a row of numbers describing his
situation. A model learns from these which player is about to get a kill.
"""

from __future__ import annotations

import math

import numpy as np

FEATURES = [
    "walk_nearest_enemy", "straight_nearest_enemy",
    "enemies_walk_60", "enemies_walk_90", "enemies_walk_150",
    "visible_enemies_150", "visible_enemies_any", "visible_nearest_enemy",
    "mates_walk_90", "mates_alive", "enemies_alive",
    "speed", "closing_speed",
    "round_time", "held",
    "door_A", "door_B", "door_mid",
    "x", "y",
]


def player_features(frame: dict, prev: dict | None, prev_dt: float, summit, closed: tuple,
                    round_time: float, held: set[str]) -> dict[str, list[float]]:
    """{player: [values in FEATURES order]} for every player in `frame`.

    frame/prev: {player: (x, y, side)}; prev is the frame ~prev_dt seconds earlier.
    """
    out = {}
    for p, (x, y, side) in frame.items():
        enemies = [(q, qx, qy) for q, (qx, qy, qs) in frame.items() if qs != side]
        mates = [(q, qx, qy) for q, (qx, qy, qs) in frame.items() if qs == side and q != p]
        walk = [summit.walking_distance((x, y), (qx, qy), closed) for _, qx, qy in enemies]
        straight = [math.hypot(qx - x, qy - y) for _, qx, qy in enemies]
        vis = [s for (_, qx, qy), s in zip(enemies, straight) if summit.line_of_sight((x, y), (qx, qy), closed)]
        mate_walk = [summit.walking_distance((x, y), (qx, qy), closed) for _, qx, qy in mates]
        speed = closing = 0.0
        if prev and p in prev and prev_dt > 0:
            px, py, _ = prev[p]
            speed = math.hypot(x - px, y - py) / prev_dt
            if enemies:
                # how fast the walk to the (currently) nearest enemy shrank
                i = int(np.argmin(walk))
                q = enemies[i][0]
                if q in prev:
                    before = summit.walking_distance((px, py), prev[q][:2], closed)
                    closing = (before - walk[i]) / prev_dt
        big = 999.0
        out[p] = [
            min(walk) if walk else big, min(straight) if straight else big,
            sum(d <= 60 for d in walk), sum(d <= 90 for d in walk), sum(d <= 150 for d in walk),
            sum(s <= 150 for s in vis), len(vis), min(vis) if vis else big,
            sum(d <= 90 for d in mate_walk), len(mates) + 1, len(enemies),
            speed, closing,
            round_time, float(p in held),
            float(closed[0]) if closed else 0.0, float(closed[1]) if len(closed) > 1 else 0.0,
            float(closed[2]) if len(closed) > 2 else 0.0,
            x, y,
        ]
    return out
