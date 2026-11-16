"""Which agent each player is on, read from the kill feed portraits.

Every kill feed row ends with the victim's agent portrait, drawn MIRRORED
(facing the killer). We already know the victim's name from the kill feed
reader, so matching that portrait against valorant-api.com's killfeedPortrait
art (flipped) and voting over a player's deaths gives his agent. On the Grand
Final this scores ~0.90 per match and every player's vote is unanimous.

Agent art comes from scripts/get_agent_icons.py (configs/agents/<Agent>/).
"""

from __future__ import annotations

import collections
import glob
from pathlib import Path

import cv2
import numpy as np

from .crops import with_margin
from .killfeed import (MAX_ROWS, ROW_H, ROW_STEP, ROW_Y0, match_pair, split_row,
                       victim_name_area, white_text)
from .regions import Region
from .roster import ocr_line


HEIGHTS = (28, 31, 34)            # portrait height on a 1080p frame (art is 2:1)


def load_killfeed_art(folder: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{agent: (mirrored BGR, mirrored mask)} from configs/agents/*/killfeed.png."""
    art = {}
    for p in sorted(glob.glob(str(Path(folder) / "*" / "killfeed.png"))):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        bgr = np.ascontiguousarray(im[:, ::-1, :3])
        mask = np.ascontiguousarray((im[:, ::-1, 3] > 128).astype(np.uint8) * 255)
        art[Path(p).parent.name] = (bgr, mask)
    if not art:
        raise FileNotFoundError(f"No agent art in {folder}. Run scripts/get_agent_icons.py first.")
    return art


def best_agent(window: np.ndarray, art: dict, scale: float = 1.0) -> tuple[float, str | None]:
    best = (-1.0, None)
    for name, (bgr, mask) in art.items():
        for h0 in HEIGHTS:
            h = round(h0 * scale)
            w = 2 * h
            if window.shape[0] < h or window.shape[1] < w:
                continue
            t = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            r = cv2.matchTemplate(window, t, cv2.TM_CCOEFF_NORMED, mask=m)
            r[~np.isfinite(r)] = -1
            if r.max() > best[0]:
                best = (float(r.max()), name)
    return best


def read_agents(map_dir: str | Path, regions: dict[str, Region], roster: dict[str, str],
                art: dict, every: int = 6, min_score: float = 0.7) -> dict[str, dict]:
    """{player: {"agent", "votes", "of", "score"}} from the map's kill feed crops."""
    import json
    d = Path(map_dir)
    m = json.loads((d / "manifest.json").read_text())
    fw, fh = m["video_info"]["width"], m["video_info"]["height"]
    s = fw / 1920
    _, by, _, _ = with_margin(regions["kill_feed"], m["margin"]).pixels(fw, fh)
    votes = collections.defaultdict(collections.Counter)
    scores = collections.defaultdict(list)
    for p in sorted((d / "kill_feed").iterdir())[::every]:
        img = cv2.imread(str(p))
        for k in range(MAX_ROWS):
            y0 = round((ROW_Y0 + ROW_STEP * k) * s) - by
            y1 = y0 + round(ROW_H * s)
            if y1 > img.shape[0]:
                break
            parts = split_row(img[y0:y1])
            if parts is None:
                continue
            victim = match_pair("", ocr_line(white_text(victim_name_area(parts[1]))), roster)[1]
            if victim is None:
                continue
            row = img[max(0, y0 - 3):y1 + 3]
            sc, agent = best_agent(row[:, round(395 * s):], art, s)
            if agent and sc >= min_score:
                votes[victim][agent] += 1
                scores[victim].append(sc)
    out = {}
    for player in roster:
        c = votes[player]
        if c:
            agent, n = c.most_common(1)[0]
            out[player] = {"agent": agent, "votes": n, "of": sum(c.values()),
                           "score": round(float(np.median(scores[player])), 2)}
        else:
            out[player] = {"agent": None, "votes": 0, "of": 0, "score": 0.0}
    return out
