"""Which agents are playing, read from the player cards (live mode).

The VCT overlay shows 5 player cards per team at the screen's sides. Each
card has the agent's portrait in its top corner (facing inwards: mirrored on
the right team) and a health bar in the team's colour (teal or red, the same
colours as the minimap rings). The portraits are far bigger than the 14 px
minimap icons, so they tell look-alike agents apart.

The cards crop each face their own way, so templates are learned from real
cards (scripts/build_card_art.py -> configs/cards/<Agent>.png, facing right).
Positions are in team_left / team_right crop pixels (crops cut with an 8%
margin, 1080p), measured on Champions 2026 frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SLOT_Y0, SLOT_STEP = 50, 104.3          # top of the first portrait, distance between cards
PORTRAIT_X = {"left": 17, "right": 271}  # left edge of the portrait in the crop
PORTRAIT_W, PORTRAIT_H = 66, 36          # portrait box, from 5 px above the card top
INNER = 7                                # template = portrait minus this margin: room for small shifts
TEAL = lambda h: (h > 75) & (h < 100)
RED = lambda h: (h < 8) | (h > 168)


def _box(side: str, slot: int) -> tuple[int, int]:
    return PORTRAIT_X[side], round(SLOT_Y0 + SLOT_STEP * slot) - 5


def portrait(crop: np.ndarray, side: str, slot: int) -> np.ndarray:
    """One card's portrait box, facing right (right-team portraits flipped)."""
    x, y = _box(side, slot)
    p = crop[y:y + PORTRAIT_H, x:x + PORTRAIT_W]
    if p.shape[:2] != (PORTRAIT_H, PORTRAIT_W):
        p = cv2.copyMakeBorder(p, 0, PORTRAIT_H - p.shape[0], 0, PORTRAIT_W - p.shape[1], cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(p[:, ::-1]) if side == "right" else p


def load_card_art(folder: str | Path) -> dict[str, np.ndarray]:
    """{agent: template} (the inner part of the learned portrait)."""
    out = {}
    for p in sorted(Path(folder).glob("*.png")):
        im = cv2.imread(str(p))
        out[p.stem] = np.ascontiguousarray(im[INNER:-INNER, INNER:-INNER])
    if not out:
        raise FileNotFoundError(f"No card art in {folder}. Run scripts/build_card_art.py.")
    return out


def read_slot(crop: np.ndarray, side: str, slot: int, art: dict) -> list[tuple[float, str]]:
    """[(score, agent)] best first, for one card's portrait."""
    win = portrait(crop, side, slot)
    scores = []
    for agent, t in art.items():
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        scores.append((float(np.nan_to_num(r, nan=-1).max()), agent))
    return sorted(scores, reverse=True)


def team_colour(crop: np.ndarray) -> str | None:
    """The cards' health-bar colour: 'teal', 'red', or None if unclear."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    strong = (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    teal, red = int((TEAL(hsv[..., 0]) & strong).sum()), int((RED(hsv[..., 0]) & strong).sum())
    if max(teal, red) < 300 or min(teal, red) * 3 > max(teal, red):
        return None
    return "teal" if teal > red else "red"
