"""Each player's health, read from the VCT player cards.

The cards show HP as a number ("100", "84") at a fixed spot; OCR reads it with
digits only. During the buy phase the cards use a bigger layout and the number
isn't there; dead players' cards show other text. Both are filtered out by the
caller using what we already know (round timing, deaths).

In live use the same numbers would come from the spectator top bar instead:
the model only sees "player X has N HP", not where it was read.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from .regions import Region
from .roster import _box_in_crop, ocr_line

HP_X = {"team_left": (283, 326), "team_right": (1594, 1638)}   # 1080p frame px
NAME_Y0, CARD_STEP, BOX_H = 558, 106, 26
DIGITS = "--psm 7 -c tessedit_char_whitelist=0123456789"


def read_hp(map_dir: str | Path, sample: int, regions: dict[str, Region],
            slots: dict[tuple[str, int], str], margin: float, fw: int, fh: int) -> dict[str, int]:
    """{player: hp} from one card sample; players whose number doesn't read as 1-100 are left out.

    slots: {(side "left"/"right", card index 0-4): player}.
    """
    out = {}
    d = Path(map_dir)
    for side, key in (("left", "team_left"), ("right", "team_right")):
        img = cv2.imread(str(d / key / f"{sample:06d}.jpg"))
        if img is None:
            continue
        for k in range(5):
            p = slots.get((side, k))
            if p is None:
                continue
            box = (HP_X[key][0], NAME_Y0 + CARD_STEP * k, HP_X[key][1], NAME_Y0 + CARD_STEP * k + BOX_H)
            x0, y0, x1, y1 = _box_in_crop(box, regions[key], margin, fw, fh)
            txt = ocr_line(img[max(0, y0):y1, max(0, x0):x1], DIGITS)
            if txt.isdigit() and 1 <= int(txt) <= 100:
                out[p] = int(txt)
    return out


def card_slots(roster_json: str | Path) -> dict[tuple[str, int], str]:
    r = json.loads(Path(roster_json).read_text())
    return {(p["side"], p["slot"]): p["name"] for p in r["players"]}
