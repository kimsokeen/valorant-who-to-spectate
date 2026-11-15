"""Read the 10 player names (and team tags) for a map from its crops.

The broadcast's player cards show each name in a fixed spot. Reading them with
OCR (Tesseract) on a few dozen samples and taking the most common reading gives
the exact spelling the broadcast uses, which is also what the kill feed shows.
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import os
import shutil

import pytesseract

# On Windows the installer doesn't add Tesseract to PATH; find it in the usual place.
if shutil.which("tesseract") is None:
    for _p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
               r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

from .crops import with_margin
from .regions import Region

# Name text boxes on a 1920x1080 frame (measured on the Champions 2026 overlay).
# Cards are 106 px apart; the first name sits at y=558..584.
NAME_X = {"team_left": (82, 240), "team_right": (1680, 1842)}
NAME_Y0, NAME_H, CARD_STEP = 558, 26, 106
# Team tags in the score bar ("VIT", "LOUD").
TAG_BOX = {"left": (722, 10, 792, 38), "right": (1140, 12, 1192, 36)}
RECHECK_BELOW = 0.8   # names read less consistently than this get re-read one by one
REF_W, REF_H = 1920, 1080


def ocr_line(img: np.ndarray, config: str = "--psm 7") -> str:
    """OCR one line of light text on a dark background."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(b, config=config).strip()


def ocr_lines(imgs: list[np.ndarray]) -> list[str]:
    """OCR many one-line images in a single Tesseract call (much faster).

    Images are prepared like ocr_line, stacked vertically with gaps, and read
    as a block; each output line is one image (lines that fail are dropped).
    """
    prepped = []
    for img in imgs:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        prepped.append(b)
    if not prepped:
        return []
    w = max(b.shape[1] for b in prepped)
    gap = np.full((30, w), 255, np.uint8)
    rows = []
    for b in prepped:
        rows += [np.pad(b, ((0, 0), (0, w - b.shape[1])), constant_values=255), gap]
    text = pytesseract.image_to_string(np.vstack(rows), config="--psm 6")
    return [line.strip() for line in text.splitlines() if line.strip()]


def clean_name(text: str) -> str:
    """Drop OCR junk around a name: 'tkzin,' -> 'tkzin', '| Darker' -> 'Darker'."""
    text = re.sub(r"[^A-Za-z0-9 _.\-]", " ", text)
    words = [w.strip("._-") for w in text.split()]
    # junk is usually a stray 1-2 character "word" at either end
    while len(words) > 1 and len(words[0]) <= 2 and not words[0].isdigit():
        words.pop(0)
    while len(words) > 1 and len(words[-1]) <= 1:
        words.pop()
    return " ".join(w for w in words if w)


def _box_in_crop(box_px, region: Region, margin: float, frame_w: int, frame_h: int):
    """Convert a 1080p frame box to coordinates inside a saved (margined) crop."""
    s = frame_w / REF_W
    bx, by, _, _ = with_margin(region, margin).pixels(frame_w, frame_h)
    x0, y0, x1, y1 = (round(v * s) for v in box_px)
    return x0 - bx, y0 - by, x1 - bx, y1 - by


def build_roster(map_dir: str | Path, regions: dict[str, Region], samples: int = 40,
                 seed: int = 0) -> dict:
    """Most common reading of each name slot and team tag, per half.

    Halves are split at the longest gap in gameplay (halftime). If a team's
    names come out the same in both halves, the cards didn't swap sides.
    """
    d = Path(map_dir)
    m = json.loads((d / "manifest.json").read_text())
    fw, fh = m["video_info"]["width"], m["video_info"]["height"]
    times = {}
    for line in (d / "index.csv").read_text().splitlines()[1:]:
        s, t, _ = line.split(",")
        times[int(s)] = float(t)
    order = sorted(times)
    gaps = [(times[b] - times[a], b) for a, b in zip(order, order[1:])]
    halftime_sample = max(gaps)[1] if gaps else order[-1] + 1

    files = sorted(p.name for p in (d / "team_left").iterdir())
    halves = {"first": [f for f in files if int(f[:6]) < halftime_sample],
              "second": [f for f in files if int(f[:6]) >= halftime_sample]}
    rng = random.Random(seed)
    out = {"map": d.name, "halftime_time_s": times.get(halftime_sample), "halves": {}}
    for half, pool in halves.items():
        pick = rng.sample(pool, min(samples, len(pool)))
        res = {}
        for side in ("team_left", "team_right"):
            slots = [[] for _ in range(5)]
            for f in pick:
                img = cv2.imread(str(d / side / f))
                for k in range(5):
                    box = (NAME_X[side][0], NAME_Y0 + CARD_STEP * k,
                           NAME_X[side][1], NAME_Y0 + CARD_STEP * k + NAME_H)
                    x0, y0, x1, y1 = _box_in_crop(box, regions[side], m["margin"], fw, fh)
                    slots[k].append(img[max(0, y0):y1, max(0, x0):x1])
            res[side] = []
            for crops in slots:
                v = collections.Counter(n for n in map(clean_name, ocr_lines(crops)) if n)
                n = len(crops)
                if not v or v.most_common(1)[0][1] / n < RECHECK_BELOW:
                    # slower but more accurate: one image per call
                    crops = crops[:12]
                    n = len(crops)
                    v = collections.Counter(x for x in (clean_name(ocr_line(c)) for c in crops) if x)
                res[side].append({"name": v.most_common(1)[0][0] if v else "",
                                  "agreement": round(v.most_common(1)[0][1] / n, 2) if v else 0.0})
        tags = {}
        for side, key in (("left", "team_left"), ("right", "team_right")):
            x0, y0, x1, y1 = _box_in_crop(TAG_BOX[side], regions["hud_top"], m["margin"], fw, fh)
            crops = [cv2.imread(str(d / "hud_top" / f.replace(".png", ".jpg")))[max(0, y0):y1, max(0, x0):x1]
                     for f in pick[:15]]
            caps = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            c = collections.Counter(t for t in (clean_name(ocr_line(x, caps)) for x in crops[:8]) if t)
            tags[side] = c.most_common(1)[0][0] if c else ""
        res["tags"] = tags
        out["halves"][half] = res
    return out
