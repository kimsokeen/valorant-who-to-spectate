"""Tell live gameplay apart from arena shots, player cams, replays and breaks.

During live play the broadcast HUD shows fixed text that never changes
between matches, e.g. "CURRENT:" in the top-left and "ROUND" above the timer.
We keep a small picture ("marker") of each and check whether it is on screen.

A marker can be `slides: true`: its row is fixed but it can move left/right.
"CURRENT:" does this: on map 2+ of a series the previous map is listed first,
pushing it right (x=20 px on map 1, x=222 px on map 2).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .regions import Region

SEARCH_PAD = 6  # pixels of wiggle room around each marker


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _scale(img: np.ndarray, frame_w: int) -> np.ndarray:
    """Markers are stored at 1920 px wide; rescale for 720p etc."""
    if frame_w == 1920:
        return img
    f = frame_w / 1920
    return cv2.resize(img, (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f))))


def frames_at(path: str | Path, times_s) -> list[tuple[float, np.ndarray]]:
    """Read single frames at the given times (seeking, so sparse reads are fast)."""
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if ok:
            out.append((t, img))
    cap.release()
    return out


class GameplayDetector:
    def __init__(self, markers: dict[str, tuple[Region, np.ndarray]], threshold: float = 0.7,
                 sliding: set[str] | None = None):
        self.markers = markers
        self.threshold = threshold
        self.sliding = sliding or set()

    @classmethod
    def load(cls, config_path: str | Path) -> "GameplayDetector":
        config_path = Path(config_path)
        data = yaml.safe_load(config_path.read_text())
        g = data["gameplay"]
        markers, sliding = {}, set()
        for name, box in g["markers"].items():
            box = dict(box)
            if box.pop("slides", False):
                sliding.add(name)
            img = cv2.imread(str(config_path.parent / "markers" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing marker image for '{name}'. Run scripts/build_markers.py first.")
            markers[name] = (Region(name, **box), img)
        return cls(markers, g.get("threshold", 0.7), sliding)

    def score(self, frame: np.ndarray) -> float:
        """Lowest match score over all markers (1.0 = perfect). All must match."""
        h, w = frame.shape[:2]
        gray = _gray(frame)
        scores = []
        for name, (region, marker) in self.markers.items():
            x, y, rw, rh = region.pixels(w, h)
            x0, y0 = max(0, x - SEARCH_PAD), max(0, y - SEARCH_PAD)
            if name in self.sliding:  # search the whole row
                window = gray[y0:y + rh + SEARCH_PAD, :]
            else:
                window = gray[y0:y + rh + SEARCH_PAD, x0:x + rw + SEARCH_PAD]
            tmpl = _scale(marker, w)
            if window.shape[0] < tmpl.shape[0] or window.shape[1] < tmpl.shape[1]:
                return 0.0
            scores.append(float(cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED).max()))
        return min(scores)

    def is_gameplay(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


def build_markers(video: str | Path, regions: dict[str, Region], samples: int = 300) -> dict[str, np.ndarray]:
    """Learn what each marker looks like from a VOD, without labelling anything.

    Takes `samples` frames spread over the video, crops each marker region, and
    picks the crop that matches the most other crops. Gameplay is most of a map,
    so that crop is the in-game HUD, not a camera shot.
    """
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames = [img for _, img in frames_at(video, np.linspace(5, duration - 5, samples))]
    out = {}
    for name, region in regions.items():
        crops = []
        for img in frames:
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, round(img.shape[0] * 1920 / img.shape[1])))
            crops.append(_gray(region.crop(img)).astype(np.float32))
        flat = np.stack([c.ravel() - c.mean() for c in crops])
        norms = np.linalg.norm(flat, axis=1) + 1e-6
        sim = (flat @ flat.T) / np.outer(norms, norms)
        votes = (sim > 0.8).sum(axis=1)
        out[name] = crops[int(votes.argmax())].astype(np.uint8)
    return out
