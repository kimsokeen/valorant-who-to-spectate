"""Screen regions of the broadcast overlay (minimap, HUD bar, kill feed...).

Regions are stored as fractions of the frame (0.0 to 1.0) so the same config
works for 720p and 1080p VODs. Calibrate them once per broadcast overlay with
notebooks/01_get_data.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class Region:
    name: str
    x: float  # left edge, fraction of frame width
    y: float  # top edge, fraction of frame height
    w: float  # width, fraction of frame width
    h: float  # height, fraction of frame height

    def pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) in pixels for a frame of the given size."""
        return (
            round(self.x * frame_w),
            round(self.y * frame_h),
            round(self.w * frame_w),
            round(self.h * frame_h),
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0, y0, rw, rh = self.pixels(w, h)
        return image[y0:y0 + rh, x0:x0 + rw]


def load_regions(path: str | Path) -> dict[str, Region]:
    data = yaml.safe_load(Path(path).read_text())
    return {name: Region(name, **box) for name, box in data["regions"].items()}


def draw_regions(image: np.ndarray, regions: dict[str, Region]) -> np.ndarray:
    """Copy of the frame with every region outlined and labelled.

    Use this to check a calibration by eye.
    """
    out = image.copy()
    h, w = out.shape[:2]
    for region in regions.values():
        x0, y0, rw, rh = region.pixels(w, h)
        cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 255), 2)
        cv2.putText(out, region.name, (x0 + 4, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def draw_grid(image: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Copy of the frame with a labelled grid every `step` (as a fraction).

    Read region coordinates straight off the grid when calibrating.
    """
    out = image.copy()
    h, w = out.shape[:2]
    n = int(round(1 / step))
    for i in range(1, n):
        f = i * step
        x, y = round(f * w), round(f * h)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, f"{f:.2f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(out, f"{f:.2f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out
