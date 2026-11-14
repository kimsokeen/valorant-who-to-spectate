"""Read frames from a video at a fixed rate.

Positions on the minimap don't change much in 1/60 of a second, so we only
look at a few frames per second. This keeps processing fast enough for Colab
now, and for live mode later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    time_s: float      # seconds from the start of the video
    index: int         # frame number in the original video
    image: np.ndarray  # BGR image, as OpenCV loads it


def video_info(path: str | Path) -> dict:
    """Basic facts about a video file: fps, frame count, size, duration."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frames": count,
        "width": width,
        "height": height,
        "duration_s": count / fps if fps else 0.0,
    }


def sample_frames(
    path: str | Path,
    every_s: float = 0.5,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[Frame]:
    """Yield one frame every `every_s` seconds between start_s and end_s.

    Uses grab() to skip frames cheaply and only decodes the ones we keep.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        cap.release()
        raise ValueError(f"Video has no frame rate: {path}")

    step = max(1, round(every_s * fps))
    index = int(start_s * fps)
    last = int(end_s * fps) if end_s is not None else None
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    try:
        while last is None or index <= last:
            ok, image = cap.read()
            if not ok:
                break
            yield Frame(time_s=index / fps, index=index, image=image)
            for _ in range(step - 1):  # skip frames without decoding them
                if not cap.grab():
                    return
            index += step
    finally:
        cap.release()


def save_frames(
    path: str | Path,
    out_dir: str | Path,
    every_s: float = 5.0,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Save sampled frames as JPGs named by timestamp. Handy for calibration."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for frame in sample_frames(path, every_s, start_s, end_s):
        name = out / f"t{frame.time_s:08.1f}s.jpg"
        cv2.imwrite(str(name), frame.image)
        saved.append(name)
        if max_frames and len(saved) >= max_frames:
            break
    return saved
