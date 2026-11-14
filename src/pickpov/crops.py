"""Save small crops of the screen regions instead of whole frames.

A 45-minute map at 4 frames per second is ~10,800 frames. As full 1080p
images that is bigger than the video itself, but the regions we care about
(minimap, HUD bar, kill feed, POV name) are a small part of the screen.

Output for one map:

    <out_dir>/
        manifest.json      settings used, video facts, crop count
        index.csv          one row per sample: sample, time_s, frame_index
        preview.jpg        a few samples side by side, to check by eye
        <region>/000000.jpg  one image per sample, per region
                             (slow regions only every `slow_every`-th sample)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .gameplay import GameplayDetector
from .regions import Region
from .video import sample_frames, video_info

# Regions that change slowly are saved only every `slow_every`-th sample.
# Kill feed entries stay on screen for several seconds and the player cards
# (alive, health, money) only need checking once a second.
DEFAULT_SLOW_REGIONS = ("kill_feed", "pov_name", "round_timer", "team_left", "team_right")

# Lossless PNG is ~4x bigger than JPG at quality 95; use it only if text reads badly.
DEFAULT_PNG_REGIONS: tuple[str, ...] = ()


def with_margin(region: Region, margin: float) -> Region:
    """Grow a region by `margin` of its own size on every side, kept on screen.

    Icons at the edge of the minimap get cut off by a tight box; a margin
    is cheap insurance.
    """
    dx, dy = region.w * margin, region.h * margin
    x0, y0 = max(0.0, region.x - dx), max(0.0, region.y - dy)
    x1, y1 = min(1.0, region.x + region.w + dx), min(1.0, region.y + region.h + dy)
    return Region(region.name, x0, y0, x1 - x0, y1 - y0)


def contact_sheet(rows: list[list[np.ndarray]], height: int = 160) -> np.ndarray:
    """Lay out crops as a grid: one row per sample, one column per region."""
    scaled = [[cv2.resize(c, (max(1, round(c.shape[1] * height / c.shape[0])), height))
               for c in row] for row in rows]
    col_w = [max(row[i].shape[1] for row in scaled) for i in range(len(scaled[0]))]
    gap = 6
    sheet = np.full((len(scaled) * (height + gap) + gap,
                     sum(col_w) + gap * (len(col_w) + 1), 3), 40, np.uint8)
    for r, row in enumerate(scaled):
        x = gap
        y = gap + r * (height + gap)
        for i, crop in enumerate(row):
            sheet[y:y + height, x:x + crop.shape[1]] = crop
            x += col_w[i] + gap
    return sheet


def extract_crops(
    video: str | Path,
    regions: dict[str, Region],
    out_dir: str | Path,
    every_s: float = 0.25,
    margin: float = 0.08,
    start_s: float = 0.0,
    end_s: float | None = None,
    slow_regions: tuple[str, ...] = DEFAULT_SLOW_REGIONS,
    slow_every: int = 4,
    png_regions: tuple[str, ...] = DEFAULT_PNG_REGIONS,
    jpg_quality: int = 95,
    preview_samples: int = 6,
    detector: GameplayDetector | None = None,
) -> dict:
    """Crop every region from frames sampled every `every_s` seconds.

    With a `detector`, frames that aren't live gameplay (arena shots, player
    cams, replays, scoreboards) are skipped and only counted.

    Returns the manifest (also written to out_dir/manifest.json).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = {name: with_margin(r, margin) for name, r in regions.items()}
    for name in boxes:
        (out / name).mkdir(exist_ok=True)

    info = video_info(video)
    count = 0
    skipped = 0
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "time_s", "frame_index"])
        for frame in sample_frames(video, every_s, start_s, end_s):
            if detector is not None and not detector.is_gameplay(frame.image):
                skipped += 1
                continue
            for name, box in boxes.items():
                if name in slow_regions and count % slow_every:
                    continue
                crop = box.crop(frame.image)
                if name in png_regions:
                    cv2.imwrite(str(out / name / f"{count:06d}.png"), crop)
                else:
                    cv2.imwrite(str(out / name / f"{count:06d}.jpg"), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            writer.writerow([count, f"{frame.time_s:.3f}", frame.index])
            count += 1

    if count == 0:
        raise RuntimeError(f"No gameplay frames found in {video}; nothing was saved."
                           f" ({skipped} non-gameplay frames skipped)")

    # preview: evenly spaced samples, re-read from disk so it shows what was saved
    # only samples where every region was saved (multiples of slow_every)
    full = np.arange(0, count, slow_every)
    picks = full[np.linspace(0, len(full) - 1, min(preview_samples, len(full))).round().astype(int)]
    rows = []
    for s in picks:
        row = []
        for name in boxes:
            ext = "png" if name in png_regions else "jpg"
            row.append(cv2.imread(str(out / name / f"{s:06d}.{ext}")))
        rows.append(row)
    cv2.imwrite(str(out / "preview.jpg"), contact_sheet(rows))

    manifest = {
        "video": Path(video).name,
        "video_info": info,
        "every_s": every_s,
        "margin": margin,
        "start_s": start_s,
        "end_s": end_s,
        "samples": count,
        "skipped_non_gameplay": skipped,
        "gameplay_share": count / (count + skipped),
        "regions": [name for name in boxes],
        "boxes_with_margin": {n: asdict(b) for n, b in boxes.items()},
        "slow_regions": list(slow_regions),
        "slow_every": slow_every,
        "png_regions": list(png_regions),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def folder_size_mb(path: str | Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6
