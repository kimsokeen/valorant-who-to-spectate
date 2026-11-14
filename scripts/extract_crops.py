"""Crop the screen regions from one video, at a few samples per second.

Example:
    python scripts/extract_crops.py vods/2026-10-18_loud_vs_team-vitality_map1.mp4
    python scripts/extract_crops.py vods/2026-10-18_loud_vs_team-vitality_map1.mp4 --start 1200 --end 1260   # 1-minute test

Writes crops/<video name>/ with one folder per region, index.csv, manifest.json
and preview.jpg. Open preview.jpg and check every box before deleting the video.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.crops import extract_crops, folder_size_mb  # noqa: E402
from pickpov.gameplay import GameplayDetector  # noqa: E402
from pickpov.regions import load_regions  # noqa: E402

DEFAULT_REGIONS = ROOT / "configs" / "broadcast_regions.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--regions", default=str(DEFAULT_REGIONS))
    parser.add_argument("--every", type=float, default=0.25, help="seconds between samples (0.25 = 4 per second)")
    parser.add_argument("--margin", type=float, default=0.08, help="extra border around each region, as a fraction of its size")
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, default=None, help="seconds")
    parser.add_argument("--out", default=None, help="default: crops/<video name>/")
    parser.add_argument("--all-frames", action="store_true", help="don't skip non-gameplay frames")
    args = parser.parse_args()

    out = Path(args.out) if args.out else ROOT / "crops" / Path(args.video).stem
    t0 = time.time()
    detector = None if args.all_frames else GameplayDetector.load(args.regions)
    manifest = extract_crops(args.video, load_regions(args.regions), out,
                             every_s=args.every, margin=args.margin,
                             start_s=args.start, end_s=args.end, detector=detector)
    print(f"{manifest['samples']} gameplay samples ({manifest['gameplay_share']:.0%}), "
          f"{manifest['skipped_non_gameplay']} skipped, {time.time() - t0:.0f}s, "
          f"{folder_size_mb(out):.0f} MB -> {out}")
    print(f"Check {out / 'preview.jpg'} before deleting the video.")


if __name__ == "__main__":
    main()
