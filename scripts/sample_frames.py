"""Save frames from a VOD as JPGs, for calibrating regions and eyeballing data.

Example:
    python scripts/sample_frames.py vods/2026-10-18_loud_vs_team-vitality_map1.mp4 --every 10 --max 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.video import save_frames, video_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--every", type=float, default=10.0, help="seconds between frames")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--max", type=int, default=None, help="stop after this many frames")
    parser.add_argument("--out", default=None, help="default: frames/<video name>/")
    args = parser.parse_args()

    info = video_info(args.video)
    print(f"{info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['duration_s'] / 60:.1f} min")
    out = args.out or Path("frames") / Path(args.video).stem
    saved = save_frames(args.video, out, args.every, args.start, args.end, args.max)
    print(f"Saved {len(saved)} frames to {out}")


if __name__ == "__main__":
    main()
