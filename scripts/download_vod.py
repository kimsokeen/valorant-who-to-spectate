"""Download a VCT VOD (or just part of it) from YouTube for personal study.

Examples:
    python scripts/download_vod.py URL
    python scripts/download_vod.py URL --start 1:02:30 --end 1:45:00 --name 2026-10-18_loud_vs_team-vitality_map1

Saves a 1080p (or best available up to 1080p) mp4 into vods/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="YouTube URL of the VOD")
    parser.add_argument("--start", help="start time, e.g. 1:02:30 (optional)")
    parser.add_argument("--end", help="end time, e.g. 1:45:00 (optional)")
    parser.add_argument("--name", default="%(title)s", help="output file name, without .mp4")
    parser.add_argument("--out", default="vods", help="output folder (default: vods)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp", args.url,
        # Prefer H.264 (avc1): OpenCV reads it everywhere, unlike AV1/VP9.
        "-f", "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]"
              "/bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", str(out / f"{args.name}.%(ext)s"),
    ]
    if args.start or args.end:
        cmd += ["--download-sections", f"*{args.start or '0'}-{args.end or 'inf'}"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
