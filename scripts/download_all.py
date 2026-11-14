"""Download every VOD listed in a maps CSV, one call to download_vod.py per row.

Fill in these CSV columns first:
    vod_url    YouTube link (rows without one are skipped)
    vod_start  optional, e.g. 0:12:00
    vod_end    optional, e.g. 0:55:00

File names are built automatically, e.g. 2026-10-18_loud_vs_team-vitality_map1.
Files that already exist in the output folder are skipped, so it is safe to re-run.

Example:
    python scripts/download_all.py data/champions2026_summit_maps.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("download_vod.py")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="maps CSV with vod_url / vod_start / vod_end columns")
    parser.add_argument("--out", default="vods", help="output folder (default: vods)")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = skipped = failed = 0
    for row in rows:
        name = (f"{row['date']}_{slug(row['team_a'])}_vs_{slug(row['team_b'])}"
                f"_map{row['map_number']}")
        url = (row.get("vod_url") or "").strip()
        if not url:
            print(f"[skip] {name}: no vod_url")
            skipped += 1
            continue
        if (Path(args.out) / f"{name}.mp4").exists():
            print(f"[skip] {name}: already downloaded")
            skipped += 1
            continue

        cmd = [sys.executable, str(SCRIPT), url, "--name", name, "--out", args.out]
        if (row.get("vod_start") or "").strip():
            cmd += ["--start", row["vod_start"].strip()]
        if (row.get("vod_end") or "").strip():
            cmd += ["--end", row["vod_end"].strip()]

        print(f"[get ] {name}")
        try:
            subprocess.run(cmd, check=True)
            done += 1
        except subprocess.CalledProcessError:
            print(f"[fail] {name}")
            failed += 1

    print(f"\nDownloaded {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
