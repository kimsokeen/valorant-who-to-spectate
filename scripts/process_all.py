"""Build the crop dataset in batches: download 5 -> crop 5 -> review 5 -> next 5.

Downloading and cropping need no input, so you can walk away. Questions only
come at the end of each batch, one after another, so you answer them in one sitting.

    python scripts/process_all.py data/champions2026_summit_maps.csv
    python scripts/process_all.py data/champions2026_summit_maps.csv --batch 3
    python scripts/process_all.py data/champions2026_summit_maps.csv --yes   # never ask

Each batch:
  1. Download every map in the batch that isn't on disk yet (no questions)
  2. Crop gameplay from each one (no questions)
  3. Review: for each map, show its numbers and preview.jpg path, ask
     "delete video? [y/N]" back to back

A video is never deleted, and you're never asked about it, if:
  - cropping failed or covered less than 95% of the video
  - gameplay was outside 40-90% of the video (the Grand Final map was 65%)
  - the map is in --keep (the Grand Final map, the screen reader's test case)
Maps already cropped (crops/<name>/manifest.json exists) are skipped, so it is
safe to stop and re-run. Disk use peaks at about 5 videos (~10 GB).

Before the first run: calibrate configs/broadcast_regions.yaml and set
`calibrated: true`, and fill in vod_url / vod_start / vod_end in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from download_all import slug  # noqa: E402  (same file names as download_all.py)
from pickpov.crops import extract_crops  # noqa: E402
from pickpov.gameplay import GameplayDetector  # noqa: E402
from pickpov.regions import load_regions  # noqa: E402

DEFAULT_KEEP = ["2026-10-18_loud_vs_team-vitality_map1"]


def map_name(row: dict) -> str:
    return (f"{row['date']}_{slug(row['team_a'])}_vs_{slug(row['team_b'])}"
            f"_map{row['map_number']}")


def download(row: dict, name: str, vods: Path) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / "download_vod.py"),
           row["vod_url"].strip(), "--name", name, "--out", str(vods)]
    if (row.get("vod_start") or "").strip():
        cmd += ["--start", row["vod_start"].strip()]
    if (row.get("vod_end") or "").strip():
        cmd += ["--end", row["vod_end"].strip()]
    return subprocess.run(cmd).returncode == 0 and (vods / f"{name}.mp4").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--regions", default=str(ROOT / "configs" / "broadcast_regions.yaml"))
    parser.add_argument("--vods", default=str(ROOT / "vods"))
    parser.add_argument("--crops", default=str(ROOT / "crops"))
    parser.add_argument("--batch", type=int, default=5, help="maps per batch (default 5)")
    parser.add_argument("--every", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--keep", nargs="*", default=DEFAULT_KEEP,
                        help="map names whose video is never deleted")
    parser.add_argument("--yes", action="store_true", help="delete without asking")
    parser.add_argument("--no-delete", action="store_true", help="crop only, keep every video")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.regions).read_text())
    if not config.get("calibrated"):
        sys.exit(f"{args.regions} is not calibrated yet. Check the boxes on real frames, "
                 "then set `calibrated: true` in it.")
    regions = load_regions(args.regions)
    detector = GameplayDetector.load(args.regions)
    vods, crops = Path(args.vods), Path(args.crops)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    todo = []
    for row in rows:
        name = map_name(row)
        if (crops / name / "manifest.json").exists():
            print(f"[done] {name}")
        elif (vods / f"{name}.mp4").exists() or (row.get("vod_url") or "").strip():
            todo.append((row, name))
        else:
            print(f"[skip] {name}: no vod_url")
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"\n{len(todo)} maps to do, in {len(batches)} batch(es) of up to {args.batch}\n")

    for b, batch in enumerate(batches, 1):
        print(f"===== Batch {b}/{len(batches)}: downloading =====")
        ready = []
        for row, name in batch:
            video = vods / f"{name}.mp4"
            if video.exists() or download(row, name, vods):
                ready.append((name, video))
            else:
                print(f"[fail] {name}: download failed, skipped this run")

        print(f"\n===== Batch {b}/{len(batches)}: cropping =====")
        results = []
        for name, video in ready:
            print(f"[crop] {name}")
            try:
                m = extract_crops(video, regions, crops / name, every_s=args.every,
                                  margin=args.margin, detector=detector)
            except Exception as e:  # keep the video if anything went wrong
                print(f"[fail] {name}: {e}")
                results.append((name, video, None, f"failed: {e}"))
                continue
            seen = m["samples"] + m["skipped_non_gameplay"]
            expected = m["video_info"]["duration_s"] / args.every
            covered = seen / expected if expected else 0
            share = m["gameplay_share"]
            note = None
            if covered < 0.95:
                note = f"only {covered:.0%} of the video was read"
            elif not 0.4 <= share <= 0.9:
                note = f"unusual gameplay share {share:.0%}"
            results.append((name, video, m, note))
            print(f"       {m['samples']} gameplay samples, {share:.0%} gameplay")

        print(f"\n===== Batch {b}/{len(batches)}: review =====")
        for name, video, m, note in results:
            if m is None or note:
                print(f"[keep] {name}: {note} -> video kept, check preview")
                continue
            print(f"\n{name}: {m['samples']} samples, {m['gameplay_share']:.0%} gameplay")
            print(f"  preview: {crops / name / 'preview.jpg'}")
            if args.no_delete or name in args.keep:
                print("  video kept")
                continue
            if not args.yes:
                if input(f"  Delete {video.name}? [y/N] ").strip().lower() != "y":
                    print("  video kept")
                    continue
            video.unlink()
            print("  video deleted")
        print()

    print("All batches done.")


if __name__ == "__main__":
    main()
