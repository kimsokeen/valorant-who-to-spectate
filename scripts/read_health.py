"""Read each player's health from the player cards, at the moments the model needs.

    python scripts/read_health.py --jobs 8

Only the card frame just before each moment in data/features/ is read (the
model needs health there, not every second), which keeps this fast.
Writes data/health/<map>.csv: sample, time_s, player, hp.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.health import card_slots, read_hp  # noqa: E402
from pickpov.regions import load_regions  # noqa: E402


def process_map(name: str) -> str:
    d = ROOT / "crops" / name
    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    m = json.loads((d / "manifest.json").read_text())
    fw, fh = m["video_info"]["width"], m["video_info"]["height"]
    have = sorted(int(p.stem) for p in (d / "team_left").iterdir())
    times = {}
    with open(d / "index.csv") as f:
        for r in csv.DictReader(f):
            times[int(r["sample"])] = float(r["time_s"])
    card_t = [times[s] for s in have]
    with open(ROOT / "data" / "features" / f"{name}.csv", newline="", encoding="utf-8") as f:
        moments = sorted({float(r["time_s"]) for r in csv.DictReader(f)})
    need = set()
    for t in moments:
        i = bisect.bisect_right(card_t, t) - 1
        if i >= 0 and t - card_t[i] <= 1.5:
            need.add(have[i])
    slots = card_slots(ROOT / "data" / "rosters" / f"{name}.json")
    out = ROOT / "data" / "health"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_s", "player", "hp"])
        for s in sorted(need):
            for p, hp in read_hp(d, s, regions, slots, m["margin"], fw, fh).items():
                w.writerow([s, round(times[s], 2), p, hp])
                n += 1
    return f"{name}: {len(need)} card frames read, {n} health values"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    names = sorted(p.stem for p in (ROOT / "data" / "features").glob("*.csv"))
    if args.only:
        names = [n for n in names if n in args.only]
    if args.jobs <= 1:
        for n in names:
            print(process_map(n), flush=True)
    else:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(process_map, names):
                print(line, flush=True)


if __name__ == "__main__":
    main()
