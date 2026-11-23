"""Who the real broadcast was showing, every second, on every map.

    python scripts/read_pov.py --jobs 8

Writes data/pov/<map>.csv (sample, time_s, player) from the pov_name crops.
An empty player means the label couldn't be read that second.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.killfeed import load_roster  # noqa: E402
from pickpov.pov import read_pov  # noqa: E402


def process_map(name: str) -> str:
    players = list(load_roster(ROOT / "data" / "rosters" / f"{name}.json"))
    rows = read_pov(ROOT / "crops" / name, players)
    out = ROOT / "data" / "pov"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_s", "player"])
        for s, t, p in rows:
            w.writerow([s, round(t, 2), p or ""])
    read = sum(1 for _, _, p in rows if p)
    switches = sum(1 for a, b in zip(rows, rows[1:]) if a[2] and b[2] and a[2] != b[2])
    return f"{name}: {read}/{len(rows)} seconds read ({read / max(len(rows), 1):.0%}), {switches} POV switches"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    names = sorted(p.stem for p in (ROOT / "data" / "rosters").glob("*.json"))
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
