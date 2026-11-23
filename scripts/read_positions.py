"""Where every player is, every half second, on every map (from the minimap).

    python scripts/read_positions.py --jobs 8                  # all maps
    python scripts/read_positions.py --only 2026-10-18_loud_vs_team-vitality_map1
    python scripts/read_positions.py --preview 2026-10-18_loud_vs_team-vitality_map1

Needs: crops/, data/rosters/ with agents (read_agents.py), data/kills/ and
data/rounds/ (read_killfeed.py), configs/agents/ (get_agent_icons.py).

Writes data/positions/<map>.csv, one row per player per sample:
    sample, time_s, round, player, side, agent, x, y, score, status
x, y are minimap-crop pixels (the same frame for every Summit map here).
status "seen" = found in this frame; "held" = not found, so his last position
from up to HOLD_S seconds earlier is kept (icons blink out in fights).

--preview writes a few minimap images with every found player marked, to
frames/positions_check/<map>/, so you can check them by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.minimap import read_positions  # noqa: E402

HOLD_S = 1.5
FIELDS = ["sample", "time_s", "round", "player", "side", "agent", "x", "y", "score", "status"]


def _paths(name: str) -> tuple:
    return (ROOT / "crops" / name, ROOT / "data" / "rosters" / f"{name}.json",
            ROOT / "data" / "kills" / f"{name}.csv", ROOT / "data" / "rounds" / f"{name}.csv")


def process_map(name: str, every: int) -> str:
    crops, roster, kills, rounds = _paths(name)
    info = json.loads(roster.read_text())
    meta = {p["name"]: (p["side"], info["agents"][p["name"]]["agent"]) for p in info["players"]}
    out = ROOT / "data" / "positions"
    out.mkdir(parents=True, exist_ok=True)
    last: dict[str, tuple] = {}          # player -> (x, y, score, time, round)
    seen = held = frames = 0
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for s, t, rnd, pos in read_positions(crops, roster, kills, rounds, ROOT / "configs" / "agents", every=every):
            frames += 1
            dead_now = set()
            for p, (x, y, sc) in pos.items():
                w.writerow([s, round(t, 2), rnd, p, *meta[p], x, y, sc, "seen"])
                last[p] = (x, y, sc, t, rnd)
                seen += 1
            for p, (x, y, sc, lt, lr) in list(last.items()):
                if p in pos:
                    continue
                if lr == rnd and t - lt <= HOLD_S:
                    w.writerow([s, round(t, 2), rnd, p, *meta[p], x, y, sc, "held"])
                    held += 1
                elif lr != rnd or t - lt > HOLD_S:
                    dead_now.add(p)
            for p in dead_now:
                last.pop(p, None)
    return f"{name}: {frames} frames, {seen} positions seen + {held} held ({(seen + held) / max(frames, 1):.1f} players per frame)"


def preview(name: str, count: int = 8) -> None:
    crops, roster, kills, rounds = _paths(name)
    out = ROOT / "frames" / "positions_check" / name
    out.mkdir(parents=True, exist_ok=True)
    idx_len = sum(1 for _ in open(crops / "index.csv")) - 1
    step = max(1, idx_len // count)
    for k, (s, t, rnd, pos) in enumerate(read_positions(crops, roster, kills, rounds, ROOT / "configs" / "agents", every=step)):
        img = cv2.imread(str(crops / "minimap" / f"{s:06d}.jpg"))
        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for p, (x, y, sc) in pos.items():
            cv2.circle(big, (2 * x, 2 * y), 16, (0, 255, 255), 2)
            cv2.putText(big, p, (2 * x + 18, 2 * y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(big, f"round {rnd + 1}  t={t:.0f}s  found {len(pos)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"t{t:07.1f}.jpg"), big)
        if k + 1 >= count:
            break
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--every", type=int, default=2, help="use every Nth sample (2 = 2 per second)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--preview", nargs="*", help="write check images for these maps and stop")
    args = parser.parse_args()
    if args.preview:
        for n in args.preview:
            preview(n)
        return
    names = sorted(p.stem for p in (ROOT / "data" / "rosters").glob("*.json"))
    if args.only:
        names = [n for n in names if n in args.only]
    if args.jobs <= 1:
        for n in names:
            print(process_map(n, args.every), flush=True)
    else:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(process_map, names, [args.every] * len(names)):
                print(line, flush=True)


if __name__ == "__main__":
    main()
