"""Feature table for the learned picker: one row per alive player, at moments
just before kills.

    python scripts/build_features.py                 # all maps
    python scripts/build_features.py --only <map> ...

For every kill with a known killer, the frames nearest to 2.0, 2.5 and 3.0 s
before it become examples. Each located player in that frame gets a row with
the features from pickpov.features and label = 1 if he is the killer.
Writes data/features/<map>.csv.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.features import FEATURES, player_features  # noqa: E402
from pickpov.walls import Summit  # noqa: E402

LEADS = (2.0, 2.5, 3.0)
MAX_GAP = 0.6


def build(name: str, summit: Summit) -> int:
    frames = {}      # t -> (sample, round, {player: (x, y, side)}, held set)
    with open(ROOT / "data" / "positions" / f"{name}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = float(r["time_s"])
            fr = frames.setdefault(t, [int(r["sample"]), int(r["round"]), {}, set()])
            fr[2][r["player"]] = (float(r["x"]), float(r["y"]), r["side"])
            if r["status"] == "held":
                fr[3].add(r["player"])
    times = sorted(frames)
    round_start = {}
    with open(ROOT / "data" / "rounds" / f"{name}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            round_start[int(r["round"])] = float(r["start_s"])
    with open(ROOT / "data" / "kills" / f"{name}.csv", newline="", encoding="utf-8") as f:
        kills = [k for k in csv.DictReader(f) if k["killer"]]

    rows = []
    for kid, k in enumerate(kills):
        kt, rnd = float(k["time_s"]), int(k["round"])
        for lead in LEADS:
            target = kt - lead
            i = bisect.bisect_left(times, target)
            cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
            if not cands:
                continue
            j = min(cands, key=lambda j: abs(times[j] - target))
            t = times[j]
            sample, fr_round, frame, held = frames[t]
            if abs(t - target) > MAX_GAP or fr_round != rnd or not frame:
                continue
            prev_t = times[j - 2] if j >= 2 else None          # ~1 s earlier (2 per second)
            prev = frames[prev_t][2] if prev_t is not None and frames[prev_t][1] == rnd else None
            img = cv2.imread(str(ROOT / "crops" / name / "minimap" / f"{sample:06d}.jpg"))
            closed = summit.door_states(img) if img is not None else (False, False, False)
            feats = player_features(frame, prev, (t - prev_t) if prev else 0.0, summit, closed,
                                    t - round_start.get(rnd, t), held)
            for p, vals in feats.items():
                rows.append([name, kid, lead, round(t, 2), rnd, p, frame[p][2], int(p == k["killer"]),
                             int(p in (k["killer"], k["victim"]))] + [round(v, 3) for v in vals])
    out = ROOT / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map", "kill_id", "lead", "time_s", "round", "player", "side", "is_killer", "in_fight"] + FEATURES)
        w.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    summit = Summit(ROOT / "configs" / "maps" / "summit_walkable.png", ROOT / "configs" / "maps" / "summit_doors.json",
                    cache_dir=ROOT / "configs" / "maps" / "cache")
    names = sorted(p.stem for p in (ROOT / "data" / "positions").glob("*.csv"))
    if args.only:
        names = [n for n in names if n in args.only]
    for n in names:
        print(f"{n}: {build(n, summit)} rows", flush=True)


if __name__ == "__main__":
    main()
