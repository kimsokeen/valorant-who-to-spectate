"""Read every kill (round, time, killer, victim) from the kill feed crops.

    python scripts/read_killfeed.py                    # all maps with a roster
    python scripts/read_killfeed.py --only 2026-10-18_loud_vs_team-vitality_map1
    python scripts/read_killfeed.py --redo --jobs 4          # 4 maps at once (4+ core PC)

Needs data/rosters/<map>.json (run scripts/build_rosters.py first).
Rounds and their winners come from score changes in the HUD (pickpov.rounds).
If that doesn't match the rounds played, it falls back to gaps in gameplay plus
merging/splitting by kills.

Writes data/kills/<map>.csv and data/rounds/<map>.csv, and prints a check per map:
  - rounds found vs rounds played (from the final score in the maps CSV)
  - rounds where one team lost all 5 players (the winner is certain there)
  - rows that couldn't be matched to a player
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pickpov.killfeed import (drop_duplicate_deaths, load_roster, merge_split_rounds,  # noqa: E402
                              read_map, split_joined_rounds)
from pickpov.regions import load_regions  # noqa: E402
from pickpov.rounds import detect_rounds, finish_last_round  # noqa: E402
from process_all import map_name  # noqa: E402

FIELDS = ["round", "time_s", "killer", "killer_side", "victim", "victim_side", "seen"]


def check(res: dict) -> dict:
    by_round = {}
    for k in res["kills"]:
        by_round.setdefault(k["round"], []).append(k)
    wipes = {"left": 0, "right": 0}
    too_many = 0
    for ks in by_round.values():
        if len(ks) > 10:
            too_many += 1
        for side in ("left", "right"):
            if sum(k["victim_side"] == side for k in ks) == 5:
                wipes["left" if side == "right" else "right"] += 1
    no_killer = sum(k["killer"] is None for k in res["kills"])
    return {"rounds": res["rounds"], "kills": len(res["kills"]), "wipes": wipes,
            "rounds_over_10_kills": too_many, "kills_without_killer": no_killer,
            "unreadable_rows": res["unreadable_rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(ROOT / "data" / "champions2026_summit_maps.csv"))
    parser.add_argument("--crops", default=str(ROOT / "crops"))
    parser.add_argument("--rosters", default=str(ROOT / "data" / "rosters"))
    parser.add_argument("--out", default=str(ROOT / "data" / "kills"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="maps to read at the same time (e.g. 4 on a 4-core PC)")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    played = {map_name(r): int(r["score_a"]) + int(r["score_b"]) for r in rows}
    finals = {map_name(r): (int(r["score_a"]), int(r["score_b"])) for r in rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo = []
    for roster_file in sorted(Path(args.rosters).glob("*.json")):
        name = roster_file.stem
        if args.only and name not in args.only:
            continue
        if (out / f"{name}.csv").exists() and not args.redo:
            continue
        todo.append(roster_file)

    jobs = [(rf, str(args.crops), str(out), played.get(rf.stem), finals.get(rf.stem)) for rf in todo]
    if args.jobs <= 1:
        for j in jobs:
            print(process_map(*j), flush=True)
    else:
        # several maps at once, one per CPU core; results print as each map finishes.
        # Tesseract would otherwise start several threads per call and the
        # processes would fight over the cores.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(process_map, *zip(*jobs)) if jobs else []:
                print(line, flush=True)


def process_map(roster_file: Path, crops: str, out: str, played_n: int | None,
                final: tuple[int, int] | None) -> str:
    """Read one map's kills and rounds, write its CSVs, return the summary text."""
    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    out = Path(out)
    name = roster_file.stem
    target = out / f"{name}.csv"
    info = detect_rounds(Path(crops) / name, regions)
    finish_last_round(info, final)
    use_score = played_n is not None and info["rounds"] == played_n
    res = read_map(Path(crops) / name, regions, load_roster(roster_file),
                   round_of=info["round_of"] if use_score else None)
    merged = split = 0
    if not use_score and played_n is not None:          # fallback: gaps + kill logic
        merged = merge_split_rounds(res, played_n)
        split = split_joined_rounds(res, played_n)
    dropped = drop_duplicate_deaths(res)
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["kills"])
    rounds_dir = out.parent / "rounds"
    rounds_dir.mkdir(exist_ok=True)
    with open(rounds_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "start_s", "end_s", "winner_side"])
        for r, (a, b) in enumerate(res["spans"]):
            w.writerow([r, round(a, 2), round(b, 2), (info["winners"].get(r, "") if use_score else "")])
    c = check(res)
    flag = "" if c["rounds"] == played_n else f"  <-- {played_n} rounds were played"
    how = (f" [rounds from score: {info['left_wins']}-{info['right_wins']}]" if use_score
           else f" [score gave {info['rounds']} rounds; used gaps instead]")
    note = how + (f" (merged {merged} split round{'s' if merged > 1 else ''})" if merged else "") + \
        (f" (split {split} joined round{'s' if split > 1 else ''})" if split else "") + \
        (f" (dropped {dropped} duplicate death{'s' if dropped > 1 else ''})" if dropped else "")
    return (f"{name}: {c['kills']} kills in {c['rounds']} rounds{note}{flag}\n"
            f"    team wipes left/right {c['wipes']['left']}/{c['wipes']['right']}, "
            f"kills with unknown killer {c['kills_without_killer']}, "
            f"unreadable rows {c['unreadable_rows']}, rounds with >10 kills {c['rounds_over_10_kills']}")

if __name__ == "__main__":
    main()
