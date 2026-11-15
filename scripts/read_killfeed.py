"""Read every kill (round, time, killer, victim) from the kill feed crops.

    python scripts/read_killfeed.py                    # all maps with a roster
    python scripts/read_killfeed.py --only 2026-10-18_loud_vs_team-vitality_map1

Needs data/rosters/<map>.json (run scripts/build_rosters.py first).
Writes data/kills/<map>.csv and prints a check per map:
  - rounds found vs rounds played (from the final score in the maps CSV)
  - rounds where one team lost all 5 players (the winner is certain there)
  - rows that couldn't be matched to a player
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pickpov.killfeed import load_roster, merge_split_rounds, read_map, split_joined_rounds  # noqa: E402
from pickpov.regions import load_regions  # noqa: E402
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
    args = parser.parse_args()

    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    with open(args.csv, newline="", encoding="utf-8") as f:
        played = {map_name(r): int(r["score_a"]) + int(r["score_b"]) for r in csv.DictReader(f)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for roster_file in sorted(Path(args.rosters).glob("*.json")):
        name = roster_file.stem
        if args.only and name not in args.only:
            continue
        target = out / f"{name}.csv"
        if target.exists() and not args.redo:
            continue
        res = read_map(Path(args.crops) / name, regions, load_roster(roster_file))
        merged = merge_split_rounds(res, played[name]) if name in played else 0
        split = split_joined_rounds(res, played[name]) if name in played else 0
        with open(target, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(res["kills"])
        c = check(res)
        flag = "" if c["rounds"] == played.get(name) else f"  <-- {played.get(name)} rounds were played"
        note = (f" (merged {merged} split round{'s' if merged > 1 else ''})" if merged else "") + \
               (f" (split {split} joined round{'s' if split > 1 else ''})" if split else "")
        print(f"{name}: {c['kills']} kills in {c['rounds']} rounds{note}{flag}\n"
              f"    team wipes left/right {c['wipes']['left']}/{c['wipes']['right']}, "
              f"kills with unknown killer {c['kills_without_killer']}, "
              f"unreadable rows {c['unreadable_rows']}, rounds with >10 kills {c['rounds_over_10_kills']}")


if __name__ == "__main__":
    main()
