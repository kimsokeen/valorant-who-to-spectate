"""Score pickers (and the real broadcast, once read_pov.py has run) on every map:
how often were they on the killer before the kill?

    python scripts/evaluate_pickers.py               # lead 2.5 s
    python scripts/evaluate_pickers.py --lead 1 2 3 5

Prints a table per map and overall, and writes data/results/pickers.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2  # noqa: E402
from pickpov.picker import (load_frames, pick_contested_walking, pick_most_contested,  # noqa: E402
                           pick_most_visible, pick_nearest_enemy, pick_nearest_walking)
from pickpov.walls import Summit  # noqa: E402
from pickpov.pov_data import load_pov, pov_at  # noqa: E402
from pickpov.scoring import frames_needed, load_kills, run_picker, score_picks  # noqa: E402

PICKERS = {"nearest_enemy": pick_nearest_enemy, "most_contested": pick_most_contested}
# These use walls and doors; door states are read from the minimap crop of each
# frame they're asked about, so they only run on the frames scoring needs.
WALL_PICKERS = {"most_visible": pick_most_visible, "nearest_walking": pick_nearest_walking,
                "contested_walking": pick_contested_walking}


def wall_picks(name: str, frames: list, kills: list, leads: list[float], summit: Summit) -> dict:
    samples = {}
    with open(ROOT / "data" / "positions" / f"{name}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples[float(r["time_s"])] = int(r["sample"])
    need = set().union(*(frames_needed(frames, kills, lead) for lead in leads))
    by_t = {t: fr for t, _, fr in frames}
    out = {pname: {} for pname in WALL_PICKERS}
    for t in sorted(need):
        img = cv2.imread(str(ROOT / "crops" / name / "minimap" / f"{samples[t]:06d}.jpg"))
        closed = summit.door_states(img) if img is not None else ()
        for pname, picker in WALL_PICKERS.items():
            out[pname][t] = picker(by_t[t], summit, closed)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead", type=float, nargs="*", default=[2.5])
    args = parser.parse_args()
    maps = sorted(p.stem for p in (ROOT / "data" / "positions").glob("*.csv"))
    summit = Summit(ROOT / "configs" / "maps" / "summit_walkable.png", ROOT / "configs" / "maps" / "summit_doors.json",
                    cache_dir=ROOT / "configs" / "maps" / "cache")
    rows = []
    for name in maps:
        frames = load_frames(ROOT / "data" / "positions" / f"{name}.csv")
        kills = load_kills(ROOT / "data" / "kills" / f"{name}.csv")
        contestants = {pname: run_picker(frames, picker) for pname, picker in PICKERS.items()}
        contestants.update(wall_picks(name, frames, kills, args.lead, summit))
        pov_csv = ROOT / "data" / "pov" / f"{name}.csv"
        if pov_csv.exists():     # the real broadcast, read by scripts/read_pov.py
            contestants["broadcast"] = pov_at(load_pov(pov_csv), [t for t, _, _ in frames])
        for pname, picks in contestants.items():
            for lead in args.lead:
                s = score_picks(frames, kills, picks, lead=lead, skip_unknown=(pname == "broadcast"))
                rows.append({"map": name, "picker": pname, "lead_s": lead, **s})
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pickers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for pname in dict.fromkeys(r["picker"] for r in rows):
        for lead in args.lead:
            sel = [r for r in rows if r["picker"] == pname and r["lead_s"] == lead]
            n = sum(r["kills_scored"] for r in sel)
            avg = lambda key: sum(r[key] * r["kills_scored"] for r in sel) / n
            print(f"\n{pname}, {lead:g} s before the kill ({n} kills on {len(sel)} maps)")
            print(f"  caught the killer: {avg('caught_killer'):.1%}   (random pick: {avg('chance_killer'):.1%})")
            print(f"  caught the fight:  {avg('caught_fight'):.1%}   (random pick: {avg('chance_fight'):.1%})")
            if len(args.lead) == 1:
                for r in sel:
                    print(f"    {r['map'][:44]:44s} killer {r['caught_killer']:5.1%}  fight {r['caught_fight']:5.1%}  ({r['kills_scored']} kills)")


if __name__ == "__main__":
    main()
