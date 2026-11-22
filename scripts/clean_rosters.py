"""Clean up player names: exactly 5 per team per map, one spelling per player.

    python scripts/clean_rosters.py            # show what would change
    python scripts/clean_rosters.py --apply    # write the changes

1. Per map, each card slot keeps ONE name. When the two halves read a slot
   differently ("ZmijKK" vs "07 o PI"), keep the one found in that map's kills.
2. Across maps, spellings of the same player are unified ("Timatino" ->
   "Timotino"), but only within a team: map sides sharing 3+ exact names are
   the same team. Look-alike names on different teams stay apart
   (Ivy of Nongshim vs invy of Paper Rex).

Writes "players" (5 per side) into data/rosters/<map>.json, renames players
in the roster's "agents" and in data/kills/<map>.csv, and saves the spelling
changes to data/name_fixes.json. Raw per-half readings are kept as they were.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDES = (("left", "team_left"), ("right", "team_right"))


def kill_counts(kills_csv: Path) -> Counter:
    c = Counter()
    if kills_csv.exists():
        with open(kills_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c[r["victim"]] += 1
                if r["killer"]:
                    c[r["killer"]] += 1
    return c


def pick_slots(roster: dict, kills: Counter) -> list[dict]:
    players = []
    for side, key in SIDES:
        for k in range(5):
            cands = {}
            for half in roster["halves"].values():
                p = half[key][k]
                if p["name"]:
                    cands[p["name"]] = max(cands.get(p["name"], 0), p["agreement"])
            if not cands:
                continue
            name = max(cands, key=lambda n: (kills[n], cands[n]))
            players.append({"side": side, "slot": k, "name": name})
    return players


def spelling_fixes(rosters: dict[str, list[dict]]) -> dict[str, str]:
    """{variant: canonical} for spellings of the same player on the same team."""
    teams = []   # each: list of name sets (one per map side)
    for players in rosters.values():
        for side in ("left", "right"):
            names = {p["name"] for p in players if p["side"] == side}
            for t in teams:
                if any(len(names & other) >= 3 for other in t):
                    t.append(names)
                    break
            else:
                teams.append([names])
    fixes = {}
    for t in teams:
        count = Counter(n for names in t for n in names)
        ordered = sorted(count, key=lambda n: -count[n])
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if b in fixes or a in fixes:
                    continue
                if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.75:
                    fixes[b] = a      # b is rarer (or equal): use a's spelling
    return fixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    roster_dir, kills_dir = ROOT / "data" / "rosters", ROOT / "data" / "kills"
    files = sorted(roster_dir.glob("*.json"))
    data = {f.stem: json.loads(f.read_text()) for f in files}
    picked = {n: pick_slots(r, kill_counts(kills_dir / f"{n}.csv")) for n, r in data.items()}
    fixes = spelling_fixes(picked)

    for n, r in data.items():
        before = {p["name"] for h in r["halves"].values() for _, key in SIDES for p in h[key] if p["name"]}
        after = [fixes.get(p["name"], p["name"]) for p in picked[n]]
        dropped = sorted(before - {p["name"] for p in picked[n]})
        renamed = sorted({(p["name"], fixes[p["name"]]) for p in picked[n] if p["name"] in fixes})
        if dropped or renamed:
            print(f"{n}: " + "; ".join([f"dropped {d!r}" for d in dropped] + [f"{a!r} -> {b!r}" for a, b in renamed]))
        assert len(after) == 10 and len(set(after)) == 10, f"{n}: not 10 distinct players: {after}"

    print(f"\nspelling fixes: {fixes or 'none'}")
    if not args.apply:
        print("\nNothing written. Run again with --apply to save these changes.")
        return

    for n, r in data.items():
        r["players"] = [dict(p, name=fixes.get(p["name"], p["name"])) for p in picked[n]]
        keep = {p["name"] for p in r["players"]}
        if "agents" in r:
            r["agents"] = {fixes.get(k, k): v for k, v in r["agents"].items() if fixes.get(k, k) in keep}
        (roster_dir / f"{n}.json").write_text(json.dumps(r, indent=2))
        kc = kills_dir / f"{n}.csv"
        if kc.exists():
            with open(kc, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                fields = list(rows[0].keys()) if rows else []
            for row in rows:
                for col in ("killer", "victim"):
                    row[col] = fixes.get(row[col], row[col])
            if rows:
                with open(kc, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
    (ROOT / "data" / "name_fixes.json").write_text(json.dumps(fixes, indent=2))
    print(f"\nSaved: {len(data)} rosters, kills CSVs, data/name_fixes.json")


if __name__ == "__main__":
    main()
