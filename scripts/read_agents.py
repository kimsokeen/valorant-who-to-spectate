"""Which agent each player is on, for every map (from kill feed portraits).

    python scripts/read_agents.py                 # all maps
    python scripts/read_agents.py --jobs 8        # several maps at once
    python scripts/read_agents.py --only 2026-10-18_loud_vs_team-vitality_map1

Needs data/rosters/ (build_rosters.py) and configs/agents/ (get_agent_icons.py).
Writes the agent into each player's entry in data/rosters/<map>.json and prints
a summary. Unanimous votes are what you want to see.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.agents import load_killfeed_art, read_agents  # noqa: E402
from pickpov.killfeed import load_roster  # noqa: E402
from pickpov.regions import load_regions  # noqa: E402


def process_map(roster_file: Path, crops: str) -> str:
    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    art = load_killfeed_art(ROOT / "configs" / "agents")
    roster = load_roster(roster_file)
    agents = read_agents(Path(crops) / roster_file.stem, regions, roster, art)
    data = json.loads(roster_file.read_text())
    data["agents"] = agents
    roster_file.write_text(json.dumps(data, indent=2))
    line = []
    for side in ("left", "right"):
        line.append(", ".join(f"{p} {a['agent'] or '?'} {a['votes']}/{a['of']}"
                              for p, a in agents.items() if roster[p] == side))
    return f"{roster_file.stem}\n    left:  {line[0]}\n    right: {line[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", default=str(ROOT / "crops"))
    parser.add_argument("--rosters", default=str(ROOT / "data" / "rosters"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    files = [f for f in sorted(Path(args.rosters).glob("*.json")) if not args.only or f.stem in args.only]
    if args.jobs <= 1:
        for f in files:
            print(process_map(f, args.crops), flush=True)
    else:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(process_map, files, [args.crops] * len(files)):
                print(line, flush=True)


if __name__ == "__main__":
    main()
