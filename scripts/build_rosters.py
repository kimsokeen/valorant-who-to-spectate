"""Read player names and team tags for every cropped map.

    python scripts/build_rosters.py                 # all maps in crops/
    python scripts/build_rosters.py --only 2026-10-18_loud_vs_team-vitality_map1

Writes data/rosters/<map>.json and prints a summary. A name with low
"agreement" (share of samples that read it the same way) needs a look.
Needs Tesseract installed: https://github.com/UB-Mannheim/tesseract/wiki (Windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.regions import load_regions  # noqa: E402
from pickpov.roster import build_roster  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", default=str(ROOT / "crops"))
    parser.add_argument("--out", default=str(ROOT / "data" / "rosters"))
    parser.add_argument("--only", nargs="*", help="map names to do (default: all)")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--redo", action="store_true", help="rebuild maps that already have a roster")
    args = parser.parse_args()

    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = sorted(p for p in Path(args.crops).iterdir()
                  if (p / "manifest.json").exists() and not p.name.startswith("_"))
    if args.only:
        maps = [p for p in maps if p.name in args.only]
    for d in maps:
        target = out / f"{d.name}.json"
        if target.exists() and not args.redo:
            continue
        r = build_roster(d, regions, samples=args.samples)
        target.write_text(json.dumps(r, indent=2))
        h1, h2 = r["halves"]["first"], r["halves"]["second"]
        same = all(h1[s][k]["name"] == h2[s][k]["name"] for s in ("team_left", "team_right") for k in range(5))
        low = min(p["agreement"] for h in (h1, h2) for s in ("team_left", "team_right") for p in h[s])
        print(f"{d.name}: {h1['tags']['left']} {[p['name'] for p in h1['team_left']]} vs "
              f"{h1['tags']['right']} {[p['name'] for p in h1['team_right']]}"
              f" | halves match: {same} | lowest agreement {low:.0%}")


if __name__ == "__main__":
    main()
