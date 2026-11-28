"""Learn what each agent's portrait looks like on the player cards.

    python scripts/build_card_art.py        # -> configs/cards/<Agent>.png

The cards crop each agent's face differently from valorant-api.com's art (on
Chamber only the glasses show), so live mode matches against real card
portraits. The offline rosters say which agent is in each card slot on each
map; the per-pixel median over many frames (dead players' grey cards and
overlays drop out) is that agent's template, stored facing right as on the
left team's cards.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.cards import portrait  # noqa: E402

FRAMES_PER_MAP = 40


def main() -> None:
    per_agent = defaultdict(list)
    for rf in sorted((ROOT / "data" / "rosters").glob("*.json")):
        r = json.loads(rf.read_text())
        d = ROOT / "crops" / rf.stem
        if not d.exists():
            continue
        samples = [int(x["sample"]) for x in csv.DictReader(open(d / "index.csv"))
                   if (d / "team_left" / f"{int(x['sample']):06d}.jpg").exists()]
        picks = samples[:: max(1, len(samples) // FRAMES_PER_MAP)][:FRAMES_PER_MAP]
        for p in r["players"]:
            agent = r.get("agents", {}).get(p["name"], {}).get("agent")
            if not agent:
                continue
            imgs = []
            for s in picks:
                crop = cv2.imread(str(d / f"team_{p['side']}" / f"{s:06d}.jpg"))
                if crop is not None:
                    imgs.append(portrait(crop, p["side"], p["slot"]))
            if imgs:
                per_agent[agent].append(np.median(np.stack(imgs), axis=0))
    out = ROOT / "configs" / "cards"
    out.mkdir(parents=True, exist_ok=True)
    for agent, meds in sorted(per_agent.items()):
        cv2.imwrite(str(out / f"{agent}.png"), np.median(np.stack(meds), axis=0).astype(np.uint8))
        print(f"{agent:10s} from {len(meds)} card slots")


if __name__ == "__main__":
    main()
