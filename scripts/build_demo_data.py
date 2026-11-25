"""Build the small files the demo app (app.py) reads, so it needs no video or crops.

    python scripts/train_lgbm.py          # -> data/results/lgbm_scores.csv
    python scripts/build_demo_data.py     # -> data/demo/moments.csv, configs/maps/summit_minimap.png

moments.csv: one row per located player, 2.5 s before every scored kill, with
PickPOV's leave-one-map-out score (so every pick shown comes from a model that
never saw that map), the killer, the victim and who the real broadcast showed.

summit_minimap.png: a clean Summit minimap, the per-pixel median of minimap
crops from every map, which removes the player icons. Kept if it already
exists unless --background is given (it needs crops/, which isn't in git).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.pov_data import load_pov, pov_at  # noqa: E402

LEAD = 2.5
BACKGROUND = ROOT / "configs" / "maps" / "summit_minimap.png"


def build_background(per_map: int = 25) -> None:
    import cv2
    import numpy as np

    rng = random.Random(1)
    imgs = []
    for d in sorted((ROOT / "crops").iterdir()):
        files = sorted((d / "minimap").glob("*.jpg"))
        for f in rng.sample(files, min(per_map, len(files))):
            img = cv2.imread(str(f))
            if img is not None and (not imgs or img.shape == imgs[0].shape):
                imgs.append(img)
    if not imgs:
        sys.exit("no minimap crops found in crops/*/minimap/")
    bg = np.median(np.stack(imgs), axis=0).astype("uint8")
    # outside the walkable floor (plus a margin for the walls' outlines) is game scene and HUD text: blank it
    floor = cv2.imread(str(ROOT / "configs" / "maps" / "summit_walkable.png"), cv2.IMREAD_GRAYSCALE)
    floor = cv2.resize(floor, bg.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    near = cv2.dilate((floor > 0).astype("uint8"), np.ones((9, 9), "uint8")) > 0
    bg[~near] = (40, 41, 42)
    cv2.imwrite(str(BACKGROUND), bg)
    print(f"background from {len(imgs)} crops -> {BACKGROUND.relative_to(ROOT)}")


def build_moments() -> None:
    scores = {}
    with open(ROOT / "data" / "results" / "lgbm_scores.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[(r["map"], int(r["kill_id"]), r["player"])] = float(r["score"])
    maps = sorted({k[0] for k in scores})
    rows = []
    for m in maps:
        agents = {n: a.get("agent", "") for n, a in
                  json.loads((ROOT / "data" / "rosters" / f"{m}.json").read_text()).get("agents", {}).items()}
        with open(ROOT / "data" / "kills" / f"{m}.csv", newline="", encoding="utf-8") as f:
            kills = [k for k in csv.DictReader(f) if k["killer"]]    # kill_id = index among kills with a killer
        with open(ROOT / "data" / "features" / f"{m}.csv", newline="", encoding="utf-8") as f:
            feats = [r for r in csv.DictReader(f) if float(r["lead"]) == LEAD]
        pov = pov_at(load_pov(ROOT / "data" / "pov" / f"{m}.csv"), sorted({float(r["time_s"]) for r in feats}))
        for r in feats:
            kid = int(r["kill_id"])
            k = kills[kid]
            rows.append({
                "map": m, "kill_id": kid, "round": int(k["round"]), "kill_time_s": float(k["time_s"]),
                "moment_s": float(r["time_s"]), "player": r["player"], "side": r["side"],
                "agent": agents.get(r["player"], ""), "x": float(r["x"]), "y": float(r["y"]),
                "score": scores[(m, kid, r["player"])], "killer": k["killer"], "victim": k["victim"],
                "broadcast": pov.get(float(r["time_s"])) or "",
            })
    out = ROOT / "data" / "demo" / "moments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len({(r["map"], r["kill_id"]) for r in rows})
    print(f"{len(rows)} player rows, {n} kills, {len(maps)} maps -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", action="store_true", help="rebuild the minimap background from crops/")
    args = ap.parse_args()
    if args.background or not BACKGROUND.exists():
        build_background()
    build_moments()


if __name__ == "__main__":
    main()
