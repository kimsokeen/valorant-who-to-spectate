"""Train and test the learned picker, always testing on maps it never saw.

    python scripts/train_model.py

Leave-one-map-out: for each of the 18 maps, train on the other 17 and pick on
this one, 2.5 s before each kill. Prints caught-the-killer / caught-the-fight
next to the best rule (scored on the same moments) and saves a model trained
on all maps to models/softmax_picker.json.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.features import FEATURES  # noqa: E402
from pickpov.model import USED, SoftmaxPicker, transform  # noqa: E402

TEST_LEAD = 2.5


def load():
    rows, meta = [], []
    for p in sorted((ROOT / "data" / "features").glob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([float(r[c]) for c in FEATURES])
                meta.append((r["map"], int(r["kill_id"]), float(r["lead"]), r["player"],
                             int(r["is_killer"]), int(r["in_fight"])))
    return np.array(rows), meta


def evaluate(scores, meta, idx):
    """Per kill (at TEST_LEAD): did the top-scored player = killer / in the fight?"""
    best = {}
    for i in idx:
        m, kid, lead, p, ik, fi = meta[i]
        if lead != TEST_LEAD:
            continue
        key = (m, kid)
        if key not in best or scores[i] > best[key][0]:
            best[key] = (scores[i], ik, fi)
    n = len(best)
    return n, sum(v[1] for v in best.values()) / n, sum(v[2] for v in best.values()) / n


def main() -> None:
    raw, meta = load()
    X = transform(raw)
    maps = np.array([m[0] for m in meta])
    groups = np.array([hash((m[0], m[1], m[2])) % (2 ** 62) for m in meta])
    order = np.argsort(groups, kind="stable")
    y = np.array([m[4] for m in meta])
    col = {f: i for i, f in enumerate(FEATURES)}
    rule = raw[:, col["enemies_walk_90"]] * 1000 - raw[:, col["walk_nearest_enemy"]]   # best rule as a score
    model_scores = np.zeros(len(meta))
    for m in sorted(set(maps)):
        tr = order[maps[order] != m]
        model = SoftmaxPicker().fit(X[tr], groups[tr], y[tr])
        te = np.flatnonzero(maps == m)
        model_scores[te] = model.score(X[te])
    idx = np.arange(len(meta))
    n, k_r, f_r = evaluate(rule, meta, idx)
    _, k_m, f_m = evaluate(model_scores, meta, idx)
    print(f"{n} kills, {TEST_LEAD} s before each, every map scored by a model that never saw it:")
    print(f"  best rule (most enemies within 90 px walk): caught killer {k_r:.1%}, fight {f_r:.1%}")
    print(f"  learned model:                              caught killer {k_m:.1%}, fight {f_m:.1%}")
    print("  (real broadcast on its readable kills: 27.7% / 55.2%)")
    per = defaultdict(list)
    for m in sorted(set(maps)):
        idx_m = np.flatnonzero(maps == m)
        per[m] = (evaluate(rule, meta, idx_m)[1], evaluate(model_scores, meta, idx_m)[1])
    better = sum(1 for a, b in per.values() if b > a)
    print(f"  model beats the rule on {better} of {len(per)} maps")
    final = SoftmaxPicker().fit(X[order], groups[order], y[order])
    (ROOT / "models").mkdir(exist_ok=True)
    final.save(ROOT / "models" / "softmax_picker.json")
    print("\nweights (standardised; positive = more likely the next killer):")
    for f, w in sorted(zip(USED, final.w), key=lambda x: -abs(x[1])):
        print(f"  {f:22s} {w:+.2f}")


if __name__ == "__main__":
    main()
