"""Train and test a LightGBM picker, always testing on maps it never saw.

    pip install lightgbm
    python scripts/train_lgbm.py
    python scripts/train_lgbm.py --dry-run      # only check the data, no LightGBM needed

LightGBM's ranking mode (LambdaRank) scores the players of each moment against
each other, like the linear model, but it can learn combinations ("close AND in
this spot") and map hot spots, so it also gets each player's x, y and agent role.

Two versions are compared:
  killer only : the killer is the only right answer (relevance 1)
  fight aware : killer 2, victim 1, everyone else 0
Leave-one-map-out at 2.5 s before each kill, next to the best rule and the
linear model on the same moments. The final model (all maps) is saved to
models/lgbm_picker.txt with its feature importances.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.features import FEATURES  # noqa: E402
from pickpov.model import SoftmaxPicker, transform  # noqa: E402
from pickpov.pov_data import load_pov, pov_at  # noqa: E402

TEST_LEAD = 2.5
# LightGBM's own interface (lgb.train) - no scikit-learn needed
PARAMS = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1, "seed": 1}
ROLES = ["Duelist", "Initiator", "Controller", "Sentinel"]


def load():
    roles = {}
    rf = ROOT / "configs" / "agents" / "roles.json"
    if rf.exists():
        roles = json.loads(rf.read_text())
    else:
        print("note: configs/agents/roles.json missing (re-run scripts/get_agent_icons.py); roles left blank")
    agent_of = {}
    for p in (ROOT / "data" / "rosters").glob("*.json"):
        r = json.loads(p.read_text())
        for name, a in r.get("agents", {}).items():
            agent_of[(p.stem, name)] = a.get("agent")
    raw, extra, meta, hp = [], [], [], []
    for p in sorted((ROOT / "data" / "features").glob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                raw.append([float(r[c]) for c in FEATURES])
                hp.append([float(r.get(c) or -1) for c in ("hp", "enemy_hp", "hp_adv")])
                role = roles.get(agent_of.get((r["map"], r["player"])) or "", "")
                extra.append([float(role == x) for x in ROLES])
                meta.append((r["map"], int(r["kill_id"]), float(r["lead"]), r["player"],
                             int(r["is_killer"]), int(r["in_fight"]), float(r["time_s"])))
    return np.array(raw), np.array(extra), meta, np.array(hp)


def head_to_head(scores, meta):
    """Model vs the real broadcast on exactly the same kills (where the
    broadcast's POV was readable), with a sign test on the disagreements."""
    from math import comb
    pick, killer_of, t_of = {}, {}, {}
    # the killer comes from the kill list itself: he may not have been located on the
    # minimap at that moment (then the model can't pick him, but the broadcast can)
    for m in {x[0] for x in meta}:
        with open(ROOT / "data" / "kills" / f"{m}.csv", newline="", encoding="utf-8") as f:
            for kid, k in enumerate(r for r in csv.DictReader(f) if r["killer"]):
                killer_of[(m, kid)] = k["killer"]
    for i, (m, kid, lead, p, ik, fi, t) in enumerate(meta):
        if lead != TEST_LEAD:
            continue
        key = (m, kid)
        t_of[key] = t
        if key not in pick or scores[i] > pick[key][0]:
            pick[key] = (scores[i], p)
    both = model_only = broad_only = neither = 0
    for m in sorted({k[0] for k in pick}):
        f = ROOT / "data" / "pov" / f"{m}.csv"
        if not f.exists():
            continue
        keys = [k for k in pick if k[0] == m]
        pov = pov_at(load_pov(f), [t_of[k] for k in keys])
        for k in keys:
            b = pov.get(t_of[k])
            if b is None:
                continue
            mr, br = pick[k][1] == killer_of.get(k), b == killer_of.get(k)
            both += mr and br; model_only += mr and not br; broad_only += br and not mr; neither += not (mr or br)
    n = both + model_only + broad_only + neither
    d = model_only + broad_only
    k = min(model_only, broad_only)
    p = min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d) if d else 1.0
    return n, both, model_only, broad_only, neither, p


def evaluate(scores, meta, idx):
    best = {}
    for i in idx:
        m, kid, lead, p, ik, fi = meta[i][:6]
        if lead != TEST_LEAD:
            continue
        key = (m, kid)
        if key not in best or scores[i] > best[key][0]:
            best[key] = (scores[i], ik, fi)
    n = len(best)
    return n, sum(v[1] for v in best.values()) / n, sum(v[2] for v in best.values()) / n


_LOG: list[str] = []


def print(*args, **kw):  # noqa: A001 - also keep every printed line for data/results/lgbm.txt
    line = " ".join(str(a) for a in args)
    _LOG.append(line)
    builtins.print(*args, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    raw, extra, meta, hp = load()
    names = FEATURES + [f"role_{r.lower()}" for r in ROLES]
    X = np.hstack([raw, extra])
    has_health = bool((hp[:, 0] >= 0).any())
    if has_health:
        print(f"health known for {np.mean(hp[:, 0] >= 0):.0%} of rows -> also testing a model WITH health")
    maps = np.array([m[0] for m in meta])
    gkey = np.array([f"{m[0]}|{m[1]}|{m[2]}" for m in meta])
    order = np.lexsort((gkey,))                       # rows of one moment next to each other
    killer = np.array([m[4] for m in meta])
    fight = np.array([m[5] for m in meta])
    col = {f: i for i, f in enumerate(FEATURES)}
    rule = raw[:, col["enemies_walk_90"]] * 1000 - raw[:, col["walk_nearest_enemy"]]
    print(f"{len(meta)} rows, {len(set(gkey))} moments, {len(set(maps))} maps, {X.shape[1]} features")
    print(f"role coverage: {np.mean(extra.sum(1) > 0):.0%} of rows have a role")
    if args.dry_run:
        print("dry run: data looks fine")
        return
    import lightgbm as lgb

    def group_sizes(idx):
        _, counts = np.unique(gkey[idx], return_counts=True)       # gkey[idx] is sorted (idx from `order`)
        return counts

    variants = {"killer only": killer, "fight aware": 2 * killer + (fight - killer)}
    Xh = np.hstack([X, np.where(hp < 0, np.nan, hp)])      # NaN = unknown; LightGBM handles it
    names_h = names + ["hp", "enemy_hp", "hp_adv"]
    pred_health = np.zeros(len(meta))
    lin = np.zeros(len(meta))
    preds = {v: np.zeros(len(meta)) for v in variants}
    Xlin = transform(raw)
    gnum = np.unique(gkey, return_inverse=True)[1]
    for m in sorted(set(maps)):
        tr = order[maps[order] != m]
        te = np.flatnonzero(maps == m)
        lin[te] = SoftmaxPicker().fit(Xlin[tr], gnum[tr], killer[tr]).score(Xlin[te])
        for v, rel in variants.items():
            ds = lgb.Dataset(X[tr], label=rel[tr], group=group_sizes(tr), feature_name=names)
            booster = lgb.train(PARAMS, ds, num_boost_round=args.trees)
            preds[v][te] = booster.predict(X[te])
        if has_health:
            ds = lgb.Dataset(Xh[tr], label=variants["fight aware"][tr], group=group_sizes(tr), feature_name=names_h)
            pred_health[te] = lgb.train(PARAMS, ds, num_boost_round=args.trees).predict(Xh[te])
        print(f"  tested on {m}", flush=True)

    idx = np.arange(len(meta))
    print(f"\n{evaluate(rule, meta, idx)[0]} kills, {TEST_LEAD} s before each, every map tested by a model that never saw it:")
    rows_out = [("best rule", rule), ("linear model", lin)] + [(f"LightGBM, {v}", p) for v, p in preds.items()]
    if has_health:
        rows_out.append(("LightGBM + health", pred_health))
    for label, sc in rows_out:
        n, k, f_ = evaluate(sc, meta, idx)
        print(f"  {label:24s} caught killer {k:5.1%}   caught fight {f_:5.1%}")
    print("  real broadcast (its readable kills): 27.7% / 55.2%")
    for v, p in preds.items():
        wins = sum(evaluate(p, meta, np.flatnonzero(maps == m))[1] > evaluate(rule, meta, np.flatnonzero(maps == m))[1]
                   for m in sorted(set(maps)))
        print(f"  LightGBM {v}: beats the rule on {wins} of {len(set(maps))} maps")
    per_map = []                                       # one row per map and picker, for the results report
    for m in sorted(set(maps)):
        mi = np.flatnonzero(maps == m)
        for label, sc in rows_out:
            n, k, f_ = evaluate(sc, meta, mi)
            per_map.append({"map": m, "picker": label, "kills_scored": n,
                            "caught_killer": round(k, 4), "caught_fight": round(f_, 4)})

    best_v = max(preds, key=lambda v: evaluate(preds[v], meta, idx)[1])
    n, both, mo, bo, nei, pval = head_to_head(preds[best_v], meta)
    print(f"\nHead-to-head on the same {n} kills (broadcast POV readable), caught the killer {TEST_LEAD} s before:")
    print(f"  LightGBM ({best_v}): {(both + mo) / n:.1%}   real broadcast: {(both + bo) / n:.1%}")
    print(f"  both right {both}, only the model {mo}, only the broadcast {bo}, neither {nei}")
    verdict = "a real difference" if pval < 0.05 else "NOT a clear difference (could be chance)"
    print(f"  sign test on the {mo + bo} kills where they disagree: p = {pval:.2f} -> {verdict}")
    if has_health:
        n, both, mo, bo, nei, pval = head_to_head(pred_health, meta)
        print(f"  LightGBM + health: {(both + mo) / n:.1%} vs broadcast {(both + bo) / n:.1%} (p = {pval:.2f})")
    ds = lgb.Dataset(X[order], label=variants[best_v][order], group=group_sizes(order), feature_name=names)
    final = lgb.train(PARAMS, ds, num_boost_round=args.trees)
    (ROOT / "models").mkdir(exist_ok=True)
    final.save_model(str(ROOT / "models" / "lgbm_picker.txt"))
    imp = final.feature_importance("gain")
    print(f"\nsaved models/lgbm_picker.txt ({best_v}); what it relies on most (share of total gain):")
    for name, g in sorted(zip(names, imp), key=lambda x: -x[1])[:12]:
        print(f"  {name:22s} {g / imp.sum():5.1%}")
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lgbm.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    with open(out / "lgbm_per_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_map[0]))
        w.writeheader()
        w.writerows(per_map)
    with open(out / "lgbm_scores.csv", "w", newline="", encoding="utf-8") as f:
        # every player's leave-one-map-out score 2.5 s before each kill (fight aware): the demo app shows these
        w = csv.writer(f)
        w.writerow(["map", "kill_id", "player", "score"])
        for i, (m, kid, lead, p) in enumerate(x[:4] for x in meta):
            if lead == TEST_LEAD:
                w.writerow([m, kid, p, round(float(preds["fight aware"][i]), 5)])
    with open(out / "lgbm_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain_share"])
        for name, g in sorted(zip(names, imp), key=lambda x: -x[1]):
            w.writerow([name, round(g / imp.sum(), 4)])
    builtins.print("results saved to data/results/lgbm.txt")


if __name__ == "__main__":
    main()
