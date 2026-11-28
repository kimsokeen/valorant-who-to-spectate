"""Train the fight-aware LightGBM picker on every map except one, for a fair live-mode test.

    python scripts/train_lgbm_without.py <map>      # -> models/lgbm_without_<map>.txt

Same data, features and settings as scripts/train_lgbm.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_lgbm as T  # noqa: E402


def main(name: str) -> None:
    import lightgbm as lgb
    raw, extra, meta, _ = T.load()
    X = np.hstack([raw, extra])
    names = T.FEATURES + [f"role_{r.lower()}" for r in T.ROLES]
    maps = np.array([m[0] for m in meta])
    gkey = np.array([f"{m[0]}|{m[1]}|{m[2]}" for m in meta])
    killer = np.array([m[4] for m in meta])
    fight = np.array([m[5] for m in meta])
    rel = 2 * killer + (fight - killer)
    order = np.lexsort((gkey,))
    tr = order[maps[order] != name]
    if len(tr) == len(order):
        sys.exit(f"no map named {name} in data/features/")
    _, groups = np.unique(gkey[tr], return_counts=True)
    booster = lgb.train(T.PARAMS, lgb.Dataset(X[tr], label=rel[tr], group=groups, feature_name=names),
                        num_boost_round=300)
    out = ROOT / "models" / f"lgbm_without_{name}.txt"
    booster.save_model(str(out))
    print(f"trained on {len(set(maps[tr]))} maps -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
