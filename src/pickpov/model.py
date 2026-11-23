"""A learned picker: score every player at a moment, pick the highest.

The model is a "conditional logit" (softmax over the players in one frame):
    P(player i is the next killer) = exp(w . x_i) / sum_j exp(w . x_j)
It's trained to give the real killer the highest share, with plain numpy
gradient descent, so it runs anywhere. Features are transformed first
(log of distances, scaled speeds) and standardised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURES

# x, y need a non-linear model (hot spots on the map); left out of this linear one
USED = [f for f in FEATURES if f not in ("x", "y")]
CAP = 400.0


def transform(rows: np.ndarray) -> np.ndarray:
    """rows: (n, len(FEATURES)) raw values -> (n, len(USED)) model inputs."""
    col = {f: i for i, f in enumerate(FEATURES)}
    out = []
    for f in USED:
        v = rows[:, col[f]].astype(np.float64)
        if f in ("walk_nearest_enemy", "straight_nearest_enemy", "visible_nearest_enemy"):
            v = np.log1p(np.minimum(v, CAP))
        elif f in ("speed", "closing_speed"):
            v = np.clip(v, -200, 200) / 50.0
        elif f == "round_time":
            v = np.clip(v, 0, 150) / 100.0
        out.append(v)
    return np.stack(out, 1)


class SoftmaxPicker:
    def __init__(self, l2: float = 1e-3, lr: float = 0.5, iters: int = 400):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.mean = self.std = self.w = None

    def fit(self, X: np.ndarray, groups: np.ndarray, y: np.ndarray) -> "SoftmaxPicker":
        """X: (n, d) transformed features; groups: frame id per row; y: 1 for the killer."""
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mean) / self.std
        # keep only frames that contain the killer
        has = {g for g, t in zip(groups, y) if t}
        keep = np.array([g in has for g in groups])
        Z, g, t = Z[keep], groups[keep], y[keep].astype(float)
        uniq, gi = np.unique(g, return_inverse=True)
        w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            s = Z @ w
            s = s - np.maximum.reduceat(s, np.r_[0, np.flatnonzero(np.diff(gi)) + 1])[gi] if np.all(np.diff(gi) >= 0) else s
            e = np.exp(s - s.max())
            denom = np.bincount(gi, e)
            p = e / denom[gi]
            grad = Z.T @ (p - t) / len(uniq) + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.w

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"features": USED, "mean": self.mean.tolist(),
                                         "std": self.std.tolist(), "w": self.w.tolist()}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxPicker":
        d = json.loads(Path(path).read_text())
        m = cls()
        m.mean, m.std, m.w = np.array(d["mean"]), np.array(d["std"]), np.array(d["w"])
        return m
