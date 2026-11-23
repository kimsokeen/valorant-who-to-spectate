"""Walls, doors, line of sight and walking distance on Summit's minimap.

configs/maps/summit_walkable.png : walkable floor (doors drawn open), built from
    the voted floor of all 18 maps plus where players actually stand (sites).
configs/maps/summit_doors.json   : the pixels of the A, mid and B doors. A closed
    door is drawn as a black bar, so its state is read from each frame.

line_of_sight : the straight line between two players crosses no wall and no
    closed door (a pixel or two of slack at the ends: icon centres aren't exact).
walking_distance : shortest path around walls (and closed doors), on 6 px cells.
    All cell-to-cell distances are computed once per combination of door states
    and cached, so per-frame queries are lookups.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import cv2
import numpy as np

CELL = 6
DOOR_DARK = 60          # grey level below which a door pixel counts as black
DOOR_CLOSED = 0.4       # share of black door pixels that means "closed"


class Summit:
    def __init__(self, walkable_png: str | Path, doors_json: str | Path, cache_dir: str | Path | None = None):
        self.walk = cv2.imread(str(walkable_png), cv2.IMREAD_GRAYSCALE) > 0
        self.doors = {k: np.array(v) for k, v in json.loads(Path(doors_json).read_text())["doors"].items()}
        self.door_names = sorted(self.doors)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._masks: dict[tuple, np.ndarray] = {}
        self._dist: dict[tuple, tuple] = {}

    # ---------- doors ----------
    def door_states(self, minimap_bgr: np.ndarray) -> tuple[bool, ...]:
        """(closed?) per door, in self.door_names order, read from one minimap image."""
        g = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY) if minimap_bgr.ndim == 3 else minimap_bgr
        return tuple(bool((g[px[:, 1], px[:, 0]] < DOOR_DARK).mean() > DOOR_CLOSED) for px in
                     (self.doors[k] for k in self.door_names))

    def mask(self, closed: tuple[bool, ...] = ()) -> np.ndarray:
        closed = tuple(closed) or (False,) * len(self.door_names)
        if closed not in self._masks:
            m = self.walk.copy()
            for name, c in zip(self.door_names, closed):
                if c:
                    px = self.doors[name]
                    m[px[:, 1], px[:, 0]] = False
            self._masks[closed] = m
        return self._masks[closed]

    # ---------- line of sight ----------
    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float],
                      closed: tuple[bool, ...] = (), end_slack: float = 3.0, max_blocked: int = 1) -> bool:
        m = self.mask(closed)
        (x0, y0), (x1, y1) = a, b
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length <= 2 * end_slack:
            return True
        n = int(length * 2)
        ts = np.linspace(end_slack / length, 1 - end_slack / length, max(n, 2))
        xs = np.clip(np.round(x0 + (x1 - x0) * ts).astype(int), 0, m.shape[1] - 1)
        ys = np.clip(np.round(y0 + (y1 - y0) * ts).astype(int), 0, m.shape[0] - 1)
        blocked = ~m[ys, xs]
        # count distinct blocked pixels, not samples
        return len({(x, y) for x, y, bl in zip(xs, ys, blocked) if bl}) <= max_blocked

    # ---------- walking distance ----------
    def _graph(self, closed: tuple[bool, ...]):
        m = self.mask(closed)
        H, W = m.shape
        gh, gw = H // CELL, W // CELL
        cells = m[:gh * CELL, :gw * CELL].reshape(gh, CELL, gw, CELL).mean(axis=(1, 3)) >= 0.5
        ids = -np.ones((gh, gw), int)
        ys, xs = np.where(cells)
        ids[ys, xs] = np.arange(len(ys))
        return cells, ids, ys, xs

    def _distances(self, closed: tuple[bool, ...]):
        closed = tuple(closed) or (False,) * len(self.door_names)
        if closed in self._dist:
            return self._dist[closed]
        tag = "".join("C" if c else "o" for c in closed)
        cache = self.cache_dir / f"summit_walk_{tag}.npz" if self.cache_dir else None
        cells, ids, ys, xs = self._graph(closed)
        if cache and cache.exists():
            D = np.load(cache)["D"]
        else:
            n = len(ys)
            nbrs = [[] for _ in range(n)]
            for i, (y, x) in enumerate(zip(ys, xs)):
                for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < cells.shape[0] and 0 <= xx < cells.shape[1] and ids[yy, xx] >= 0:
                        j = ids[yy, xx]
                        nbrs[i].append((j, w)); nbrs[j].append((i, w))
            D = np.full((n, n), np.inf, np.float32)
            for s in range(n):
                dist = D[s]
                dist[s] = 0.0
                pq = [(0.0, s)]
                while pq:
                    d, u = heapq.heappop(pq)
                    if d > dist[u]:
                        continue
                    for v, w in nbrs[u]:
                        nd = d + w
                        if nd < dist[v]:
                            dist[v] = nd
                            heapq.heappush(pq, (nd, v))
            D *= CELL
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, D=D)
        self._dist[closed] = (D, ids, ys, xs)
        return self._dist[closed]

    def _cell(self, ids, ys, xs, p):
        cx, cy = int(p[0]) // CELL, int(p[1]) // CELL
        if 0 <= cy < ids.shape[0] and 0 <= cx < ids.shape[1] and ids[cy, cx] >= 0:
            return ids[cy, cx]
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))     # nearest walkable cell
        return k

    def walking_distance(self, a, b, closed: tuple[bool, ...] = ()) -> float:
        D, ids, ys, xs = self._distances(closed)
        return float(D[self._cell(ids, ys, xs, a), self._cell(ids, ys, xs, b)])
