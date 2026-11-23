"""Where each player is, from the minimap.

The minimap draws each player as a ~14 px agent portrait inside a ring in his
team's colour: teal or red. Which team is teal changes with sides (it swaps at
halftime and every overtime round), so it's worked out per round.
We already know the 10 agents on the map and who plays them (pickpov.agents),
so for each agent we:

  1. template-match valorant-api.com's minimapPortrait art (14-15 px),
  2. keep peaks on the map drawing (not the game scene showing through),
  3. check the ring around each peak: real icons have 30-115 ring-coloured
     pixels, look-alikes in the background have ~0. The ring colour also says
     which team, which separates mirror picks (both teams on Neon).

Dead players (known from the kill feed) are not searched for.
Positions are in minimap-crop pixels; every Summit map on this overlay uses
the same crop, so positions are comparable across maps.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

SIZES = (14, 15)          # icon size on a 1080p broadcast, px
PEAKS_PER_AGENT = 4       # candidates checked per agent per frame
MIN_MATCH = 0.55          # template score
MIN_RING = 15             # pixels of the team colour in the ring
STRONG_MATCH = 0.75       # a portrait match this good counts even with a faint ring (not for mirror picks)
RING = (7, 12)            # ring radius range around the icon centre, px
TEAM_HUE = {"teal": lambda h: (h > 75) & (h < 100), "red": lambda h: (h < 8) | (h > 168)}


def load_icon_art(folder: str | Path, agents: set[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    art = {}
    for a in agents:
        im = cv2.imread(str(Path(folder) / a / "minimap.png"), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"No minimap art for {a}. Run scripts/get_agent_icons.py.")
        art[a] = [(cv2.resize(im[..., :3], (s, s), interpolation=cv2.INTER_AREA),
                   cv2.resize((im[..., 3] > 128).astype(np.uint8) * 255, (s, s), interpolation=cv2.INTER_NEAREST))
                  for s in SIZES]
    return art


def map_area(images: list[np.ndarray]) -> np.ndarray:
    """Where the map drawing is: pixels that stay grey across many frames.

    The game scene behind the semi-transparent minimap changes constantly;
    the map floor doesn't. Dilated so icons on the edges of corridors count.
    """
    gray = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in images]).astype(np.float32)
    sat = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2HSV)[..., 1] for i in images]).astype(np.float32)
    med, std, s = np.median(gray, 0), gray.std(0), np.median(sat, 0)
    floor = ((std < 28) & (med > 90) & (med < 215) & (s < 45)).astype(np.uint8)
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(floor, np.ones((15, 15), np.uint8)) > 0


def ring_colours(hsv: np.ndarray, x: int, y: int) -> dict[str, int]:
    r0, r1 = RING
    y0, y1, x0, x1 = max(0, y - r1), y + r1 + 1, max(0, x - r1), x + r1 + 1
    win = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
    d = np.hypot(xx - x, yy - y)
    ann = (d >= r0) & (d <= r1)
    h, s, v = win[..., 0][ann].astype(int), win[..., 1][ann], win[..., 2][ann]
    strong = (s > 90) & (v > 100)
    return {team: int((f(h) & strong).sum()) for team, f in TEAM_HUE.items()}


def find_players(img: np.ndarray, area: np.ndarray, art: dict, players: dict[str, tuple[str, str]],
                 team_colour: dict[str, str], alive: set[str]) -> dict[str, tuple[int, int, float]]:
    """{player: (x, y, score)} for alive players found in one minimap image.

    players: {name: (agent, side)}; team_colour: {side: "teal" | "red"}.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    found: dict[str, tuple[int, int, float]] = {}
    by_agent: dict[str, list[str]] = {}
    for p, (a, _) in players.items():
        if p in alive:
            by_agent.setdefault(a, []).append(p)
    for agent, who in by_agent.items():
        best = None
        for t, m in art[agent]:
            r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED, mask=m)
            r[~np.isfinite(r)] = -1
            sz = t.shape[0]
            r = r * area[sz // 2: sz // 2 + r.shape[0], sz // 2: sz // 2 + r.shape[1]]
            best = r if best is None or best.shape != r.shape else np.maximum(best, r)
        sz = SIZES[0]
        res = best.copy()
        for _ in range(PEAKS_PER_AGENT):
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < MIN_MATCH:
                break
            x, y = loc[0] + sz // 2, loc[1] + sz // 2
            cv2.circle(res, loc, 9, -1.0, -1)
            ring = ring_colours(hsv, x, y)
            mirror = len({players[p][1] for p in who}) > 1   # same agent on both teams
            for p in who:
                side = players[p][1]
                mine, other = ring[team_colour[side]], ring[team_colour["right" if side == "left" else "left"]]
                # In a fight an enemy icon sits right next to yours, so the ring also
                # catches the other team's colour. Only mirror picks need "mine > other".
                # In fights the ring is often faint (flashes, overlapping icons): a very
                # strong portrait match counts on its own unless the ring is clearly the
                # other team's. Mirror picks always need the ring to tell the teams apart.
                ring_ok = mine >= MIN_RING and (not mirror or mine > other)
                strong_ok = not mirror and score >= STRONG_MATCH and other < MIN_RING
                if p not in found and (ring_ok or strong_ok):
                    found[p] = (x, y, round(float(score), 3))
                    break
            if all(p in found for p in who):
                break
    return found


def deaths_by_round(kills_csv: Path, rounds_csv: Path) -> tuple[dict, list]:
    """({round: {player: death_time}}, [(round, start_s, end_s)])."""
    deaths: dict[int, dict[str, float]] = {}
    with open(kills_csv, newline="", encoding="utf-8") as f:
        for k in csv.DictReader(f):
            deaths.setdefault(int(k["round"]), {}).setdefault(k["victim"], float(k["time_s"]))
    with open(rounds_csv, newline="", encoding="utf-8") as f:
        spans = [(int(r["round"]), float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(f)]
    return deaths, spans


COLOURINGS = ({"left": "teal", "right": "red"}, {"left": "red", "right": "teal"})


def round_colouring(images: list[np.ndarray], area: np.ndarray, art: dict,
                    players: dict[str, tuple[str, str]], alive: set[str]) -> dict[str, str]:
    """Which team is teal this round: the colouring that finds more players."""
    counts = [sum(len(find_players(img, area, art, players, c, alive)) for img in images)
              for c in COLOURINGS]
    return COLOURINGS[int(np.argmax(counts))]


def read_positions(map_dir: str | Path, roster_json: str | Path, kills_csv: str | Path,
                   rounds_csv: str | Path, art_folder: str | Path, every: int = 1,
                   max_samples: int | None = None, start_s: float = 0.0):
    """Yield (sample, time_s, round, {player: (x, y, score)}) for a map."""
    d = Path(map_dir)
    r = json.loads(Path(roster_json).read_text())
    players = {p["name"]: (r["agents"][p["name"]]["agent"], p["side"]) for p in r["players"]
               if r.get("agents", {}).get(p["name"], {}).get("agent")}
    art = load_icon_art(art_folder, {a for a, _ in players.values()})
    idx = []
    with open(d / "index.csv") as f:
        for row in csv.DictReader(f):
            idx.append((int(row["sample"]), float(row["time_s"])))
    area = map_area([cv2.imread(str(d / "minimap" / f"{s:06d}.jpg")) for s, _ in idx[::max(1, len(idx) // 120)]])
    deaths, spans = deaths_by_round(Path(kills_csv), Path(rounds_csv))

    def round_of(t: float) -> int:
        for r0, a, b in spans:
            if a - 1 <= t <= b + 1:
                return r0
        return min(spans, key=lambda x: min(abs(t - x[1]), abs(t - x[2])))[0] if spans else 0

    colouring: dict[int, dict[str, str]] = {}
    n = 0
    for s, t in idx[::every]:
        if t < start_s:
            continue
        if max_samples and n >= max_samples:
            break
        n += 1
        rnd = round_of(t)
        dead = {p for p, dt in deaths.get(rnd, {}).items() if dt <= t}
        if rnd not in colouring:
            # early in the round, before deaths: up to 6 frames from its first 40 s
            r0, a, b = next(x for x in spans if x[0] == rnd) if spans else (0, t, t)
            sample_frames = [x for x in idx if a + 3 <= x[1] <= min(b, a + 40)][::12][:6] or [(s, t)]
            imgs = [cv2.imread(str(d / "minimap" / f"{x[0]:06d}.jpg")) for x in sample_frames]
            alive0 = {p for p in players if deaths.get(rnd, {}).get(p, 1e9) > sample_frames[-1][1]}
            colouring[rnd] = round_colouring(imgs, area, art, players, alive0)
        img = cv2.imread(str(d / "minimap" / f"{s:06d}.jpg"))
        yield s, t, rnd, find_players(img, area, art, players, colouring[rnd], set(players) - dead)
