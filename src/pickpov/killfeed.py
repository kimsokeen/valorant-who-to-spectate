"""Read kills (killer, victim, time) from kill feed crops.

Each kill feed row on the Champions 2026 overlay (1920x1080) is 34 px tall,
rows are 39 px apart and the first starts at y=97:

    [assists] [killer portrait] TAG killer [weapon] [icon] TAG victim [victim portrait]
              <--------- killer team colour ---------><--- victim team colour --->

The two teams' bars have different colours (teal vs red), so each row is split
where the colour changes and each side is read separately. Names are matched
against the map's roster, so OCR slips ("Tuk xo", a cut-off tag) still resolve.

A victim can only die once per round, so a kill is identified by
(round, victim): the same row stays on screen for several seconds and is
counted once, at the time it first appeared.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from .crops import with_margin
from .regions import Region
from .roster import ocr_line

ROW_Y0, ROW_STEP, ROW_H, MAX_ROWS = 97, 39, 34, 6   # 1080p frame pixels
ROUND_GAP_S = 5.0     # a gap this long between gameplay samples = new round
MATCH_MIN = 0.7       # minimum name similarity to accept a match
SAME_DEATH_S = 15.0   # sightings of a victim further apart than this are separate deaths


# ---------- names ----------

def similarity(text: str, name: str) -> float:
    """Best match of `name` anywhere inside `text` (0..1), ignoring case."""
    text, name = text.lower(), name.lower()
    if not text or not name:
        return 0.0
    if name in text:
        return 1.0
    n = len(name)
    best = 0.0
    for i in range(max(1, len(text) - n + 1)):
        for L in (n - 1, n, n + 1):
            if L > 0:
                best = max(best, SequenceMatcher(None, text[i:i + L], name).ratio())
    return best


def match_player(text: str, players: list[str]) -> tuple[str | None, float]:
    scored = sorted(((similarity(text, p), p) for p in players), reverse=True)
    if not scored or scored[0][0] < MATCH_MIN:
        return None, scored[0][0] if scored else 0.0
    # reject ties between two different players
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.02 and scored[1][1] != scored[0][1]:
        return None, scored[0][0]
    return scored[0][1], scored[0][0]


def match_pair(killer_text: str, victim_text: str,
               roster: dict[str, str]) -> tuple[str | None, str | None]:
    """Best (killer, victim) with the two on opposite teams.

    Killer and victim bars always have different team colours, so a pair from
    the same team means a misread (e.g. "Derke" read as "Darker"). The victim
    is required; the killer may be None (spike, fall damage, unreadable).
    """
    ks = {p: similarity(killer_text, p) for p in roster}
    vs = {p: similarity(victim_text, p) for p in roster}
    best, pair = 0.0, (None, None)
    for v, sv in vs.items():
        if sv < MATCH_MIN:
            continue
        opp = [(sk, k) for k, sk in ks.items() if roster[k] != roster[v] and sk >= MATCH_MIN]
        sk, k = max(opp) if opp else (0.0, None)
        score = sv + sk
        if score > best:
            best, pair = score, (k, v)
    return pair


def load_roster(path: str | Path) -> dict[str, str]:
    """{player name: 'left' | 'right'} from data/rosters/<map>.json."""
    r = json.loads(Path(path).read_text())
    out = {}
    for half in r["halves"].values():
        for side, key in (("left", "team_left"), ("right", "team_right")):
            for p in half[key]:
                if p["name"]:
                    out.setdefault(p["name"], side)
    return out


# ---------- one row ----------

def _bar_colours(row: np.ndarray) -> np.ndarray:
    """Per column: 1 = teal/green bar, 2 = red/pink bar, 0 = neither."""
    hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    teal = ((h > 60) & (h < 100) & (s > 60) & (v > 100)).sum(axis=0)
    red = (((h < 10) | (h > 160)) & (s > 80) & (v > 120)).sum(axis=0)
    need = row.shape[0] * 0.35
    out = np.zeros(row.shape[1], int)
    out[teal > need] = 1
    out[(red > need) & (red > teal)] = 2
    return out


def split_row(row: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a kill feed row into (killer part, victim part), or None if no row."""
    col = _bar_colours(row)
    if (col > 0).mean() < 0.25:              # no kill row here
        return None
    xs = np.where(col > 0)[0]
    victim_colour = col[xs[-1]]
    # victim part: the rightmost run of the victim colour (gaps for icons allowed)
    i = len(xs) - 1
    while i > 0 and (col[xs[i - 1]] == victim_colour or xs[i] - xs[i - 1] > 3 and col[xs[i - 1]] == 0):
        i -= 1
    split = xs[i]
    killer_cols = np.where(col[:split] > 0)[0]
    if len(killer_cols) < 20 or split > row.shape[1] - 20:
        return None
    return row[:, killer_cols[0]:split], row[:, split:]


def white_text(img: np.ndarray) -> np.ndarray:
    """Keep only the white name text (and icons), as light-on-dark for ocr_line."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < 70) & (hsv[..., 2] > 185)).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ---------- a whole map ----------

def read_map(map_dir: str | Path, regions: dict[str, Region], roster: dict[str, str],
             max_samples: int | None = None) -> dict:
    d = Path(map_dir)
    m = json.loads((d / "manifest.json").read_text())
    fw, fh = m["video_info"]["width"], m["video_info"]["height"]
    s = fw / 1920
    bx, by, _, _ = with_margin(regions["kill_feed"], m["margin"]).pixels(fw, fh)

    times = {}
    with open(d / "index.csv") as f:
        for r in csv.DictReader(f):
            times[int(r["sample"])] = float(r["time_s"])
    order = sorted(times)
    round_of, rnd = {}, 0
    spans = [[times[order[0]], times[order[0]]]] if order else []
    for a, b in zip([None] + order[:-1], order):
        if a is not None and times[b] - times[a] > ROUND_GAP_S:
            rnd += 1
            spans.append([times[b], times[b]])
        round_of[b] = rnd
        spans[-1][1] = times[b]

    sightings = defaultdict(list)    # (round, victim) -> [(time, killer or None)]
    unread = 0
    files = sorted((d / "kill_feed").iterdir())
    if max_samples:
        files = files[:max_samples]
    for p in files:
        sample = int(p.stem)
        img = cv2.imread(str(p))
        for k in range(MAX_ROWS):
            y0 = round((ROW_Y0 + ROW_STEP * k) * s) - by
            y1 = y0 + round(ROW_H * s)
            if y1 > img.shape[0]:
                break
            parts = split_row(img[y0:y1])
            if parts is None:
                continue    # one awkward row mustn't hide the rows below it
            # No caching of OCR results between rows: rows that look alike at low
            # resolution ("T1 iZu" vs "T1 stax") got each other's names.
            victim_text = ocr_line(white_text(parts[1]))
            _, guess = match_pair("", victim_text, roster)
            known = sightings.get((round_of[sample], guess), [])
            if guess and sum(1 for _, k in known if k) >= 3:
                killer, victim = None, guess      # killer already read 3 times; save time
            else:
                killer, victim = match_pair(ocr_line(white_text(parts[0])), victim_text, roster)
            if victim is None:
                unread += 1
                continue
            sightings[(round_of[sample], victim)].append((times[sample], killer))

    kills = []
    for (rnd, victim), seen in sightings.items():
        # a kill feed row stays ~5 s; sightings far apart are two deaths, which
        # means two rounds ran together (no break in gameplay between them)
        seen.sort(key=lambda s: s[0])   # by time only (killer may be None)
        groups = [[seen[0]]]
        for t, k in seen[1:]:
            if t - groups[-1][-1][0] > SAME_DEATH_S:
                groups.append([])
            groups[-1].append((t, k))
        for g in groups:
            killers = Counter(k for _, k in g if k)
            killer = killers.most_common(1)[0][0] if killers else None
            kills.append({
                "round": rnd, "time_s": round(g[0][0], 2), "victim": victim,
                "victim_side": roster[victim], "killer": killer,
                "killer_side": roster.get(killer) if killer else None,
                "seen": len(g),
            })
    kills.sort(key=lambda k: k["time_s"])
    return {"map": d.name, "rounds": rnd + 1, "spans": spans, "kills": kills,
            "unreadable_rows": unread}


def split_joined_rounds(res: dict, rounds_played: int) -> int:
    """Split stretches that hold two rounds (no gameplay break between them).

    Sign: a player dies twice in one stretch. Split at the longest quiet
    period between his two deaths (the buy phase). Returns splits made.
    """
    splits = 0
    while res["rounds"] < rounds_played:
        by_round = defaultdict(list)
        for k in res["kills"]:
            by_round[k["round"]].append(k)
        target = None
        for r in range(res["rounds"]):
            ks = sorted(by_round[r], key=lambda k: k["time_s"])
            seen = {}
            for k in ks:
                if k["victim"] in seen:
                    first = seen[k["victim"]]
                    between = [x["time_s"] for x in ks if first["time_s"] <= x["time_s"] <= k["time_s"]]
                    gaps = [(b - a, (a + b) / 2) for a, b in zip(between, between[1:])]
                    target = (r, max(gaps)[1])
                    break
                seen[k["victim"]] = k
            if target:
                break
        if target is None:
            break
        r, cut = target
        for k in res["kills"]:
            if k["round"] > r or (k["round"] == r and k["time_s"] > cut):
                k["round"] += 1
        res["spans"].insert(r + 1, [cut, res["spans"][r][1]])
        res["spans"][r][1] = cut
        res["rounds"] += 1
        splits += 1
    return splits


def drop_duplicate_deaths(res: dict) -> int:
    """Nobody dies twice in a round: keep the best-supported death, drop the rest.

    Run after the round fixes. A second "death" left in a round is a misread
    (e.g. "Derke" seen once as "Darker"). Keep the one with a known killer and
    the most sightings. Returns how many were dropped.
    """
    best = {}
    for k in res["kills"]:
        key = (k["round"], k["victim"])
        score = (k["killer"] is not None, k["seen"])
        if key not in best or score > best[key][0]:
            best[key] = (score, id(k))
    keep = {v[1] for v in best.values()}
    before = len(res["kills"])
    res["kills"] = [k for k in res["kills"] if id(k) in keep]
    return before - len(res["kills"])


def merge_split_rounds(res: dict, rounds_played: int) -> int:
    """Join stretches that are really one round split by a pause or replay.

    Stretches come from gaps in gameplay, so a mid-round pause splits a round in
    two. Pieces of one round never share a victim (nobody dies twice in a round)
    and the first piece can't end with a whole team dead. While there are more
    stretches than rounds played, merge the neighbouring pair that fits that
    and has the shortest gap between them. Returns how many merges were made.
    """
    merges = 0
    while res["rounds"] > rounds_played:
        by_round = defaultdict(list)
        for k in res["kills"]:
            by_round[k["round"]].append(k)
        best = None
        for r in range(res["rounds"] - 1):
            a, b = by_round[r], by_round[r + 1]
            if {k["victim"] for k in a} & {k["victim"] for k in b}:
                continue
            if any(sum(k["victim_side"] == side for k in a) == 5 for side in ("left", "right")):
                continue
            if len(a) + len(b) > 10:
                continue
            gap = res["spans"][r + 1][0] - res["spans"][r][1]
            if best is None or gap < best[0]:
                best = (gap, r)
        if best is None:
            break
        r = best[1]
        for k in res["kills"]:
            if k["round"] > r:
                k["round"] -= 1
        res["spans"][r][1] = res["spans"][r + 1][1]
        del res["spans"][r + 1]
        res["rounds"] -= 1
        merges += 1
    return merges
