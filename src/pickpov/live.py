"""Live mode: from screen frames to "watch this player", a couple of times a second.

The offline pipeline knows the whole VOD in advance (roster, kills, rounds).
Live, we only have the frames seen so far, so:

  agents   : found from the minimap itself. At the start of each round every
             agent's portrait is searched over the whole minimap (slow, on all
             CPU cores); the 5 agents seen most often in each team colour are
             the round's players. A player is "Chamber (teal)".
  tracking : after that each player is searched only near his last position
             (fast). A player missing for a while gets a full search again.
  rounds   : the score digits, as in pickpov.rounds: a gameplay stretch whose
             score differs from the previous stretch's is a new round.
  alive    : whoever is on the minimap (a player missing up to 1.5 s keeps his
             last position, marked held, as offline).

Each frame then goes through the same features (pickpov.features) and the
saved LightGBM model as offline, and the players are ranked.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import cards
from . import minimap as mm
from .crops import with_margin
from .features import FEATURES, player_features
from .gameplay import GameplayDetector
from .regions import load_regions
from .rounds import CHANGED_PCT, PIXEL_DELTA, SCORE_BOX, STRETCH_GAP_S
from .walls import Summit

ROLES = ["Duelist", "Initiator", "Controller", "Sentinel"]   # same order as scripts/train_lgbm.py
MARGIN = 0.08              # crops were cut with this margin; positions are in those crop pixels
SETUP_FRAMES = 4          # full searches (with any sighting) before the roster may be fixed
SETUP_MAX = 12             # ... and the most, after which whatever was found is used
SETUP_MIN_SEEN = 3         # sightings needed to count as a player of the round
TRACK_RADIUS = 40          # px searched around the last position (players move < 30 px per 0.5 s)
HOLD_S = 1.5               # a missing player keeps his last position this long
RESEARCH_EVERY_S = 2.0     # how often a missing player gets a full-minimap search
PREV_S = 1.0               # movement features compare with the frame ~1 s earlier
COLOURS = ("teal", "red")
MAP_AGENTS_AFTER_ROUNDS = 3   # rounds of evidence before the map's 10 agents are fixed
MIRROR_SHARE = 0.3           # seen in both colours in this share of frames = a mirror pick
CARD_MIN_SCORE = 0.5         # a card portrait match below this is ignored (grey dead card, overlay)
CARD_MIN_FRAMES = 3          # card reads before the cards' roster is used
WATCH_MIN_S = 1.5            # stay on the watched player at least this long ...
SWITCH_MARGIN = 0.0          # ... then switch to whoever is top (0.2 halved switches but cost ~2 points on replays)
DEAD_READS = 2               # grey card this many reads in a row = dead (switch at once)



def to_1080p(frame: np.ndarray) -> np.ndarray:
    """The centred 16:9 picture of a frame, at 1920x1080.

    A fullscreen video on a 16:10 (or 21:9) screen has black bars; cut them off
    instead of stretching them in, or every overlay box lands off target."""
    h, w = frame.shape[:2]
    if h * 16 > w * 9 + 16:                      # too tall: bars top and bottom
        vh = round(w * 9 / 16)
        frame = frame[(h - vh) // 2:(h - vh) // 2 + vh]
    elif w * 9 > h * 16 + 9:                     # too wide: bars left and right
        vw = round(h * 16 / 9)
        frame = frame[:, (w - vw) // 2:(w - vw) // 2 + vw]
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame

@dataclass
class Pick:
    """One processed frame."""
    t: float
    status: str                                    # "not gameplay", "finding agents", "tracking"
    round: int = 0
    round_time: float = 0.0
    positions: dict = field(default_factory=dict)  # player -> (x, y, colour, held)
    ranking: list = field(default_factory=list)    # [(player, score)], best first
    doors: tuple = ()
    ms: float = 0.0                                # processing time
    watch: str | None = None                       # the steady pick: who to spectate now
    numbers: dict = field(default_factory=dict)    # player -> spectator number "1".."9", "0"

    @property
    def top(self) -> str | None:
        return self.ranking[0][0] if self.ranking else None


class LiveReader:
    def __init__(self, root: str | Path, threads: int = 8, model: str | Path | None = None):
        import lightgbm as lgb      # after cv2: both bring OpenMP; this order is fine on Windows

        root = Path(root)
        regions = load_regions(root / "configs" / "broadcast_regions.yaml")
        self.box_minimap = with_margin(regions["minimap"], MARGIN)
        self.box_hud = with_margin(regions["hud_top"], MARGIN)
        self.box_cards = {s: with_margin(regions[f"team_{s}"], MARGIN) for s in ("left", "right")}
        self.card_art = cards.load_card_art(root / "configs" / "cards") if (root / "configs" / "cards").exists() else {}
        self.detector = GameplayDetector.load(root / "configs" / "broadcast_regions.yaml")
        agents_dir = root / "configs" / "agents"
        self.all_agents = sorted(p.name for p in agents_dir.iterdir() if (p / "minimap.png").exists())
        if not self.all_agents:
            raise FileNotFoundError("No agent art in configs/agents/. Run scripts/get_agent_icons.py.")
        self.art = mm.load_icon_art(agents_dir, set(self.all_agents))
        self.roles = json.loads((agents_dir / "roles.json").read_text()) if (agents_dir / "roles.json").exists() else {}
        maps = root / "configs" / "maps"
        self.summit = Summit(maps / "summit_walkable.png", maps / "summit_doors.json", cache_dir=maps / "cache")
        walk = (cv2.imread(str(maps / "summit_walkable.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        self._walk = cv2.dilate(walk, np.ones((15, 15), np.uint8)) > 0
        self.area = None                            # sized to the first minimap crop
        self.model = lgb.Booster(model_file=str(model or root / "models" / "lgbm_picker.txt"))
        self.pool = ThreadPoolExecutor(threads)
        self.reset_match()

    # ---------- state ----------
    def reset_match(self) -> None:
        self.round = -1
        self.round_start = None
        self.last_gameplay_t = None
        self.stretch_start = None
        self.score_ref = None          # score area of the previous stretch
        self.score_pending = None      # this stretch's, taken a few seconds in
        self.match_agents: Counter = Counter()      # agent -> rounds it was in the roster
        # evidence over the whole map: agents never change, so after a few rounds the
        # map's 10 agents (mirror picks counted twice) are fixed and setup only finds colours
        self.ev_frames = 0
        self.ev_any: Counter = Counter()            # frames an agent was seen in some colour
        self.ev_both: Counter = Counter()           # ... in both colours at once (a mirror pick)
        self.map_agents: list[str] = []             # fixed 10 slots, e.g. ["Neon", "Neon", "Omen", ...]
        # player cards: per (side, slot) the summed match score of each agent, over the map
        self.card_votes: dict[tuple[str, int], Counter] = {(s, i): Counter() for s in ("left", "right") for i in range(5)}
        self.card_frames = 0
        self.cards_work = False
        self.new_round(0.0)

    def new_round(self, t: float) -> None:
        self.round += 1
        self.round_start = t
        self.roster: list[tuple[str, str]] = []     # [(agent, colour)] once known
        self.sightings: Counter = Counter()
        self.sight_score: Counter = Counter()
        self.setup_done = 0            # full searches that saw anything
        self.setup_tries = 0
        self.tracks: dict[str, tuple[float, float, float]] = {}   # player -> (x, y, last seen t)
        self.last_full: dict[str, float] = {}
        self.history: deque = deque(maxlen=12)      # (t, {player: (x, y, colour)})
        self.watching: tuple[str, float] | None = None   # (player, since t)
        self.card_dark: Counter = Counter()          # (side, slot) -> grey reads in a row
        self.card_colour: dict[str, Counter] = {"left": Counter(), "right": Counter()}   # this round

    # ---------- frames ----------
    def crops(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Minimap and HUD crops from a full broadcast frame (any size)."""
        frame = to_1080p(frame)
        return self.box_minimap.crop(frame), self.box_hud.crop(frame)

    def process_frame(self, t: float, frame: np.ndarray) -> Pick:
        frame = to_1080p(frame)
        if not self.detector.is_gameplay(frame):
            return Pick(t, "not gameplay", self.round)
        mini, hud = self.crops(frame)
        team = {s: b.crop(frame) for s, b in self.box_cards.items()}
        return self.process(t, mini, hud, team)

    def process(self, t: float, mini: np.ndarray, hud: np.ndarray, team: dict | None = None) -> Pick:
        """One gameplay frame: minimap + HUD crops (as cut by extract_crops), and the
        team_left / team_right card crops when available ({"left": img, "right": img})."""
        t0 = time.perf_counter()
        if self.area is None or self.area.shape != mini.shape[:2]:
            self.area = cv2.resize(self._walk.astype(np.uint8), mini.shape[1::-1],
                                   interpolation=cv2.INTER_NEAREST) > 0
        self._rounds(t, hud)
        hsv = cv2.cvtColor(mini, cv2.COLOR_BGR2HSV)
        if team and self.card_art:
            self._read_cards(team)
        card_roster = self._card_roster()
        if card_roster:
            self.cards_work = True
            if card_roster != self.roster:
                self.roster = card_roster
        elif not self.roster and not self.cards_work:
            self._setup(mini, hsv)      # no usable cards on this broadcast: find agents on the minimap
        # slot numbers, as on the cards: left team 1-5 top to bottom, right team 6-9, 0
        numbers = ({f"{a} ({c})": str((i + 1) % 10) for i, (a, c) in enumerate(self.roster)}
                   if self.cards_work else {})
        status = "tracking" if self.roster else "finding agents"
        found = self._track(t, mini, hsv) if self.roster else {}
        pick = Pick(t, status, self.round, t - self.round_start, ms=0.0, numbers=numbers)
        if found:
            pick.doors = self.summit.door_states(mini)
            pick.positions, pick.ranking = self._rank(t, found, pick.doors)
            pick.watch = self._steady(t, pick, numbers)
        pick.ms = (time.perf_counter() - t0) * 1000
        return pick

    # ---------- rounds ----------
    def _score_area(self, hud: np.ndarray, side: str) -> np.ndarray:
        hx, hy, _, _ = self.box_hud.pixels(1920, 1080)
        x0, y0, x1, y1 = SCORE_BOX[side]
        return cv2.cvtColor(hud[y0 - hy:y1 - hy, x0 - hx:x1 - hx], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _rounds(self, t: float, hud: np.ndarray) -> None:
        if self.last_gameplay_t is None or t - self.last_gameplay_t > STRETCH_GAP_S:
            if self.score_pending is not None:
                self.score_ref = self.score_pending
            self.stretch_start, self.score_pending = t, None
            if self.last_gameplay_t is None:
                self.round_start = t
        self.last_gameplay_t = t
        # a few seconds into a stretch the score has settled: compare with the previous stretch
        if self.score_pending is None and t - self.stretch_start >= 3:
            self.score_pending = {s: self._score_area(hud, s) for s in SCORE_BOX}
            if self.score_ref is not None:
                changed = any((np.abs(self.score_pending[s] - self.score_ref[s]) > PIXEL_DELTA).mean() * 100
                              > CHANGED_PCT for s in SCORE_BOX)
                if changed:
                    self.new_round(self.stretch_start)
                    self.round_start = self.stretch_start

    # ---------- agents from the player cards ----------
    def _read_cards(self, team: dict) -> None:
        for side, crop in team.items():
            if crop is None:
                continue
            c = cards.team_colour(crop)
            if c:
                self.card_colour[side][c] += 1
            for slot in range(5):
                for score, agent in cards.read_slot(crop, side, slot, self.card_art)[:1]:
                    if score >= CARD_MIN_SCORE:
                        self.card_votes[(side, slot)][agent] += score
                        self.card_dark[(side, slot)] = 0
                    else:
                        self.card_dark[(side, slot)] += 1     # grey (dead) card, or covered
        self.card_frames += 1

    def _card_roster(self) -> list[tuple[str, str]]:
        """[(agent, colour)] from the cards, once every slot and both colours are clear."""
        if self.card_frames < CARD_MIN_FRAMES:
            return []
        colour = {}
        for side, votes in self.card_colour.items():
            if not votes:
                return []
            colour[side] = votes.most_common(1)[0][0]
        if colour["left"] == colour["right"]:
            return []
        roster = []
        for (side, slot), votes in self.card_votes.items():
            if not votes:
                return []           # a slot never read (an agent with no card art yet): use the minimap
            roster.append((votes.most_common(1)[0][0], colour[side]))
        return roster

    # ---------- agents from the minimap (fallback) ----------
    def _peaks(self, mini, hsv, agent, box=None, peaks=mm.PEAKS_PER_AGENT):
        """[(x, y, score, {colour: ring pixels})] for one agent, in a box or the whole minimap."""
        x0, y0, x1, y1 = box or (0, 0, mini.shape[1], mini.shape[0])
        img = mini[y0:y1, x0:x1]
        best = None
        for tm, m in self.art[agent]:
            if img.shape[0] < tm.shape[0] or img.shape[1] < tm.shape[1]:
                return []
            r = cv2.matchTemplate(img, tm, cv2.TM_CCOEFF_NORMED, mask=m)
            r[~np.isfinite(r)] = -1
            sz = tm.shape[0]
            r = r * self.area[y0 + sz // 2: y0 + sz // 2 + r.shape[0], x0 + sz // 2: x0 + sz // 2 + r.shape[1]]
            best = r if best is None or best.shape != r.shape else np.maximum(best, r)
        out, sz = [], mm.SIZES[0]
        for _ in range(peaks):
            _, score, _, loc = cv2.minMaxLoc(best)
            if score < mm.MIN_MATCH:
                break
            cv2.circle(best, loc, 9, -1.0, -1)
            x, y = x0 + loc[0] + sz // 2, y0 + loc[1] + sz // 2
            out.append((x, y, float(score), mm.ring_colours(hsv, x, y)))
        return out

    @staticmethod
    def _colour_of(score: float, ring: dict, mirror: bool, want: str | None = None) -> str | None:
        """The team colour of a match, or None if it isn't a real icon (same tests as minimap.find_players)."""
        for c in ([want] if want else COLOURS):
            mine, other = ring[c], ring[COLOURS[1 - COLOURS.index(c)]]
            if mine >= mm.MIN_RING and (not mirror or mine > other):
                return c
            if not mirror and score >= mm.STRONG_MATCH and other < mm.MIN_RING:
                return c
        return None

    def _setup(self, mini, hsv) -> None:
        """A full search for every agent: count which (agent, colour) pairs are on the map."""
        # the first search of every round looks for all agents, so the map's agents keep
        # gathering evidence (and a wrong pick can be corrected); later ones only the map's
        full = not self.map_agents or self.setup_tries == 0
        search = self.all_agents if full else sorted(set(self.map_agents))
        results = list(self.pool.map(lambda a: (a, self._peaks(mini, hsv, a)), search))
        any_seen = False
        frame_colours: dict[str, set] = {}
        for agent, peaks in results:
            seen: dict[str, float] = {}
            for x, y, score, ring in peaks:
                for c in COLOURS:     # each colour judged on its own, so a mirror pick counts twice
                    mine, other = ring[c], ring[COLOURS[1 - COLOURS.index(c)]]
                    if mine >= mm.MIN_RING and mine >= 2 * other:
                        seen[c] = max(seen.get(c, 0.0), score)
            for c, s in seen.items():
                self.sightings[(agent, c)] += 1
                self.sight_score[(agent, c)] += s
                any_seen = True
            frame_colours[agent] = set(seen)
        self.setup_tries += 1
        self.setup_done += any_seen
        if any_seen and full:
            self.ev_frames += 1
            for a, cs in frame_colours.items():
                self.ev_any[a] += bool(cs)
                self.ev_both[a] += len(cs) == 2
            if self.map_agents:
                self._fix_map_agents()          # re-decided with the new evidence

        if self.map_agents:
            roster = self._colours_for_map_agents()
        else:
            def key(ac):     # seen most often; then agents from earlier rounds; then match quality
                return (self.sightings[ac], self.match_agents[ac[0]] > 0, self.sight_score[ac])

            roster = []
            for c in COLOURS:
                team = sorted((ac for ac, n in self.sightings.items() if ac[1] == c and n >= SETUP_MIN_SEEN),
                              key=key, reverse=True)
                roster += team[:5]
        if (self.setup_done >= SETUP_FRAMES and len(roster) == 10) or self.setup_tries >= SETUP_MAX:
            self.roster = roster or []
            if len(roster) == 10:
                self.match_agents.update(a for a, _ in roster)
            elif not roster:
                self.setup_tries = 0            # nothing at all yet (not live play?): keep looking
            if not self.map_agents and self.round >= MAP_AGENTS_AFTER_ROUNDS and self.ev_frames >= 8:
                self._fix_map_agents()

    def _fix_map_agents(self) -> None:
        """The map's 10 agents from all setup frames so far; mirror picks fill two slots."""
        slots = []
        for a, n in self.ev_any.most_common():
            slots += [a] * (2 if self.ev_both[a] >= MIRROR_SHARE * n else 1)
            if len(slots) >= 10:
                break
        self.map_agents = slots[:10]

    def _colours_for_map_agents(self) -> list[tuple[str, str]]:
        """This round's (agent, colour) for the fixed agents: 5 per colour."""
        roster, count = [], Counter()
        mirrors = {a for a, n in Counter(self.map_agents).items() if n == 2}
        for a in mirrors:
            roster += [(a, "teal"), (a, "red")]
            count.update(COLOURS)
        singles = [a for a in dict.fromkeys(self.map_agents) if a not in mirrors]
        # most certain first: the bigger the gap between the colours' sightings
        singles.sort(key=lambda a: -abs(self.sightings[(a, "teal")] - self.sightings[(a, "red")]))
        for a in singles:
            want = max(COLOURS, key=lambda c: self.sightings[(a, c)])
            c = want if count[want] < 5 else COLOURS[1 - COLOURS.index(want)]
            roster.append((a, c))
            count[c] += 1
        return roster

    # ---------- tracking ----------
    def _track(self, t: float, mini, hsv) -> dict[str, tuple[float, float, str, bool]]:
        agents_twice = {a for a, n in Counter(a for a, _ in self.roster).items() if n > 1}
        jobs = []
        for agent, colour in self.roster:
            p = f"{agent} ({colour})"
            last = self.tracks.get(p)
            if last and t - last[2] <= HOLD_S:
                x, y = int(last[0]), int(last[1])
                box = (max(0, x - TRACK_RADIUS), max(0, y - TRACK_RADIUS),
                       min(mini.shape[1], x + TRACK_RADIUS), min(mini.shape[0], y + TRACK_RADIUS))
                jobs.append((p, agent, colour, box, 2))
            elif t - self.last_full.get(p, -1e9) >= RESEARCH_EVERY_S or last is None:
                self.last_full[p] = t
                jobs.append((p, agent, colour, None, mm.PEAKS_PER_AGENT))
        results = self.pool.map(lambda j: (j, self._peaks(mini, hsv, j[1], j[3], j[4])), jobs)
        taken: list[tuple[float, float]] = []
        for (p, agent, colour, _, _), peaks in sorted(results, key=lambda r: r[0][3] is None):
            for x, y, score, ring in peaks:
                if self._colour_of(score, ring, agent in agents_twice, colour) == colour \
                        and all(abs(x - a) + abs(y - b) > 6 for a, b in taken):
                    self.tracks[p] = (x, y, t)
                    taken.append((x, y))
                    break
        out = {}
        for agent, colour in self.roster:
            p = f"{agent} ({colour})"
            last = self.tracks.get(p)
            if last and t - last[2] <= HOLD_S:
                out[p] = (last[0], last[1], colour, last[2] < t)
        return out

    # ---------- ranking ----------
    def _rank(self, t, found, closed):
        frame = {p: (x, y, c) for p, (x, y, c, _) in found.items()}
        held = {p for p, v in found.items() if v[3]}
        prev = None
        for pt, pf in self.history:
            if t - pt >= PREV_S - 0.25:
                prev, prev_t = pf, pt
        self.history.append((t, frame))
        feats = player_features(frame, prev, (t - prev_t) if prev else 0.0, self.summit, closed,
                                t - self.round_start, held)
        names = list(feats)
        X = np.array([feats[p] + [float(self.roles.get(p.split(" (")[0], "") == r) for r in ROLES]
                      for p in names])
        scores = self.model.predict(X)
        ranking = sorted(zip(names, (float(s) for s in scores)), key=lambda x: -x[1])
        return found, ranking

    # ---------- steady pick ----------
    def _dead(self, player: str, pick: Pick, numbers: dict) -> bool:
        if player not in pick.positions:              # off the minimap longer than HOLD_S
            return True
        n = numbers.get(player)
        if n is not None:
            i = (int(n) - 1) % 10
            if self.card_dark[("left" if i < 5 else "right", i % 5)] >= DEAD_READS:
                return True
        return False

    def _steady(self, t: float, pick: Pick, numbers: dict) -> str | None:
        """The model's top player, but held: at least WATCH_MIN_S on one player, then
        a switch only to a clearly higher one; at once if the watched player dies."""
        score = dict(pick.ranking)
        alive = [(p, s) for p, s in pick.ranking if not self._dead(p, pick, numbers)]
        best = alive[0][0] if alive else pick.top
        cur = self.watching
        if cur and cur[0] in score and not self._dead(cur[0], pick, numbers):
            if t - cur[1] < WATCH_MIN_S or score[best] < score[cur[0]] + SWITCH_MARGIN:
                return cur[0]
        if not cur or best != cur[0]:
            self.watching = (best, t)
        return best

    def close(self) -> None:
        self.pool.shutdown(wait=False)


assert len(FEATURES) + len(ROLES) == 24, "the saved model expects 24 inputs"
