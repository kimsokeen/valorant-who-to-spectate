"""Check a live-mode replay log against the offline data of the same map.

    python scripts/live.py --replay crops/<map> --speed 0 --no-window --log data/live/<map>.csv
    python scripts/check_live.py <map> [model]     # model: the one live mode used

Reports, per map:
  agents    : rounds where live mode found exactly the map's 10 agents
  positions : share of the offline positions that live mode also found
              (same agent, within 12 px, at the same time)
  picks     : caught the killer 2.5 s before each kill, live vs offline
Live players are "Agent (colour)"; offline ones have names. They're matched by
agent (and by position for mirror picks).
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAD, MAX_GAP, NEAR = 2.5, 0.6, 12


def main(name: str, model: str | None = None) -> None:
    roster = json.loads((ROOT / "data" / "rosters" / f"{name}.json").read_text())
    agent_of = {p: a["agent"] for p, a in roster["agents"].items()}
    live = defaultdict(dict)          # t -> {player: (x, y, rank)}
    live_round = {}
    watch = {}                        # t -> the steady pick
    for r in csv.DictReader(open(ROOT / "data" / "live" / f"{name}.csv", encoding="utf-8")):
        t = float(r["time_s"])
        live_round[t] = int(r["round"])
        if r["player"]:
            live[t][r["player"]] = (float(r["x"]), float(r["y"]), int(r["rank"] or 99))
            if r.get("watch") == "1":
                watch[t] = r["player"]
    times = sorted(live_round)
    off = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "data" / "positions" / f"{name}.csv", encoding="utf-8")):
        off[float(r["time_s"])][r["player"]] = (float(r["x"]), float(r["y"]))

    def at(t):
        i = bisect.bisect_left(times, t)
        c = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(c, key=lambda j: abs(times[j] - t)) if c else None
        return times[j] if j is not None and abs(times[j] - t) <= MAX_GAP else None

    def match(t_off, lt):
        """{offline player: live player} for one moment."""
        out, used = {}, set()
        for p, (x, y) in off[t_off].items():
            best = min(((abs(lx - x) + abs(ly - y), q) for q, (lx, ly, _) in live[lt].items()
                        if q.split(" (")[0] == agent_of.get(p) and q not in used), default=None)
            if best and best[0] <= NEAR * 1.5:
                out[p] = best[1]
                used.add(best[1])
        return out

    # agents: per live round, the agents found vs the map's 10
    truth = sorted(agent_of.values())
    by_round = defaultdict(set)
    for t in times:
        for q in live[t]:
            by_round[live_round[t]].add(q)
    exact = sum(sorted(q.split(" (")[0] for q in s) == truth for s in by_round.values())
    # positions
    hit = total = 0
    for t_off in off:
        lt = at(t_off)
        if lt is None:
            continue
        total += len(off[t_off])
        hit += len(match(t_off, lt))
    # picks
    kills = [k for k in csv.DictReader(open(ROOT / "data" / "kills" / f"{name}.csv", encoding="utf-8")) if k["killer"]]
    caught = caught_w = scored = 0
    for k in kills:
        lt = at(float(k["time_s"]) - LEAD)
        if lt is None or not live[lt]:
            continue
        top = min(live[lt], key=lambda q: live[lt][q][2])
        scored += 1
        # the killer's live identity: same agent, nearest to his offline position if known
        t_off = min(off, key=lambda u: abs(u - lt)) if off else None
        m = match(t_off, lt) if t_off is not None and abs(t_off - lt) <= MAX_GAP else {}

        def is_killer(q):
            return m.get(k["killer"]) == q or (k["killer"] not in m and q.split(" (")[0] == agent_of.get(k["killer"])
                                              and sum(u.split(" (")[0] == agent_of.get(k["killer"]) for u in live[lt]) == 1)
        caught += is_killer(top)
        caught_w += is_killer(watch.get(lt, top))
    # switches per minute of tracked play, raw top pick vs the steady one
    tops = [min(live[t], key=lambda q: live[t][q][2]) for t in times if live[t]]
    steady = [watch.get(t) for t in times if live[t]]
    minutes = len(tops) * 0.5 / 60
    sw = lambda xs: sum(a != b for a, b in zip(xs, xs[1:])) / max(minutes, 1e-9)
    print(f"{name}")
    print(f"  agents   : all 10 right in {exact} of {len(by_round)} rounds")
    print(f"  positions: {hit / max(total, 1):.0%} of {total} offline positions found live")
    print(f"  picks    : caught the killer {caught / max(scored, 1):.1%} of {scored} kills (of {len(kills)})")
    if watch:
        print(f"  steady   : caught the killer {caught_w / max(scored, 1):.1%}; switches per minute "
              f"{sw(steady):.0f} steady vs {sw(tops):.0f} raw")
    base, n_base = offline_same_model(name, agent_of, model)
    print(f"  baseline : the same saved model on the offline data: {base:.1%} of {n_base} kills "
          f"(the gap is what live reading loses)")


def offline_same_model(name: str, agent_of: dict, model_path: str | None = None) -> tuple[float, int]:
    """Caught the killer with models/lgbm_picker.txt on this map's offline feature table.

    The saved model was trained on every map, this one included, so both this and
    the live number are optimistic; only their difference matters here."""
    import numpy as np
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT / "src"))
    from pickpov.features import FEATURES
    from pickpov.live import ROLES
    roles = json.loads((ROOT / "configs" / "agents" / "roles.json").read_text())
    model = lgb.Booster(model_file=str(model_path or ROOT / "models" / "lgbm_picker.txt"))
    best = {}
    for r in csv.DictReader(open(ROOT / "data" / "features" / f"{name}.csv", encoding="utf-8")):
        if float(r["lead"]) != LEAD:
            continue
        x = [float(r[c]) for c in FEATURES] + [float(roles.get(agent_of.get(r["player"], ""), "") == ro) for ro in ROLES]
        s = float(model.predict(np.array([x]))[0])
        k = int(r["kill_id"])
        if k not in best or s > best[k][0]:
            best[k] = (s, int(r["is_killer"]))
    return sum(v[1] for v in best.values()) / max(len(best), 1), len(best)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
