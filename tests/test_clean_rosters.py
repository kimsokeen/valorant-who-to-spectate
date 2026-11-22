"""Roster clean-up rules (no files needed)."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_rosters import pick_slots, spelling_fixes  # noqa: E402


def _half(left, right, agree=0.8):
    return {"team_left": [{"name": n, "agreement": agree} for n in left],
            "team_right": [{"name": n, "agreement": agree} for n in right]}


def test_slot_keeps_the_name_seen_in_kills():
    roster = {"halves": {"first": _half(["a", "b", "c", "d", "ZmijKK"], list("vwxyz"), 0.2),
                         "second": _half(["a", "b", "c", "d", "07 o PI"], list("vwxyz"), 0.2)}}
    players = pick_slots(roster, Counter({"ZmijKK": 9}))
    assert [p["name"] for p in players if p["side"] == "left"] == ["a", "b", "c", "d", "ZmijKK"]


def test_same_team_spellings_merge_but_other_teams_dont():
    def side(names, s):
        return [{"side": s, "slot": i, "name": n} for i, n in enumerate(names)]
    rosters = {
        "m1": side(["Cryo", "bang", "Vora", "Asuna", "Timotino"], "left") + side(["Dambi", "Xross", "Rb", "Francis", "Ivy"], "right"),
        "m2": side(["Cryo", "bang", "Vora", "Asuna", "Timotino"], "left") + side(["a", "b", "c", "d", "e"], "right"),
        "m3": side(["Cryo", "bang", "Vora", "Asuna", "Timatino"], "right") + side(["fOrsakeN", "something", "d4v41", "Jinggg", "invy"], "left"),
    }
    fixes = spelling_fixes(rosters)
    assert fixes == {"Timatino": "Timotino"}      # Ivy and invy are different players
