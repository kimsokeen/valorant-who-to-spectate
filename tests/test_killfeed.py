"""Kill feed name matching (no images or Tesseract needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.killfeed import match_pair, match_player, similarity  # noqa: E402

ROSTER = {"Chronicle": "left", "Derke": "left", "Jamppi": "left", "Sayonara": "left", "PROFEK": "left",
          "Darker": "right", "luk xo": "right", "Erde": "right", "tkzin": "right", "DaviH": "right"}


def test_ocr_slips_still_match():
    assert match_player("Tuk xo", list(ROSTER))[0] == "luk xo"
    assert match_player("IT Derke", list(ROSTER))[0] == "Derke"      # cut-off team tag
    assert match_player("~~~", list(ROSTER))[0] is None


def test_pair_must_be_opposite_teams():
    # "Darker" and "Derke" look alike; a LOUD killer can't kill LOUD's Darker
    assert match_pair("LOUD luk xo ===", "VIT Darke", ROSTER) == ("luk xo", "Derke")
    assert match_pair("VIT Jamppi", "LOUD Darker", ROSTER) == ("Jamppi", "Darker")


def test_victim_without_readable_killer():
    assert match_pair("####", "LOUD Erde", ROSTER) == (None, "Erde")


def test_similarity_bounds():
    assert similarity("VIT Jamppi", "Jamppi") == 1.0
    assert similarity("", "Jamppi") == 0.0


def _kill(rnd, victim, side):
    return {"round": rnd, "victim": victim, "victim_side": side, "killer": None, "time_s": 0}


def test_merges_a_round_split_by_a_pause():
    # round 1 was split in two by a pause: no shared victims, no wipe in the first piece
    kills = [_kill(0, "Derke", "left"), _kill(0, "Erde", "right"),
             _kill(1, "Jamppi", "left"), _kill(1, "tkzin", "right"),
             _kill(2, "DaviH", "right"),
             _kill(3, "Derke", "left"), _kill(3, "Erde", "right")]
    res = {"rounds": 4, "kills": kills, "spans": [[0, 60], [100, 130], [135, 160], [200, 260]]}
    from pickpov.killfeed import merge_split_rounds
    assert merge_split_rounds(res, 3) == 1
    assert res["rounds"] == 3
    assert [k["round"] for k in kills] == [0, 0, 1, 1, 1, 2, 2]   # rounds 1 and 2 joined (5 s gap)


def test_never_merges_rounds_that_share_a_victim():
    kills = [_kill(0, "Derke", "left"), _kill(1, "Derke", "left")]
    res = {"rounds": 2, "kills": kills, "spans": [[0, 60], [62, 120]]}
    from pickpov.killfeed import merge_split_rounds
    assert merge_split_rounds(res, 1) == 0


def test_splits_two_rounds_that_ran_together():
    from pickpov.killfeed import split_joined_rounds
    k = lambda t, v, side: {"round": 0, "time_s": t, "victim": v, "victim_side": side, "killer": None}
    # Derke dies at 10 s and again at 120 s: two rounds, buy phase is the 30..100 s gap
    kills = [k(10, "Derke", "left"), k(20, "Erde", "right"), k(30, "tkzin", "right"),
             k(100, "Jamppi", "left"), k(120, "Derke", "left")]
    res = {"rounds": 1, "kills": kills, "spans": [[0, 130]]}
    assert split_joined_rounds(res, 2) == 1
    assert [x["round"] for x in kills] == [0, 0, 0, 1, 1]


def test_drops_a_misread_second_death():
    from pickpov.killfeed import drop_duplicate_deaths
    kills = [{"round": 2, "victim": "Darker", "killer": None, "seen": 1, "time_s": 233},
             {"round": 2, "victim": "Darker", "killer": "PROFEK", "seen": 5, "time_s": 271},
             {"round": 3, "victim": "Darker", "killer": "Derke", "seen": 3, "time_s": 400}]
    res = {"kills": kills}
    assert drop_duplicate_deaths(res) == 1
    assert [(k["round"], k["killer"]) for k in res["kills"]] == [(2, "PROFEK"), (3, "Derke")]


PRX_T1 = {"Meteor": "left", "iZu": "left", "stax": "left", "Munchkin": "left", "BuZz": "left",
          "fOrsakeN": "right", "something": "right", "d4v41": "right", "Jinggg": "right", "invy": "right"}


def test_lookalike_characters_in_names():
    # real OCR readings of "d4v41" and "Meteor" from the T1 vs Paper Rex VOD
    assert match_pair("4) PRX dével WP tS", "TV Zu a -", PRX_T1) == ("d4v41", "iZu")
    assert match_pair("~~) PRX d4val DPE ZS", "71 Mateor et", PRX_T1) == ("d4v41", "Meteor")


def test_exact_match_beats_lookalike():
    assert match_pair("VIT Jamppi", "LOUD Darker", ROSTER) == ("Jamppi", "Darker")
    assert match_pair("LOUD luk xo", "VIT Derke", ROSTER) == ("luk xo", "Derke")
