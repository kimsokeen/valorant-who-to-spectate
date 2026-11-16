"""Last-round winner from the final score (no images needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.rounds import finish_last_round  # noqa: E402


def test_last_round_goes_to_the_map_winner():
    info = {"rounds": 24, "winners": {r: "left" for r in range(11)} | {r: "right" for r in range(11, 23)},
            "left_wins": 11, "right_wins": 12}
    finish_last_round(info, (13, 11))
    assert info["winners"][23] == "right" and info["right_wins"] == 13


def test_unknown_when_counts_dont_fit():
    info = {"rounds": 20, "winners": {}, "left_wins": 6, "right_wins": 11}
    finish_last_round(info, (13, 7))
    assert 19 not in info["winners"]
