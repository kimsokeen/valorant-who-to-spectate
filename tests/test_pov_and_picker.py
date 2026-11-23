"""Broadcast POV smoothing and the rule pickers (no files needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.picker import pick_most_contested, pick_nearest_enemy  # noqa: E402
from pickpov.pov import pov_at, remove_flicker  # noqa: E402
from pickpov.scoring import score_picks  # noqa: E402


def test_one_second_flicker_is_removed():
    pov = [(1, "Darker"), (2, "Darker"), (3, "Erde"), (4, "Darker"), (5, "tkzin"), (6, "tkzin")]
    assert [p for _, p in remove_flicker(pov)] == ["Darker", "Darker", "Darker", "Darker", "tkzin", "tkzin"]


def test_pov_fills_short_gaps_only():
    pov = [(10.0, "Erde"), (11.0, None), (20.0, None)]
    got = pov_at(pov, [10.5, 12.5, 14.0])
    assert got == {10.5: "Erde", 12.5: "Erde", 14.0: None}       # 4 s gap is too long


def test_pickers():
    frame = {"a": (0, 0, "left"), "b": (100, 0, "left"),
             "x": (110, 0, "right"), "y": (120, 0, "right"), "z": (400, 400, "right")}
    assert pick_nearest_enemy(frame) in ("b", "x")                # the closest pair
    assert pick_most_contested(frame) == "b"                      # two enemies within 60 px


def test_scoring_counts_killer_and_fight():
    frames = [(10.0, 0, {"a": (0, 0, "left"), "x": (10, 0, "right"), "y": (50, 0, "right")})]
    kills = [{"killer": "a", "victim": "x", "time_s": 12.5, "round": 0},
             {"killer": "y", "victim": "a", "time_s": 12.5, "round": 0}]
    s = score_picks(frames, kills, {10.0: "a"}, lead=2.5)
    assert s["kills_scored"] == 2 and s["caught_killer"] == 0.5 and s["caught_fight"] == 1.0
