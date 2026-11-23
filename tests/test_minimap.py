"""Minimap player finding on a synthetic minimap (no agent art needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.minimap import SIZES, find_players, ring_colours  # noqa: E402

TEAL, RED = (200, 180, 20), (40, 30, 230)          # BGR


def _icon(size):
    """A made-up 'agent portrait': a few distinct blobs, with a full mask."""
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (22, 24), 9, (60, 120, 230), -1)
    cv2.rectangle(img, (34, 30), (50, 52), (240, 240, 240), -1)
    cv2.circle(img, (40, 14), 6, (30, 200, 60), -1)
    return (cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
            np.full((size, size), 255, np.uint8))


def _minimap(icons):
    img = np.full((200, 200, 3), 130, np.uint8)       # grey "map floor"
    for (x, y), colour in icons:
        cv2.circle(img, (x, y), 10, colour, 3)         # team ring
        t, _ = _icon(14)
        img[y - 7:y + 7, x - 7:x + 7] = t
    return img


ART = {"Fake": [_icon(s) for s in SIZES]}
AREA = np.ones((200, 200), bool)


def test_ring_colour_is_measured():
    img = _minimap([((60, 60), TEAL)])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rc = ring_colours(hsv, 60, 60)
    assert rc["teal"] > 30 and rc["red"] == 0


def test_mirror_pick_is_split_by_ring_colour():
    img = _minimap([((60, 60), TEAL), ((140, 120), RED)])
    players = {"A": ("Fake", "left"), "B": ("Fake", "right")}
    found = find_players(img, AREA, ART, players, {"left": "teal", "right": "red"}, {"A", "B"})
    assert abs(found["A"][0] - 60) <= 2 and abs(found["A"][1] - 60) <= 2
    assert abs(found["B"][0] - 140) <= 2 and abs(found["B"][1] - 120) <= 2
    # after halftime the colours swap: the same picture means the opposite
    swapped = find_players(img, AREA, ART, players, {"left": "red", "right": "teal"}, {"A", "B"})
    assert abs(swapped["A"][0] - 140) <= 2 and abs(swapped["B"][0] - 60) <= 2


def test_dead_players_are_not_searched():
    img = _minimap([((60, 60), TEAL)])
    found = find_players(img, AREA, ART, {"A": ("Fake", "left")}, {"left": "teal", "right": "red"}, set())
    assert found == {}
