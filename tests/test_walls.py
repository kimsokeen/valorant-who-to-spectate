"""Line of sight and walking distance on a tiny synthetic map with one door."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.walls import Summit  # noqa: E402


def _map(tmp_path):
    # two rooms (left, right) joined by a corridor at the top; a wall between
    # them with a door in it at y 100-110
    walk = np.zeros((200, 200), np.uint8)
    walk[20:180, 20:90] = 255           # left room
    walk[20:180, 110:180] = 255         # right room
    walk[20:40, 90:110] = 255           # top corridor
    walk[100:111, 90:110] = 255         # door opening (drawn open)
    cv2.imwrite(str(tmp_path / "walk.png"), walk)
    door = [[x, y] for x in range(90, 110) for y in range(100, 111)]
    (tmp_path / "doors.json").write_text(json.dumps({"doors": {"mid": door}}))
    return Summit(tmp_path / "walk.png", tmp_path / "doors.json")


def test_wall_blocks_sight_and_forces_a_detour(tmp_path):
    s = _map(tmp_path)
    a, b = (60, 150), (150, 150)                       # either side of the wall, below the door
    assert not s.line_of_sight(a, b)
    assert s.walking_distance(a, b, (False,)) > 1.2 * 90   # detour through the open door
    assert s.walking_distance(a, b, (True,)) > 2.0 * 90    # door closed: all the way round the top


def test_door_open_vs_closed(tmp_path):
    s = _map(tmp_path)
    a, b = (60, 105), (150, 105)                       # straight through the door
    assert s.line_of_sight(a, b, (False,))
    assert not s.line_of_sight(a, b, (True,))
    assert s.walking_distance(a, b, (True,)) > s.walking_distance(a, b, (False,)) + 50


def test_door_state_is_read_from_black_pixels(tmp_path):
    s = _map(tmp_path)
    img = np.full((200, 200, 3), 150, np.uint8)
    assert s.door_states(img) == (False,)
    img[100:111, 90:110] = 10                           # black bar = closed
    assert s.door_states(img) == (True,)
