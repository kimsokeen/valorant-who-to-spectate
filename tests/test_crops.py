"""Crop extraction on a tiny synthetic video."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pickpov.crops import extract_crops, with_margin  # noqa: E402
from pickpov.regions import Region  # noqa: E402


@pytest.fixture
def tiny_video(tmp_path):
    """4 seconds at 30 fps, 320x180."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 180))
    for i in range(120):
        writer.write(np.full((180, 320, 3), i * 2, dtype=np.uint8))
    writer.release()
    return path


def test_margin_grows_and_stays_on_screen():
    grown = with_margin(Region("m", 0.1, 0.1, 0.5, 0.5), 0.1)
    assert grown.x == pytest.approx(0.05) and grown.w == pytest.approx(0.6)
    edge = with_margin(Region("m", 0.0, 0.0, 1.0, 1.0), 0.2)
    assert (edge.x, edge.y, edge.w, edge.h) == (0.0, 0.0, 1.0, 1.0)


def test_extract_crops(tiny_video, tmp_path):
    regions = {
        "minimap": Region("minimap", 0.0, 0.0, 0.3, 0.4),
        "kill_feed": Region("kill_feed", 0.7, 0.1, 0.25, 0.25),
    }
    out = tmp_path / "crops"
    manifest = extract_crops(tiny_video, regions, out, every_s=0.5, slow_every=4)

    assert manifest["samples"] == 8                        # 4 s at 2 per second
    assert len(list((out / "minimap").glob("*.jpg"))) == 8
    assert len(list((out / "kill_feed").glob("*.jpg"))) == 2   # slow region: every 4th sample
    assert (out / "preview.jpg").exists() and (out / "index.csv").exists()
    assert json.loads((out / "manifest.json").read_text())["slow_every"] == 4


def test_map_names_match_download_all():
    from process_all import map_name
    row = {"date": "2026-10-18", "team_a": "LOUD", "team_b": "Team Vitality", "map_number": "1"}
    assert map_name(row) == "2026-10-18_loud_vs_team-vitality_map1"
