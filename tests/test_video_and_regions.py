"""Tests run on a tiny synthetic video, so no VOD is needed."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.regions import Region, draw_grid, draw_regions, load_regions  # noqa: E402
from pickpov.video import sample_frames, save_frames, video_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_video(tmp_path):
    """4 seconds at 30 fps; each frame's brightness encodes its frame number."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
    for i in range(120):
        writer.write(np.full((90, 160, 3), i * 2, dtype=np.uint8))
    writer.release()
    return path


def test_video_info(tiny_video):
    info = video_info(tiny_video)
    assert info["width"] == 160 and info["height"] == 90
    assert info["duration_s"] == pytest.approx(4.0, abs=0.1)


def test_sample_every_half_second(tiny_video):
    frames = list(sample_frames(tiny_video, every_s=0.5))
    assert [f.index for f in frames] == list(range(0, 120, 15))
    assert frames[1].time_s == pytest.approx(0.5)


def test_sample_window(tiny_video):
    frames = list(sample_frames(tiny_video, every_s=1.0, start_s=1.0, end_s=3.0))
    assert [round(f.time_s, 2) for f in frames] == [1.0, 2.0, 3.0]


def test_save_frames(tiny_video, tmp_path):
    saved = save_frames(tiny_video, tmp_path / "out", every_s=1.0, max_frames=2)
    assert len(saved) == 2 and all(p.exists() for p in saved)


def test_region_crop_scales_with_resolution():
    region = Region("minimap", x=0.5, y=0.5, w=0.25, h=0.5)
    assert region.crop(np.zeros((1080, 1920, 3), np.uint8)).shape == (540, 480, 3)
    assert region.crop(np.zeros((720, 1280, 3), np.uint8)).shape == (360, 320, 3)


def test_config_loads_and_draws():
    regions = load_regions(ROOT / "configs" / "broadcast_regions.yaml")
    assert {"minimap", "kill_feed", "hud_top", "pov_name"} <= set(regions)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert draw_regions(frame, regions).shape == frame.shape
    assert draw_grid(frame).shape == frame.shape
