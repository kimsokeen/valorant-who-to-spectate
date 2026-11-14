"""Learn the gameplay markers (HUD text that only shows during live play).

Run once per broadcast overlay, on any VOD from that event:
    python scripts/build_markers.py vods/2026-10-18_loud_vs_team-vitality_map1.mp4

Saves configs/markers/<name>.png. Open them: each should be clean HUD text
("CURRENT:", "ROUND"). Then it prints how much of the video counts as gameplay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.gameplay import GameplayDetector, build_markers, frames_at  # noqa: E402
from pickpov.regions import Region  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--config", default=str(ROOT / "configs" / "broadcast_regions.yaml"))
    parser.add_argument("--check-every", type=float, default=10.0, help="seconds between checks in the report")
    args = parser.parse_args()

    config_path = Path(args.config)
    data = yaml.safe_load(config_path.read_text())
    regions = {n: Region(n, **{k: v for k, v in b.items() if k != "slides"})
               for n, b in data["gameplay"]["markers"].items()}
    markers = build_markers(args.video, regions)
    (config_path.parent / "markers").mkdir(exist_ok=True)
    for name, img in markers.items():
        cv2.imwrite(str(config_path.parent / "markers" / f"{name}.png"), img)
        print(f"saved configs/markers/{name}.png")

    det = GameplayDetector.load(config_path)
    cap = cv2.VideoCapture(args.video)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    times = np.arange(0, duration, args.check_every)
    scores = [(t, det.score(img)) for t, img in frames_at(args.video, times)]
    live = [t for t, s in scores if s >= det.threshold]
    print(f"\n{len(live)}/{len(scores)} checks are gameplay ({len(live) / len(scores):.0%})")
    hist = np.histogram([s for _, s in scores], bins=10, range=(0, 1))[0]
    print("score histogram (0.0 -> 1.0):", list(hist))

    # gameplay stretches
    spans, start, prev = [], None, None
    for t, s in scores:
        if s >= det.threshold:
            start = t if start is None else start
            prev = t
        elif start is not None:
            spans.append((start, prev)); start = None
    if start is not None:
        spans.append((start, prev))
    fmt = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"
    print(f"{len(spans)} gameplay stretches:", ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in spans))


if __name__ == "__main__":
    main()
