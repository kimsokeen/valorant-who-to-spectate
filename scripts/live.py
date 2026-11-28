"""PickPOV live: watch a VCT broadcast on screen and suggest who to spectate.

    python scripts/live.py                       # capture the main screen
    python scripts/live.py --monitor 2           # another screen
    python scripts/live.py --region 0 0 1920 1080   # part of a screen (left, top, width, height)
    python scripts/live.py --replay crops/<map> --speed 4 --log data/live/<map>.csv   # test on saved crops

Play the broadcast (YouTube, Twitch) fullscreen or in a 16:9 window and give
its position with --region. Summit only, VCT Champions 2026 overlay.
A small always-on-top window shows the minimap as PickPOV reads it and who
to watch; press q in it (or close it) to stop. The first few seconds of each round are spent finding
the 10 agents.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pickpov.live import LiveReader, Pick  # noqa: E402

COLOUR_BGR = {"teal": (160, 175, 27), "red": (72, 73, 227)}
PICK_BGR = (214, 120, 42)


def screen_frames(monitor: int, region, every_s: float):
    import mss

    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        mon = sct.monitors[monitor]
        if region:
            mon = {"left": mon["left"] + region[0], "top": mon["top"] + region[1],
                   "width": region[2], "height": region[3]}
        t0 = time.perf_counter()
        while True:
            start = time.perf_counter()
            img = np.asarray(sct.grab(mon))[..., :3]
            yield start - t0, np.ascontiguousarray(img)
            time.sleep(max(0.0, every_s - (time.perf_counter() - start)))


def replay_crops(folder: Path, every: int, speed: float):
    """Saved crops (minimap and hud_top 4/s, player cards 1/s) as if they were arriving live."""
    rows = list(csv.DictReader(open(folder / "index.csv")))[::every]
    t_start, wall = float(rows[0]["time_s"]), time.perf_counter()
    for r in rows:
        t = float(r["time_s"])
        if speed > 0:
            time.sleep(max(0.0, (t - t_start) / speed - (time.perf_counter() - wall)))
        s = int(r["sample"])
        mini = cv2.imread(str(folder / "minimap" / f"{s:06d}.jpg"))
        hud = cv2.imread(str(folder / "hud_top" / f"{s:06d}.jpg"))
        paths = {side: folder / f"team_{side}" / f"{s:06d}.jpg" for side in ("left", "right")}
        team = {side: cv2.imread(str(p)) if p.exists() else None for side, p in paths.items()}
        if mini is not None and hud is not None:
            yield t, mini, hud, (team if all(v is not None for v in team.values()) else None)


def label(pick: Pick, p: str) -> str:
    """'7' when the cards gave slot numbers, else the agent."""
    return pick.numbers.get(p) or p.split(" (")[0]


def draw(mini: np.ndarray, pick: Pick, scale: int = 2) -> np.ndarray:
    img = cv2.resize(mini, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for p, (x, y, colour, held) in pick.positions.items():
        c = (int(x * scale), int(y * scale))
        if p == pick.watch:
            cv2.circle(img, c, 22, PICK_BGR, 4, cv2.LINE_AA)
        cv2.circle(img, c, 13, COLOUR_BGR[colour], -1 if not held else 2, cv2.LINE_AA)
        text = label(pick, p)
        if len(text) == 1:          # a number: centred in the dot
            (w, h), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.putText(img, text, (c[0] - w // 2, c[1] + h // 2), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            for col, th in (((20, 20, 20), 3), ((255, 255, 255), 1)):
                cv2.putText(img, text, (c[0] + 16, c[1] + 5), font, 0.5, col, th, cv2.LINE_AA)
    bar = np.full((90, img.shape[1], 3), 30, np.uint8)
    if pick.watch:
        n = pick.numbers.get(pick.watch)
        agent, colour = pick.watch.split(" (")[0], pick.watch.split(" (")[1].rstrip(")")
        big = f"WATCH  {n}" if n else f"WATCH  {agent}"
        cv2.putText(bar, big, (14, 52), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(big, font, 1.5, 3)
        cv2.putText(bar, f"{agent}, {colour}" if n else colour, (28 + w, 50), font, 0.7,
                    COLOUR_BGR[colour], 2, cv2.LINE_AA)
        nxt = [label(pick, p) for p, _ in pick.ranking if p != pick.watch][:2]
        cv2.putText(bar, "next: " + ", ".join(nxt), (14, 80), font, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, pick.status, (14, 52), font, 0.9, (190, 190, 190), 2, cv2.LINE_AA)
    info = f"round {pick.round + 1}  {pick.round_time:5.1f}s  {pick.ms:4.0f} ms"
    cv2.putText(bar, info, (img.shape[1] - 250, 80), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


class Window:
    """A small always-on-top window (Tkinter: this OpenCV build has no window support)."""

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("PickPOV live")
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.open = True
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("q", lambda e: self.close())

    def show(self, bgr: np.ndarray) -> bool:
        from PIL import Image, ImageTk
        if not self.open:
            return False
        self.photo = ImageTk.PhotoImage(Image.fromarray(bgr[..., ::-1]))
        self.label.configure(image=self.photo)
        self.root.update()
        return self.open

    def close(self):
        self.open = False
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monitor", type=int, default=1, help="screen number (1 = main)")
    ap.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "W", "H"))
    ap.add_argument("--every", type=float, default=0.5, help="seconds between screen captures")
    ap.add_argument("--replay", type=Path, help="a crops/<map> folder instead of the screen")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed; 0 = as fast as possible")
    ap.add_argument("--start", type=float, default=0.0, help="replay: skip to this time (s)")
    ap.add_argument("--model", type=Path, help="another LightGBM model (default models/lgbm_picker.txt)")
    ap.add_argument("--log", type=Path, help="write every frame's positions and ranking to this CSV")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    reader = LiveReader(ROOT, model=args.model)
    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "w", newline="", encoding="utf-8", buffering=1)    # line-buffered
        log = csv.writer(log_file)
        log.writerow(["time_s", "round", "round_time", "status", "player", "x", "y", "held", "score", "rank", "watch", "number", "ms"])
    window = None if args.no_window else Window()

    if args.replay:
        source = ((t, *reader_input) for t, *reader_input in replay_crops(args.replay, 2, args.speed)
                  if t >= args.start)
        step = lambda item: (item[1], reader.process(item[0], item[1], item[2], item[3]))
    else:
        source = screen_frames(args.monitor, args.region, args.every)
        step = lambda item: (reader.crops(item[1])[0], reader.process_frame(item[0], item[1]))

    n, slow = 0, []
    try:
        for item in source:
            mini, pick = step(item)
            n += 1
            slow.append(pick.ms)
            if log:
                score = dict(pick.ranking)
                rank = {p: i + 1 for i, (p, _) in enumerate(pick.ranking)}
                for p, (x, y, _, held) in pick.positions.items():
                    log.writerow([f"{pick.t:.2f}", pick.round, f"{pick.round_time:.2f}", pick.status, p,
                                  x, y, int(held), round(score.get(p, 0), 4), rank.get(p, ""),
                                  int(p == pick.watch), pick.numbers.get(p, ""), round(pick.ms)])
                if not pick.positions:
                    log.writerow([f"{pick.t:.2f}", pick.round, f"{pick.round_time:.2f}", pick.status,
                                  "", "", "", "", "", "", "", "", round(pick.ms)])
            if window:
                if not window.show(draw(mini, pick)):
                    break
            elif n % 200 == 0:
                print(f"  t={pick.t:7.1f}s round {pick.round + 1} {pick.status:15s} "
                      f"median {np.median(slow[-200:]):.0f} ms/frame", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        if window:
            window.close()
    if slow:
        print(f"{n} frames, median {np.median(slow):.0f} ms, 95th percentile {np.percentile(slow, 95):.0f} ms per frame")


if __name__ == "__main__":
    main()
