"""PickPOV demo: who should the camera follow? Kill by kill on Summit.

    pip install gradio
    python app.py            # opens http://127.0.0.1:7860

Reads only data/demo/moments.csv and configs/maps/summit_minimap.png
(built by scripts/build_demo_data.py), so it runs without any video.
Every pick shown comes from a model that never saw that map.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
MOMENTS = pd.read_csv(ROOT / "data" / "demo" / "moments.csv", keep_default_na=False)
BACKGROUND = Image.open(ROOT / "configs" / "maps" / "summit_minimap.png").convert("RGB")
SCALE = 2
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 15)
FONT_SMALL = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 13)

SIDE = {"left": (27, 175, 160), "right": (227, 73, 72)}          # teal and red, like the broadcast minimap
PICKPOV, BROADCAST, KILL = (42, 120, 214), (235, 104, 52), (255, 255, 255)
CAPS = {"loud": "LOUD", "nrg": "NRG", "t1": "T1", "g2": "G2", "jd": "JD", "fut": "FUT"}
FILTERS = ["All kills", "PickPOV right", "Broadcast right", "Only PickPOV right", "Only broadcast right"]


def map_label(m: str) -> str:
    date, rest = m.split("_", 1)
    teams, n = rest.rsplit("_map", 1)
    a, b = (" ".join(CAPS.get(w, w.capitalize()) for w in t.split("-")) for t in teams.split("_vs_"))
    d = pd.Timestamp(date)
    return f"{a} vs {b} · map {n} · {d:%b} {d.day}"


def build_kills() -> pd.DataFrame:
    """One row per kill: PickPOV's pick (top score), the broadcast's, and who was right."""
    top = MOMENTS.sort_values("score", ascending=False).groupby(["map", "kill_id"], sort=False).first()
    k = top.reset_index()[["map", "kill_id", "round", "kill_time_s", "killer", "victim", "broadcast", "player"]]
    k = k.rename(columns={"player": "pick"}).sort_values(["map", "kill_id"]).reset_index(drop=True)
    k["pick_right"] = k.pick == k.killer
    k["broadcast_known"] = k.broadcast != ""
    k["broadcast_right"] = k.broadcast == k.killer
    return k


KILLS = build_kills()
MAPS = {map_label(m): m for m in sorted(KILLS["map"].unique(), key=lambda m: m.split("_")[0])}


def filtered(map_key: str, which: str) -> pd.DataFrame:
    k = KILLS[KILLS["map"] == MAPS[map_key]]
    if which != "All kills":
        k = k[k.broadcast_known]
    return {
        "All kills": k,
        "PickPOV right": k[k.pick_right],
        "Broadcast right": k[k.broadcast_right],
        "Only PickPOV right": k[k.pick_right & ~k.broadcast_right],
        "Only broadcast right": k[k.broadcast_right & ~k.pick_right],
    }[which]


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def draw(rows: pd.DataFrame, kill: pd.Series) -> Image.Image:
    img = BACKGROUND.resize((BACKGROUND.width * SCALE, BACKGROUND.height * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    taken = [(r.x * SCALE - 10, r.y * SCALE - 10, r.x * SCALE + 10, r.y * SCALE + 10) for _, r in rows.iterrows()]
    for _, r in rows.iterrows():
        x, y = r.x * SCALE, r.y * SCALE
        if r.player == kill.broadcast:
            d.ellipse((x - 21, y - 21, x + 21, y + 21), outline=BROADCAST, width=5)
        if r.player == kill.pick:
            d.ellipse((x - 15, y - 15, x + 15, y + 15), outline=PICKPOV, width=5)
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=SIDE[r.side], outline=(20, 20, 20), width=2)
        if r.player == kill.killer:
            d.line((x - 5, y, x + 5, y), fill=KILL, width=2)
            d.line((x, y - 5, x, y + 5), fill=KILL, width=2)
        if r.player == kill.victim:
            d.line((x - 4, y - 4, x + 4, y + 4), fill=KILL, width=2)
            d.line((x - 4, y + 4, x + 4, y - 4), fill=KILL, width=2)
        tag = " (killer)" if r.player == kill.killer else " (victim)" if r.player == kill.victim else ""
        label = f"{r.player}{tag}"
        w = d.textlength(label, font=FONT)
        # try right, left, above, below, then further out, and take the first spot no other label or dot covers
        spots = [(x + 24, y - 9), (x - 24 - w, y - 9), (x - w / 2, y - 34), (x - w / 2, y + 16),
                 (x + 24, y - 28), (x + 24, y + 10), (x - 24 - w, y - 28), (x - 24 - w, y + 10),
                 (x - w / 2, y - 52), (x - w / 2, y + 34)]
        free = lambda b: all(b[2] < t[0] or b[0] > t[2] or b[3] < t[1] or b[1] > t[3] for t in taken)
        tx, ty = next((s for s in spots if free((s[0], s[1], s[0] + w, s[1] + 18))), spots[0])
        taken.append((tx, ty, tx + w, ty + 18))
        d.text((tx, ty), label, font=FONT, fill=(255, 255, 255), stroke_width=3, stroke_fill=(20, 20, 20))
    return img


def verdict(right: bool, known: bool = True) -> str:
    return "not readable" if not known else ("✅ on the killer" if right else "❌ not the killer")


def show(map_key: str, which: str, pos: int):
    ks = filtered(map_key, which)
    all_k = KILLS[KILLS["map"] == MAPS[map_key]]
    shared = all_k[all_k.broadcast_known]
    summary = (f"**This map:** PickPOV caught the killer 2.5 s early on **{all_k.pick_right.mean():.0%}** "
               f"of {len(all_k)} kills. On the {len(shared)} kills where the broadcast's label was readable: "
               f"PickPOV **{shared.pick_right.mean():.0%}**, real broadcast **{shared.broadcast_right.mean():.0%}**.")
    if ks.empty:
        return None, "No kills match this filter on this map.", None, summary, gr.update(maximum=1, value=1)
    pos = max(1, min(int(pos), len(ks)))
    kill = ks.iloc[pos - 1]
    rows = MOMENTS[(MOMENTS["map"] == kill["map"]) & (MOMENTS.kill_id == kill.kill_id)]
    rows = rows.sort_values("score", ascending=False)
    text = (f"### Round {kill['round'] + 1}, {mmss(kill.kill_time_s)} into the VOD: "
            f"**{kill.killer}** kills **{kill.victim}**\n\n"
            f"2.5 s before the kill:\n\n"
            f"- 🔵 **PickPOV** watches **{kill.pick}**: {verdict(kill.pick_right)}\n"
            f"- 🟠 **Real broadcast** shows **{kill.broadcast or '?'}**: "
            f"{verdict(kill.broadcast_right, kill.broadcast_known)}\n\n"
            f"Kill {pos} of {len(ks)} · {which}")
    table = pd.DataFrame({
        "Rank": range(1, len(rows) + 1),
        "Player": rows.player.values,
        "Agent": rows.agent.values,
        "PickPOV score": rows.score.round(2).values,
        "Role in this kill": ["killer" if p == kill.killer else "victim" if p == kill.victim else ""
                              for p in rows.player],
    })
    return draw(rows, kill), text, table, summary, gr.update(maximum=len(ks), value=pos)


def step(map_key, which, pos, delta):
    n = len(filtered(map_key, which))
    return max(1, min(int(pos) + delta, max(n, 1)))


INTRO = """# PickPOV: who should the camera follow?
An AI observer for Valorant. Two and a half seconds before each kill on Summit at VCT Champions 2026,
PickPOV picks one player to watch, using only the minimap. Overall it is on the future killer **26.6%** of the time,
against **27.7%** for the real broadcast on the same kills (not a significant difference, p = 0.40).
Every pick here comes from a model that **never saw that map**.
"""

LEGEND = """**On the map:** 🔵 blue ring = PickPOV's pick · 🟠 orange ring = what the broadcast showed ·
➕ = killer · ✖ = victim · teal and red = the two teams. Only players found on the minimap are drawn."""

with gr.Blocks(title="PickPOV demo") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        map_dd = gr.Dropdown(list(MAPS), value=list(MAPS)[-1], label="Map")
        which = gr.Radio(FILTERS, value="All kills", label="Show")
    summary = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=3):
            image = gr.Image(type="pil", label="Minimap 2.5 s before the kill", show_label=True, height=620)
            gr.Markdown(LEGEND)
        with gr.Column(scale=2):
            with gr.Row():
                prev_b = gr.Button("◀ Previous kill")
                next_b = gr.Button("Next kill ▶")
            pos = gr.Slider(1, 200, value=1, step=1, label="Kill")
            text = gr.Markdown()
            table = gr.Dataframe(label="PickPOV's ranking of the players it could see", interactive=False)

    outs = [image, text, table, summary, pos]
    map_dd.change(lambda m, w: show(m, w, 1), [map_dd, which], outs)
    which.change(lambda m, w: show(m, w, 1), [map_dd, which], outs)
    pos.release(show, [map_dd, which, pos], outs)
    prev_b.click(lambda m, w, p: show(m, w, step(m, w, p, -1)), [map_dd, which, pos], outs)
    next_b.click(lambda m, w, p: show(m, w, step(m, w, p, +1)), [map_dd, which, pos], outs)
    demo.load(lambda m, w: show(m, w, 1), [map_dd, which], outs)

if __name__ == "__main__":
    demo.launch()
