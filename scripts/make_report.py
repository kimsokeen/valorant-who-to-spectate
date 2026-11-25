"""Draw the charts for RESULTS.md from the saved results.

    python scripts/evaluate_pickers.py --lead 0 2.5    # -> data/results/pickers.csv
    python scripts/train_lgbm.py                       # -> data/results/lgbm*.{txt,csv}
    python scripts/make_report.py                      # -> docs/figures/*.png

Nothing is retrained here: it only reads data/results/ and prints the numbers
it drew, so they can be checked against RESULTS.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
OUT = ROOT / "docs" / "figures"
LEAD = 2.5
LEVEL = 0.5          # per-map gaps within half a point count as level

# colours: PickPOV blue, broadcast orange, everything else grey
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8d8c86", "#e4e3df", "#fcfcfb"
MODEL, BROADCAST, BOTH, OTHER = "#2a78d6", "#eb6834", "#6250d6", "#c9c8c2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150,
})

RULE_NAMES = {"nearest_enemy": "Nearest enemy", "most_contested": "Most enemies within 60 px",
              "nearest_walking": "Nearest enemy by walking", "most_visible": "Most enemies in sight",
              "contested_walking": "Most enemies within a 90 px walk"}


def tidy(ax, xmax):
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def title(fig, main, sub):
    fig.text(0.01, 0.97, main, fontsize=13, weight="bold", va="top")
    fig.text(0.01, 0.97 - 0.075 * 6 / fig.get_figheight(), sub, fontsize=9.5, color=INK2, va="top")


def lgbm_summary():
    """Numbers from data/results/lgbm.txt (the model rows, the head-to-head)."""
    text = (RES / "lgbm.txt").read_text(encoding="utf-8")
    models = {m[0].strip(): float(m[1]) for m in re.findall(r"^\s{2}(\S.*?)\s+caught killer\s+([\d.]+)%", text, re.M)}
    h = re.search(r"both right (\d+), only the model (\d+), only the broadcast (\d+), neither (\d+)", text)
    p = re.search(r"disagree: p = ([\d.]+)", text)
    return models, tuple(int(x) for x in h.groups()), float(p.group(1))


def pooled(df, col):
    return 100 * (df[col] * df.kills_scored).sum() / df.kills_scored.sum()


def chart_ladder(pk, models):
    d = pk[pk.lead_s == LEAD]
    rules = d[d.picker != "broadcast"]
    rows = [("Random pick", pooled(rules[rules.picker == "nearest_enemy"], "chance_killer"), OTHER)]
    for key, name in RULE_NAMES.items():
        rows.append((f"Rule: {name}", pooled(d[d.picker == key], "caught_killer"), OTHER))
    rows.sort(key=lambda r: r[1])
    rows += [("Linear model", models["linear model"], OTHER),
             ("PickPOV (LightGBM, fight aware)", models["LightGBM, fight aware"], MODEL),
             ("Real VCT broadcast", pooled(d[d.picker == "broadcast"], "caught_killer"), BROADCAST)]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.38, right=0.95, top=0.80, bottom=0.10)
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for i, (_, v, c) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9.5,
                color=INK if c != OTHER else INK2, weight="bold" if c != OTHER else "normal")
    ax.set_yticks(list(y), [r[0] for r in rows])
    tidy(ax, 32)
    title(fig, "PickPOV is almost level with the real observers",
          "Camera already on the killer 2.5 s before the kill (2,544 kills; broadcast: the 2,259 it could be read on)")
    fig.savefig(OUT / "pickers.png")
    plt.close(fig)
    return rows


def chart_head_to_head(h, p):
    both, mo, bo, nei = h
    n = sum(h)
    parts = [("Both right", both, BOTH), ("Only PickPOV", mo, MODEL), ("Only the broadcast", bo, BROADCAST),
             ("Neither", nei, OTHER)]
    fig, ax = plt.subplots(figsize=(8, 1.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.56, bottom=0.02)
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.text(left + v / 2, 0, f"{v:,}", ha="center", va="center", fontsize=10,
                color="white" if c != OTHER else INK, weight="bold")
        ax.text(left + v / 2, -0.52, name, ha="center", va="top", fontsize=9, color=INK2)
        left += v
    ax.set_xlim(0, n)
    ax.set_ylim(-0.9, 0.4)
    ax.axis("off")
    title(fig, "They catch different kills",
          f"Same {n:,} kills, camera on the killer 2.5 s before. On the {mo + bo} where they disagree, "
          f"a sign test gives p = {p:.2f}")
    fig.savefig(OUT / "head_to_head.png")
    plt.close(fig)


def chart_per_map(pk, pm):
    model = pm[pm.picker == "LightGBM, fight aware"].set_index("map").caught_killer * 100
    b = pk[(pk.lead_s == LEAD) & (pk.picker == "broadcast")].set_index("map").caught_killer * 100
    d = pd.DataFrame({"model": model, "broadcast": b}).dropna()
    d["gap"] = d.model - d.broadcast
    d = d.sort_values("gap")

    caps = {"loud": "LOUD", "nrg": "NRG", "t1": "T1", "g2": "G2", "jd": "JD", "fut": "FUT", "xi": "Xi", "lai": "Lai"}

    def label(m):        # 2026-10-18_loud_vs_team-vitality_map1 -> LOUD vs Team Vitality, map 1 (Oct 18)
        date, rest = m.split("_", 1)
        teams, n = rest.rsplit("_map", 1)
        a, b_ = teams.split("_vs_")
        nice = lambda t: " ".join(caps.get(w, w.capitalize()) for w in t.split("-"))
        return f"{nice(a)} vs {nice(b_)}, map {n} ({pd.Timestamp(date):%b} {pd.Timestamp(date).day})"

    fig, ax = plt.subplots(figsize=(8, 6.8))
    fig.subplots_adjust(left=0.43, right=0.96, top=0.80, bottom=0.07)
    y = range(len(d))
    ax.hlines(list(y), d[["model", "broadcast"]].min(1), d[["model", "broadcast"]].max(1), color=GRID, linewidth=3)
    ax.scatter(d.broadcast, y, s=46, color=BROADCAST, zorder=3, edgecolor=SURFACE, linewidth=1.5,
               label="Real VCT broadcast")
    ax.scatter(d.model, y, s=46, color=MODEL, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="PickPOV")
    ax.set_yticks(list(y), [label(m) for m in d.index], fontsize=9)
    tidy(ax, 45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    wins = int((d.gap > LEVEL).sum())
    level = int((d.gap.abs() <= LEVEL).sum())
    title(fig, f"PickPOV is ahead on {wins} of {len(d)} maps, level on {level}",
          "Camera on the killer 2.5 s before each kill, per map; PickPOV tested on maps it never saw")
    fig.savefig(OUT / "per_map.png")
    plt.close(fig)
    return d


def chart_importance(imp):
    names = {"walk_nearest_enemy": "Walking distance to nearest enemy", "x": "Map position x",
             "y": "Map position y", "round_time": "Round time", "speed": "Speed",
             "straight_nearest_enemy": "Straight-line distance to nearest enemy",
             "enemies_walk_90": "Enemies within a 90 px walk", "closing_speed": "Moving toward an enemy",
             "enemies_alive": "Enemies alive", "mates_walk_90": "Teammates within a 90 px walk",
             "visible_nearest_enemy": "Nearest enemy in sight", "mates_alive": "Teammates alive"}
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(left=0.40, right=0.95, top=0.80, bottom=0.10)
    y = range(len(top))
    ax.barh(list(y), top.gain_share * 100, color=MODEL, height=0.62)
    for i, v in enumerate(top.gain_share * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(list(y), [names.get(f, f) for f in top.feature])
    tidy(ax, 27)
    title(fig, "Distance to the enemy and where you stand matter most",
          "Share of the final model's total gain, top 10 of 24 features")
    fig.savefig(OUT / "importance.png")
    plt.close(fig)
    return top


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pk = pd.read_csv(RES / "pickers.csv")
    pm = pd.read_csv(RES / "lgbm_per_map.csv")
    imp = pd.read_csv(RES / "lgbm_importance.csv")
    models, h, p = lgbm_summary()

    print("caught the killer 2.5 s before:")
    for name, v, _ in chart_ladder(pk, models):
        print(f"  {name:45s} {v:5.1f}%")
    b0 = pk[(pk.lead_s == 0) & (pk.picker == "broadcast")]
    print(f"  broadcast at the moment of the kill          {pooled(b0, 'caught_killer'):5.1f}%")
    chart_head_to_head(h, p)
    print(f"head-to-head: both {h[0]}, only PickPOV {h[1]}, only broadcast {h[2]}, neither {h[3]}, p = {p}")
    d = chart_per_map(pk, pm)
    print(f"per map: PickPOV ahead on {(d.gap > LEVEL).sum()}, level on {(d.gap.abs() <= LEVEL).sum()}, "
          f"behind on {(d.gap < -LEVEL).sum()} of {len(d)}, "
          f"gap from {d.gap.min():+.1f} to {d.gap.max():+.1f} points")
    chart_importance(imp)
    print(f"charts saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
