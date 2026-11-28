# PickPOV

An AI observer for Valorant. PickPOV reads the minimap and HUD and suggests which player to spectate, aiming to be on the next killer 2–3 seconds before the kill.

- **End goal:** live, for any Valorant game: watch the screen, read the map, suggest who to watch.
- **v1 (now):** offline on VCT Champions 2026 VODs, Summit only, to prove the picker works.
- **Design rule:** the picker works on map positions, not screen pixels, so only the screen reader changes between sources (broadcast, replays, live).

**Results:** 2.5 s before a kill, PickPOV is on the killer 26.6% of the time against 27.7% for the real VCT broadcast on the same kills, a gap that isn't statistically significant (p = 0.40). Full report: [RESULTS.md](RESULTS.md).

**Demo:** `python app.py`, then open http://127.0.0.1:7860. Pick a map and step through its kills: the minimap 2.5 s before each kill, PickPOV's pick, what the broadcast showed, and who got the kill. It reads only `data/demo/moments.csv` and a clean Summit minimap, both made by `scripts/build_demo_data.py`. The minimap is cut from broadcast crops, so it isn't in git: run that script once on a machine that has `crops/`.

Project plan: https://claude.ai/artifact/ViEjjaMNen2KCbezofSDi2

## Dataset

Summit was the most played map at VALORANT Champions 2026 Shanghai: 18 of 86 maps (389 rounds). All 18 are listed in `data/champions2026_summit_maps.csv` with match links; fill in `vod_url` as you find each VOD on the official VCT YouTube channel.

| Map | Times played |
| --- | --- |
| Summit | 18 |
| Lotus | 15 |
| Ascent | 13 |
| Sunset | 12 |
| Haven | 10 |
| Split | 10 |
| Abyss | 8 |

Counted from the full results on [THESPIKE.GG](https://www.thespike.gg/events/results/valorant-champions-tour-2026-champions-shanghai/4233).

## Layout

```
PickPOV/
├── configs/broadcast_regions.yaml   screen regions of the broadcast overlay (calibrate in week 1)
├── data/                            match lists and, later, extracted positions and kills
├── notebooks/01_get_data.ipynb      Colab: get a VOD, sample frames, calibrate regions
├── scripts/download_vod.py          download a VOD or one section of it (yt-dlp)
├── scripts/download_all.py          download every VOD listed in the maps CSV
├── scripts/sample_frames.py         save a few full frames as JPGs (for calibration)
├── scripts/extract_crops.py         crop the screen regions from one video
├── scripts/process_all.py           in batches of 5: download, crop, then ask about deleting
├── src/pickpov/video.py             read frames at a fixed rate
├── src/pickpov/regions.py           crop and draw screen regions
├── src/pickpov/crops.py             save region crops, preview sheet, manifest
└── tests/                           run with: pytest
```

Not in git: `vods/`, `frames/` and `crops/` (too big).

## Getting started

On your PC:

```
pip install -r requirements.txt
pytest
python scripts/download_vod.py <youtube-url> --start 0:12:00 --end 0:55:00 --name 2026-10-18_loud_vs_team-vitality_map1
python scripts/sample_frames.py vods/2026-10-18_loud_vs_team-vitality_map1.mp4 --every 20 --max 40
```

On Colab: upload the `PickPOV` folder to `My Drive/PickPOV` and open `notebooks/01_get_data.ipynb`. If YouTube blocks the download on Colab, download on your PC and upload the mp4 to `My Drive/PickPOV/vods/`.

## Building the dataset: in batches

Full videos are ~2 GB each. `process_all.py` works in batches of 5: it downloads all 5 and crops only the gameplay from each with no questions, then asks about deleting each video back to back. Disk use peaks at about 10 GB.

| Region | Samples | Why |
| --- | --- | --- |
| minimap, hud_top | 4 per second | positions change fast |
| kill_feed, pov_name, round_timer, team_left, team_right | 1 per second | kill feed entries stay for seconds; alive and health only need a check per second |

That is about 17 MB per minute, roughly 0.9 GB for a 50-minute map instead of 2 GB of video.

Safeguards:
- It refuses to run until `calibrated: true` is set in `configs/broadcast_regions.yaml`. Crops from wrong boxes are useless once the video is gone.
- Every crop box gets an 8% margin so icons at the edges aren't cut off.
- Each map gets a `preview.jpg`; at the end of each batch you're asked about each video in turn (`--yes` skips the questions).
- Frames that aren't live gameplay (arena shots, player cams, replays) are skipped, detected from HUD text in `configs/markers/`.
- The video is kept if the crops cover less than 95% of it, or if anything fails.
- The Grand Final Summit map video is never deleted: it's the test case for the screen reader.
- Re-running skips maps that are already cropped. Any deleted map can be re-downloaded from its `vod_url`.

```
python scripts/sample_frames.py vods/<map>.mp4 --every 30 --max 20      # 1. frames to calibrate on
python scripts/extract_crops.py vods/<map>.mp4 --start 1200 --end 1260  # 2. 1-minute crop test, check preview.jpg
python scripts/process_all.py data/champions2026_summit_maps.csv        # 3. all maps, 5 at a time
```

## Reading the screen (week 2)

Tesseract OCR is needed: install it from https://github.com/UB-Mannheim/tesseract/wiki (keep the default folder).

```
python scripts/build_rosters.py     # 10 player names per map, from the player cards -> data/rosters/
python scripts/read_killfeed.py     # every kill: round, time, killer, victim -> data/kills/
```

The kill feed reader splits each row where the team colour changes, reads killer and victim separately, and matches them to the map's roster, so OCR slips ("Tuk xo") and cut-off tags still resolve. Killer and victim must be on opposite teams. Each kill is counted once per (round, victim), at the moment it first appears.

Rounds and their winners come from the HUD score: a team's score digits look identical until that team wins a round, so comparing the score area between gameplay stretches shows when a round ended and who won it, with no OCR. On 17 of 18 maps this gives exactly the real round count and final score (overtime included).

Names are also compared in a look-alike form (4 = a = é, 1 = l = i, 0 = o), so OCR readings like "dével" still match d4v41; an exact match always wins.

Checked against official stats: luk xo's 26 kills on Summit in the upper bracket final were read as 25.

## Player positions (minimap)

```
python scripts/get_agent_icons.py           # agent art from valorant-api.com -> configs/agents/ (not in git)
python scripts/read_agents.py --jobs 8      # each player's agent, from mirrored kill feed portraits
python scripts/read_positions.py --jobs 8   # every player's minimap position, 2 per second -> data/positions/
```

Each player is a ~14 px agent portrait in a ring of his team's colour. Knowing the map's 10 agents, the reader template-matches only those, keeps matches on the map drawing (the game scene shows through the minimap), and checks the ring: real icons have 30+ ring-coloured pixels, look-alikes ~0. Which team is teal swaps with sides, so it's decided per round. The ring colour also separates mirror picks.

Checked on three maps: 77-92% of alive players found per frame, with smooth tracks (median move 3-5 px per half second, under 2% jumps over 40 px). At kills, the victim is the killer's nearest enemy 65% of the time against 36-46% by chance. In fights icons blink out, so a missing player keeps his last position for up to 1.5 s, marked `held`.

## Walls, doors and the rule pickers (week 3)

`configs/maps/summit_walkable.png` is Summit's walkable floor, voted across all 18 maps plus where players actually stand (both sites); 99.7% of 460k detected positions land on it. `configs/maps/summit_doors.json` holds the A, mid and B doors: a closed door is drawn as a black bar, so each frame's door states are read from the minimap (closed 24%, 61% and 23% of the time). `pickpov.walls` gives line of sight and walking distance around walls and closed doors.

```
python scripts/read_pov.py --jobs 8          # who the real broadcast was showing, every second
python scripts/evaluate_pickers.py --lead 0 2.5
```

Over 2,544 kills, 2.5 s before each kill:

| Picker | Caught the killer | Caught the fight |
| --- | --- | --- |
| Random pick | 14.6% | 29.0% |
| Nearest enemy (straight line) | 20.8% | 42.6% |
| Most enemies within 60 px (straight line) | 21.8% | 46.7% |
| Most enemies within 90 px walking distance | **24.0%** | **50.9%** |
| Real broadcast | 27.7% | 55.2% |

At the moment of the kill, the real broadcast was on the killer 45.0% of the time.

Known limitation: boxes and small cover aren't in the floor map. They're faint on the minimap, and players often shoot over or around them, so marking them as solid walls would be wrong in a different way; the effect on line of sight is small. Height (stairs, ledges) is also lost on the flat minimap. Just before kills, a killer can see his victim 70% of the time but only 12% of his other enemies.

## Live mode

Play a VCT broadcast on Summit (YouTube or Twitch, live or VOD) fullscreen, or in a 16:9 window given with `--region`, and run:

```
python scripts/build_card_art.py     # once: agent portraits from the player cards -> configs/cards/
python scripts/live.py               # or: --monitor 2, --region LEFT TOP W H
```

A small always-on-top window shows the minimap as PickPOV reads it and who to watch, twice a second, as a player number: the left team's cards are 1–5 top to bottom, the right team's 6–9 and 0. A fullscreen video on a screen that isn't 16:9 (e.g. 2560×1600) is fine: the black bars are cut off before reading.

Picks are held: PickPOV stays on a player at least 1.5 s, then moves to whoever is top, or at once if the watched player dies (grey card, or gone from the minimap). On the two replayed maps below, that costs no accuracy (29.9% held vs 29.6% unheld, both maps together) and cuts switching from 38 to 25 a minute. The real broadcast switches about 11 times a minute; slower settings tried (longer holds, or switching only to a clearly better player) got down to 14 a minute but lost 2–4 points. Agents and team colours come from the player cards (the 14 px minimap icons can't tell look-alikes apart); then each player is tracked near his last minimap position, so a frame takes ~100–120 ms. Rounds come from the score digits, as offline.

Tested by replaying two maps' saved crops through the same code (`--replay crops/<map> --speed 0 --log data/live/<map>.csv`, then `scripts/check_live.py`), with a model trained without that map (`scripts/train_lgbm_without.py`):

| Map | All 10 agents right | Offline positions found live | Caught the killer, live | Same model, offline |
| --- | --- | --- | --- | --- |
| JD Gaming vs FUT Esports, map 2 | 21 of 23 rounds | 85% | 28.1% | 29.0% |
| LOUD vs Team Vitality, map 1 | 23 of 24 rounds | 86% | 31.1% | 29.2% |

Limits: Summit and the Champions 2026 overlay only. Card portraits exist for the 21 agents played on Summit in the dataset; a broadcast with another agent falls back to finding agents on the minimap, which mixes up look-alikes.

## Roadmap

1. **Data** — VODs, calibrated regions, skip replays and caster cams
2. **Read the screen** — minimap positions, alive players, kill feed, broadcast POV
3. **Rule baseline** — nearest-enemy picker, scored against the broadcast
4. **ML model** — LightGBM on per-player features
5. **Ship v1** — Gradio app, results report
6. **Live mode** — screen capture of a broadcast (done; Summit only)

Then: the in-game spectator client, and more maps.

VODs are downloaded for personal study only. Not affiliated with Riot Games.
