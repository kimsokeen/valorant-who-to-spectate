# PickPOV

An AI observer for Valorant. PickPOV reads the minimap and HUD and suggests which player to spectate, aiming to be on the next killer 2–3 seconds before the kill.

- **End goal:** live, for any Valorant game: watch the screen, read the map, suggest who to watch.
- **v1 (now):** offline on VCT Champions 2026 VODs, Summit only, to prove the picker works.
- **Design rule:** the picker works on map positions, not screen pixels, so only the screen reader changes between sources (broadcast, replays, live).

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

## Roadmap

1. **Data** — VODs, calibrated regions, skip replays and caster cams
2. **Read the screen** — minimap positions, alive players, kill feed, broadcast POV
3. **Rule baseline** — nearest-enemy picker, scored against the broadcast
4. **ML model** — LightGBM on per-player features
5. **Ship v1** — Gradio app, results report

Then: live mode via screen capture.

VODs are downloaded for personal study only. Not affiliated with Riot Games.
