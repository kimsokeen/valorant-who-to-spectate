"""Download every agent's minimap and kill feed icons from valorant-api.com.

    python scripts/get_agent_icons.py

Saves configs/agents/<Agent>/minimap.png and killfeed.png. These are Riot's
art, so they stay out of git (see .gitignore); this script recreates them.
valorant-api.com is a fan-run service, not affiliated with Riot Games.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
FILES = {"minimapPortrait": "minimap.png", "killfeedPortrait": "killfeed.png",
         "displayIconSmall": "icon.png"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "PickPOV/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main() -> None:
    agents = json.loads(fetch(API))["data"]
    out = ROOT / "configs" / "agents"
    for a in sorted(agents, key=lambda a: a["displayName"]):
        folder = out / a["displayName"].replace("/", "-")
        folder.mkdir(parents=True, exist_ok=True)
        got = []
        for field, fname in FILES.items():
            if a.get(field):
                (folder / fname).write_bytes(fetch(a[field]))
                got.append(fname)
        print(f"{a['displayName']:10s} {', '.join(got)}")
    print(f"\n{len(agents)} agents saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
