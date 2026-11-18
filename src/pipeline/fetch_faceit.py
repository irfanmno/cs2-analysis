"""Pull my FACEIT CS2 match history + per-match stats and cache them as raw JSON.

Usage:
    python src/pipeline/fetch_faceit.py            # fetch anything not cached yet
    python src/pipeline/fetch_faceit.py --refresh  # re-download everything

Output (gitignored):
    data/raw/faceit/player.json
    data/raw/faceit/history.json
    data/raw/faceit/matches/<match_id>.json   (match details: map, teams, demo_url)
    data/raw/faceit/stats/<match_id>.json     (per-player scoreboard stats)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://open.faceit.com/data/v4"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "faceit"
GAME = "cs2"
# CS2 history starts in 2023; FACEIT's history endpoint only looks back a short
# window by default, so pass an explicit start.
HISTORY_FROM = 1672531200  # 2023-01-01 UTC


class FaceitClient:
    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def get(self, path: str, **params) -> dict:
        for attempt in range(5):
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=30)
            if resp.status_code == 429:  # rate limited: back off and retry
                wait = 2 ** attempt
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                sys.exit("FACEIT rejected the API key (401). Check FACEIT_API_KEY in .env.")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Gave up on {path} after repeated rate limits")

    def player(self, nickname: str) -> dict:
        return self.get("/players", nickname=nickname, game=GAME)

    def history(self, player_id: str) -> list[dict]:
        items, offset, limit = [], 0, 100
        while True:
            page = self.get(
                f"/players/{player_id}/history",
                game=GAME, offset=offset, limit=limit, **{"from": HISTORY_FROM},
            )
            batch = page.get("items", [])
            items.extend(batch)
            if len(batch) < limit:
                return items
            offset += limit


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download cached matches")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("FACEIT_API_KEY", "").strip()
    nickname = os.getenv("FACEIT_NICKNAME", "MlgSly").strip()
    if not api_key:
        sys.exit("No FACEIT_API_KEY found. Copy .env.example to .env and paste your key in.")

    client = FaceitClient(api_key)

    player = client.player(nickname)
    save_json(RAW_DIR / "player.json", player)
    player_id = player["player_id"]
    cs2 = player.get("games", {}).get(GAME, {})
    print(f"{nickname}: level {cs2.get('skill_level')}, elo {cs2.get('faceit_elo')}")

    history = client.history(player_id)
    save_json(RAW_DIR / "history.json", history)
    print(f"Found {len(history)} CS2 matches in history")

    fetched = skipped = failed = 0
    for i, item in enumerate(history, 1):
        match_id = item["match_id"]
        match_path = RAW_DIR / "matches" / f"{match_id}.json"
        stats_path = RAW_DIR / "stats" / f"{match_id}.json"
        if not args.refresh and match_path.exists() and stats_path.exists():
            skipped += 1
            continue
        try:
            save_json(match_path, client.get(f"/matches/{match_id}"))
            save_json(stats_path, client.get(f"/matches/{match_id}/stats"))
            fetched += 1
        except requests.HTTPError as e:
            # e.g. cancelled matches have no stats
            print(f"  [{i}/{len(history)}] {match_id}: {e.response.status_code}, skipping")
            failed += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(history)}] ...")

    print(f"Done. fetched={fetched} cached={skipped} failed={failed} -> {RAW_DIR}")


if __name__ == "__main__":
    main()
