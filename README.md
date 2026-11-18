# cs2-analysis

> 🚧 Work in progress. The proper README (screenshots, findings, what I learned) comes once there's something to show.

I've played Counter-Strike since I was 15. I have 139 CS2 matches on FACEIT and a lot of them did not go well. This repo is me using data to figure out why I lose rounds, changing something, and checking honestly whether it helped.

## Plan

1. **Match history:** pull every FACEIT CS2 match and look at trends (maps, K/D, win rate over time).
2. **Demos:** parse recent demo files for round-level data (economy, utility, first kills, positions).
3. **Why do I lose rounds?** Build a round-win model and see what actually matters for *my* games.
4. **Where do I die?** Heatmaps by map and side.
5. **Ongoing A/B tests on myself:** change one thing about how I play, measure it from my demos, keep or drop it. Whatever happens gets reported (log in `PORTFOLIO.md`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # then paste your FACEIT API key into .env
python src/fetch_faceit.py
```

You can get a FACEIT API key free at [developers.faceit.com](https://developers.faceit.com) (server-side key).

## Layout

```
src/            scripts (data fetching, parsing, features)
notebooks/      exploration and analysis
data/raw/       FACEIT JSON + .dem files (gitignored)
data/processed/ cleaned tables (gitignored)
JOURNAL.md      dev log: dead ends included
```
