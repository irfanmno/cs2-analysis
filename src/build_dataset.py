"""Turn the raw FACEIT JSON into one tidy table: one row per match, from my point of view.

Usage:
    python src/build_dataset.py   # writes data/processed/matches.parquet
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "faceit"
OUT = ROOT / "data" / "processed" / "matches.parquet"

SESSION_GAP_HOURS = 2  # a new session starts if I haven't queued for this long


def snake(name: str) -> str:
    """'Headshots %' -> 'headshots_pct', 'K/D Ratio' -> 'k_d_ratio'."""
    name = name.replace("%", "pct")
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def to_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def match_row(item: dict, my_id: str) -> dict | None:
    match_id = item["match_id"]
    stats_path = RAW / "stats" / f"{match_id}.json"
    details_path = RAW / "matches" / f"{match_id}.json"
    if not stats_path.exists() or not details_path.exists():
        return None

    stats = load_json(stats_path)["rounds"][0]  # FACEIT queue = best of 1
    details = load_json(details_path)

    my_team = opp_team = None
    me = None
    for team in stats["teams"]:
        for p in team["players"]:
            if p["player_id"] == my_id:
                my_team, me = team, p
    if me is None:
        return None
    opp_team = next(t for t in stats["teams"] if t is not my_team)

    # skill levels from the match roster (each player's FACEIT level at queue time)
    levels = {}
    teammates = []
    my_level = None
    for faction in details["teams"].values():
        ids = [r["player_id"] for r in faction["roster"]]
        for r in faction["roster"]:
            if r["player_id"] == my_id:
                my_level = r.get("game_skill_level") or None  # 0 = missing in the API
        lvls = [r.get("game_skill_level") for r in faction["roster"]]
        lvls = [lv for lv in lvls if lv]  # drop None and 0 (missing)
        key = "mine" if my_id in ids else "opp"
        levels[key] = sum(lvls) / len(lvls) if lvls else None
        if key == "mine":
            teammates = [i for i in ids if i != my_id]

    my_ts, opp_ts = my_team["team_stats"], opp_team["team_stats"]
    row = {
        "match_id": match_id,
        "started_at": pd.to_datetime(item["started_at"], unit="s", utc=True),
        "map": stats["round_stats"]["Map"],
        "rounds": int(stats["round_stats"]["Rounds"]),
        "won": int(my_ts["Team Win"]) == 1,
        "my_score": int(my_ts["Final Score"]),
        "opp_score": int(opp_ts["Final Score"]),
        "my_first_half": int(my_ts["First Half Score"]),
        "opp_first_half": int(opp_ts["First Half Score"]),
        # note: the API's "premade" flag is always False in my data, so parties are
        # inferred from repeat teammates instead (see n_regulars below)
        "my_level": my_level,
        "my_team_avg_level": levels.get("mine"),
        "opp_team_avg_level": levels.get("opp"),
        "teammates": teammates,
        "faceit_url": item.get("faceit_url", "").replace("{lang}", "en"),
    }
    # my scoreboard stats, all numeric
    for k, v in me["player_stats"].items():
        row[snake(k)] = to_num(v)

    # where I ranked on my own team (1 = top fragger)
    kills = sorted((float(p["player_stats"]["Kills"]) for p in my_team["players"]), reverse=True)
    row["team_kill_rank"] = kills.index(float(me["player_stats"]["Kills"])) + 1
    return row


def build() -> pd.DataFrame:
    player = load_json(RAW / "player.json")
    my_id = player["player_id"]
    history = load_json(RAW / "history.json")

    rows = [r for item in history if (r := match_row(item, my_id))]
    df = pd.DataFrame(rows).sort_values("started_at").reset_index(drop=True)

    df["started_at_local"] = df["started_at"].dt.tz_convert("Asia/Singapore")
    df["round_diff"] = df["my_score"] - df["opp_score"]
    df["level_gap"] = df["my_team_avg_level"] - df["opp_team_avg_level"]
    df["match_no"] = range(1, len(df) + 1)

    # sessions: consecutive matches with < SESSION_GAP_HOURS between starts
    gap = df["started_at"].diff().dt.total_seconds().div(3600)
    df["session_id"] = (gap.isna() | (gap > SESSION_GAP_HOURS)).cumsum()
    df["game_in_session"] = df.groupby("session_id").cumcount() + 1
    df["prev_won"] = df.groupby("session_id")["won"].shift(1)  # NaN for 1st game of a session

    # how often I queued with each teammate (to spot my regular stack)
    counts = df["teammates"].explode().value_counts()
    regulars = set(counts[counts >= 5].index)
    df["n_regulars"] = df["teammates"].apply(lambda t: sum(i in regulars for i in t))
    return df


def build_teammates() -> pd.DataFrame:
    """One row per player per match, for everyone on MY team (me included)."""
    my_id = load_json(RAW / "player.json")["player_id"]
    rows = []
    for item in load_json(RAW / "history.json"):
        path = RAW / "stats" / f"{item['match_id']}.json"
        if not path.exists():
            continue
        stats = load_json(path)["rounds"][0]
        team = next((t for t in stats["teams"] if any(p["player_id"] == my_id for p in t["players"])), None)
        if team is None:
            continue
        # top fragger = most kills, ADR as tie-break
        top = max(team["players"], key=lambda p: (float(p["player_stats"]["Kills"]), float(p["player_stats"]["ADR"])))
        for p in team["players"]:
            row = {
                "match_id": item["match_id"],
                "player_id": p["player_id"],
                "nickname": p["nickname"],
                "is_me": p["player_id"] == my_id,
                "won": team["team_stats"]["Team Win"] == "1",
                "is_top_frag": p is top,
            }
            row.update({snake(k): to_num(v) for k, v in p["player_stats"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} matches -> {OUT}")
    mates = build_teammates()
    mates.to_parquet(OUT.parent / "teammates.parquet", index=False)
    print(f"{len(mates)} player-match rows -> {OUT.parent / 'teammates.parquet'}")
    print(df[["started_at_local", "map", "won", "my_score", "opp_score", "kills", "deaths", "adr"]].tail())
