"""Export tidy CSVs for a dashboard tool (Tableau Public / Power BI can't read parquet).

Writes to dashboard/data/:
    matches.csv        one row per FACEIT match (all CS2 matches, demo or not)
    rounds.csv         one row per round with a good demo, my team's point of view
    player_rounds.csv  one row per player per round, my team only, with win % added (WPA)
    kills.csv          one row per kill with map coordinates (for death/kill maps)

Run after the normal pipeline (and parse_loadouts.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment2 import kill_wpa  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "data"
KEY = ["match_id", "round"]
REGULARS = {"MlgSly", "Sad_X", "Inbail", "Toe_bias", "XxcanonboixX"}
PERIODS = [("2026-11-09", "team experiment"), ("2026-11-17", "after")]   # anything earlier = "before"


def period(ts: pd.Series) -> pd.Series:
    out = pd.Series("before", index=ts.index)
    for start, label in PERIODS:
        out[ts >= pd.Timestamp(start, tz="Asia/Singapore")] = label
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = set(pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"])

    # ---- matches
    m = pd.read_parquet(P / "matches.parquet")
    m["date"] = m["started_at_local"].dt.tz_localize(None)
    m["map"] = m["map"].str.replace("de_", "").str.title()
    m["result"] = np.where(m["won"], "Win", "Loss")
    m["margin"] = m["my_score"] - m["opp_score"]
    m["has_demo"] = m["match_id"].isin(ok)
    m["period"] = period(m["started_at_local"])
    m["match_no"] = m["date"].rank(method="first").astype(int)
    m[["match_id", "match_no", "date", "period", "map", "result", "my_score", "opp_score", "margin", "my_level",
       "my_team_avg_level", "opp_team_avg_level", "kills", "assists", "adr", "k_d_ratio", "headshots_pct",
       "first_kills", "has_demo", "faceit_url"]].to_csv(OUT / "matches.csv", index=False)

    # ---- rounds (good demos only)
    rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok")
    rf = rf.merge(m[["match_id", "date", "period"]], on="match_id")
    r = rf.assign(
        map=rf["map"].str.replace("de_", "").str.title(),
        round_result=np.where(rf["my_team_won"], "Win", "Loss"),
        opening=np.where(rf["opening_ours"], "We won it", "They won it"),
        first_death_traded=np.where(rf["opening_ours"], np.nan, rf["opening_traded"].astype(float)),
    )
    r[["match_id", "date", "period", "map", "round", "half", "is_pistol", "is_overtime", "my_side", "round_result",
       "reason", "my_team_buy", "enemy_buy", "my_team_equip", "enemy_equip", "opening", "opening_time_s",
       "opening_place", "first_death_traded", "bomb_planted", "plant_site", "my_util_before_contact",
       "enemy_util_before_contact", "duration_s"]].to_csv(OUT / "rounds.csv", index=False)

    # ---- player rounds (my team), with WPA
    kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end").sort_values(KEY + ["tick"])
    k = kill_wpa(rf, kills, pd.Series(True, index=rf.index))
    credit = k[k["attacker_is_my_team"] & ~k["victim_is_my_team"]].groupby(KEY + ["attacker_name"])["wpa"].sum()
    cost = k[k["victim_is_my_team"]].groupby(KEY + ["victim_name"])["wpa"].sum()
    pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok and is_my_team")
    pr = pr.merge(r[KEY + ["date", "period", "map", "is_pistol", "round_result", "my_team_buy"]], on=KEY)
    lo = pd.read_parquet(P / "loadouts.parquet")[KEY + ["steamid", "primary", "weapon_class", "wealth_at_start"]]
    pr = pr.merge(lo, on=KEY + ["steamid"], how="left")
    idx = pd.MultiIndex.from_frame(pr[KEY + ["name"]])
    pr["kill_wpa"] = credit.reindex(idx).fillna(0).to_numpy().round(2)
    pr["death_wpa"] = cost.reindex(idx).fillna(0).to_numpy().round(2)
    pr["net_wpa"] = (pr["kill_wpa"] + pr["death_wpa"]).round(2)
    pr["player"] = np.where(pr["name"].isin(REGULARS), pr["name"], "Other")
    pr["util_thrown"] = pr[["flash_thrown", "smoke_thrown", "he_thrown", "molotov_thrown"]].sum(axis=1)
    pr[["match_id", "date", "period", "map", "round", "is_pistol", "side", "round_result", "my_team_buy", "name", "player",
        "is_me", "kills", "deaths", "assists", "damage", "kast", "survived", "death_time_s", "opening_kills",
        "opening_deaths", "trade_kills", "traded_deaths", "headshot_kills", "util_thrown", "enemies_flashed",
        "primary", "weapon_class", "wealth_at_start", "kill_wpa", "death_wpa", "net_wpa"]].to_csv(OUT / "player_rounds.csv", index=False)

    # ---- kills with coordinates
    kk = k.merge(r[KEY + ["date", "period", "map"]], on=KEY, suffixes=("_raw", ""))
    kk["our_kill"] = kk["attacker_is_my_team"] & ~kk["victim_is_my_team"]
    kk[["match_id", "date", "period", "map", "round", "time_s", "my_side", "attacker_name", "victim_name", "our_kill",
        "attacker_is_me", "victim_is_me", "weapon", "headshot", "is_opening", "traded", "attacker_last_place_name",
        "victim_last_place_name", "attacker_X", "attacker_Y", "victim_X", "victim_Y", "wpa"]].round(2).to_csv(OUT / "kills.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name:20s} {len(pd.read_csv(f)):>7,} rows  {f.stat().st_size / 1e6:5.1f} MB")


if __name__ == "__main__":
    main()
