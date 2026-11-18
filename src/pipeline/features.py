"""Round-level features from my team's point of view, built on top of parse_demos.py output.

Usage:
    python src/pipeline/features.py

Writes:
    data/processed/round_features.parquet   one row per round (my team), only demos that passed the quality check
    data/processed/deaths.parquet           one row per death of a player on my team, with context
                                            (when, where, traded?, nearest teammate distance, ...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"

ISOLATED_UNITS = 800   # no living teammate within this distance when you die = "died alone"
EARLY_S = 20           # deaths in the first 20 s of a round count as "early"
UTIL = ["smoke", "flash", "molotov", "he"]


def load(name: str, ok: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(P / f"{name}.parquet")
    return df[df["match_id"].isin(ok)].copy()


def death_context(kills: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """For each death, the distance to the nearest living teammate (from the last position sample)."""
    deaths = kills.copy()
    deaths["sample_tick"] = deaths["tick"] // 32 * 32  # positions are sampled on a 32-tick grid offset by start+1
    # align: take the latest position sample at or before the death tick for each round
    pos_ticks = pos[["match_id", "round", "tick"]].drop_duplicates().sort_values("tick")
    deaths = deaths.sort_values("tick")
    deaths = pd.merge_asof(deaths, pos_ticks.rename(columns={"tick": "pos_tick"}), left_on="tick", right_on="pos_tick",
                           by=["match_id", "round"], direction="backward")
    snap = pos[["match_id", "round", "tick", "steamid", "side", "X", "Y"]].rename(columns={"tick": "pos_tick"})
    m = deaths[["match_id", "round", "pos_tick", "tick", "victim_steamid", "victim_side", "victim_X", "victim_Y"]].merge(
        snap, on=["match_id", "round", "pos_tick"], how="left")
    m = m[(m["side"] == m["victim_side"]) & (m["steamid"] != m["victim_steamid"])]
    m["dist"] = np.hypot(m["X"] - m["victim_X"], m["Y"] - m["victim_Y"])
    nearest = m.groupby(["match_id", "round", "tick", "victim_steamid"])["dist"].min().rename("nearest_teammate")
    deaths = deaths.merge(nearest.reset_index(), on=["match_id", "round", "tick", "victim_steamid"], how="left")
    # no teammate alive at all -> infinitely alone
    deaths["nearest_teammate"] = deaths["nearest_teammate"].fillna(np.inf)
    deaths["isolated"] = deaths["nearest_teammate"] > ISOLATED_UNITS
    return deaths.drop(columns=["sample_tick"])


def spread_at(pos: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Team shape `seconds` into the round: how spread out my team is and how many distinct spots it covers."""
    s = pos[pos["is_my_team"]].copy()
    s["dt"] = (s["time_s"] - seconds).abs()
    s = s[s["dt"] <= 0.3]
    s = s[s["dt"] == s.groupby(["match_id", "round"])["dt"].transform("min")]
    g = s.groupby(["match_id", "round"])
    cx, cy = g["X"].transform("mean"), g["Y"].transform("mean")
    s["d_centroid"] = np.hypot(s["X"] - cx, s["Y"] - cy)
    out = g.agg(alive=("steamid", "nunique"), places=("place", "nunique"))
    out["spread"] = s.groupby(["match_id", "round"])["d_centroid"].mean()
    # biggest group standing in the same named spot
    out["max_stack"] = s.groupby(["match_id", "round", "place"]).size().groupby(level=[0, 1]).max()
    return out.add_suffix(f"_{int(seconds)}s")


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(P / "demo_quality.parquet")
    ok = set(q.loc[q["ok"], "match_id"])

    rounds = load("rounds", ok)
    kills = load("kills", ok)
    grenades = load("grenades", ok)
    pos = load("positions", ok)

    key = ["match_id", "round"]
    rf = rounds.set_index(key)[[
        "map", "half", "is_pistol", "is_overtime", "my_side", "my_team_won", "reason", "duration_s",
        "my_team_buy", "enemy_buy", "my_team_equip", "enemy_equip", "my_score_before", "enemy_score_before",
        "bomb_planted", "plant_time_s", "plant_site",
    ]].copy()
    rf["equip_diff"] = rf["my_team_equip"] - rf["enemy_equip"]

    live = kills[~kills["after_round_end"]].sort_values("tick")

    # ---- opening duel ---------------------------------------------------------------------
    first = live.groupby(key).first()
    rf["opening_ours"] = ~first["victim_is_my_team"]                      # enemy died first
    rf["opening_time_s"] = first["time_s"]
    rf["opening_place"] = np.where(first["victim_is_my_team"], first["victim_last_place_name"],
                                   first["attacker_last_place_name"])
    rf["opening_victim"] = first["victim_name"]
    rf["opening_traded"] = first["traded"]

    # ---- deaths on my team ----------------------------------------------------------------
    deaths = death_context(live[live["victim_is_my_team"]], pos)
    deaths = deaths.merge(rounds[key + ["map", "my_side", "my_team_won", "my_team_buy"]], on=key, how="left")
    d = deaths.groupby(key)
    rf["my_deaths"] = d.size()
    rf["my_deaths_early"] = deaths[deaths["time_s"] <= EARLY_S].groupby(key).size()
    rf["my_untraded_deaths"] = deaths[~deaths["traded"]].groupby(key).size()
    rf["my_isolated_deaths"] = deaths[deaths["isolated"]].groupby(key).size()
    rf["my_isolated_untraded"] = deaths[deaths["isolated"] & ~deaths["traded"]].groupby(key).size()

    # ---- kills by my team -----------------------------------------------------------------
    mine = live[live["attacker_is_my_team"] & ~live["is_world"] & ~live["is_teamkill"]]
    rf["my_kills"] = mine.groupby(key).size()
    rf["my_trade_kills"] = mine[mine["is_trade_kill"]].groupby(key).size()

    # ---- man advantage: did we convert / throw a 5v4? ---------------------------------------
    second = live.groupby(key).nth(1).set_index(key) if len(live) else live
    rf["second_death_ours"] = second["victim_is_my_team"]

    # ---- utility --------------------------------------------------------------------------
    g = grenades[~grenades["after_round_end"]].merge(first[["tick"]].rename(columns={"tick": "opening_tick"}),
                                                     left_on=key, right_index=True, how="left")
    g["before_contact"] = g["tick"] < g["opening_tick"].fillna(np.inf)
    gm, ge = g[g["thrower_is_my_team"]], g[~g["thrower_is_my_team"]]
    for t in UTIL:
        rf[f"my_{t}"] = gm[gm["type"] == t].groupby(key).size()
        rf[f"enemy_{t}"] = ge[ge["type"] == t].groupby(key).size()
    rf["my_util"] = gm[gm["type"].isin(UTIL)].groupby(key).size()
    rf["enemy_util"] = ge[ge["type"].isin(UTIL)].groupby(key).size()
    rf["my_util_before_contact"] = gm[gm["type"].isin(UTIL) & gm["before_contact"]].groupby(key).size()
    rf["enemy_util_before_contact"] = ge[ge["type"].isin(UTIL) & ge["before_contact"]].groupby(key).size()
    rf["my_enemies_flashed"] = gm.groupby(key)["enemies_flashed"].sum()
    rf["my_team_flashed"] = gm.groupby(key)["teammates_flashed"].sum()

    # ---- team shape -----------------------------------------------------------------------
    rf = rf.join(spread_at(pos, 15)).join(spread_at(pos, 30))

    count_cols = [c for c in rf.columns if c.startswith(("my_deaths", "my_untraded", "my_isolated", "my_kills",
                                                         "my_trade", "my_util", "enemy_util", "my_enemies", "my_team_flashed"))
                  or c in [f"my_{t}" for t in UTIL] + [f"enemy_{t}" for t in UTIL]]
    rf[count_cols] = rf[count_cols].fillna(0)
    rf = rf.reset_index()
    return rf, deaths


if __name__ == "__main__":
    rf, deaths = build()
    rf.to_parquet(P / "round_features.parquet", index=False)
    deaths.to_parquet(P / "deaths.parquet", index=False)
    print(f"{len(rf)} rounds -> round_features.parquet ({rf.shape[1]} columns)")
    print(f"{len(deaths)} deaths on my team -> deaths.parquet")
