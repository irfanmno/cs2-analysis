"""Score the habit experiment: baseline matches vs experiment matches. See experiments/test1_team_habits.md for the plan.

Usage:
    python src/experiments/test1_team_habits.py                      # baseline only (before the start date in EXPERIMENT_START)
    python src/experiments/test1_team_habits.py --start 2026-11-10   # baseline = before, experiment = on/after this date (SGT)

Run the normal pipeline first so new demos are in data/processed:
    fetch_faceit -> build_dataset -> prepare_demos -> parse_demos -> features

Only demos that pass the quality check count, and only matches with at least MIN_REGULARS of the
regular stack (me included), so the baseline and the experiment compare like with like.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"

EXPERIMENT_START = "2026-11-09"   # first day of the experiment (Asia/Singapore); matches before this are the baseline
REGULARS = {"MlgSly", "Sad_X", "Inbail", "Toe_bias", "XxcanonboixX"}
MIN_REGULARS = 3
KEY = ["match_id", "round"]
N_BOOT = 2000


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"]
    rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok and not is_pistol")
    kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end")
    pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok")
    pos = pd.read_parquet(P / "positions.parquet", columns=["match_id", "round", "time_s", "name", "side", "is_my_team", "X", "Y"])
    pos = pos[pos["match_id"].isin(ok) & pos["is_my_team"]]
    matches = pd.read_parquet(P / "matches.parquet")[["match_id", "started_at_local"]]
    nades = pd.read_parquet(P / "grenades.parquet").query(
        "match_id in @ok and not after_round_end and thrower_is_me and type in ['smoke', 'flash', 'he', 'molotov']")

    regs = pr[pr["is_my_team"] & pr["name"].isin(REGULARS)].groupby("match_id")["name"].nunique()
    keep = regs[regs >= MIN_REGULARS].index
    rf = rf[rf["match_id"].isin(keep)].merge(matches, on="match_id")
    return rf, kills, pr, pos, nades


def per_round(rf: pd.DataFrame, kills: pd.DataFrame, pr: pd.DataFrame, pos: pd.DataFrame, nades: pd.DataFrame) -> pd.DataFrame:
    """One row per gun round with every metric the plan names."""
    r = rf.copy()
    r["we_died_first"] = ~r["opening_ours"]
    r["first_death_traded"] = np.where(r["we_died_first"], r["opening_traded"].astype(float), np.nan)
    r["untraded_first_death"] = r["we_died_first"] & ~r["opening_traded"].astype(bool)

    # opponents' first death traded (control: same matches, same lobbies)
    r["they_died_first_traded"] = np.where(r["opening_ours"], r["opening_traded"].astype(float), np.nan)

    # me: where am I at 0:15 on CT, and do I die early in Mirage Middle?
    spawn = pos[pos["time_s"] < 1].merge(r[KEY + ["map"]], on=KEY).groupby(["map", "side"])[["X", "Y"]].median()
    at15 = pos[(pos["time_s"] >= 14.9) & (pos["time_s"] < 15.4)].merge(r[KEY + ["map"]], on=KEY)
    at15 = at15.join(spawn.add_suffix("_spawn"), on=["map", "side"])
    at15["d"] = np.hypot(at15["X"] - at15["X_spawn"], at15["Y"] - at15["Y_spawn"])
    g = at15.groupby(KEY)["d"]
    at15 = at15[g.transform("size") == 5]
    at15["forward"] = at15.groupby(KEY)["d"].rank(ascending=False, method="first") == 1
    me15 = at15[at15["name"] == "MlgSly"].set_index(KEY)["forward"].rename("me_furthest_forward_015")
    r = r.join(me15, on=KEY)

    my_deaths = kills[kills["victim_is_me"]]
    early_mid = my_deaths[(my_deaths["victim_last_place_name"].isin(["Middle", "TopofMid"])) & (my_deaths["time_s"] < 30)]
    r["me_early_mid_death"] = r.set_index(KEY).index.isin(early_mid.set_index(KEY).index)

    # me: nades I throw before the first kill (only rounds I played)
    open_tick = kills[kills["is_opening"]].groupby(KEY)["tick"].min().rename("open_tick")
    n = nades.join(open_tick, on=KEY)
    my_pre = n[n["tick"] < n["open_tick"].fillna(np.inf)].groupby(KEY).size().rename("my_nades_pre")
    played = pr[pr["is_me"]].set_index(KEY).index
    r["my_nades_before_contact"] = np.where(r.set_index(KEY).index.isin(played),
                                            r.join(my_pre, on=KEY)["my_nades_pre"].fillna(0), np.nan)
    return r


METRICS = {
    # name: (column, side filter, description)
    "H1 first death traded (T)": ("first_death_traded", "T", "share of our first deaths that get traded"),
    "H1 first death traded (CT)": ("first_death_traded", "CT", ""),
    "H1 first death traded (both)": ("first_death_traded", None, "PRIMARY"),
    "H1 control: their first death traded": ("they_died_first_traded", None, "should NOT move"),
    "H2 our nades before first kill (T)": ("my_util_before_contact", "T", "per gun round"),
    "H2 our nades before first kill (CT)": ("my_util_before_contact", "CT", ""),
    "H2 my nades before first kill (T)": ("my_nades_before_contact", "T", "me as util lead"),
    "H2 control: their nades before first kill": ("enemy_util_before_contact", None, "should NOT move"),
    "H3 me furthest forward at 0:15 (CT)": ("me_furthest_forward_015", "CT", "share of my CT rounds"),
    "H3 me dying early in mid (CT, Mirage)": ("me_early_mid_death", "CT-mirage", "share of Mirage CT rounds"),
    "outcome: untraded first death": ("untraded_first_death", None, "share of gun rounds"),
    "outcome: round win (T)": ("my_team_won", "T", "underpowered, report only"),
    "outcome: round win (CT)": ("my_team_won", "CT", "underpowered, report only"),
}


def subset(r: pd.DataFrame, side: str | None) -> pd.DataFrame:
    if side == "CT-mirage":
        return r[(r["my_side"] == "CT") & (r["map"] == "de_mirage")]
    return r if side is None else r[r["my_side"] == side]


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, rng) -> tuple[float, float]:
    """95% CI for mean(b) - mean(a), resampling whole matches."""
    def match_sums(d):
        x = d[["match_id", col]].dropna()
        return x.groupby("match_id")[col].agg(["sum", "size"])
    sa, sb = match_sums(a), match_sums(b)
    if sa.empty or sb.empty:
        return np.nan, np.nan
    diffs = []
    for _ in range(N_BOOT):
        ra = sa.iloc[rng.integers(0, len(sa), len(sa))].sum()
        rb = sb.iloc[rng.integers(0, len(sb), len(sb))].sum()
        diffs.append(rb["sum"] / rb["size"] - ra["sum"] / ra["size"])
    return tuple(np.nanpercentile(diffs, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=EXPERIMENT_START)
    args = ap.parse_args()

    rf, kills, pr, pos, nades = load()
    r = per_round(rf, kills, pr, pos, nades)
    start = pd.Timestamp(args.start, tz="Asia/Singapore")
    base, exp = r[r["started_at_local"] < start], r[r["started_at_local"] >= start]
    rng = np.random.default_rng(0)

    print(f"baseline: {base['match_id'].nunique()} matches, {len(base)} gun rounds (before {args.start})")
    print(f"experiment: {exp['match_id'].nunique()} matches, {len(exp)} gun rounds\n")
    rows = []
    for name, (col, side, note) in METRICS.items():
        a, b = subset(base, side), subset(exp, side)
        row = {"metric": name, "baseline": a[col].astype(float).mean(), "n_base": int(a[col].notna().sum())}
        if len(b):
            row.update(experiment=b[col].astype(float).mean(), n_exp=int(b[col].notna().sum()))
            row["change"] = row["experiment"] - row["baseline"]
            row["ci_low"], row["ci_high"] = boot_diff(a, b, col, rng)
        row["note"] = note
        rows.append(row)
    out = pd.DataFrame(rows).set_index("metric")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3))


if __name__ == "__main__":
    main()
