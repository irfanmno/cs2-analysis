"""Score experiment 2 (EXPERIMENT_2.md): MAC-10 vs AK on my eligible T rounds, assigned by a random card.

Usage:
    python src/experiment2.py                       # score everything since the lock
    python src/experiment2.py --compliance-only     # the 10-match check: did I follow the card? (no outcomes shown)
    python src/experiment2.py --dry-run 2026-10-01  # pretend the lock was earlier, to test the code on old demos

Run first: fetch_faceit -> build_dataset -> prepare_demos -> parse_demos -> features -> parse_loadouts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
KEY = ["match_id", "round"]
SGT = "Asia/Singapore"

LOCK = pd.Timestamp("2026-11-17 18:00", tz=SGT)   # matches that START after this count
WEALTH_MIN = 3700                                  # AK ($2,700) + armour ($1,000): "I can afford a rifle"
SEED = 20261117
N_BOOT = 2000


def card() -> dict[int, str]:
    """Random weapon per round number (balanced within each half), from a fixed seed. Odd matches use it, even matches flip it."""
    rng = np.random.default_rng(SEED)
    out: dict[int, str] = {}
    for block in [range(2, 13), range(14, 25), range(25, 31)]:
        b = list(block)
        arms = (["MAC-10", "AK"] * len(b))[: len(b)]
        rng.shuffle(arms)
        out.update(dict(zip(b, arms)))
    return out


def kill_wpa(rf: pd.DataFrame, kills: pd.DataFrame, fit_mask: pd.Series) -> pd.DataFrame:
    """Every kill with its win-probability change for my team (x100), nb04 model, fitted only on rounds where fit_mask is True."""
    k = kills.merge(rf[KEY + ["my_side", "my_team_won", "equip_diff", "bomb_planted", "plant_time_s"]], on=KEY)
    k["us_died"] = k["victim_is_my_team"].astype(int)
    k["them_died"] = 1 - k["us_died"]
    g = k.groupby(KEY)
    k["us_after"] = 5 - g["us_died"].cumsum()
    k["them_after"] = 5 - g["them_died"].cumsum()
    k["us_before"] = k["us_after"] + k["us_died"]
    k["them_before"] = k["them_after"] + k["them_died"]
    k["planted"] = k["bomb_planted"] & (k["time_s"] > k["plant_time_s"])

    def state(us, them, side, planted, eq):
        X = pd.DataFrame({"us": us, "them": them, "diff": us - them, "ct": (side == "CT").astype(int),
                          "planted": planted.astype(int), "equip": eq / 10_000})
        X["planted_ct"] = X["planted"] * X["ct"]
        X["diff_ct"] = X["diff"] * X["ct"]
        X["us_dead"] = (X["us"] == 0).astype(int)
        X["them_dead"] = (X["them"] == 0).astype(int)
        return X

    fit_ids = set(rf.loc[fit_mask, "match_id"])
    b = rf[rf["match_id"].isin(fit_ids)]
    kb = k[k["match_id"].isin(fit_ids)]
    five = pd.Series(5, index=b.index)
    X = pd.concat([state(five, five, b["my_side"], pd.Series(False, index=b.index), b["equip_diff"]),
                   state(kb["us_after"], kb["them_after"], kb["my_side"], kb["planted"], kb["equip_diff"])])
    y = pd.concat([b["my_team_won"], kb["my_team_won"]]).astype(int)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    after = model.predict_proba(state(k["us_after"], k["them_after"], k["my_side"], k["planted"], k["equip_diff"]))[:, 1]
    before = model.predict_proba(state(k["us_before"], k["them_before"], k["my_side"], k["planted"], k["equip_diff"]))[:, 1]
    k["wpa"] = (after - before) * 100
    return k


def wpa_per_round(rf: pd.DataFrame, kills: pd.DataFrame, fit_mask: pd.Series) -> pd.Series:
    """My net win-probability added per round (x100): my kills' WPA plus my deaths' WPA."""
    k = kill_wpa(rf, kills, fit_mask)
    credit = k[k["attacker_is_me"] & ~k["victim_is_my_team"]].groupby(KEY)["wpa"].sum()
    cost = k[k["victim_is_me"]].groupby(KEY)["wpa"].sum()
    return credit.add(cost, fill_value=0)


def build(lock: pd.Timestamp) -> pd.DataFrame:
    ok = set(pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"])
    matches = pd.read_parquet(P / "matches.parquet")[["match_id", "started_at_local", "map"]]
    after = matches[matches["started_at_local"] >= lock].sort_values("started_at_local").reset_index(drop=True)
    after["match_no"] = np.arange(1, len(after) + 1)    # counts every FACEIT match after the lock, demo or not

    rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok")
    rf = rf.merge(matches[["match_id", "started_at_local"]], on="match_id")
    kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end").sort_values(KEY + ["tick"])
    wpa = wpa_per_round(rf, kills, rf["started_at_local"] < lock)

    pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok and is_me")
    lo = pd.read_parquet(P / "loadouts.parquet")
    me = pr.merge(rf[KEY + ["map", "my_side", "my_team_won", "opening_ours", "duration_s"]], on=KEY)
    me = me.merge(lo[KEY + ["steamid", "weapon_class", "primary", "wealth_at_start"]], on=KEY + ["steamid"], how="left")
    me = me.merge(after[["match_id", "match_no"]], on="match_id")   # only matches after the lock

    c = card()
    me = me[(me["my_side"] == "T") & me["round"].isin(c.keys()) & (me["wealth_at_start"] >= WEALTH_MIN)].copy()
    me["arm"] = [c[r] if n % 2 == 1 else {"MAC-10": "AK", "AK": "MAC-10"}[c[r]] for r, n in zip(me["round"], me["match_no"])]
    me["followed"] = np.where(me["arm"] == "MAC-10", me["weapon_class"] == "smg", me["weapon_class"] == "rifle")

    me["wpa"] = me.set_index(KEY).index.map(wpa).fillna(0).to_numpy()
    me["opening_won"] = me["opening_ours"].astype(float)
    me["round_won"] = me["my_team_won"].astype(float)
    me["first_death"] = me["opening_deaths"].astype(float)

    # how far from T spawn I am at 0:15 (median T spawn per map, from pre-lock rounds)
    pos = pd.read_parquet(P / "positions.parquet", columns=["match_id", "round", "time_s", "is_me", "side", "X", "Y"])
    pos = pos[pos["is_me"] & (pos["side"] == "T")].merge(rf[KEY + ["map"]], on=KEY)
    spawn = pos[pos["time_s"] < 1].groupby("map")[["X", "Y"]].median()
    p15 = pos[(pos["time_s"] >= 14.9) & (pos["time_s"] < 15.4)].join(spawn.add_suffix("_s"), on="map")
    p15["dist_015"] = np.hypot(p15["X"] - p15["X_s"], p15["Y"] - p15["Y_s"])
    me = me.join(p15.groupby(KEY)["dist_015"].first(), on=KEY)
    return me


def diff_ci(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """MAC-10 arm minus AK arm, 95% interval from resampling whole matches."""
    d = df.dropna(subset=[col])
    est = d.loc[d["arm"] == "MAC-10", col].mean() - d.loc[d["arm"] == "AK", col].mean()
    ids = d["match_id"].unique()
    groups = {m: g for m, g in d.groupby("match_id")}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        s = pd.concat([groups[m] for m in rng.choice(ids, len(ids))])
        boots.append(s.loc[s["arm"] == "MAC-10", col].mean() - s.loc[s["arm"] == "AK", col].mean())
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compliance-only", action="store_true")
    ap.add_argument("--dry-run", help="pretend the lock was at this date (SGT), to test the code")
    args = ap.parse_args()
    lock = pd.Timestamp(args.dry_run, tz=SGT) if args.dry_run else LOCK

    me = build(lock)
    n_matches = me["match_id"].nunique()
    print(f"lock {lock:%Y-%m-%d %H:%M} SGT | {n_matches} matches with eligible T rounds | {len(me)} eligible rounds"
          + ("  [DRY RUN: old demos, nobody followed a card]" if args.dry_run else ""))
    if me.empty:
        return

    comp = me.groupby("arm").agg(rounds=("followed", "size"), followed=("followed", "mean"))
    print("\nCompliance (held the card's gun 10 s into the round):")
    print(comp.round(2).to_string())
    print(f"overall: {me['followed'].mean():.0%}  | maps: {me.groupby('match_id')['map'].first().value_counts().to_dict()}")
    if args.compliance_only:
        return

    rows = []
    for col, label in [("wpa", "PRIMARY my net WPA per T round (win %)"), ("opening_won", "team wins opening duel"),
                       ("round_won", "team wins round (report only)"), ("dist_015", "my distance from spawn at 0:15"),
                       ("damage", "my damage"), ("kast", "my KAST"), ("first_death", "I'm our first death")]:
        est, lo, hi = diff_ci(me, col)
        rows.append({"metric": label,
                     "MAC-10": me.loc[me["arm"] == "MAC-10", col].mean(), "AK": me.loc[me["arm"] == "AK", col].mean(),
                     "diff": est, "ci_low": lo, "ci_high": hi})
    print("\nIntention to treat (by the card, whatever I actually held):")
    print(pd.DataFrame(rows).set_index("metric").round(3).to_string())

    half = me["match_no"] <= me["match_no"].median()
    print("\nLearning-curve check, primary metric by half of the experiment:")
    for name, part in [("first half", me[half]), ("second half", me[~half])]:
        est, lo, hi = diff_ci(part, "wpa")
        print(f"  {name}: {est:+.2f} ({lo:+.2f} to {hi:+.2f}), {len(part)} rounds")


if __name__ == "__main__":
    main()
