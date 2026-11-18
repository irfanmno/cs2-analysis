"""Exploratory: how do my aggressive CT rounds go, and does it matter if it's the first push to that spot in the match?

"Aggressive" = I'm the furthest player from our spawn at 0:15 (same rule as experiment 1, H3), i.e. I pushed up.
"First push" = the first time in that match I'm furthest forward in that callout area; later ones at the same area = repeats.
Impact = my net WPA per round (win % added by my kills minus my deaths), nb04 model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from wpa import wpa_per_round  # same WPA model as nb04  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
KEY = ["match_id", "round"]

ok = set(pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"])
rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok")
kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end").sort_values(KEY + ["tick"])
wpa = wpa_per_round(rf, kills, pd.Series(True, index=rf.index))

pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok and is_me")
me = pr.merge(rf[KEY + ["map", "my_side", "my_team_won", "is_pistol", "my_team_buy"]], on=KEY)
me = me[(me["my_side"] == "CT") & ~me["is_pistol"]].copy()
me["wpa"] = me.set_index(KEY).index.map(wpa).fillna(0).to_numpy()

# who is furthest forward at 0:15
pos = pd.read_parquet(P / "positions.parquet", columns=["match_id", "round", "time_s", "name", "is_me", "is_my_team", "side", "X", "Y", "place"])
pos = pos[pos["match_id"].isin(ok) & pos["is_my_team"] & (pos["side"] == "CT")].merge(rf[KEY + ["map"]], on=KEY)
spawn = pos[pos["time_s"] < 1].groupby("map")[["X", "Y"]].median()
p15 = pos[(pos["time_s"] >= 14.9) & (pos["time_s"] < 15.4)].join(spawn.add_suffix("_s"), on="map")
p15["d"] = np.hypot(p15["X"] - p15["X_s"], p15["Y"] - p15["Y_s"])
p15 = p15.drop_duplicates(KEY + ["name"])
p15 = p15[p15.groupby(KEY)["d"].transform("size") == 5]
p15["forward"] = p15.groupby(KEY)["d"].rank(ascending=False, method="first") == 1
mine15 = p15[p15["is_me"]].set_index(KEY)[["forward", "place"]]
me = me.join(mine15, on=KEY).dropna(subset=["forward"])
me["forward"] = me["forward"].astype(bool)

# "good spawn" = of my team, I spawned closest to the enemy (T) spawn, i.e. the shortest route to forward spots
allpos = pd.read_parquet(P / "positions.parquet", columns=["match_id", "round", "time_s", "name", "is_me", "side", "X", "Y"])
allpos = allpos[allpos["match_id"].isin(ok) & (allpos["time_s"] < 1)].merge(rf[KEY + ["map"]], on=KEY)
t_spawn = allpos[allpos["side"] == "T"].groupby("map")[["X", "Y"]].median()
ct0 = allpos[(allpos["side"] == "CT")].drop_duplicates(KEY + ["name"]).join(t_spawn.add_suffix("_t"), on="map")
ct0["to_enemy"] = np.hypot(ct0["X"] - ct0["X_t"], ct0["Y"] - ct0["Y_t"])
ct0["spawn_rank"] = ct0.groupby(KEY)["to_enemy"].rank(method="first")
me = me.join(ct0[ct0["is_me"]].set_index(KEY)["spawn_rank"], on=KEY)
me["good_spawn"] = me["spawn_rank"] == 1

# first push to that area in the match vs repeats
me = me.sort_values(KEY)
fw = me[me["forward"]].copy()
fw["nth_push_here"] = fw.groupby(["match_id", "place"]).cumcount() + 1
fw["nth_push_any"] = fw.groupby("match_id").cumcount() + 1
me = me.join(fw.set_index(KEY)[["nth_push_here", "nth_push_any"]], on=KEY)
me["style"] = np.select([~me["forward"], me["nth_push_here"] == 1], ["not pushing", "push: 1st time at that spot"],
                        "push: repeat at that spot")
me["kills_2plus"] = (me["kills"] >= 2).astype(float)
me["won"] = me["my_team_won"].astype(float)


def boot_diff(df: pd.DataFrame, col: str, a: str, b: str, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    ids = df["match_id"].unique()
    g = {k: v for k, v in df.groupby("match_id")}
    out = []
    for _ in range(n):
        s = pd.concat([g[k] for k in rng.choice(ids, len(ids))])
        out.append(s.loc[s["style"] == a, col].mean() - s.loc[s["style"] == b, col].mean())
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only matches starting on/after this date (SGT), e.g. 2026-11-17 for trial 3")
    ap.add_argument("--until", help="only matches starting before this date (SGT)")
    args = ap.parse_args()
    starts = pd.read_parquet(P / "matches.parquet").set_index("match_id")["started_at_local"]
    if args.since:
        me = me[me["match_id"].map(starts) >= pd.Timestamp(args.since, tz="Asia/Singapore")]
    if args.until:
        me = me[me["match_id"].map(starts) < pd.Timestamp(args.until, tz="Asia/Singapore")]
    if args.since or args.until:
        print(f"Only matches from {args.since or 'the start'} to {args.until or 'now'}")
        if me.empty:
            print("No matches yet.")
            raise SystemExit
    pd.set_option("display.width", 200)
    agg = dict(rounds=("wpa", "size"), my_wpa=("wpa", "mean"), kills=("kills", "mean"), two_plus_kills=("kills_2plus", "mean"),
               died=("deaths", "mean"), round_won=("won", "mean"))
    print(f"My CT gun rounds with positions: {len(me)} in {me['match_id'].nunique()} matches\n")
    print(me.groupby("style").agg(**agg).round(2).to_string())
    for col in ["wpa", "kills_2plus", "deaths", "won"]:
        lo, hi = boot_diff(me, col, "push: 1st time at that spot", "push: repeat at that spot")
        lo2, hi2 = boot_diff(me, col, "push: 1st time at that spot", "not pushing")
        print(f"{col:12s} 1st push - repeat: {lo:+.2f} to {hi:+.2f} | 1st push - not pushing: {lo2:+.2f} to {hi2:+.2f}")

    print("\nGood spawn only (I spawned closest to the enemy): pushed vs didn't push")
    gs = me[me["good_spawn"]].assign(style=lambda d: np.where(d["forward"], "good spawn + pushed", "good spawn + didn't push"))
    print(gs.groupby("style").agg(**agg).round(2).to_string())
    if gs["style"].nunique() == 2:
        for col in ["wpa", "deaths", "won"]:
            lo, hi = boot_diff(gs, col, "good spawn + pushed", "good spawn + didn't push")
            print(f"  {col:8s} pushed - didn't: {lo:+.2f} to {hi:+.2f}")
    print(f"\nWhole CT side, all rounds: my WPA per round {me['wpa'].mean():+.2f} over {me['match_id'].nunique()} matches; "
          f"per-match average {me.groupby('match_id')['wpa'].mean().mean():+.2f}")

    print("\nHow often I push per match (CT gun rounds):", me.groupby("match_id")["forward"].sum().describe().round(1).to_dict())
    print("\nBy how many pushes I'd already made that match (any spot):")
    print(me[me["forward"]].assign(nth=lambda d: d["nth_push_any"].clip(upper=4).astype(int).astype(str).replace("4", "4+"))
          .groupby("nth").agg(**agg).round(2).to_string())
    print("\nWhere I push from (Mirage, first push vs repeat):")
    mir = me[(me["map"] == "de_mirage") & me["forward"]]
    print(mir.groupby(["place", "style"]).agg(rounds=("wpa", "size"), my_wpa=("wpa", "mean")).unstack().query("`('rounds', 'push: 1st time at that spot')` >= 5").round(2).to_string())
