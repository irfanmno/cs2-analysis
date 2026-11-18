"""Exploratory: what do opponents do in matches where they blow us out?

Blowout = lost by 7+ rounds. Compared with close losses (2-6) and wins. Only things decided at or before
the first kill are compared (buys, guns, utility before contact, where/when the first fight happens),
because anything after the first kill is mostly a consequence of who won it (nb05 lesson).
Intervals: bootstrap over matches, blowout minus close loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
KEY = ["match_id", "round"]

ok = set(pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"])
m = pd.read_parquet(P / "matches.parquet").query("match_id in @ok")
m["margin"] = m["my_score"] - m["opp_score"]
m["group"] = np.select([m["margin"] <= -7, m["margin"] < 0], ["blowout loss", "close loss"], "win")
m["level_gap"] = m["opp_team_avg_level"] - m["my_team_avg_level"]

rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok and not is_pistol")
rf = rf.merge(m[["match_id", "group"]], on="match_id")
pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok")
lo = pd.read_parquet(P / "loadouts.parquet").merge(pr[KEY + ["steamid", "is_my_team"]], on=KEY + ["steamid"])
kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end")

# guns and utility they carry 10 s in (decided in freeze time)
en = lo[~lo["is_my_team"]].groupby(KEY).agg(their_awps=("weapon_class", lambda s: (s == "awp").sum()),
                                           their_util_carried=("n_util", "sum"))
us = lo[lo["is_my_team"]].groupby(KEY).agg(our_awps=("weapon_class", lambda s: (s == "awp").sum()))
rf = rf.join(en, on=KEY).join(us, on=KEY)

op = kills[kills["is_opening"]].set_index(KEY)
rf = rf.join(op[["weapon", "attacker_is_my_team"]].rename(columns={"weapon": "opening_weapon"}), on=KEY)
rf["they_won_opening"] = (~rf["opening_ours"]).astype(float)
rf["their_opening_with_awp"] = ((~rf["opening_ours"]) & (rf["opening_weapon"] == "awp")).astype(float)
rf["their_full_buy"] = (rf["enemy_buy"] == "full").astype(float)
rf["we_full_buy"] = (rf["my_team_buy"] == "full").astype(float)
rf["they_have_awp"] = (rf["their_awps"] > 0).astype(float)

METRICS = [
    ("their_full_buy", "they full buy"),
    ("we_full_buy", "we full buy"),
    ("they_have_awp", "they have an AWP"),
    ("their_util_carried", "utility they carry"),
    ("enemy_util_before_contact", "their nades before first kill"),
    ("my_util_before_contact", "our nades before first kill"),
    ("opening_time_s", "first kill happens at (s)"),
    ("they_won_opening", "THEY win the opening duel"),
    ("their_opening_with_awp", "they open with an AWP"),
]


def boot(df: pd.DataFrame, col: str, a: str, b: str, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    ga = {k: v[col].dropna().to_numpy() for k, v in df[df["group"] == a].groupby("match_id")}
    gb = {k: v[col].dropna().to_numpy() for k, v in df[df["group"] == b].groupby("match_id")}
    ka, kb = list(ga), list(gb)
    d = [np.concatenate([ga[k] for k in rng.choice(ka, len(ka))]).mean()
         - np.concatenate([gb[k] for k in rng.choice(kb, len(kb))]).mean() for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in METRICS:
        means = df.groupby("group")[col].mean()
        lo_, hi_ = boot(df, col, "blowout loss", "close loss")
        rows.append({"metric": label, **means.round(2).to_dict(), "blowout - close (95%)": f"{lo_:+.2f} to {hi_:+.2f}"})
    return pd.DataFrame(rows).set_index("metric")[["blowout loss", "close loss", "win", "blowout - close (95%)"]]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(m.groupby("group").agg(matches=("match_id", "size"), level_gap=("level_gap", "mean"),
                                 maps=("map", lambda s: s.str.replace("de_", "").value_counts().to_dict())).to_string())
    print(f"\nAll gun rounds ({len(rf)}):")
    print(table(rf).to_string())
    for side in ["CT", "T"]:
        print(f"\nWe are {side}:")
        print(table(rf[rf["my_side"] == side]).to_string())

    # where do they win the opening duel, blowouts vs close losses (Mirage, our side)
    print("\nWhere they win the opening duel (top places, share of their opening wins):")
    theirs = kills[kills["is_opening"] & ~kills["attacker_is_my_team"]].merge(rf[KEY + ["group", "map", "my_side"]], on=KEY)
    for (mp, side), d in theirs[theirs["group"] != "win"].groupby(["map", "my_side"]):
        if len(d) < 40:
            continue
        t = d.groupby("group")["victim_last_place_name"].value_counts(normalize=True).unstack(0).fillna(0)
        if "blowout loss" not in t:
            continue
        print(f"\n{mp} (we are {side}), n = {d.groupby('group').size().to_dict()}")
        print(t.sort_values("blowout loss", ascending=False).head(5).round(2).to_string())

    # economy is a consequence of losing rounds, so redo it with both teams on a full buy
    both_full = rf[(rf["my_team_buy"] == "full") & (rf["enemy_buy"] == "full")]
    print(f"\nBoth teams full buy only ({len(both_full)} rounds):")
    print(table(both_full).to_string())

    # how blowouts start: pistols, early rounds, second half
    allr = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok").merge(m[["match_id", "group"]], on="match_id")
    allr["won"] = allr["my_team_won"].astype(float)
    print("\nHow it goes over the match (our round win rate):")
    print(pd.DataFrame({
        "pistols (1, 13)": allr[allr["round"].isin([1, 13])].groupby("group")["won"].mean(),
        "rounds 2-4": allr[allr["round"].between(2, 4)].groupby("group")["won"].mean(),
        "rounds 5-12": allr[allr["round"].between(5, 12)].groupby("group")["won"].mean(),
        "rounds 14-16": allr[allr["round"].between(14, 16)].groupby("group")["won"].mean(),
        "rounds 17-24": allr[allr["round"].between(17, 24)].groupby("group")["won"].mean(),
    }).round(2).to_string())
    bl = m[m["group"] == "blowout loss"]
    print("\nBlowouts, first-half score and final:", [f"{a}-{b} -> {c}-{d}" for a, b, c, d in
          bl[["my_first_half", "opp_first_half", "my_score", "opp_score"]].astype(int).values])
