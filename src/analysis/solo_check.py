"""Exploratory: did MlgSly's solo versions of the habits show up? Baseline vs week 1, match bootstrap."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
KEY = ["match_id", "round"]
START = pd.Timestamp("2026-11-09", tz="Asia/Singapore")
R = {"MlgSly", "Sad_X", "Inbail", "Toe_bias", "XxcanonboixX"}

ok = pd.read_parquet(P / "demo_quality.parquet").query("ok")["match_id"]
pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok")
regs = pr[pr.is_my_team & pr.name.isin(R)].groupby("match_id")["name"].nunique()
keep = regs[regs >= 3].index
m = pd.read_parquet(P / "matches.parquet")[["match_id", "started_at_local"]]
rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @keep and not is_pistol").merge(m, on="match_id")
rf["exp"] = rf.started_at_local >= START
kills = pd.read_parquet(P / "kills.parquet").query("match_id in @keep and not after_round_end")
me = pr[pr.is_me].set_index(KEY)

r = rf.join(me[["survived", "death_time_s", "kast", "trade_kills", "kills", "damage"]], on=KEY)
r = r[r.survived.notna()]

# first death is ours and traded BY ME
op = kills[kills.is_opening].set_index(KEY)
r = r.join(op[["victim_is_me", "tick"]].rename(columns={"tick": "op_tick"}), on=KEY)
tk = kills[kills.is_trade_kill & kills.attacker_is_me][KEY + ["tick"]]
tk = tk.merge(r[KEY + ["op_tick"]], on=KEY)
tk = tk[(tk.tick > tk.op_tick) & (tk.tick <= tk.op_tick + 5 * 64)]
r["i_traded_first"] = r.set_index(KEY).index.isin(tk.set_index(KEY).index)
teammate_first = (~r.opening_ours) & (r.victim_is_me == False)
r["i_traded_first_m"] = np.where(teammate_first, r.i_traded_first.astype(float), np.nan)
r["first_death_traded_m"] = np.where(~r.opening_ours, r.opening_traded.astype(float), np.nan)
r["i_died_first"] = np.where(~r.opening_ours, (r.victim_is_me == True).astype(float), np.nan)
r["dt"] = r.death_time_s.fillna(r.duration_s)  # seconds alive

# distance from me to the teammate who died first, at that moment
pos = pd.read_parquet(P / "positions.parquet", columns=["match_id", "round", "time_s", "is_me", "X", "Y", "place", "side", "is_my_team"])
pos = pos[pos.match_id.isin(keep)]
fd = kills[kills.is_opening & kills.victim_is_my_team & ~kills.victim_is_me][KEY + ["time_s", "victim_X", "victim_Y"]]
mp = pos[pos.is_me].merge(fd, on=KEY, suffixes=("", "_d"))
mp = mp[(mp.time_s <= mp.time_s_d) & (mp.time_s > mp.time_s_d - 0.6)]
mp["dist"] = np.hypot(mp.X - mp.victim_X, mp.Y - mp.victim_Y)
r = r.join(mp.groupby(KEY).dist.last().rename("my_dist_to_first_death"), on=KEY)

# 2+ mid smokes before the first kill (our T rounds, Mirage/Inferno/Dust2), same rule as nb05
MID_SMOKE = {"de_mirage": {"SnipersNest", "TopofMid", "Middle", "Connector", "Catwalk", "Jungle"},
             "de_inferno": {"Middle", "TopofMid", "SecondMid", "Arch", "Library"},
             "de_dust2": {"Middle", "MidDoors", "TopofMid", "Catwalk", "ShortStairs", "CTSpawn"}}
nd = pd.read_parquet(P / "grenades.parquet").query("match_id in @keep and not after_round_end and type=='smoke' and thrower_is_my_team and thrower_side=='T'")
nd = nd.merge(rf[KEY + ["map"]], on=KEY).join(op["tick"].rename("op_tick"), on=KEY)
nd = nd[(nd.tick < nd.op_tick.fillna(np.inf)) & nd["map"].isin(MID_SMOKE)].copy()
nd["land_place"] = ""
for mpn, sub in nd.groupby("map"):
    pp = pos[(pos.match_id.isin(rf[rf["map"] == mpn].match_id))].dropna(subset=["place"])
    pp = pp.sample(min(len(pp), 150_000), random_state=0)
    _, i = cKDTree(pp[["X", "Y"]].values).query(sub[["land_X", "land_Y"]].values)
    nd.loc[sub.index, "land_place"] = pp["place"].values[i]
nd = nd[[lp in MID_SMOKE[mm] for lp, mm in zip(nd.land_place, nd["map"])]]
r = r.join(nd.groupby(KEY).size().rename("mid_smokes"), on=KEY)
three = r["map"].isin(MID_SMOKE) & (r.my_side == "T")
r["two_plus_mid"] = np.where(three, (r.mid_smokes.fillna(0) >= 2).astype(float), np.nan)


def boot(df, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = {k: v[col].dropna().to_numpy() for k, v in df.groupby("match_id")}
    b = [k for k in g if df.loc[df.match_id == k, "exp"].iloc[0] == False]
    e = [k for k in g if df.loc[df.match_id == k, "exp"].iloc[0] == True]
    est = lambda ks: np.concatenate([g[k] for k in ks]).mean()
    d = [est(rng.choice(e, len(e))) - est(rng.choice(b, len(b))) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


rows = []
for side in ["T", "CT"]:
    s = r[r.my_side == side]
    for col, lab in [("i_traded_first_m", "I trade the teammate who died first"),
                     ("my_dist_to_first_death", "my distance to that teammate (units)"),
                     ("i_died_first", "I'm our first death"),
                     ("survived", "I survive the round"),
                     ("dt", "seconds I stay alive"),
                     ("kast", "my KAST"),
                     ("damage", "my damage per round"),
                     ("two_plus_mid", "we throw 2+ mid smokes before first kill")]:
        if s[col].notna().sum() == 0:
            continue
        b, e = s[~s.exp][col].dropna(), s[s.exp][col].dropna()
        lo, hi = boot(s, col)
        rows.append((side, lab, round(b.mean(), 3), len(b), round(e.mean(), 3), len(e), round(lo, 3), round(hi, 3)))
out = pd.DataFrame(rows, columns=["side", "metric", "base", "n_b", "week1", "n_e", "ci_lo", "ci_hi"])
pd.set_option("display.width", 200)
print(out.to_string(index=False))
t = r[three].copy(); t["mid_smokes"] = t.mid_smokes.fillna(0)
print(t.groupby(["map", "exp"]).agg(rounds=("two_plus_mid", "size"), two_plus=("two_plus_mid", "mean"), mean_mid=("mid_smokes", "mean"), any_mid=("mid_smokes", lambda s: (s > 0).mean())).round(2))
ms = nd.merge(t[KEY + ["exp"]], on=KEY); print(ms.groupby(["exp"]).thrower_is_me.mean().round(2))
