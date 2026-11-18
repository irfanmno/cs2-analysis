"""Exploratory: my T side (weapon, utility, WPA) baseline vs week 1, plus power inputs for a coin-flip test. Not part of the locked scorer."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "data" / "processed"; KEY = ["match_id", "round"]
START = pd.Timestamp("2026-11-09", tz="Asia/Singapore")
ok = set(pd.read_parquet(P / "demo_quality.parquet").query("ok").match_id)
rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @ok")
m = pd.read_parquet(P / "matches.parquet")[["match_id", "started_at_local"]]
rf = rf.merge(m, on="match_id"); rf["exp"] = rf.started_at_local >= START
kills = pd.read_parquet(P / "kills.parquet").query("match_id in @ok and not after_round_end").sort_values(KEY + ["tick"])
pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @ok")
lo = pd.read_parquet(P / "loadouts.parquet")
nades = pd.read_parquet(P / "grenades.parquet").query("match_id in @ok and not after_round_end and thrower_is_me and type in ['smoke','flash','he','molotov']")
dmg = pd.read_parquet(P / "damage.parquet").query("match_id in @ok")

# WPA model fitted on baseline rounds only
k = kills.merge(rf[KEY + ["my_side", "my_team_won", "equip_diff", "bomb_planted", "plant_time_s", "exp"]], on=KEY)
k["us_died"] = k.victim_is_my_team.astype(int); k["them_died"] = 1 - k.us_died
g = k.groupby(KEY); k["us_after"] = 5 - g.us_died.cumsum(); k["them_after"] = 5 - g.them_died.cumsum()
k["us_before"] = k.us_after + k.us_died; k["them_before"] = k.them_after + k.them_died
k["planted"] = k.bomb_planted & (k.time_s > k.plant_time_s)
def state(us, them, side, planted, eq):
    X = pd.DataFrame({"us": us, "them": them, "diff": us - them, "ct": (side == "CT").astype(int), "planted": planted.astype(int), "equip": eq / 10000})
    X["planted_ct"] = X.planted * X.ct; X["diff_ct"] = X["diff"] * X.ct; X["us_dead"] = (X.us == 0).astype(int); X["them_dead"] = (X.them == 0).astype(int)
    return X
b = rf[~rf.exp]; kb = k[~k.exp]
five = pd.Series(5, index=b.index)
X = pd.concat([state(five, five, b.my_side, pd.Series(False, index=b.index), b.equip_diff), state(kb.us_after, kb.them_after, kb.my_side, kb.planted, kb.equip_diff)])
y = pd.concat([b.my_team_won, kb.my_team_won]).astype(int)
mdl = LogisticRegression(max_iter=2000).fit(X, y)
k["wpa"] = mdl.predict_proba(state(k.us_after, k.them_after, k.my_side, k.planted, k.equip_diff))[:, 1] - mdl.predict_proba(state(k.us_before, k.them_before, k.my_side, k.planted, k.equip_diff))[:, 1]
credit = k[k.attacker_is_me & ~k.victim_is_my_team].groupby(KEY).wpa.sum()
cost = k[k.victim_is_me].groupby(KEY).wpa.sum()

me = pr[pr.is_me].merge(rf[KEY + ["map", "is_pistol", "my_team_buy", "exp"]], on=KEY)
me = me[~me.is_pistol]
me["wpa"] = me.set_index(KEY).index.map(credit.add(cost, fill_value=0)).fillna(0).to_numpy() * 100
me = me.merge(lo[["match_id", "round", "name", "primary", "weapon_class", "n_util"]], on=KEY + ["name"], how="left")
me["smg"] = (me.weapon_class == "smg").astype(float)
me["util"] = me.flash_thrown + me.smoke_thrown + me.he_thrown + me.molotov_thrown

# my first engagement: first damage I deal or take
eng = dmg[(dmg.attacker_is_me | dmg.victim_is_me) & ~dmg.after_round_end & ~(dmg.attacker_is_my_team & dmg.victim_is_my_team)].groupby(KEY).tick.min().rename("eng_tick")
n = nades.join(eng, on=KEY)
pre = n[n.tick < n.eng_tick.fillna(np.inf)]
me = me.join(pre.groupby(KEY).size().rename("util_before_my_fight"), on=KEY)
me["util_before_my_fight"] = me.util_before_my_fight.fillna(0)
fl = pre[pre.type == "flash"].groupby(KEY).size().rename("flash_pre"); me = me.join(fl, on=KEY); me["flash_pre"] = me.flash_pre.fillna(0)
sm = pre[pre.type == "smoke"].groupby(KEY).size().rename("smoke_pre"); me = me.join(sm, on=KEY); me["smoke_pre"] = me.smoke_pre.fillna(0)

pd.set_option("display.width", 220)
t = me[me.side == "T"]
print("T gun rounds:", t.groupby("exp").size().to_dict())
print(t.groupby(["exp", "my_team_buy"]).smg.agg(["size", "mean"]).round(2))
print(t[t.weapon_class.isin(["smg", "rifle"])].groupby(["exp", "primary"]).size().unstack(0).fillna(0).astype(int).query("`True`>0 or `False`>20"))
cols = ["util", "util_before_my_fight", "flash_pre", "smoke_pre", "enemies_flashed", "n_util", "wpa", "damage", "kast", "survived"]
print(t.groupby("exp")[cols].mean().round(2).T)
print("\nBaseline only, T, by weapon class (full+force buys):")
tb = t[t.my_team_buy.isin(["full", "force"]) & t.weapon_class.isin(["smg", "rifle", "awp"])]
print(tb.groupby(["exp", "weapon_class"]).agg(rounds=("wpa", "size"), wpa=("wpa", "mean"), dmg=("damage", "mean"), kast=("kast", "mean"), kills=("kills", "mean"), died_first=("opening_deaths", "mean")).round(2))
print("\n--- power inputs (baseline, T full+force buys with smg/rifle) ---")
tf = t[t.my_team_buy.isin(["full", "force"])]
print("T full+force rounds per match:", round(tf.groupby("match_id").size().mean(), 1))
op = kills[kills.is_opening].set_index(KEY)
tf = tf.join(op.attacker_is_my_team.rename("we_won_opening"), on=KEY)
for c in ["wpa", "damage", "kast", "we_won_opening", "util_before_my_fight"]:
    s = tf[~tf.exp][c].astype(float)
    print(f"{c:22s} mean {s.mean():7.2f}  sd {s.std():7.2f}")
from scipy.stats import norm
z = norm.ppf(0.975) + norm.ppf(0.8)
for c in ["wpa", "damage", "kast", "we_won_opening"]:
    sd = tf[~tf.exp][c].astype(float).std()
    for matches in [20, 40]:
        n_arm = matches * len(tf) / tf.match_id.nunique() / 2
        print(f"  {c}: {matches} matches -> {n_arm:.0f} rounds/arm, detectable diff (80% power) = {z*sd*np.sqrt(2/n_arm):.2f}")
