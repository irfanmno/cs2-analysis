import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 04: Where and when do I have the most impact?

**Question:** when do my kills (and deaths) actually swing rounds, and where on the map am I worth the most?

Kills aren't equal. Getting the 5th kill when it's 4v1 does almost nothing; the opening kill in a 5v5 does a lot.
So instead of counting kills I use **win probability added (WPA)**:

1. Fit a simple model of *"how likely are we to win this round right now?"* from the state of the round: players alive on each side, side, bomb planted, and the money difference.
2. Every kill moves that number. A kill by me gets the change as **credit**; my death gets the change as a **cost**.
3. Add it up per round. +0.10 means I moved our win chance by 10 percentage points that round.

It's the same idea sports analysts use for "win probability added" in baseball and the NFL. Same 64 demos as before.

> 📝 = my own take.
""")

code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
P = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

quality = pd.read_parquet(P / "demo_quality.parquet")
OK = set(quality.loc[quality["ok"], "match_id"])
KEY = ["match_id", "round"]

rf = pd.read_parquet(P / "round_features.parquet").query("match_id in @OK")
kills = pd.read_parquet(P / "kills.parquet").query("match_id in @OK and not after_round_end").sort_values(KEY + ["tick"])
pr = pd.read_parquet(P / "player_rounds.parquet").query("match_id in @OK")
pos = pd.read_parquet(P / "positions.parquet").query("match_id in @OK")

REGS = ["MlgSly", "Sad_X", "Inbail", "Toe_bias", "XxcanonboixX"]
MAPS = {m: m.replace("de_", "").title() for m in rf["map"].unique()}

plt.rcParams.update({
    "figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})
WIN, LOSS, NEUTRAL, US, THEM = "#2a9d8f", "#e76f51", "#6c757d", "#264653", "#e9c46a"

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
""")

md("## 1. The win-probability model")
code("""
k = kills.merge(rf[KEY + ["map", "my_side", "my_team_won", "equip_diff", "bomb_planted", "plant_time_s", "is_pistol"]], on=KEY)
k["us_died"] = k["victim_is_my_team"].astype(int)
k["them_died"] = 1 - k["us_died"]
g = k.groupby(KEY)
k["us_after"] = 5 - g["us_died"].cumsum()
k["them_after"] = 5 - g["them_died"].cumsum()
k["us_before"] = k["us_after"] + k["us_died"]
k["them_before"] = k["them_after"] + k["them_died"]
k["planted"] = k["bomb_planted"] & (k["time_s"] > k["plant_time_s"])

def state(us, them, side, planted, equip_diff):
    X = pd.DataFrame({"us": us, "them": them, "diff": us - them, "ct": (side == "CT").astype(int),
                      "planted": planted.astype(int), "equip": equip_diff / 10_000})
    X["planted_ct"] = X["planted"] * X["ct"]
    X["diff_ct"] = X["diff"] * X["ct"]
    X["us_dead"] = (X["us"] == 0).astype(int)
    X["them_dead"] = (X["them"] == 0).astype(int)
    return X

five = pd.Series(5, index=rf.index)
X = pd.concat([state(five, five, rf["my_side"], pd.Series(False, index=rf.index), rf["equip_diff"]),   # round start
               state(k["us_after"], k["them_after"], k["my_side"], k["planted"], k["equip_diff"])])  # after every kill
y = pd.concat([rf["my_team_won"], k["my_team_won"]]).astype(int)
groups = pd.concat([rf["match_id"], k["match_id"]])

wp_model = LogisticRegression(max_iter=2000)
p = cross_val_predict(wp_model, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
print(f"out-of-sample AUC {roc_auc_score(y, p):.3f}, Brier {brier_score_loss(y, p):.3f}  (grouped by match)")

cal = pd.DataFrame({"predicted": p, "actual": y.values})
cal["bucket"] = pd.cut(cal["predicted"], np.linspace(0, 1, 11))
cal = cal.groupby("bucket", observed=True).agg(n=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"))
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1)
ax.plot(cal["predicted"], cal["actual"], "o-", color=US)
ax.set_xlabel("predicted win chance"); ax.set_ylabel("actual win rate")
ax.set_title("Calibration: the model's % are honest")
save("04_wp_calibration")
""")
code("""
wp_model.fit(X, y)
k["wp_before"] = wp_model.predict_proba(state(k["us_before"], k["them_before"], k["my_side"], k["planted"], k["equip_diff"]))[:, 1]
k["wp_after"] = wp_model.predict_proba(state(k["us_after"], k["them_after"], k["my_side"], k["planted"], k["equip_diff"]))[:, 1]
k["wpa"] = k["wp_after"] - k["wp_before"]   # from my team's point of view

# per player per round: credit for kills + cost of dying
credit = k[k["attacker_is_my_team"] & ~k["victim_is_my_team"]].groupby(KEY + ["attacker_name"])["wpa"].sum().rename("kill_wpa")
cost = k[k["victim_is_my_team"]].groupby(KEY + ["victim_name"])["wpa"].sum().rename("death_wpa")
mine = pr[pr["is_my_team"] & pr["name"].isin(REGS)].merge(rf[KEY + ["map", "is_pistol"]], on=KEY)
mine = mine.join(credit, on=KEY + ["name"]).join(cost, on=KEY + ["name"]).fillna({"kill_wpa": 0, "death_wpa": 0})
mine["net_wpa"] = mine["kill_wpa"] + mine["death_wpa"]

k.loc[k["attacker_is_my_team"] & ~k["victim_is_my_team"], "wpa"].describe().round(3)
""")
md("""
The model is simple, but it's **well calibrated**: when it says 70%, we win about 70% of the time, out of sample.
An average kill is worth about **+0.17** (17 percentage points).

*Limits:* it doesn't know about positions, health or time left (apart from the plant), and kill credit goes fully to the killer (no credit for the flash or the trade setup).
""")

md("## 2. Who adds the most per round?")
code("""
per100 = (mine.groupby(["name", "side"])["net_wpa"].mean().unstack() * 100)
per100["overall"] = mine.groupby("name")["net_wpa"].mean() * 100
per100["rounds"] = mine.groupby("name").size()
per100.sort_values("overall", ascending=False).round(1)
""")
code("""
t = per100.sort_values("overall")
y = np.arange(len(t))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(y + 0.2, t["T"], height=0.38, color=LOSS, label="T")
ax.barh(y - 0.2, t["CT"], height=0.38, color=US, label="CT")
ax.set_yticks(y, t.index)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("net win % added per round (kills minus deaths)")
ax.set_title("Impact per round, by side")
ax.legend()
save("04_wpa_by_player")
""")
md("""
Read it as "percentage points of round win added per round". **+4 means that, on average, I push our win chance up 4 points every round I play.**

- **I'm the team's biggest impact on CT** (about +4.3 per round), but only about break-even on T (+0.5).
- **Sad_X is the best T-side player** (+3.0) and good on CT too.
- Toe_bias and XxcanonboixX are negative on T: their deaths cost more than their kills add.

> 📝 **My take:** yes. I think my strong point is my game sense/knowledge. My aim and reaction time are above average, but not as good as Sad_X and Inbail. But because of my game sense my aim is a lot better as well, because I know where to pre-aim and what to expect. At FACEIT 5 and 6 it seemed easier to open rounds by purely having aim fights, but now I feel like on T side it's hard for me to open, because not only are players better at aiming, their game sense probably matches or is close to mine.
""")
code("""
# checking "opening rounds got harder as I levelled up": FACEIT's own entry stats (all 138 matches, both sides)
m = pd.read_parquet(P / "matches.parquet").query("my_level > 0")
m["level"] = m["my_level"].clip(4, 7).astype(int).map({4: "4", 5: "5", 6: "6", 7: "7–8"})
lv = m.groupby("level").agg(matches=("won", "size"), opening_duels=("entry_count", "sum"),
                            opening_wins=("entry_wins", "sum"), rounds=("rounds", "sum"))
lv["opening_duel_win_pct"] = lv["opening_wins"] / lv["opening_duels"]
lv["opening_duels_per_round"] = lv["opening_duels"] / lv["rounds"]
lv.round(3)
""")
md("""
**It's a slow slide, not a cliff:** I win about **56%** of my opening duels at level 4–5, **55%** at 6 and **53%** at 7–8, while taking about the same number of them per round.
FACEIT doesn't split this by side, so I can't confirm it's T specifically, and the steps are small (about 100–200 duels per level).
It does fit the story: I'm still winning more than half, just by less as the aim *and* the game sense around me get better.
""")

md("## 3. What kind of kills do I get?")
code("""
def kill_type(r):
    if r["is_opening"]:
        return "1 opening kill"
    if r["planted"]:
        return "5 after the plant"
    if r["us_before"] <= 2:
        return "4 we're down to 1–2"
    if r["is_trade_kill"]:
        return "2 trade kill"
    return "3 other mid-round"

ok = k[k["attacker_is_my_team"] & ~k["victim_is_my_team"] & k["attacker_name"].isin(REGS)].copy()
ok["type"] = ok.apply(kill_type, axis=1)
rounds_played = mine.groupby("name").size()

value = ok.groupby("type")["wpa"].mean().rename("avg value of one kill")
per_player = ok.groupby(["type", "attacker_name"])["wpa"].sum().unstack().div(rounds_played) * 100
display(value.round(3).to_frame())
per_player.round(1)
""")
code("""
types = per_player.index.tolist()
cols = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
order = per_player.sum().sort_values().index
fig, ax = plt.subplots(figsize=(9, 3.8))
left = np.zeros(len(order))
for typ, col in zip(types, cols):
    vals = per_player.loc[typ, order].values
    ax.barh(order, vals, left=left, color=col, label=typ[2:])
    left += vals
ax.set_xlabel("win % added per round from kills, by kind of kill")
ax.set_title("Where each of us gets our value")
ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.45))
save("04_kill_types")
""")
md("""
- **Opening kills are worth the most each** (~+0.23), and trade kills and late-round kills are next.
- **My value mostly comes from mid-round kills**, the fights after the opening duel when it's still 4v4 or 3v4. I have the most of those on the team.
- **Inbail** gets the most from opening kills, which fits him taking first fights on CT (notebook 02).
- **Sad_X** gets the most from **late-round and post-plant kills**: he's our closer.
""")

md("## 4. Passive or aggressive? Where I stand at 0:15")
code("""
p = pos[pos["is_my_team"]].merge(rf.loc[~rf["is_pistol"], KEY + ["map", "my_team_won"]], on=KEY)
spawn = p[p["time_s"] < 1].groupby(["map", "side"])[["X", "Y"]].median()
at15 = p[(p["time_s"] >= 14.9) & (p["time_s"] < 15.4)].join(spawn.add_suffix("_spawn"), on=["map", "side"])
at15["from_spawn"] = np.hypot(at15["X"] - at15["X_spawn"], at15["Y"] - at15["Y_spawn"])
g15 = at15.groupby(KEY)["from_spawn"]
at15["rank"] = g15.rank(ascending=False, method="first")
at15 = at15[g15.transform("size") == 5]   # all 5 alive at 0:15
at15["position"] = np.select([at15["rank"] == 1, at15["rank"] == 5], ["furthest forward", "furthest back"], "in between")
at15 = at15.merge(mine[KEY + ["name", "net_wpa"]], on=KEY + ["name"])

display(pd.crosstab([at15["name"], at15["side"]], at15["position"], normalize="index").round(2))
me = at15[at15["name"] == "MlgSly"]
me.groupby(["side", "position"]).agg(rounds=("round", "size"), net_wpa_per_round=("net_wpa", "mean"),
                                     round_win=("my_team_won", "mean")).round(3)
""")
md("""
"Forward" here means the furthest from our own spawn at 0:15 (straight-line distance, so it's rough). It's a proxy for who's playing aggressive.
XxcanonboixX shows up as "forward" a lot just because lurking takes him far from spawn.

**For me:** on **CT** I'm worth the most when I'm the one **furthest back** (+6.1 per round), and I'm actually *negative* when I'm the furthest forward (−1.1, only 54 rounds).
On **T** it's the other way round: I'm worth the most when I'm **furthest forward** (+4.8 vs +0.9 in between).
Small samples, but it hints that I should be the **anchor/rotator on CT** and **one of the first in on T**.

> 📝 **My take:** I tend to entry on T because my teammates are mostly taking too long, or not taking enough space.
""")

md("## 5. Mirage: where am I worth the most?")
code("""
km = k[k["map"] == "de_mirage"]
got = km[km["attacker_is_me"] & ~km["victim_is_my_team"]].rename(columns={"attacker_last_place_name": "place"})
lost = km[km["victim_is_me"]].rename(columns={"victim_last_place_name": "place"})
spot = (got.groupby(["my_side", "place"])["wpa"].agg(kills="size", kill_wpa="sum")
        .join(lost.groupby(["my_side", "place"])["wpa"].agg(deaths="size", death_wpa="sum"), how="outer").fillna(0))
spot["net"] = spot["kill_wpa"] + spot["death_wpa"]
spot = spot[(spot["kills"] + spot["deaths"]) >= 10].sort_values("net")
spot.round(2)
""")
code("""
mir_ids = rf.loc[rf["map"] == "de_mirage", "match_id"].unique()
bg = pos[pos["match_id"].isin(mir_ids)].sample(120_000, random_state=1)
mpos = pos[pos["match_id"].isin(mir_ids) & pos["is_me"]]
labels_xy = bg[bg["place"] != ""].groupby("place")[["X", "Y"]].median()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, side in zip(axes, ["T", "CT"]):
    ax.scatter(bg["X"], bg["Y"], s=0.3, color="#e3e3e3", rasterized=True)
    m = mpos[(mpos["side"] == side) & (mpos["time_s"] <= 40)]
    ax.hexbin(m["X"], m["Y"], gridsize=60, cmap="Blues", mincnt=3, alpha=0.8)
    kk = got[got["my_side"] == side]
    dd = lost[lost["my_side"] == side]
    ax.scatter(kk["attacker_X"], kk["attacker_Y"], s=14, color=WIN, edgecolor="white", lw=0.3, label=f"where I got a kill ({len(kk)})")
    ax.scatter(dd["victim_X"], dd["victim_Y"], s=14, color=LOSS, marker="x", lw=1, label=f"where I died ({len(dd)})")
    for place in ["TopofMid", "Middle", "SnipersNest", "PalaceInterior", "TRamp", "Apartments", "BombsiteA",
                  "BombsiteB", "Catwalk", "Connector", "Jungle", "Underpass", "CTSpawn", "Stairs"]:
        if place in labels_xy.index:
            ax.text(*labels_xy.loc[place], place, fontsize=8, ha="center", alpha=0.8)
    ax.set_title(f"Me on Mirage {side}: where I stand (first 0:40), kill and die")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower left", fontsize=8)
save("04_mirage_me")
""")
md("""
Blue = where I spend the first 40 seconds. Green dots = where I got kills, red crosses = where I died.

- **Best spot by far: CT Short/Catwalk** (+5.6 win % in total, 55 kills vs 24 deaths). CT A site and CT spawn (rotating into A) are next.
- **Worst CT spots: Jungle and Middle.** I trade about one for one there, and the deaths cost more than the kills add.
- **On T, Top of Mid is a pure coin flip** (31 kills, 32 deaths, net 0). That's the spot from notebook 02 where we die first to the Window AWP. Middle and Palace are positive for me.

> 📝 **My take:** on Mirage CT I tend to prefer playing where the most action is, so I can gain info (e.g. Short/Connector). I never play B or A anchor. Mid is always smoked off, so I don't play Window either. I'm guessing I go Jungle to rotate.
""")
code("""
# checking "I go Jungle to rotate": when do my CT fights in each spot happen?
ct = km[km["my_side"] == "CT"]
rows = []
for place in ["Catwalk", "Middle", "Connector", "Jungle", "BombsiteA"]:
    d = ct[ct["victim_is_me"] & (ct["victim_last_place_name"] == place)]
    kk = ct[ct["attacker_is_me"] & (ct["attacker_last_place_name"] == place)]
    rows.append(dict(spot=place, my_kills=len(kk), my_deaths=len(d), median_death_time_s=d["time_s"].median(),
                     deaths_that_were_the_opening=d["is_opening"].mean(), deaths_after_plant=d["planted"].mean()))
pd.DataFrame(rows).set_index("spot").round(2)
""")
md("""
**Yes, Jungle is a rotation spot for me.** My Jungle deaths happen late (median ~0:39), **none** of them is the opening death, and about a fifth come after the plant.
Compare Middle: deaths there come early (~0:20) and a quarter are the opening duel. Those are the forward mid fights that cost me.
So the pattern is: **Short = my info spot and where I'm best; Middle = early fights I lose too often; Jungle = mid-rotation fights.**
""")

md("""
## Summary: answer to question 3

1. **I'm the team's biggest impact on CT** (+4.3 win % per round) and about **break-even on T** (+0.5). Sad_X is our best T player (+3.0).
2. **My value comes from mid-round kills**, not opening duels. Inbail is our opener, Sad_X is our closer.
3. **On CT I'm worth the most playing back** (anchor/rotator) and the least when I'm the most forward. On T it flips: I'm worth more going in first.
4. **Mirage:** CT Short is my best spot. CT Jungle/Middle and T Top of Mid are where I trade one for one.

**For the experiment:** on CT, stop taking the forward mid fights and play Short/rotate; on T, be one of the first in, but not through Top of Mid without a plan.

> 📝 **My take, in my own words:** (3–5 bullets)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out)
print("wrote", out)
