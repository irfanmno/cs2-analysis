"""Win-probability model shared by the experiments and the dashboard export.

Same model as notebook 04: logistic regression on the round state (players alive
on each side, side, bomb planted, equipment difference). Each kill's WPA is the
change in my team's win probability it caused (x100).
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

KEY = ["match_id", "round"]


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

