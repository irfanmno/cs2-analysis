"""What everyone carried each round: primary weapon class + utility, 10 s after the round goes live.

Output: data/processed/loadouts.parquet (match_id, round, steamid, name, primary, weapon_class, n_util, inventory,
wealth_at_start = money + equipment value on the first live tick, i.e. what the player could spend that round).
Cached per match in data/interim/loadouts/. Run after parse_demos.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

ROOT = Path(__file__).resolve().parent.parent
DEMOS = ROOT / "data" / "raw" / "demos"
P = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "interim" / "loadouts"
TICKRATE = 64
AT_S = 10   # buy time is still open right after freeze end; 10 s in, the buy is done

CLASSES = {
    "smg": {"MAC-10", "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon"},
    "rifle": {"AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "SG 553", "AUG"},
    "awp": {"AWP"},
    "other_sniper": {"SSG 08", "SCAR-20", "G3SG1"},
    "heavy": {"Nova", "XM1014", "Sawed-Off", "MAG-7", "M249", "Negev"},
}
UTIL = {"Flashbang", "Smoke Grenade", "High Explosive Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade"}


def classify(inv: list[str]) -> tuple[str, str]:
    for cls, names in CLASSES.items():
        for item in inv:
            if item in names:
                return item, cls
    return "", "pistol_only"


def parse_one(match_id: str, starts: pd.DataFrame) -> pd.DataFrame:
    path = DEMOS / f"{match_id}-1-1.dem"
    parser = DemoParser(str(path))
    ticks = [int(t) + AT_S * TICKRATE for t in starts["start_tick"]]
    df = parser.parse_ticks(["inventory", "team_num"], ticks=ticks)
    df = df[df["team_num"].isin([2, 3])]
    tick_to_round = dict(zip(ticks, starts["round"]))
    df["round"] = df["tick"].map(tick_to_round)
    df["inventory"] = df["inventory"].map(lambda x: list(x) if x is not None else [])
    cls = df["inventory"].map(classify)
    df["primary"] = cls.str[0]
    df["weapon_class"] = cls.str[1]
    df["n_util"] = df["inventory"].map(lambda inv: sum(i in UTIL for i in inv))
    df["match_id"] = match_id
    df["steamid"] = df["steamid"].astype(str)

    t0 = [int(t) + 1 for t in starts["start_tick"]]
    w = parser.parse_ticks(["balance", "current_equip_value", "team_num"], ticks=t0)
    w = w[w["team_num"].isin([2, 3])]
    w["round"] = w["tick"].map(dict(zip(t0, starts["round"])))
    w["steamid"] = w["steamid"].astype(str)
    w["wealth_at_start"] = w["balance"].fillna(0) + w["current_equip_value"].fillna(0)
    df = df.merge(w[["round", "steamid", "wealth_at_start"]], on=["round", "steamid"], how="left")
    return df[["match_id", "round", "steamid", "name", "primary", "weapon_class", "n_util", "inventory", "wealth_at_start"]]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rounds = pd.read_parquet(P / "rounds.parquet")[["match_id", "round", "start_tick"]]
    parts = []
    for match_id, starts in rounds.groupby("match_id"):
        out = CACHE / f"{match_id}.parquet"
        if not out.exists():
            if not (DEMOS / f"{match_id}-1-1.dem").exists():
                continue
            parse_one(match_id, starts).to_parquet(out, index=False)
            print("parsed", match_id[:10])
        parts.append(pd.read_parquet(out))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(P / "loadouts.parquet", index=False)
    print(f"{len(df):,} player-rounds -> {P / 'loadouts.parquet'}")


if __name__ == "__main__":
    main()
