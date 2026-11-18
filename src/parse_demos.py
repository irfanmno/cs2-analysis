"""Parse CS2 demos into round-level tables (from my team's point of view).

Usage:
    python src/parse_demos.py              # parse new demos, then combine
    python src/parse_demos.py --refresh    # re-parse everything
    python src/parse_demos.py --only 1-63bb002e-...   # one match (debugging)

Per-match parquet files are cached in data/processed/demos/<table>/<match_id>.parquet,
then combined into data/processed/<table>.parquet:

    rounds        one row per round: winner, reason, sides, economy, bomb plant
    kills         every kill: who/where/when, opening kill, traded or not
    grenades      every smoke/molly/HE/flash/decoy: thrower, throw spot, landing spot, flash victims
    damage        every damage event (player_hurt)
    player_rounds one row per player per round: K/D/A, damage, survived, opening duel, trades, utility
    positions     every alive player sampled twice a second (for heatmaps / "where is everyone at 0:20")
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

ROOT = Path(__file__).resolve().parent.parent
DEMOS = ROOT / "data" / "raw" / "demos"
OUT = ROOT / "data" / "processed"
CACHE = OUT / "demos"
TABLES = ["rounds", "kills", "grenades", "damage", "player_rounds", "positions"]

TICKRATE = 64
SAMPLE_EVERY = 32          # position/state samples: every 0.5 s
TRADE_WINDOW_S = 5         # a death counts as "traded" if the killer dies within 5 s
T, CT = 2, 3
SIDE = {T: "T", CT: "CT"}

# CS2 round end reasons (RoundEndReason_t)
REASONS = {
    1: "bomb_exploded", 7: "bomb_defused", 8: "t_eliminated", 9: "ct_eliminated",
    10: "draw", 12: "time_ran_out", 15: "surrender", 17: "t_surrender", 18: "ct_surrender",
}
NADE_EVENTS = {
    "smokegrenade_detonate": ("smoke", "weapon_smokegrenade"),
    "inferno_startburn": ("molotov", "weapon_molotov|weapon_incgrenade"),
    "hegrenade_detonate": ("he", "weapon_hegrenade"),
    "flashbang_detonate": ("flash", "weapon_flashbang"),
    "decoy_started": ("decoy", "weapon_decoy"),
}
STATE_PROPS = [
    "team_num", "is_alive", "health", "X", "Y", "Z", "last_place_name",
    "current_equip_value", "balance", "round_win_status", "round_win_reason",
]


def my_steamid() -> str:
    player = json.loads((ROOT / "data" / "raw" / "faceit" / "player.json").read_text(encoding="utf-8"))
    return str(player["games"]["cs2"]["game_player_id"])


def buy_type(equip: float, round_no: int, is_ot: bool, side: str) -> str:
    """Rough team buy classification from total team equipment value at the end of freeze time.

    Thresholds are per-player averages; CT gear (M4/kits) costs more, so a CT full buy needs more.
    """
    if round_no in (1, 13) and not is_ot:
        return "pistol"
    per_player = equip / 5
    if per_player < 1500:
        return "eco"
    if per_player < (4000 if side == "CT" else 3500):
        return "force"
    return "full"


def event(parser: DemoParser, name: str, **kw) -> pd.DataFrame:
    try:
        df = parser.parse_event(name, **kw)
    except Exception:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# --------------------------------------------------------------------------------------
# per-demo parsing
# --------------------------------------------------------------------------------------
def last_tick(parser: DemoParser) -> int:
    ticks = [0]
    for name in ("cs_win_panel_match", "round_officially_ended", "player_death", "player_hurt"):
        d = event(parser, name)
        if len(d):
            ticks.append(int(d["tick"].max()))
    return max(ticks) + 10 * TICKRATE


def round_windows(parser: DemoParser) -> pd.DataFrame:
    """Every real round, built from the game's own state rather than events.

    A round = a stretch of live play (not freeze time, not warmup) that ends with a winner.
    Its number = total_rounds_played at the start + 1. Event-based detection breaks on some
    demos (missing freeze_end events, and FACEIT restoring a backup after a server issue, which
    replays rounds). Here, aborted stretches never get a winner and are dropped, and if a round
    number was played more than once, only the last (official) version is kept. This also drops
    the knife round, which is "round 1" before the real round 1.
    """
    ticks = list(range(0, last_tick(parser), SAMPLE_EVERY))
    gs = parser.parse_ticks(
        ["total_rounds_played", "is_freeze_period", "is_warmup_period", "round_win_status", "round_win_reason"],
        ticks=ticks,
    )
    gs = gs.drop_duplicates("tick").sort_values("tick").reset_index(drop=True)
    gs = gs[~gs["is_warmup_period"].astype(bool)]
    live = ~gs["is_freeze_period"].astype(bool)
    run_id = (live != live.shift()).cumsum()

    gs = gs.reset_index(drop=True)
    live = live.reset_index(drop=True)
    run_id = run_id.reset_index(drop=True)
    segs = []
    for _, g in gs[live].groupby(run_id[live]):
        # the final round flips straight into the game-over (freeze) state at the moment it is won,
        # so also look at the first sample after live play ends
        nxt = gs.loc[g.index[-1] + 1: g.index[-1] + 1]
        decided = pd.concat([g, nxt])
        decided = decided[decided["round_win_status"].isin([T, CT])]
        if decided.empty:
            continue
        d = decided.iloc[0]
        segs.append({
            "round": int(g["total_rounds_played"].iloc[0]) + 1,
            "start_tick": int(g["tick"].iloc[0]),
            "end_tick": int(d["tick"]),
            "seg_end_tick": int(g["tick"].iloc[-1]) + SAMPLE_EVERY,
            "winner": int(d["round_win_status"]),
            "reason_code": int(d["round_win_reason"]),
        })
    rw = pd.DataFrame(segs)
    if rw.empty:
        return rw
    rw = rw.drop_duplicates("round", keep="last").sort_values("start_tick").reset_index(drop=True)
    return rw


def assign_round(ticks: pd.Series, rw: pd.DataFrame) -> pd.Series:
    """Map each tick to the round whose live window contains it (NaN if none)."""
    starts = rw["start_tick"].to_numpy()
    idx = np.searchsorted(starts, ticks.to_numpy(), side="right") - 1
    ok = (idx >= 0)
    idx_c = np.clip(idx, 0, len(rw) - 1)
    ok &= ticks.to_numpy() < rw["seg_end_tick"].to_numpy()[idx_c]
    return pd.Series(np.where(ok, rw["round"].to_numpy()[idx_c], np.nan), index=ticks.index)


def parse_demo(path: Path, me: str) -> dict[str, pd.DataFrame]:
    match_id = path.name.split("-1-1")[0]
    parser = DemoParser(str(path))
    header = parser.parse_header()
    map_name = header.get("map_name", "")

    rw = round_windows(parser)
    if rw.empty:
        raise ValueError("no rounds found")

    # ---- tick samples: state during every round ---------------------------------------
    sample_ticks = sorted({
        int(t)
        for r in rw.itertuples()
        for t in range(r.start_tick + 1, r.seg_end_tick, SAMPLE_EVERY)
    })
    st = parser.parse_ticks(STATE_PROPS, ticks=sample_ticks)
    st["steamid"] = st["steamid"].astype(str)
    st = st[st["team_num"].isin([T, CT])]
    st["round"] = assign_round(st["tick"], rw)
    st = st.dropna(subset=["round"])
    st["round"] = st["round"].astype(int)

    # ---- rounds -------------------------------------------------------------------------
    first = st[st["tick"].isin(st.groupby("round")["tick"].min())]  # first sample after freeze end
    my_team_by_round = first[first["steamid"] == me].set_index("round")["team_num"]

    rounds = []
    for r in rw.itertuples():
        if r.round not in my_team_by_round.index:
            continue
        my_team = int(my_team_by_round[r.round])
        equip = first[first["round"] == r.round].groupby("team_num")["current_equip_value"].sum()
        is_ot = r.round > 24
        rounds.append({
            "match_id": match_id, "map": map_name, "round": r.round,
            "half": 1 if r.round <= 12 else (2 if r.round <= 24 else 3),
            "is_pistol": r.round in (1, 13), "is_overtime": is_ot,
            "start_tick": r.start_tick, "end_tick": r.end_tick, "seg_end_tick": r.seg_end_tick,
            "duration_s": round((r.end_tick - r.start_tick) / TICKRATE, 1),
            "my_side": SIDE[my_team],
            "winner_side": SIDE[r.winner],
            "my_team_won": r.winner == my_team,
            "reason_code": r.reason_code,
            "reason": REASONS.get(r.reason_code, str(r.reason_code)),
            "my_team_equip": float(equip.get(my_team, np.nan)),
            "enemy_equip": float(equip.get(T if my_team == CT else CT, np.nan)),
        })
    rounds = pd.DataFrame(rounds).sort_values("round").reset_index(drop=True)
    enemy_side = rounds["my_side"].map({"T": "CT", "CT": "T"})
    rounds["my_team_buy"] = [buy_type(e, r, o, s) for e, r, o, s in
                             zip(rounds["my_team_equip"], rounds["round"], rounds["is_overtime"], rounds["my_side"])]
    rounds["enemy_buy"] = [buy_type(e, r, o, s) for e, r, o, s in
                           zip(rounds["enemy_equip"], rounds["round"], rounds["is_overtime"], enemy_side)]
    won = rounds["my_team_won"].fillna(False).astype(int)
    rounds["my_score_before"] = won.cumsum() - won
    rounds["enemy_score_before"] = (1 - won).cumsum() - (1 - won)

    # helpers to attach round/time/side info to any event
    starts = rw.set_index("round")["start_tick"]
    ends = rounds.set_index("round")["end_tick"]
    my_team_num = my_team_by_round.to_dict()

    def tag(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["round"] = assign_round(df["tick"], rw)
        df = df.dropna(subset=["round"])
        df["round"] = df["round"].astype(int)
        df = df[df["round"].isin(rounds["round"])]
        df["time_s"] = ((df["tick"] - df["round"].map(starts)) / TICKRATE).round(2)
        df["after_round_end"] = df["tick"] > df["round"].map(ends)
        df.insert(0, "match_id", match_id)
        return df

    def side_cols(df: pd.DataFrame, prefix: str) -> None:
        team = df[f"{prefix}_team_num"]
        df[f"{prefix}_side"] = team.map(SIDE)
        df[f"{prefix}_is_my_team"] = team == df["round"].map(my_team_num)
        df[f"{prefix}_steamid"] = df[f"{prefix}_steamid"].astype(str)
        df[f"{prefix}_is_me"] = df[f"{prefix}_steamid"] == me

    # ---- kills --------------------------------------------------------------------------
    kills = tag(event(parser, "player_death", player=["X", "Y", "Z", "team_num", "last_place_name"]))
    keep = [
        "match_id", "round", "tick", "time_s", "after_round_end",
        "attacker_steamid", "attacker_name", "attacker_team_num", "attacker_last_place_name", "attacker_X", "attacker_Y",
        "user_steamid", "user_name", "user_team_num", "user_last_place_name", "user_X", "user_Y",
        "assister_steamid", "assister_name", "assistedflash",
        "weapon", "headshot", "thrusmoke", "attackerblind", "noscope", "penetrated", "distance",
    ]
    kills = kills[[c for c in keep if c in kills.columns]].rename(columns=lambda c: c.replace("user_", "victim_"))
    # world deaths (bomb, fall damage) and suicides have no real attacker
    no_attacker = kills["attacker_steamid"].isna() | kills["attacker_name"].isna()
    kills["attacker_team_num"] = kills["attacker_team_num"].fillna(0)
    side_cols(kills, "attacker")
    side_cols(kills, "victim")
    kills["assister_steamid"] = kills["assister_steamid"].astype(str)
    kills["is_world"] = no_attacker | (kills["attacker_steamid"] == kills["victim_steamid"])
    kills["is_teamkill"] = ~kills["is_world"] & (kills["attacker_team_num"] == kills["victim_team_num"])
    kills = kills.sort_values("tick").reset_index(drop=True)
    kills["kill_no"] = kills.groupby("round").cumcount() + 1
    kills["is_opening"] = kills["kill_no"] == 1

    # trades: victim's death is "traded" if the killer is killed by the victim's team within the window
    kills["traded"] = False
    kills["is_trade_kill"] = False
    window = TRADE_WINDOW_S * TICKRATE
    for i, k in kills.iterrows():
        if k["is_world"] or k["is_teamkill"]:
            continue
        later = kills[(kills["round"] == k["round"]) & (kills["tick"] > k["tick"]) & (kills["tick"] <= k["tick"] + window)
                      & (kills["victim_steamid"] == k["attacker_steamid"]) & (kills["attacker_team_num"] == k["victim_team_num"])]
        if len(later):
            kills.at[i, "traded"] = True
            kills.at[later.index[0], "is_trade_kill"] = True

    # ---- damage -------------------------------------------------------------------------
    dmg = tag(event(parser, "player_hurt", player=["team_num"]))
    dmg = dmg[["match_id", "round", "tick", "time_s", "after_round_end", "attacker_steamid", "attacker_name",
               "attacker_team_num", "user_steamid", "user_name", "user_team_num", "dmg_health", "health", "weapon", "hitgroup"]]
    dmg = dmg.rename(columns=lambda c: c.replace("user_", "victim_"))
    dmg["attacker_team_num"] = dmg["attacker_team_num"].fillna(0)
    side_cols(dmg, "attacker")
    side_cols(dmg, "victim")
    # damage actually dealt = victim's health before the hit minus health after (caps overkill)
    dmg = dmg.sort_values("tick").reset_index(drop=True)
    prev_hp = dmg.groupby(["round", "victim_steamid"])["health"].shift(1).fillna(100)
    dmg["dmg_real"] = (prev_hp - dmg["health"]).clip(lower=0)

    # ---- grenades -----------------------------------------------------------------------
    fires = event(parser, "weapon_fire", player=["X", "Y", "Z", "last_place_name"])
    fires = fires[fires["weapon"].str.contains("grenade|flashbang|molotov|decoy", na=False)].copy() if len(fires) else fires
    if len(fires):
        fires["user_steamid"] = fires["user_steamid"].astype(str)
    blind = event(parser, "player_blind", player=["team_num"])
    nades = []
    for ev_name, (kind, weapons) in NADE_EVENTS.items():
        d = event(parser, ev_name, player=["team_num"])
        if d.empty:
            continue
        d = d.copy()
        d["type"] = kind
        d["user_steamid"] = d["user_steamid"].astype(str)
        # throw position = thrower's latest matching weapon_fire before detonation
        throw = []
        for row in d.itertuples():
            f = fires[(fires["user_steamid"] == row.user_steamid) & (fires["tick"] < row.tick)
                      & fires["weapon"].str.fullmatch(weapons)] if len(fires) else fires
            f = f.iloc[-1] if len(f) else None
            throw.append({
                "throw_tick": f["tick"] if f is not None else np.nan,
                "throw_X": f["user_X"] if f is not None else np.nan,
                "throw_Y": f["user_Y"] if f is not None else np.nan,
                "throw_place": f["user_last_place_name"] if f is not None else None,
            })
        d = pd.concat([d.reset_index(drop=True), pd.DataFrame(throw)], axis=1)
        if kind == "flash" and len(blind):
            b = blind.copy()
            b["enemy"] = b["attacker_team_num"] != b["user_team_num"]
            agg = b.groupby("entityid").apply(lambda g: pd.Series({
                "enemies_flashed": int((g["enemy"] & (g["blind_duration"] > 0.5)).sum()),
                "teammates_flashed": int((~g["enemy"] & (g["blind_duration"] > 0.5)).sum()),
                "enemy_blind_s": float(g.loc[g["enemy"], "blind_duration"].sum()),
            }), include_groups=False)
            d = d.merge(agg, left_on="entityid", right_index=True, how="left")
            flash_cols = ["enemies_flashed", "teammates_flashed", "enemy_blind_s"]
            d[flash_cols] = d[flash_cols].fillna(0)  # flash that blinded nobody
        nades.append(d)
    grenades = pd.concat(nades, ignore_index=True) if nades else pd.DataFrame(columns=["tick"])
    grenades = tag(grenades.rename(columns={"x": "land_X", "y": "land_Y", "z": "land_Z"}))
    grenades = grenades.rename(columns={"user_steamid": "thrower_steamid", "user_name": "thrower_name",
                                        "user_team_num": "thrower_team_num"})
    side_cols(grenades, "thrower")
    grenades = grenades.drop(columns=["entityid", "user_last_place_name"], errors="ignore")

    # ---- bomb ---------------------------------------------------------------------------
    plants = tag(event(parser, "bomb_planted", player=["last_place_name"]))
    if len(plants):
        p = plants.groupby("round").first()
        rounds["bomb_planted"] = rounds["round"].isin(p.index)
        rounds["plant_time_s"] = rounds["round"].map(p["time_s"])
        rounds["plant_site"] = rounds["round"].map(p["user_last_place_name"]).str.replace("Bombsite", "", regex=False)
    else:
        rounds["bomb_planted"], rounds["plant_time_s"], rounds["plant_site"] = False, np.nan, None

    # ---- positions (alive players only) -------------------------------------------------
    pos = st[st["is_alive"].astype(bool)].copy()
    pos.insert(0, "match_id", match_id)
    pos["time_s"] = ((pos["tick"] - pos["round"].map(starts)) / TICKRATE).round(2)
    pos["side"] = pos["team_num"].map(SIDE)
    pos["is_my_team"] = pos["team_num"] == pos["round"].map(my_team_num)
    pos["is_me"] = pos["steamid"] == me
    pos = pos[["match_id", "round", "tick", "time_s", "steamid", "name", "side", "is_my_team", "is_me",
               "X", "Y", "Z", "last_place_name", "health", "current_equip_value"]].rename(columns={"last_place_name": "place"})

    # ---- player_rounds ------------------------------------------------------------------
    roster = first[["round", "steamid", "name", "team_num", "current_equip_value"]].copy()
    roster = roster[roster["round"].isin(rounds["round"])]
    roster.insert(0, "match_id", match_id)
    roster["side"] = roster["team_num"].map(SIDE)
    roster["is_my_team"] = roster["team_num"] == roster["round"].map(my_team_num)
    roster["is_me"] = roster["steamid"] == me
    key = ["round", "steamid"]
    real = kills[~kills["is_world"] & ~kills["is_teamkill"]]

    def count(df, col, name, cond=None):
        d = df if cond is None else df[cond]
        return d.groupby(["round", col]).size().rename(name).rename_axis(key)

    pr = roster.set_index(key)
    pr = pr.join(count(real, "attacker_steamid", "kills"))
    pr = pr.join(count(kills, "victim_steamid", "deaths"))
    pr = pr.join(count(real[real["assister_steamid"] != "None"], "assister_steamid", "assists"))
    pr = pr.join(count(real, "attacker_steamid", "opening_kills", real["is_opening"]))
    pr = pr.join(count(kills, "victim_steamid", "opening_deaths", kills["is_opening"]))
    pr = pr.join(count(kills, "victim_steamid", "traded_deaths", kills["traded"]))
    pr = pr.join(count(real, "attacker_steamid", "trade_kills", real["is_trade_kill"]))
    pr = pr.join(count(real, "attacker_steamid", "headshot_kills", real["headshot"]))
    enemy_dmg = dmg[dmg["attacker_team_num"] != dmg["victim_team_num"]]
    pr = pr.join(enemy_dmg.groupby(["round", "attacker_steamid"])["dmg_real"].sum().rename("damage").rename_axis(key))
    death_time = kills.groupby(["round", "victim_steamid"])["time_s"].min().rename("death_time_s").rename_axis(key)
    pr = pr.join(death_time)
    if len(grenades):
        g = grenades.pivot_table(index=["round", "thrower_steamid"], columns="type", values="tick", aggfunc="count", fill_value=0)
        g.columns = [f"{c}_thrown" for c in g.columns]
        pr = pr.join(g.rename_axis(key))
        if "enemies_flashed" in grenades:
            pr = pr.join(grenades.groupby(["round", "thrower_steamid"])["enemies_flashed"].sum().rename_axis(key))
    pr = pr.reset_index()
    num_cols = [c for c in pr.columns if c.endswith(("kills", "deaths", "assists", "_thrown")) or c in ("damage", "enemies_flashed")]
    pr[num_cols] = pr[num_cols].fillna(0)
    pr["survived"] = pr["deaths"] == 0
    pr["kast"] = (pr["kills"] > 0) | (pr["assists"] > 0) | pr["survived"] | (pr["traded_deaths"] > 0)
    pr = pr.merge(rounds[["round", "my_team_won", "winner_side"]], on="round", how="left")
    pr["won_round"] = pr["side"] == pr["winner_side"]
    pr = pr.drop(columns=["team_num", "winner_side", "my_team_won"]).rename(columns={"current_equip_value": "equip_value"})

    return {"rounds": rounds, "kills": kills, "grenades": grenades, "damage": dmg,
            "player_rounds": pr, "positions": pos}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def cached(match_id: str) -> bool:
    return all((CACHE / t / f"{match_id}.parquet").exists() for t in TABLES)


def work(path_str: str, me: str) -> tuple[str, str | None]:
    path = Path(path_str)
    match_id = path.name.split("-1-1")[0]
    try:
        tables = parse_demo(path, me)
        for name, df in tables.items():
            (CACHE / name).mkdir(parents=True, exist_ok=True)
            df.to_parquet(CACHE / name / f"{match_id}.parquet", index=False)
        n = len(tables["rounds"])
        return match_id, f"{n} rounds"
    except Exception:
        return match_id, "ERROR\n" + traceback.format_exc()


def combine() -> None:
    for name in TABLES:
        parts = sorted((CACHE / name).glob("*.parquet"))
        if not parts:
            continue
        df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"  {name:<14} {len(df):>9,} rows -> {OUT / (name + '.parquet')}")


def quality_check() -> None:
    """Compare each parsed demo with FACEIT's official scoreboard and flag the ones that don't line up.

    A few demos contain server restores (FACEIT rolled the match back after a technical issue),
    which can't always be untangled reliably. Analyses should use matches with ok == True.
    """
    matches_path = OUT / "matches.parquet"
    if not matches_path.exists():
        print("  (skipping quality check: run build_dataset.py first)")
        return
    rounds = pd.read_parquet(OUT / "rounds.parquet")
    pr = pd.read_parquet(OUT / "player_rounds.parquet")
    faceit = pd.read_parquet(matches_path).set_index("match_id")
    rows = []
    for mid, g in rounds.groupby("match_id"):
        me = pr[(pr["match_id"] == mid) & pr["is_me"]]
        n = g["round"].nunique()
        f = faceit.loc[mid] if mid in faceit.index else None
        row = {
            "match_id": mid, "map": g["map"].iloc[0], "rounds": n,
            "faceit_rounds": int(f["rounds"]) if f is not None else None,
            "wins": int(g["my_team_won"].sum()), "faceit_wins": int(f["my_score"]) if f is not None else None,
            "kills": int(me["kills"].sum()), "faceit_kills": int(f["kills"]) if f is not None else None,
            "adr": round(me["damage"].sum() / n, 1), "faceit_adr": float(f["adr"]) if f is not None else None,
        }
        row["ok"] = (f is not None and row["rounds"] == row["faceit_rounds"] and row["wins"] == row["faceit_wins"]
                     and abs(row["adr"] - row["faceit_adr"]) <= 1.5)
        rows.append(row)
    q = pd.DataFrame(rows)
    q.to_parquet(OUT / "demo_quality.parquet", index=False)
    print(f"  quality: {q['ok'].sum()}/{len(q)} demos match FACEIT (rounds, score, ADR) -> demo_quality.parquet")
    bad = q[~q["ok"]]
    if len(bad):
        print(bad[["match_id", "rounds", "faceit_rounds", "wins", "faceit_wins", "adr", "faceit_adr"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--only", help="match id to parse")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    me = my_steamid()
    demos = sorted(DEMOS.glob("*.dem"))
    if args.only:
        demos = [d for d in demos if d.name.startswith(args.only)]
    todo = [d for d in demos if args.refresh or not cached(d.name.split("-1-1")[0])]
    print(f"{len(demos)} demos, {len(todo)} to parse")

    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, str(d), me): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            match_id, msg = fut.result()
            errors += msg.startswith("ERROR")
            print(f"[{i}/{len(todo)}] {match_id[:10]} {msg}", flush=True)

    print("combining...")
    combine()
    quality_check()
    if errors:
        print(f"{errors} demo(s) failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
