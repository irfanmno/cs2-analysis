# The experiment: can changing a few habits win us more rounds?

**Status: STOPPED EARLY on 2026-11-17 after 12 of 20 matches; the team habits were never adopted (see §5a and §6).** Originally **LOCKED on 2026-11-09. The experiment starts today (2026-11-09, SGT).** Nothing in sections 1–4 changes from here. (That is the point of writing it down first: no moving the goalposts after seeing the results.) 

## 0. Why these habits

From notebooks 02–04 (64 demos, Sep–Nov 2026):

- The recurring thing in lost rounds is **losing the first fight untraded**: 37% of gun rounds, and we win only ~25% of those. A traded first death nearly doubles our chances. Only **23%** of our first deaths get traded.
- We throw **~20–25% fewer nades before the first fight** than our opponents, on both sides (T: 2.3 vs their 2.9).
- On CT, **I'm worth the most playing back/rotating** (+6.1 win % per round) and **negative when I'm the furthest forward** (−1.1). Short is my best spot; early mid fights are where I lose value.

Things that turned out *not* to be the problem (so we're not changing them): being "disjointed" in general, getting held at site entrances, flash or nade damage on its own.

## 1. The habits

| | Habit | Who | What it looks like in a game |
|---|---|---|---|
| **H1** (primary) | **Trade the first fight** | whole team | Players pair up. Whoever takes the first fight has their pair right behind them, close enough to swing within ~2 s. On T: "entry + 1", never a solo peek. On CT: no solo off-angle fights in the first 20 s. |
| **H2** | **One more nade before the first fight** | me as the util lead, plus whoever I can get on board | On T, nades go in *before* someone peeks. I already throw the most (0.8 per T round, but zero in about half my T rounds), so I take on most of the extra myself (target ~1.5 per T round) and call the rest. No homework videos: 1–2 lineups taught in a short custom game if people are up for it. |
| **H3** | **Me: play back and support on CT** | me (+ Inbail) | Inbail, our best duelist and opener, takes the forward pushes on CT when I call them. I play behind him (Short/Connector) for info, the trade and the rotate. No forward mid duels for me in the first 20 s. |

## 2. What gets measured (decided before playing)

Scored by `src/experiment.py`. Baseline = every quality-checked demo before 2026-11-09 with 3+ regulars playing (**61 matches, 1,234 gun rounds**).

| Metric | Baseline | Target | Control (should *not* move) |
|---|---|---|---|
| **H1: share of our first deaths that get traded** (primary) | **22.6%** (T 26%, CT 19%) | 35% | opponents' first deaths traded: 19.3% |
| H2: our nades before the first kill, T / CT | 2.27 / 2.81 per round | T ≥ 3.0 | opponents' nades before the first kill: 3.40 |
| H2: *my* nades before the first kill, T | 0.80 per round | ~1.5 | none (shows whether the extra comes from me or the team) |
| H3: me furthest forward at 0:15 on CT | 12.1% of CT rounds | ≤ 6% | none |
| H3: me dying in mid before 0:30 (Mirage CT) | 4.5% of rounds | lower | none (too rare to test, report only) |
| Outcome: untraded first death | 36.9% of gun rounds | lower | none |
| Outcome: round win, T / CT | 48.7% / 53.7% | report only | none |
| Outcome: match win | 52.9% (all 138 FACEIT matches) | report only | none |

**Success rule (H1, H2, H3 separately):** the habit "worked" if the 95% bootstrap interval for the change (resampling whole matches) is entirely on the right side of zero, **and** the control metric didn't move the same way. If the control moved too, it's the lobby that changed, not us.

**Round and match win rate are report-only.** To detect a 5-point round-win improvement with 80% power we'd need about **100 matches**, and match win rate is hopeless at any size we'll realistically play. I'll show them, but I won't claim the habits "won us games" from them.

## 3. How long

**20 qualifying matches with demos** (3+ regulars, demo passes the quality check). That's ~400 gun rounds and ~190 of our first deaths.

Power (two-sided, α = 0.05, vs the current baseline):

| Effect | Power at 20 matches |
|---|---|
| H1: traded first deaths 23% → 35% | 93% |
| H1: 23% → 30% (smaller effect) | 58% |
| H2: +0.75 nades per round (T, matches opponents) | 97% |
| H2: +0.5 nades per round | 74% |

## 4. Rules while it's running

1. **Save every demo** straight after the match (FACEIT deletes them after ~2 months; Downloads API still pending).
2. Every 5 matches, run the pipeline and `python src/experiment.py --start <date>` to check we're actually *doing* the habits (H1–H3 numbers). **The decision is made only at 20 matches**, not when it looks good.
3. Matches with fewer than 3 regulars are played as normal but don't count.
4. A JOURNAL entry every 5 matches: what we tried, what felt different, anything that went wrong.

## 5. Things that could fool me

- **Natural drift.** Dry run: splitting the *old* demos at a fake start date (20 Oct, 19 matches after it), with no experiment at all, already "moved" two metrics. Me furthest forward on CT dropped 14.5% → 5.4%, and our T nades dropped 2.4 → 1.9. Habits drift on their own (I'd already started playing more passive). That's why every metric gets judged by its interval, and H1/H2 have controls.
- **The lobbies get harder** as elo goes up. The controls (what opponents do in the same matches) catch some of this.
- **Map mix.** If we suddenly play more Inferno, numbers shift. Check the map split at the end.
- **Trying harder because we're being measured** (Hawthorne effect). Honestly, that's fine: the point is whether the habit helps when we do it.
- **Small samples on single maps or players.** No per-player or per-map claims unless the sample is big.

## 5a. Deviations from the plan (logged as they happen; sections 1–4 stay as written)

**2026-11-17 (after 12 qualifying matches).** The team versions of H1 and H3 never really started. Pairs didn't happen, and there were no coordinated CT pushes with Inbail. After about two days it was clear it was very hard to get the team coordinated on trades and pushes. So I switched to doing it **on my own**: trying to trade, throwing more nades on T, and staying alive longer on CT. On CT the team kept its usual 2-1-2 setup. On Mirage T I pushed for 2 mid smokes instead of 1 whenever I could, and people generally did go mid after them.

What this means for scoring:
- The locked metrics are still scored exactly as planned at 20 matches. They now answer a narrower question: *did announcing the plan change what the team does?* (So far: not for H1 and H2-team.)
- Metrics about my solo versions (do I trade the first death, how long I stay alive, how far I am from the first death) were **picked after seeing the week 1 data**. They get reported as exploratory, never as the experiment's result.

## 6. Results (stopped early at 12 qualifying matches, 2026-11-17)

**I stopped this experiment at 12 of 20 matches because the habits never got done.** Pairs didn't happen and there were no coordinated CT pushes. I couldn't make everyone play a certain way, since not everyone is willing to listen. With no team habits in place, 8 more matches would only measure the same thing again. This is my call, made after seeing the week 1 numbers, and it's written down here so it's visible.

Scored exactly as locked (`src/experiment.py --start 2026-11-09`, 12 matches, 224 gun rounds):

| Metric | Baseline | Experiment | Change (95% interval) | Verdict |
|---|---|---|---|---|
| **H1 first deaths traded** (primary) | 22.6% | 20.0% | −2.6 pts (−9.0 to +4.2) | no change |
| H1 CT only | 18.9% | 7.3% | −11.6 pts (−19.1 to −3.8) | worse |
| H1 control: their first deaths traded | 19.3% | 21.1% | +1.7 pts (−6.3 to +8.6) | flat |
| H2 our T nades before first kill | 2.27 | 2.06 | −0.22 (−0.70 to +0.32) | no change |
| H2 my T nades before first kill | 0.80 | 0.93 | +0.14 (−0.07 to +0.38) | not clear |
| H2 control: their nades before first kill | 3.40 | 3.67 | +0.27 (−0.32 to +0.77) | flat |
| H3 me furthest forward on CT at 0:15 | 12.1% | 9.2% | −2.9 pts (−10.0 to +4.8) | not clear (target 6%) |
| Round win T / CT (report only) | 48.7% / 53.7% | 40.6% / 58.5% | | underpowered |

**What it says:** announcing team habits didn't change how the team plays. The test was never really run: H1 and H3 weren't done, so this says nothing about whether *trading more* would help. It only shows that telling five friends to trade more doesn't make it happen. CT trades got worse, likely partly because I played for survival and was further from where teammates died (exploratory, see JOURNAL 2026-11-17).

**What I learned:** a habit experiment has to target something I control myself. That's where the next one starts (`EXPERIMENT_2.md`).
