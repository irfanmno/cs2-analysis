# Journal

The messy middle. Short dated entries: what I tried, what broke, what surprised me. Raw notes are fine, and this file is where the portfolio story comes from later.

---

## 2026-11-08: Day 0

- Set up the repo: Python venv, FACEIT fetch script, folder structure.
- Starting point: **139 CS2 matches on FACEIT** (+72 CS:GO, ignoring those).
- Haven't downloaded many demos yet. FACEIT only keeps demos for a limited time, so older matches are probably API stats only. **From now on, save the demo for every match.**
- Next: get a FACEIT API key, run the fetch, see what the data actually looks like.

## 2026-11-08: First data pull

- Got a FACEIT API key and ran `fetch_faceit.py`: **138 matches** (profile says 139, so one is missing), from 2025-06-19 to today. Zero failures.
- Currently level 7, 1398 elo.
- All 138 matches have a `demo_url` in the API response, but that doesn't mean the old demos can still be downloaded. Still to check.

## 2026-11-08: First look notebook

- `src/pipeline/build_dataset.py` turns the raw JSON into one row per match, and `notebooks/exploration/01_first_look.ipynb` explores it. Charts are saved to `figures/`.
- Data gotchas: FACEIT's `premade` flag is always False, so I used repeat teammates instead. 10 matches have level 0, meaning missing.
- Headline: 52.9% win rate, K/D 1.25, and I went from level 4 to 7. "Losing all my games" is... not what the data says?
- My takes: (fill in)

## 2026-11-08: Who carries, and did I stop entrying?

- Me top-fragging is *not* a win condition: we win only 45% of those matches. When the random 5th player top-frags, we win 82% (n=22).
- Sad_X top-frags 30% of games, and I do 41%. Toe_bias is on my team the most (105 games) but top-frags only 7%.
- "I stopped entrying": in the last 15 games I'm actually in *more* opening duels per round, but winning fewer, with a lower K/D and a 40% win rate. Passive play might just mean getting picked first.
- Demo links from the API don't resolve (they aren't public), so I'll need to download demos manually from the match room.

## 2026-11-08: Demo haul

- Applied for FACEIT Downloads API access (FACEIT says a reply can take up to 30 days).
- Meanwhile I downloaded demos through the match room's "Watch demo" button: **68 demos, 9 Sep to 8 Nov, 1,479 rounds**.
- FACEIT keeps demos for about **2 months**. Matches from 8 Sep and earlier have no demo button anymore, so the other 69 matches are stats only. Lesson: save demos as I go.
- Skipped the SMU eSports LAN BO3, which is club play rather than matchmaking and has 2 demos.
- Maps: Mirage 32, Dust2 14, Inferno 13, others ≤4.
- 28 GB on disk, because each demo is kept both compressed and decompressed.

## 2026-11-08: Demo parser

- `src/pipeline/parse_demos.py` turns each demo into 6 tables: rounds, kills, grenades, damage, player_rounds and positions (every alive player sampled twice a second).
- Validated against FACEIT's scoreboard. On a clean demo, K/D/A, headshots, first kills, ADR and the score all match **exactly** for all 10 players.
- Dead ends along the way:
  - ADR was too high at first, because I counted overkill damage (a 400 dmg AWP shot on a 100 HP player). Fixed by using the health actually lost.
  - Some demos are missing round-start events, and some had **server restores** (FACEIT rolled the match back and replayed rounds). I switched to the game's own round counter and keep only the last version of a replayed round.
  - That missed the **final round** of every match: the game jumps straight to the game-over screen at the moment of the win, so the win flag only appears in the first sample after live play ends.
- Result: **64/68 demos match FACEIT exactly (1,416 rounds)**. 4 restore-heavy demos are flagged in `demo_quality.parquet` and excluded.
- First teaser: when we get the opening kill we win **71%** of rounds, vs **30%** when they do. Only **20%** of my team's deaths get traded.

## 2026-11-08: Q1, why do we lose rounds?

- `src/pipeline/features.py` builds one row per round for my team, plus every death with context (nearest teammate, traded or not).
- Rule: only use things known *before* the round is decided (deaths are partly a result of losing).
- **The recurring thing is losing the first fight, untraded.** That happens in 37% of gun rounds, and we win only ~25% of them. In the model it's ×6 odds for winning the opening duel and ×2.7 for a traded first death.
- **"Disjointed" didn't hold up:** our deaths get traded slightly *more* than our opponents', and we die alone slightly less. Everyone at this level trades about 1 in 5.
- **Mirage T, Top of Mid** is our #1 first-death spot (43 times, 21% round win after), mostly to the Window AWP at ~0:11. Dead end: a Window smoke alone barely helps.
- We throw **~20–25% fewer nades before first contact** than our opponents, on both sides.
- Bug caught: the "who goes first" chart had CT labels on the wrong players (shared y-axis). Fixed.

## 2026-11-09: My takes on Q1, plus two quick checks

- Wrote my answers into notebook 02 (and the two open reactions in notebook 01).
- Roles: XxcanonboixX is our **lurk**. That makes his 12% first-death rate on T stand out more, since a lurk shouldn't usually be the first to die.
- Mirage T plan is basically "let's go mid": smoke Top of Mid ~70% of the time, Toe_bias peeks mid with the AWP, I go Top of Mid for space.
- New lead from my gut: on T executes we get **held at the site entrance**. Checkable later (where we die when 3+ of us hit the same site).
- **"Higher floor" theory checked (nb01 §9):** my 10th-percentile ADR (~66) is the highest of the regulars. Win rate climbs with how many of us have K/D ≥ 1: 2 → 28%, 3 → 66%, 4 → 96%. Me top-fragging is a bad sign because it often means I'm the only one doing well.
- **"Should we play more aggressive?" (nb02 §8):** not really. ~70–75% of first fights happen before 0:20 anyway, and timing barely changes who wins them. Winning the first fight and getting it traded is what matters.

## 2026-11-09: Do we get held at the site entrance? (nb02 §9)

- Grouped Mirage/Inferno/Dust2 callouts into each site's **entrance** and **site** zone. A "site hit" starts when 4+ of the T team are alive in one site's zones; every T death until the plant gets tagged entrance or site. I did the same for opponents on T against us, to have something to compare with.
- **Dead end:** we're no worse than anyone. Someone reaches site 88% vs 88%, plant 66% vs 63%, lose 2+ at the entrance 22% vs 23%, T round win 60% vs 60%.
- Busiest entrances for both teams: Mirage Palace, Mirage Apartments, Inferno Banana, Dust2 Long.
- Level 6 vs 7: basically the same. I can't check 5 vs 8 because demos only go back to September.
- First try used 3+ players, which flagged 83% of T rounds as "hits" (just walking past a default spot counted). 4+ is closer to an actual execute. Same conclusion either way.

- Follow-up (my take: *"even when nothing goes wrong before the hit begins, we still struggle"*): split hits by what happened before them. A clean 5v5 hit is a coin flip for **both** teams (T wins the first fight of the hit ~49%, the round 45–49%). We're slightly better than opponents there (lose 2+ at the entrance 26% vs 31%). Going in already up a player = 86% round win. Small flag: when up a player we still lose 2+ at the entrance 17% vs their 9% (n=64 vs 57).

## 2026-11-09: Q2, what enemy utility hurts us most? (notebook 03)

- Landing spots are named by the nearest spot any player stood on (KD-tree on positions), so each nade gets a callout.
- Only nades and damage from **before the first kill** count. Otherwise lost rounds look nade-heavy just because they last longer.
- **Their utility doesn't out-damage ours:** pre-contact nade damage, players blinded per flash and flash-assisted deaths are all about even.
- **Dead end:** taking 40+ chip damage before the first fight looked brutal on CT (38% round win), but with both teams on a full buy it mostly disappears. That was the economy.
- Spot test: win rate with vs without each enemy nade spot, adjusted within buy matchups, with a bootstrap over matches. 50 combos, 4 intervals clear zero (2–3 expected by luck).
- Clearest: **Mirage CT, they smoke Window** (47% of rounds): 65% → 46%, and our first death moves into Middle (Inbail 9×, me 5×). It's the mirror of notebook 02, where *our* Window smoke barely helped. Also Inferno T (they smoke Mid) and Dust2 T (they flash Long).
- No radar image, so the Mirage map background is just every position sample. It works surprisingly well.

- My answers: Mirage CT, when they smoke Window we watch Connector and Short, but Inbail likes to duel mid anyway. Inferno T, when they smoke mid "everything comes to a halt". Dust2 T, we go Long as a team, rain flashes and force it.
- Checked the halt: on Inferno T after a CT mid smoke, the first kill comes ~8 s later (0:22 vs 0:14), plants drop 70% → 54%, wins 63% → 44%. Waiting doesn't help, again.

## 2026-11-09: Q3, where and when do I have the most impact? (notebook 04)

- Built a **win-probability model** (logistic regression on players alive, side, plant, money difference), fitted on the round start and after every kill. Out of sample (grouped by match): AUC 0.87, well calibrated. Each kill and death gets a **win probability added** (WPA). An average kill is worth +0.17.
- **I'm the team's biggest impact on CT** (+4.3 win % per round) but only break-even on T (+0.5). Sad_X is the best on T (+3.0). Toe_bias and XxcanonboixX are negative on T.
- **Kinds of kill:** opening kills are worth the most each (+0.23). Most of my value comes from mid-round kills (the most on the team). Inbail = opener, Sad_X = closer (late-round + post-plant).
- **Passive vs aggressive** (who's furthest from spawn at 0:15, rough): on CT I'm worth most playing furthest back (+6.1) and negative when furthest forward (−1.1, n=54). On T it flips (+4.8 when furthest forward).
- **Mirage:** CT Short/Catwalk is my best spot (55 kills vs 24 deaths). CT Jungle/Middle are net negative. T Top of Mid is a coin flip (31 kills, 32 deaths).
- Bug caught: the first "who's furthest forward" check used a time window holding 2 samples per player, so ranks went 1–10. Fixed to one sample.

- My answers: my strong point is game sense (pre-aiming, knowing what to expect), and my aim is above average but below Sad_X and Inbail. At level 5–6 I could open rounds on pure aim; now opponents' aim *and* game sense are close to mine. On Mirage CT I play where the action is (Short/Connector), never anchor, never Window (mid is always smoked), and Jungle is probably rotating. I entry on T because teammates take too long or don't take enough space.
- Checked: my opening-duel win % slides slowly with level (56% at 4–5 → 55% at 6 → 53% at 7–8). Jungle deaths on CT are late (median 0:39) and never the opening death, so it really is rotations. Middle deaths are early (0:20) and 27% are the opening duel.

## 2026-11-09: Designing the experiment

- Wrote `experiments/test1_team_habits.md` as a pre-registration: habits, metrics, targets, controls and the success rule, fixed *before* playing.
- Habits: **H1** trade the first fight (primary, 22.6% of our first deaths traded → target 35%), **H2** one more nade before the first fight (T 2.3 → 3.0), **H3** me playing back on CT (furthest forward at 0:15: 12% → ≤6%).
- Power: 20 matches gives 93% power for H1 (23% → 35%). Round win would need ~100 matches for a 5-point change, so it's report-only.
- `src/experiments/test1_team_habits.py` scores baseline vs experiment, with a match-level bootstrap and controls (opponents' trade rate and nades in the same matches).
- **Dry run with a fake start date (20 Oct)** on old demos: two metrics "moved" with no experiment at all (my CT forward share 14.5% → 5.4%, our T nades 2.4 → 1.9). Good reminder that habits drift on their own, and the reason the controls exist.

- Tweaks after talking it through: H1 = players pair up. H3 = Inbail (our opener) takes the CT forward pushes when I call them, and I play behind him for the trade and the rotate. H2 = I'm the util lead. Nobody will watch lineup videos, and I already throw the most before contact (0.8 per T round; Toe_bias 0.28, XxcanonboixX 0.02 as lurk), so I aim for ~1.5 myself. Added "my nades before the first kill" to the scorer.

## 2026-11-09: Experiment locked + good vs bad T sides (notebook 05)

- **Experiment locked, starting today (2026-11-09).** `EXPERIMENT_START` is set in `src/experiments/test1_team_habits.py`.
- My question: how do our bad T halves (≤35% won) differ from the opponents' good T halves against us (≥60%)? 16 halves each, plus our good T halves (19) as a reference.
- Economy trap: our bad T halves are 41% force buys vs 16%, so the comparison uses full buys only.
- **Same movement speed** (~140 u/s). It's not about running faster.
- **Clear (interval excludes 0):** we clump (fewer callouts held at 0:20, 2.8 vs 3.2; biggest stack 2.4 vs 2.0), and our utility doesn't come together (most nades in any 5 s: 2.6 vs 3.7).
- **Direction only:** ~1 fewer nade before the first kill, less ground by 0:30, first smoke ~2 s later.
- **Mid smokes:** they smoke mid more often (70% vs 57%), earlier (0:08 vs 0:11), and more of them (1.9 vs 1.3, clear). On Mirage, 87% of our first mid smokes in bad halves are Window. After the mid smoke, the same number of players walk into mid, but they win the next 10 s (0.75 kills : 0.51 deaths) and we lose it (0.58 : 0.72, clear). "Decisive" = more smoke cover, then winning the fight, not speed.
- Mirage at 0:20: nobody on B in our bad halves (2% vs 6–10%), and more players still "in between".

## 2026-11-09: Within-match check, notebook 05 §5 (what survives?)

- Re-measured everything **before the first kill** and compared rounds **within the same match** (fixed effect per match × T team, money control, bootstrap over matches). 1,194 regulation gun rounds, both teams on T.
- **Most of the notebook 05 story disappeared.** Callouts covered, stack size, nades before contact and bundling: no clear link to round or opening win.
- Why it looked real: "callouts at 0:20" predicts winning only when someone already died before 0:20 (+12 pts per callout) and not at all when nobody's dead yet (0). Fewer players alive = fewer callouts. **The clumping was a consequence of losing the first fight, not a cause.** Biggest lesson so far about picking groups by their outcome.
- **Survivor (hypothesis):** 2+ mid smokes before the first kill: +10 pts opening duel (CI +2.5 to +17), +6.7 pts round (CI −1 to +15). One smoke = nothing. Same direction on all 3 maps and for both teams. But I chose the 2+ cut-off after looking → needs a coin-flip test after the current experiment.

- Follow-ups on the 2+ mid smoke finding: it doesn't assume anyone goes mid, but 77% of those rounds have 2+ T players in mid before the first kill (34% with no mid smoke), and the first kill comes later (0:18 vs 0:13). On Mirage the T opening kill then comes from mid 58% of the time (46% otherwise), usually killing the CT mid player. It's every T side: opponents do it in 18% of T rounds, we do it in 10%. Mirage combos: Window + Middle (42), Window + Connector (35), Window + Catwalk (13). Correction: Inferno/Dust2 only have ~11–12 such rounds each, so it's really a Mirage finding.

## 2026-11-09: Week 1 plan

- Realised the data only helps if we play consistently, so week 1 is: same stack and roles, fixed pairs (H1), me as util lead (H2), Inbail pushes on CT while I support (H3), and on Mirage T the default is Window + Connector/Middle smokes together, then 2 take mid.
- Maps: roughly the baseline mix (about half Mirage, the rest Inferno/Dust2), so the comparison stays like for like.
- Report back after 10 qualifying matches (we usually play ~8 a week). Save every demo.
- PROGRESS.md rewritten as a full handover.

## 2026-11-17: Experiment check-in after week 1 (12 qualifying matches)

- Downloaded 16 new demos (9 to 16 Nov) through Chrome. Navigating away from a match room cancels a demo download that's still running, so they have to be done one at a time.
- **Parser broke on the newer demos**: everything from 12 Nov onward failed with `EntityNotFound` (CS2 game update). Upgrading demoparser2 0.42.1 to 0.42.2 fixed it. Pinned in requirements.txt.
- 16 new matches: 12 qualify (3+ regulars, demo passes quality). 3 had only 2 regulars; the 16 Nov Anubis demo fails the quality check (ADR 58 vs FACEIT 50, someone left). Maps: 5 Mirage, 4 Dust2, 2 Anubis, 1 Inferno. Anubis was meant to be avoided.
- One match (9 Nov, 00:49 SGT) was probably played before the plan was agreed later that day. Scoring with and without it gives the same picture, so I've kept the locked start date (2026-11-09).
- **Are we doing the habits? (the 5-match check, not the verdict)**
  - H1 traded first deaths: 22.6% to **20.0%** (n=110). Not happening. **CT got worse: 18.9% to 7.3%** (41 first deaths, interval excludes 0). Our CT opening deaths are more isolated (46% vs 33% with no teammate within 800 units; nearest teammate 774 vs 626 units). It isn't only Inbail dying on his pushes (5 of 41), so it's not just H3 leaving him alone. Small n, direction only.
  - H2 team T nades before the first kill: 2.27 to **2.06**, flat. Mine: 0.80 to **0.93** (target ~1.5). Moving a little, not enough.
  - H3 me furthest forward on CT at 0:15: 12.1% to **9.2%** (target 6%). Zero early mid deaths on Mirage CT (was 4.5%). The most visible change.
  - Controls flat (their trades 19% to 21%, their nades 3.4 to 3.7), so the lobbies didn't change much.
  - Report-only: round win T 49% to 41%, CT 54% to 59%. Means nothing at this n.
- Next: ask what actually happened in-game with the pairs and the util. H1 needs a concrete change for the next 8 matches, or it won't be tested at all.

## 2026-11-17: What actually happened in week 1 (my account + exploratory check)

- My answers: pairs didn't happen, and no coordinated pushes with Inbail. After the first 2 days it was clear it's very hard to get the team to coordinate trades and pushes, so I did it on my own instead: try to trade, more nades on T, stay alive longer on CT. On CT I didn't change the team's setup (mostly 2-1-2), I just tried to stay alive since I'm worth more alive. On Mirage T I pushed for 2 mid smokes instead of 1 whenever I could, and they generally went mid.
- Logged as a deviation in experiments/test1_team_habits.md §5a. Locked metrics stay as they are; the solo metrics below were picked after seeing the data, so they're **exploratory**.
- **T side, I changed:** I'm our first death far less (20% to 7%, clear) and stay alive ~7 s longer (clear). I trade the teammate who died first a bit more often (7% to 13%, not clear). KAST 76% to 81% (not clear). Basically I stopped going in first and became the "+1".
- **CT side, there's a trade-off:** I stay alive ~8 s longer (clear) and survive more rounds (36% to 42%, not clear), but when a teammate dies first I'm further away (1,408 to 1,769 units, clear) and my damage dropped (98 to 83 per round, borderline). Staying alive took me out of trade range. That fits the team's CT trade rate falling to 7%, but it's a guess, not proven.
- **Mirage T mid smokes:** 2+ mid smokes before the first kill 16% to 21% of rounds (38 rounds, so ~8 rounds). But about half of Mirage T rounds still have **no** mid smoke before the first kill (47%, same as before). I throw 56% of our pre-contact mid smokes (was 50%).
- Takeaway for the write-up: team habits didn't stick without someone enforcing them, the individual changes did show up in the data.

## 2026-11-17: Stopping experiment 1, drafting an individual one

- My take: the experiment should switch to individual analysis rather than the team/stack. After week 1 it was clear I can't dictate how everyone plays, because not everyone is willing to listen.
- Also new: I'm playing the **MAC-10 on T** more, because I feel I can make more space for my teammates instead of always being stuck at entrances. And I'm using more utility on T before entering sites (flashes, close-range smokes). I don't know clear lineups, I just throw smokes where I feel they help.
- **Experiment 1 stopped at 12/20** and written up in experiments/test1_team_habits.md §6. The habits were never done, so it only shows that announcing team habits doesn't change how the team plays.
- New `src/pipeline/parse_loadouts.py`: what everyone carried 10 s into each round (about 2 s per demo).
- **The data backs up what I said** (exploratory, `src/analysis/individual_check.py`): on T full buys my SMG share went 4% → 48% (63 MAC-10 rounds vs 53 in the whole baseline). Utility thrown per T round 1.8 → 2.4, flashes before my first fight 0.50 → 0.76, enemies flashed 0.63 → 1.10. My T WPA went +0.8 → +3.8 per round, but I changed two things at once in one week.
- Can't tell from this whether the MAC-10 helps. Week 1: SMG rounds +2.8, rifle rounds +6.2, but only 37 rifle rounds. In the baseline, SMG rounds were mostly force buys. Too mixed up to separate.
- Drafted the test 2 write-up (withdrawn): coin flip on T rounds where I can afford a rifle, MAC-10 vs AK. Power is the catch: my T WPA per round has an sd of 22, so 40 matches only detects a ~6.5-point difference. The likely outcome is "no clear difference", and the draft says that before starting.

## 2026-11-17: Experiment 2 locked

- Locked the test 2 write-up (withdrawn) at 18:00 SGT. One change from the draft: a pre-generated random **card** instead of a real coin, because a coin leaves no record in the demo. Odd matches use the card, even matches flip it, so each round number gets both guns equally.
- "Can afford a rifle" = money + equipment ≥ $3,700 at the start of the round (AK + armour). Added `wealth_at_start` to `parse_loadouts.py`. Above that line I held a rifle 68% of the time and was pistol-only 4%, so the cut-off works.
- Wrote the scorer (`experiment2.py (since removed; its WPA code now lives in src/pipeline/wpa.py)`) before playing. Dry run on old demos with a fake lock (1 Oct): random labels alone gave MAC-10 "+2.8 win % per round" (−0.8 to +6.4). Good reminder of how noisy this is.

## 2026-11-17: Experiment 2 withdrawn

- I never asked for a MAC-10 vs AK test. It came out of me mentioning that I play the MAC-10 more, and it got locked before I understood what the "card" was.
- Also the design was flawed: the gun dictates what I do. If I'm on a MAC-10 on Dust2, I won't peek Long, I'll play somewhere close range. So it can't test the gun on its own.
- Withdrawn before any match counted. What stays: `parse_loadouts.py` (guns + money per round) is useful anyway.
- Lesson for the write-up: a design is only as good as the person playing it understanding it.

## 2026-11-17: Blowouts, and pushing on CT (two of my questions)

- My thoughts (unfiltered): playing CT passively helped me perform better and helped my mental state because I die less. But I feel less impactful. I used to swing hard and get 1-2 kills before dying, which felt very impactful, and since I'm mechanically better than my teammates there may be some waste in not using me more aggressively. Idea: out of 12 rounds, use 2-3 where, given a good spawn, I rush that place, and never rush the same place twice. Second question: when opponents are destroying us, do they do something more often, and is it significant?
- **Blowouts** (`src/analysis/blowouts.py`): 12 matches lost by 7+ vs 23 closer losses vs 44 wins. Only things decided at or before the first kill are compared.
  - Opponents **don't do anything measurably different**. With both teams on full buys: same utility, same nades before contact, same speed, *fewer* AWPs if anything. No level gap either (opponents' average level is the same in all three groups).
  - They win the opening duel a bit more (59% vs 50%, not clear).
  - What stands out is how it snowballs: pistols 25% won (vs 48% in close losses), behind at half in all 12, then the second half collapses (16-18% of rounds won, vs 43-57% in close losses). Our full buys drop to 49% of rounds (vs 64%) and our nades drop with them. That's the economy breaking, a consequence, not a cause.
  - Careful: the groups are picked by the final score, so "the second half goes badly" is partly built into the definition. 12 matches is small.
- **CT pushes** (`src/experiments/test3_ct_rushes.py`, "push" = I'm furthest from spawn at 0:15): only 69 of my 558 CT gun rounds (about 1 per match).
  - Push rounds: my impact +0.5 win % per round vs **+4.8** when I don't push. I die more (69% vs 56%, clear). 2+ kills in 16% of push rounds vs 24% otherwise.
  - "First time at that spot" vs "repeat": only 14 repeats, so the data can't say whether the surprise helps.
  - But past pushes weren't the planned "good spawn, once per spot" kind, so this doesn't test the idea. It only says my *unplanned* pushes haven't paid off.

## 2026-11-17: Trial 3 started (planned CT rushes)

- Agreed to try my CT rush idea for ~20 matches (experiments/test3_ct_rushes.md): when I spawn close to a spot, rush it, 2-3 times per half max, never the same spot twice. Otherwise passive as now.
- Rush rounds get picked up from the demo (furthest forward at 0:15), so I don't have to log anything.

<!-- Template:
## YYYY-MM-DD: short title
- What I did
- What went wrong / surprised me
- Screenshot or number worth keeping
- Next
-->
