# CS2 analysis: how to present it (read this when writing the website)

_The framing for the portfolio site. Keep it updated as tests finish. Details live in JOURNAL.md (diary, dead ends), experiments/*.md (each test) and PROGRESS.md (current state)._

## The framing (my words, 2026-11-17)

This isn't a typical one-off data analytics project. It's a **live project that will likely stay with me for as long as I play CS**. The way to present it: **a long-running A/B testing project on myself.** I change one thing about how I play, measure it from my own demos, and keep or drop it based on what the data says. It also gives me direction while gaming.

How it got here (the arc so far):
1. **Analysis** (notebooks 01-05): why do we lose rounds? Mostly losing the first fight untraded. Plenty of dead ends along the way (see JOURNAL).
2. **Team experiment** (experiments/test1_team_habits.md): habits for the whole stack. **Failed**, because I couldn't get five people to change how they play ("not everyone is willing to listen"). Stopped early and written up honestly.
3. **The lesson:** test things *I* control. From here on, tests are about my own play.
4. **Ongoing tests on myself** (log below).

## Test log

| # | Test | When | Status | Result |
|---|---|---|---|---|
| 1 | Team habits: trade the first fight, more nades, me playing back on CT | 9-17 Nov 2026 | Stopped at 12/20 matches | Team never adopted the habits. My solo changes (playing back on CT, more util) showed up in the data. |
| 2 | MAC-10 vs AK, decided by a random card | 17 Nov 2026 | Withdrawn before starting | I never asked for it, and the gun changes how I play, so it couldn't test the gun alone. |
| 3 | Planned CT rushes: 2-3 per half off a good spawn, never the same spot twice | from 17 Nov 2026, ~20 matches | **Running** | _pending_ |

## How I run tests (working rules, agreed 2026-11-17)

1. **One change at a time.** A new test starts only when the previous one has finished. Things I kept from earlier tests (e.g. playing back on CT) become part of my "normal".
2. **Only things I control myself.** Nothing that needs teammates to change.
3. **Write it down before playing**, in plain words: what I do in game, what gets compared, for how long.
4. **Length depends on how often it happens, not the calendar.** Something I can do in many rounds per match (a flash, a position) needs ~1-2 weeks. Something rare (a rush 2-3 times a half) needs ~20 matches. Match wins need ~100 matches, so they're never the main measure.
5. **Measure as close to the action as possible.** "Did the flash blind anyone, and did we win the fight right after?" says more, faster, than "did we win the round?".
6. **Best when I can switch it on and off within the same matches** (e.g. some rounds yes, some rounds no). Comparing this week with last week mixes in different lobbies, teammates and form.
7. **Report whatever comes out**, including "no clear difference".

## Ideas for next tests (backlog, one at a time after test 3)

**Likely to give real results** (I control it, it happens often, the demo sees it):
- **A new flash or smoke I've never used** (e.g. a pop flash for a T entry): throw it in some rounds and not in others. The demo shows where it lands, whether it blinded anyone and for how long, and who won the fight right after.
- **Playing off my spawn on CT** (go where my spawn is closest) vs a fixed position: the demo has my spawn and positions every 0.5 s.
- **Where I hold on CT** (e.g. Short vs Connector on Mirage) on alternating rounds.

**Harder to get real results:**
- Anything that needs teammates (learned the hard way in test 1).
- Settings, crosshair, sensitivity, warm-up routines: they can only change per match (not per round), there's a learning curve, and the effect shows up only in noisy numbers like K/D.
- Mental stuff (tilt, breaks between games): match-level, needs ~100 matches.
- Rare situations (clutches, 1vX): too few per week.
- Map choice: the map changes the lobby and teammates too.
