# Trial 3: planned CT rushes

**Started 2026-11-17.** My idea, agreed the same day. Every FACEIT match from 17 Nov onward counts.

## The idea (in my words)

Playing CT passively helped me perform better and helped my mental state because I die less, but I feel less impactful. I used to swing hard and get 1-2 kills before dying. Since I'm mechanically better than my teammates, there may be some waste in not using me more aggressively.

## What I do in game

- On CT, when I spawn close to a spot, I **rush that spot**.
- At most **2-3 times per half**, and **never the same spot twice** in a match.
- Every other CT round: play passive like now.
- T side: no change.

## What gets compared (decided 2026-11-17, before any trial match)

**Main number everywhere:** my win % added per round (my kills minus my deaths, weighted by how much each one changed the chance of winning the round). Also shown: how often I die, how often I get 2+ kills, round win.

**How the demo tells:**
- **Rush** = at 0:15 I'm further from our spawn than any teammate. I don't need to note anything down.
- **Good spawn** = of my team, I spawned closest to the enemy side, so I have the shortest route to the forward spots. Happens in ~1 round in 5. (Rough: it doesn't know *which* spot I had in mind.)

**Comparison A: same games.** Is a planned rush a good play in that moment?
- Good spawn + rushed vs good spawn + didn't rush, in the trial games. Same lobbies, teammates and form on both sides, so it's the fairer one.

**Comparison B: whole games.** Does adding rushes make my CT side better overall, including knock-on effects on the other rounds (opponents playing slower after getting rushed, or a failed rush costing the team money)?
- My whole CT side in the trial games vs **the 2 weeks before (9-16 Nov, 15 games, ~116 CT rounds)**. That's when I was playing passive, so it's the closest match to how I play now. Smaller, so noisier.
- The 79 older games are shown for reference only. That's when I pushed more and without using spawns: those pushes were +0.5 per round vs **+4.8** when I didn't push.

If A and B agree, that's a strong result. If they disagree (e.g. rushes win their own rounds but my CT side gets worse overall), that gets written up too.

Commands:
- Trial: `python src/ct_aggression.py --since 2026-11-17`
- Passive baseline: `python src/ct_aggression.py --since 2026-11-09 --until 2026-11-17`

## How long

About **20 matches**. That gives ~40-60 rush rounds, about as many as all my past pushes put together. Enough to see a big difference, not a small one. I'll look at the result only at the end.

## Result

_(after ~20 matches)_
