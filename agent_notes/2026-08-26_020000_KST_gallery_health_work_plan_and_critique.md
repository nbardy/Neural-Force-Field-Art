# 2026-08-26 — what the health data actually says, what work follows, and where
# this program's methodology is weak

Written after the first complete 16-piece headless audit
(`agent_notes/2026-08-25_170000_KST_health_audit_full_gallery_r2_gate.md`).
This note is the PLAN + CRITIQUE; that one is the measurement record.

Everything below separates **measured** from **hypothesis** explicitly, because
this program has twice shipped a mechanism story that a later measurement
refuted (`2026-08-18_pole_exploit_refuted_predictor_scale_blindness.md`).

---

## PART 1 — two results that reorganize the work list

### Result A (MEASURED) — 3 of the 4 direction failures are pieces with no
### anti-collapse pressure compiled at all

Cross-tabulating the audit against `pressure: GALLERY_ANTI_COLLAPSE` in the
GALLERY table (`src/main.ts`):

|  | n | max grid R1 | max grid R2 | max sat |
|---|---|---|---|---|
| pressure compiled | 9 | 0.337 | 0.613 | 0.756 |
| no pressure | 7 | **0.821** | **0.927** | 0.528 |

Every direction failure and which side it is on:

| piece | pressure? | R1 | R2 |
|---|---|---|---|
| Pixel · VecField | **no** | **0.782** | 0.556 |
| Pixel · NextFrame | **no** | **0.821** | 0.604 |
| Neural Field · Max Structure | **no** | 0.034 | **0.927** |
| Adversary · Tri WTA K=6 | yes | 0.337 | **0.613** |

So the headline is NOT "the fix stopped working". Three of four failures are
pieces the fix was **never applied to** — the four pixel critics and the three
plain Neural Field pieces have no `pressure` line. Tri6 is the only piece that
has the pressure and defeated it, and only on the nematic axis.

This is good news operationally and bad news epistemically: for weeks the
gallery has read "collapse fixed" on the strength of pieces that carry the fix,
while the pieces without it were either unmeasured (pixel critics had no grid R1
in the verdict) or measured on an axis that could not see it (R2 ungated).

### Result B (MEASURED, and it contradicts a verdict we are printing) —
### `frozen-saturated` is measuring magnitude, not frozenness

`satFrac` across the 9 pressure pieces, sorted by AC:

| ac | 1.350 | 1.265 | 1.007 | 0.972 | 0.905 | 0.685 | 0.419 | 0.345 | 0.062 |
|---|---|---|---|---|---|---|---|---|---|
| sat | 0.756 | 0.473 | 0.244 | 0.213 | 0.075 | 0.000 | 0.000 | 0.000 | 0.000 |

**Perfectly monotone — 9 for 9, Spearman ρ = 1.0.** satFrac is a near-pure
function of field magnitude on this gallery. That is unsurprising in hindsight
(satFrac counts tanh components past ±0.9, so a big field saturates) and it
means the gate at 0.3 is a *magnitude* threshold wearing a *dynamics* name.

Worse, the verdict text asserts something our own data denies. It prints:

> FROZEN GENERATOR — satFrac 0.4727 ... **the field cannot move**

while the same aggregate reports `ac 1.2648 (rising)` for `single` and
`ac 1.3501 (rising)` for `quad6`. A field whose AC is climbing is, by our own
instrument, moving. The only saturated piece whose AC is **falling** is
`Pixel · NextFrame` (0.5995, falling) — which is also the one with dc/ac 1.80,
i.e. a large near-constant field. That one may genuinely be stuck.

**"Frozen" is an inference we never measured.** Nothing in the snapshot reads
the rate of change of the WEIGHTS or of the field itself; `acTrend` is the
closest thing and it disagrees with the verdict on 2 of 3 saturated pieces.

---

## PART 2 — the work, ranked, with what would falsify each item

Ranked by (evidence strength x cheapness), not by how interesting it is.

### W1. Fix the `frozen-saturated` verdict — it is currently making a claim we
### have not measured  [instrument; cheap; blocks trusting W3]

Two honest options, pick one:

- **(a) Rename to what it measures.** `saturated` / "N% of the domain is past
  the tanh knee; the field is at its output ceiling". Drops the dynamics claim.
- **(b) Make it measure frozenness.** Require `satFrac > gate` **AND**
  `acTrend` not rising. Under (b), today's run reports `single` and `quad6` as
  saturated-but-still-moving (not failures) and `nextframe` as genuinely frozen.

Recommend **(b) with the reading from (a) in the message**, because "saturated
and still climbing" and "saturated and dying" are different problems and the
operator needs to know which. Falsifier: if a piece with rising AC and sat 0.75
is visually stuck, (b) is wrong and the gate should stay magnitude-only.

**This is the highest-priority item** because W3's entire premise — "saturation
is the dominant failure mode of this gallery" — rests on a verdict that is
provably over-claiming on two of the three pieces it fires on.

### W2. Apply the anti-collapse pressure to the 7 pieces that lack it, then
### re-audit  [artwork; cheap; directly closes 3 of 6 failures]

The four Pixel critics and the three Neural Field pieces have no `pressure`
line. Result A says the pieces that have it top out at R1 0.337 / R2 0.613 and
the pieces without it reach R1 0.821 / R2 0.927.

Do it as an EXPERIMENT, not a commit: `?advPolar=`/`?advNematic=` are already
URL knobs, so run `vecfield` and `nextframe` at the gallery default pressure and
compare R1 against today's baseline before touching the GALLERY table.

Two caveats that must not be skipped:
- The pixel critics have **no adversary block**, so whether the pressure term is
  even reachable on them needs checking in the fused path first — this may be
  code, not config.
- **Max Structure should probably NOT get it.** It optimizes W_STRUCT directly;
  a single-axis field may be exactly what that loss wants, and forcing direction
  disorder onto it is an artistic decision, not a health fix. Needs the artist.

Falsifier: pressure on, R1 stays > 0.5 → Result A is confounded (the pieces
differ in more than pressure) and the pixel critics need their own diagnosis.

### W3. Diagnose the saturation cluster — AFTER W1  [artwork; medium]

Whatever survives W1's re-scoring. Note the uncomfortable direction of the
correlation: the three highest-AC pieces in the gallery are all pressure pieces,
and AC is the thing we have been trying to RAISE. If the anti-collapse pressure
raises AC and AC drives saturation, then the collapse fix bought saturation.

**Hypothesis, NOT measured:** the direction-order penalty is cheapest to satisfy
by driving tanh outputs to the corners in a spatially varied pattern.
**Cheap control that would settle it:** run `single` at
`?advPolar=0&advNematic=0` and compare satFrac against today's 0.473. Costs one
90 s audit. Nobody has run it.

### W4. `rmsP` in the health snapshot  [instrument; expensive; highest value]

Still the outstanding recommendation from the pole-exploit refutation, and still
the only proposed metric that says WHICH SIDE of the game failed: D drifting
past ~5tau while rmsY < tau fires before a collapse completes. Everything we
currently gate on is a symptom read off the generator.

Cost is real: a new stat slot in the fused WGSL reduction, the readback in
`adversary_train.ts`, the CPU oracle in `core/gan/adversary.ts` moved in step,
and `tools/adversary_wire_test.ts` (currently another session's dirty file)
gating the parity. Do not start this without checking `git status` first.

### W5. Repeat runs — we have no variance estimate at all  [method; cheap]

Two runs of the same gallery disagreed on 2 of 10 pieces (`single` healthy ->
frozen, `tri6` frozen -> healthy). Every number in every note in this program is
n = 1. Until we know the run-to-run spread, "improved" and "regressed" are not
falsifiable at the single-piece level, and W3 cannot be evaluated.

Minimum useful version: 3 repeats of a 4-piece subset, report the spread. If the
spread on satFrac is +/-0.2, most of the piece-level history in `agent_notes/` is
noise and should be re-read with that in mind.

### W6. Record whether pressure was compiled, per piece, in the snapshot
### [instrument; cheap]

Today the auditor cannot distinguish "collapsed **despite** the pressure" from
"collapsed with **no pressure compiled**" — I had to reconstruct that by
grepping `src/main.ts`, which is exactly the kind of out-of-band reasoning the
snapshot exists to eliminate. `adv.r1 === null` is a partial proxy and it is
absent entirely on pixel pieces. Add the resolved `GamePressure` tag to the
snapshot; it makes Result A a reading instead of an archaeology exercise.

### W7. Attribute the fps regression  [perf; cheap]

41.2 (hashgrid), 41.7 (vecfield), 50.0 (struct), all 60.0 in the 2026-08-18
baseline. Above the floor of 30 so nothing failed. Three pieces moving together
suggests a shared cause (the parallel pixel critic pass, or the family-planed
hashgrid gating). Bisect across the 7 branch commits.

### W8. Ship it  [process]

`multi-guess-modularization` is **7 commits ahead of main and undeployed**. None
of the fusion, pressure, health-metric or multi-guess work is on the live site.
The longer this sits the more the deploy is a big-bang with no per-commit
attribution — and this program's whole method depends on attribution.

### W9. Doc + coverage debt  [done in part]

`AGENTS.md` updated this session for the R2 gate, 16-piece coverage and the
import guard. Remaining: `docs/ADVERSARY_STATUS.md` predates all of it.

### W10. Re-read every adversary number in this repo as a **32/16** reading
### [method; free to state, expensive to redo]

Landed by the concurrent session on 2026-08-26 and it retro-qualifies most of
this program (`AGENTS.md` §"EVERY prior adversary reading is a 32/16 reading"):
`hiddenUnits`/`featureDim` have been accepted since the port and **no caller
ever passed either**. Win EMAs, payoff curves, R₁/R₂, the NaN probes — and the
pole-exploit refutation itself — are properties of a 32/16 predictor, not of
"the adversary".

This does not overturn the refutation (a scale-blind metric is scale-blind at
any width) but it does mean **the audit in this session is one point in
predictor space, not a gallery baseline**. Anything of the form "objective X
collapses" should read "objective X collapses against a 32/16 predictor". The
cheap partial fix is to record the resolved predictor pair in the health
snapshot next to the arch fingerprint, so the qualifier travels with the data
instead of living in one paragraph of `AGENTS.md`.

### W11. `tools/adversary_strict_test.ts` is RED and predates this work
### [debt; unowned]

Documented in `AGENTS.md`, owned by nobody, and it is a strictness gate on the
adversary — the subsystem every open item above touches. A red test in the
subsystem under investigation means "the suite is green except where it
matters". Either fix it or delete it; leaving it red trains everyone to ignore
the suite.

### W12. The in-app browser pane hands the page a 0x0 canvas
### [instrument; trap]

Also newly documented: the swapchain fails and `__nffHealth.field` stays
`null` — a false negative that **reads exactly like a dead field**. This is a
`no-signal`/`nonfinite` verdict caused by the harness, not the artwork, and it
is the same class of attribution failure as the 2026-08-17 soak flake. The
auditor already guards the analogous browser-level failure with its independent
rAF counter; it should refuse to score a 0x0 canvas the same way, by name.

---

## PART 3 — critique of how this program is working

Not a list of mistakes; a list of things that will keep producing wrong
conclusions if they are not changed.

### C1. A documented failure mode without a gate is not a check

`AGENTS.md` has said "never read r1 without r2" for weeks, with a measured
example (R2 0.81 on Max Structure at R1 0.10). The gate never existed. Max
Structure sat in the gallery reading PASS at R2 0.927.

Same pattern, second instance: the grid R1 existed and simply was not wired into
the verdict for pieces with no adversary, so two pixel critics could not fail a
direction check no matter what they did.

**Rule to adopt:** a failure mode written into the docs gets a gate and a
self-test case in the same change, or it gets deleted from the docs. Prose in
`AGENTS.md` reads, to every future agent, as a check that is running.

### C2. "all" meant 10 of 16 pieces, and said so in its own comment

The `PIECES` map covered 10; the comment read "`all` = every adversary piece
plus ONE pixel critic". Every green `all` run for weeks was a claim about 62% of
the gallery presented as a claim about the gallery. Both of the worst-collapsed
fields were in the uncovered 38%.

### C3. Everything is n = 1, and we know it disagrees with itself

See W5. Two consecutive full runs disagreed on 2 of 10 verdicts. We have been
attributing single-run deltas to code changes for weeks with no idea of the
noise floor. This is the single biggest threat to every claim in `agent_notes/`.

### C4. Verdict names assert mechanisms the metrics do not measure

`frozen-saturated` prints "the field cannot move" from a threshold on a static
magnitude statistic, and today it printed that about two fields whose AC was
rising. `pole-exploit` is better — it was explicitly built to require both
halves of an ambiguous reading — but the general failure is naming a verdict
after a STORY rather than after the MEASUREMENT, and then reading the story back
out of the logs as if it were data. `laminar-collapse` and `nematic-collapse`
are safe by this standard; `dead-field` and `blown-up` are borderline.

### C5. The shipped anti-collapse pressure treats a fault that was measured to
### be on the other side of the game

The refutation note is unambiguous: ~97% of the generator's reward during a
collapse was D's failure to track scale, not the field's unpredictability. The
default-on fix penalizes the GENERATOR. It works as a symptom control (Result A
is real), but the lever the evidence points at — **tau**, already a knob at
`?advTau` — has still never been A/B'd on the live gallery.

We are one 90 s audit away from knowing whether the correct default is
"pressure on" or "tau larger", and we have not run it. That experiment should
outrank most of PART 2.

### C6. Two sessions in one worktree has now cost attribution three times
### (and NO code, verified)

**Correction to an earlier draft of this note**, which claimed the branch had
been "reset and recommitted underneath a live session". It had not. `git reflog`
shows exactly two `reset: moving to HEAD` entries — no-op mixed resets that
unstage and move nothing (`7a8d12d` → `7a8d12d`) — and one `commit (amend)` that
changed 2 lines of one agent note (`48c3a2f` → `a61273d`, the pre-amend commit
still reachable as a dangling object). **No history was rewritten.** Stating the
stronger claim from a `git status` that merely looked wrong was itself an
instance of C4: naming a mechanism the evidence did not support.

What actually happened, three times:

1. One session's `cp` clobbered another's uncommitted parallel critic pass
   (recovered from `dist` source maps).
2. `7a8d12d` swept in another session's uncommitted `?arch=` work.
3. `7a8d12d` ALSO swept in this session's uncommitted `tools/health_audit.mjs`
   work — the R2 gate, the 10→16 coverage, the import guard, 156 insertions —
   under a message about dock-selectable arch that mentions none of it.

All three are the same mechanism: **a commit-all (`git add -A` / `git commit
-a`) in a shared worktree takes whatever another session has in flight.** It is
not a rewrite and it destroys nothing; `git fsck` is clean, there are no
stashes, and every commit is reachable from HEAD.

The cost is **attribution**, which is the one thing this program's method
depends on. `git log -S 'nematic-collapse'` is the only way to find which change
introduced that gate, and it answers `7a8d12d` — a commit about something else.

**Fix:** one git worktree per session (`.claude/worktrees/` already exists and
one agent worktree is live), or at minimum `git add <explicit paths>` rather
than `-A` when any peer session is listed by `ListAgents`.

**Proposal:** one git worktree per session. The cost is disk; the alternative is
that the next collision lands on something with no `dist` source map to recover
from.

### C7. Thresholds are "measured, not guessed" — but measured once, long ago

Each gate default cites a measurement in `agent_notes/`. Those measurements come
from single runs of a codebase that has moved 7 commits since. `acDead = 1e-4`
is the clearest case: hashgrid sat at 6.7e-4 in the baseline, one order above
the gate, and read healthy — while the other session independently found
`dualStd` at ~1e-3 and *falling*, also passing. A gate that a dying field passes
by one order is not tuned; it is a floor nobody has revisited.

---

## What I would do next, in order

1. **W1** (fix the over-claiming verdict) and **C5's tau A/B** — both cheap,
   both change what the other items mean.
2. **W5** (3 repeats, 4 pieces) — establishes the noise floor everything else
   is compared against.
3. **W2** (pressure on the 7 uncovered pieces, as an experiment first).
4. **W6** + **W10**'s snapshot half (record pressure AND predictor pair in the
   snapshot) — together they make the two biggest qualifiers on today's data
   travel with the data.
5. **W7**, **W12**, then **W4** (the expensive one), **W11**, and **W8** (ship).

Not in this list, deliberately: tuning any threshold. Until W5 gives a noise
floor, every threshold change is fitted to a sample of one.
