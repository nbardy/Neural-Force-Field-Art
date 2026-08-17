# Attributing the `soak_adversary.mjs` percentile-span flake: hashgrid vs pair4

Date: 2026-08-17 (KST)
Predecessor: `agent_notes/2026-08-17_120215_KST_hashgrid_adversary_fusion.md` §3 "Gate 5"
and §6.1, which recorded 1 failure in 4 `hashgrid` runs and asked for
"both keys ×5 at `SOAK_WARMUP_MS=8000`".
Scope: measurement + attribution only. **Nothing committed. No source file
changed** — see §5 for why the tool fix was deliberately NOT applied.

Legend: **[V]** verified by running it on this machine (Apple M4 / Metal,
headless Chrome). **[H]** hypothesis.

---

## 0. Verdict

**Neither "flaky gate" nor "hashgrid regression" alone. Precisely:**

1. **The incidence is hashgrid-specific and statistically significant.** [V]
   hashgrid failed **6 of 9** runs, pair4 **0 of 8**. Fisher exact, one-sided:
   **p = 0.0068**.
2. **Warm-up is NOT the mechanism.** [V] Doubling `SOAK_WARMUP_MS` 4000 → 8000
   did not help (2/3 → 4/6). The fusion note's suggested setting is a dead end.
3. **It is NOT a perf / pipeline-compilation signature.** [V] Both keys hold
   **60.0 FPS from the very first sample** (min across all runs 59.9), rollout
   0.010 ms, optim 0.010 ms. There is no early-timing tail on either key, so the
   audit's plausible "grid-segment pipeline compilation" mechanism is
   **explicitly not supported** by the data and should not be cited.
4. **The gate is nevertheless ill-formed, and pair4 passes by one rounding
   bucket — not by a margin.** [V] The gate asserts a *strict* inequality on a
   readout the HUD prints with `toExponential(1)` (2 significant figures). At the
   instant the harness reads it, BOTH pieces sit at raw surprise ≈ 1.3–1.4;
   pair4 straddles the bucket boundary (`1.3..1.4`, relative width 0.071) and
   passes, hashgrid lands inside one bucket (`1.4..1.4`) and fails.

So: a real, measurable hashgrid-specific narrowing of the raw-surprise
distribution, amplified into a hard CI failure by a gate that cannot resolve it.
Per the task's decision rule (fix the tool only if it fails on *both* keys or is
warm-up-driven — neither holds), **the tool was left untouched** and the finding
is documented for follow-up.

---

## 1. The matrix [V]

Build: clean `git archive HEAD` (c342cb7) exported to a scratch dir and built
with `parcel build --no-scope-hoist --no-cache --public-url ./ src/index.html`.
**The main worktree was NOT used for the build** — it carries another agent's
in-flight pixel-disc changes including a block marked
`TEMP INSTRUMENTATION — remove before commit`, which would have invalidated the
measurement. Served with `python3 -m http.server 8798` on `dist/`, with an empty
`favicon.ico` added (the fusion note's known 404 → console-error false positive).

All 17 runs `… <url> 60 10`, **strictly sequential** (shared GPU).

| key | `SOAK_WARMUP_MS` | runs | failed | failure rate |
|---|---|---|---|---|
| hashgrid | 8000 | 6 (incl. 1 pilot) | 4 | **67%** |
| hashgrid | 4000 | 3 | 2 | **67%** |
| pair4 | 8000 | 5 | 0 | **0%** |
| pair4 | 4000 | 3 | 0 | **0%** |

Per-run detail (`RAW`/`UNIT` = the `p2..p98` pair the harness reads for each
metric; `sampleEQ` = how many of the 7 loop samples had `p98 == p2`):

```
hashgrid w=4000 r1 FAIL RAW=[1.1,1.2]      UNIT=[0,1300]   sampleEQ=1/7
hashgrid w=4000 r2 FAIL RAW=[1.4,1.4]EQUAL UNIT=[0,1300]   sampleEQ=2/7
hashgrid w=4000 r3 pass RAW=[1.3,1.4]      UNIT=[0,1300]   sampleEQ=0/7
hashgrid w=8000 r0 FAIL RAW=[1.4,1.4]EQUAL UNIT=[0,1300]   sampleEQ=0/7   (pilot)
hashgrid w=8000 r1 pass RAW=[0.17,0.86]    UNIT=[1.3,47]   sampleEQ=0/7
hashgrid w=8000 r2 FAIL RAW=[1.3,1.3]EQUAL UNIT=[0,1300]   sampleEQ=0/7
hashgrid w=8000 r3 FAIL RAW=[1.4,1.4]EQUAL UNIT=[0,1300]   sampleEQ=2/7
hashgrid w=8000 r4 FAIL RAW=[0.15,0.88]    UNIT=[1.1,100]  sampleEQ=2/7
hashgrid w=8000 r5 pass RAW=[1.3,1.4]      UNIT=[0,1400]   sampleEQ=0/7
pair4    w=4000 r1 pass RAW=[0.53,1.1]     UNIT=[3,96]     sampleEQ=0/7
pair4    w=4000 r2 pass RAW=[1.3,1.4]      UNIT=[0,1200]   sampleEQ=0/7
pair4    w=4000 r3 pass RAW=[0.56,0.9]     UNIT=[0,1200]   sampleEQ=0/7
pair4    w=8000 r1 pass RAW=[1.3,1.4]      UNIT=[0,1300]   sampleEQ=0/7
pair4    w=8000 r2 pass RAW=[0.14,1.1]     UNIT=[3.2,67]   sampleEQ=0/7
pair4    w=8000 r3 pass RAW=[1.3,1.4]      UNIT=[0,1200]   sampleEQ=0/7
pair4    w=8000 r4 pass RAW=[1.3,1.4]      UNIT=[0,1100]   sampleEQ=0/7
pair4    w=8000 r5 pass RAW=[1.3,1.4]      UNIT=[0,1300]   sampleEQ=0/7
```

Only two gates ever failed, both span gates, exactly as the fusion note reported:

- `RAW and PER UNIT each expose a finite non-flat percentile span`
  (`tools/soak_adversary.mjs:498-506`) — 4 hashgrid runs, 0 pair4.
- `exposed surprise percentile span has p98 > p2`
  (`tools/soak_adversary.mjs:542-547`) — 4 hashgrid runs, 0 pair4.

Every other gate passed on every run, including
**"every sample reports the fused adversary"** — the end-to-end proof of the
hashgrid fusion — and "no NaN/Infinity in telemetry".

**Note the near-miss in pair4's own numbers:** 5 of its 8 runs read exactly
`RAW=[1.3,1.4]`, relative width **0.071**. That is one 2-significant-figure
bucket wide. pair4 is not comfortably passing; it is passing by the smallest
representable margin.

---

## 2. Mechanism, traced end to end [V]

The chain, from the gate back to the number:

1. **The gate reads a display string, not a value.**
   `src/index.tsx:2175-2183` renders the readout as
   `p2 {surpriseSpan.lo.toExponential(1)} · p98 {surpriseSpan.hi.toExponential(1)}`
   — **2 significant figures**. `parseTelemetry` regexes those two rounded
   numbers back out and the gate asserts `spanHi > spanLo`. Any distribution
   whose p2 and p98 agree to 2 s.f. (relative width below roughly 7% at 1.3)
   renders as identical text and fails, no matter how healthy it is.

2. **The displayed pair is ONE frame's estimate, not an accumulation.**
   `main.ts:3978-3988` returns `...advSurStats.norm.raw`, and `raw` is
   documented in `src/draw/robust_norm.ts:125-127` as *"This frame's UNSMOOTHED
   percentile estimate — for the HUD / debugging"*. The EMA-smoothed `span` —
   the one the colormap actually uses — is a different field.

3. **That frame's sample is a single contiguous 1024-particle slice.**
   `main.ts:3255-3262` passes `advTrainer.surpriseCoverage().window` as the
   sample window; `adversary_train.ts:687-696` sets that window to
   `{ start: cursor, count: effectiveB * m }` and advances the cursor modulo the
   particle count. For the pair game (m = 2, B = 512) that is exactly 1024
   particles — equal to `RobustSpan`'s `cap` — sweeping the 90k buffer as a
   rolling window. So the gate is asserting on the p2..p98 of one 1024-particle
   chunk in one frame.

4. **The collapse question already has an exact answer, and the gate ignores
   it.** `robust_norm.ts:147-149` computes `collapsed = hi - lo < SPAN_FLOOR`
   with `SPAN_FLOOR = 1e-6`, surfaced as the `· FLAT` suffix, and
   `soak_adversary.mjs` already gates it separately
   (`surprise span never reports FLAT`). In **every** failing run that FLAT gate
   **passed** — i.e. the app never considered the span collapsed while the
   string-derived gate failed it. The failing spans sit at ≈1.35 with a spread
   up to ~0.1, which is **~10⁵× above** `SPAN_FLOOR`. The two checks disagree
   because one is exact and the other is a 2-s.f. reconstruction.

5. **Why `SOAK_WARMUP_MS` cannot fix it.** The readout is taken 400 ms after
   `exerciseColorDiagnostics` switches the color mode, and a metric switch calls
   `advSurStats.reset()` (`main.ts:3975`). Sampling runs every 8th frame with an
   async map (`surprise_points.ts:201,249`), so 400 ms ≈ 2–3 accumulator
   updates — and, per (2), only the newest one is displayed anyway. Warm-up time
   *before* that reset is irrelevant, which is exactly what the 4000-vs-8000
   columns show.

---

## 3. What actually differs between the two pieces [V]

`tools/span_dist_probe.mjs` (scratch-only, not added to the repo) samples the
same HUD readout every 250 ms for 45 s per metric instead of once:

| key | metric | reads | `p98 > p2` would fail | rel. width p10 / p50 / p90 |
|---|---|---|---|---|
| **hashgrid** | RAW | 179 | **52 (29.1%)** | **0.000** / 0.260 / 0.800 |
| pair4 | RAW | 179 | **0 (0.0%)** | 0.423 / 0.816 / 0.866 |
| hashgrid | PER UNIT | 180 | 0 (0.0%) | 0.961 / 0.994 / 0.999 |
| pair4 | PER UNIT | 180 | 0 (0.0%) | 0.990 / 0.996 / 0.999 |

So the difference is confined to the **RAW** metric, and it is large: hashgrid's
bottom decile of frames is fully degenerate at display precision, pair4's is
0.42 wide.

A second 60 s probe run shows the degenerate frames are **not** uniformly
scattered — they arrive in **multi-second bursts**:

```
DEGENERATE readouts (3 distinct): 0..0   1.4..1.4   1.3..1.3
degenerate at t(ms): 0, 33924, 34179, 34938, 35949, 36201, 36454,
                     36707, 37464, 37717, 38473, 38724, 39482, 39735
```

One contiguous ≈6 s episode (t ≈ 33.9–39.7 s) in which the whole 1024-particle
window sits at ~1.35 with under 7% spread. Burst frequency varies run to run
(29.1% of frames in one probe, 5.9% in the other), which is why the soak's
single-shot read is a coin flip on hashgrid and why the fusion note saw 1-in-4
while this matrix saw 2-in-3.

**Magnitude, and the one number worth chasing:** surprise is a distance in
normalised position space (`robust_norm.ts:59-65`), so the maximum possible
value over the unit square is **√2 ≈ 1.414**. The degenerate episodes sit at
**1.3–1.4 — at or immediately below that ceiling, for essentially every particle
in the window.** [H] The reading that fits is that during those episodes the
hashgrid predictor's residual is pegged near the domain diagonal: not a NaN, not
a collapse, but a multi-second window in which the surprise channel carries no
spatial information because everything is equally maximally surprising. This is
a hypothesis about *why*; the pegged magnitude itself is measured.

Caveat against over-reading it: pair4 also spends time at 1.3–1.4 (5 of its 8
soak reads). The difference is the *width* at that level, not the level itself.

### 3b. A second, unexplained hashgrid-only observation [V]

Across the loop samples (PER UNIT mode), **`p2 = p98 = 0.0` occurred in 7 of 63
hashgrid samples and 0 of 56 pair4 samples** — an exactly-zero 1024-particle
window, with `flatSpan` **false** (no `· FLAT`), i.e. the EMA span still held
history while that frame's window was all zeros. Example, hashgrid w=8000 r3:

```
per-sample [p2,p98]: [0,1240] [0,0] [0,0] [0,1300] [0,1260] [2.95,62.5] [2.61,55.4]
flatSpan:            false    false false false    false    false       false
coverage:            100%     100%  100%  100%     100%     100%        100%
```

The 250 ms probe did **not** reproduce an all-zero PER UNIT window in 239 reads,
so the trigger is not simply "sample often enough". **Mechanism unresolved.**
Note that the one all-zero RAW read the probe *did* capture was at t = 0, the
post-`reset()` state, and it correctly carried `· FLAT` — so these loop samples
are *not* the reset signature.

---

## 4. Reproduce

```bash
# clean build (do NOT build the dirty main worktree)
mkdir -p /tmp/soakclean && git archive HEAD | tar -x -C /tmp/soakclean
ln -s "$PWD/node_modules" /tmp/soakclean/node_modules
cd /tmp/soakclean && ./node_modules/.bin/parcel build --no-scope-hoist \
  --no-cache --public-url ./ src/index.html
touch dist/favicon.ico && (cd dist && python3 -m http.server 8798 &)

# the matrix — SEQUENTIAL, nothing else on the GPU
for i in 1 2 3 4 5; do
  SOAK_WARMUP_MS=8000 node tools/soak_adversary.mjs hashgrid \
    http://localhost:8798/index.html 60 10
done
# …same for pair4, and with SOAK_WARMUP_MS=4000
```

One soak run = **~70 s** at `60 10`.

---

## 5. Why no fix was applied, and what the fix would be

The task's decision rule was: fix `tools/soak_adversary.mjs` only if the data
says FLAKY GATE — *"fails on both keys / driven by warm-up truncation"*. It
fails on **one** key (p = 0.0068) and is **not** warm-up-driven, so the
hashgrid-specific branch applies: **document, change nothing.** No engine code
was touched either; both are other agents' territory this session.

For whoever picks this up, the tool change that the mechanism in §2 argues for
(**a proposal, not applied, not tested**) is to stop reconstructing a collapse
test from a 2-significant-figure string and defer to the exact one the app
already computes — i.e. in both gates assert
`finite(lo) && finite(hi) && hi >= lo` **plus** the absence of `· FLAT`
(`parseTelemetry` already exposes this as `flatSpan`, and `SPAN_FLOOR = 1e-6` is
the authoritative threshold), rather than `hi > lo`. That keeps every real
failure mode — an unparseable readout, an inverted span, a genuine collapse —
and drops only the rounding-noise failure. Repo policy: whoever applies it
should comment the site naming this incident.

**Do not** paper over it by raising `SOAK_WARMUP_MS`; §1 and §2.5 show that does
nothing, and the default should stay at 4000.

---

## 6. Follow-ups

1. **[V] hashgrid RAW surprise degenerates in multi-second bursts** (29.1% /
   5.9% of frames in two probes; pair4 0.0% of 179). Pegged at 1.3–1.4 against a
   √2 ≈ 1.414 ceiling. Stage: the raw surprise plane feeding
   `GpuSurpriseStats` for the default piece. Not a NaN, not a collapse
   (`SPAN_FLOOR` is 1e-6, ~10⁵× below), not a perf issue (60.0 FPS throughout) —
   a diagnostic-quality question about the default piece, worth a look before
   anyone leans on RAW surprise as a visual on it.
2. **[V] Exactly-zero surprise windows, hashgrid only** (7/63 samples vs 0/56).
   Mechanism unresolved; see §3b.
3. **The gate fix in §5** — one clean change to `tools/soak_adversary.mjs`,
   currently unowned. Until it lands, `soak_adversary.mjs hashgrid` is expected
   to fail ~2 runs in 3 on these two gates, and that failure carries no
   information about the fused hashgrid adversary.
4. The fusion note's §6.1 item ("run both keys ×5 at `SOAK_WARMUP_MS=8000`") is
   **closed by this note** — answered, and answered negatively.
