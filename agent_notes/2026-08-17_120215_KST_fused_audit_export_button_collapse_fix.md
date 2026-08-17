# Fused-kernel audit · settings export button · directional-collapse fix

**Status:** in progress — three parallel workstreams, subagents fanned out.
**Trigger:** user reports (a) default piece (Adversary · Pair · HashGrid · Curl)
runs `learn 92.7 ms (cpu·tfjs)` at ~10 fps — "I thought we did fused for all of
them"; (b) the game collapses to a single global direction ("picks a direction
and gets stuck"); (c) wants a button to export/share dock settings.

## Verified facts going in

- The hashgrid refusal is deliberate and documented at the default piece
  (`src/main.ts:2116`): `fusedAdvOk` requires
  `advect.layout.encoding.kind !== "hashgrid"` (`src/main.ts:3350`), and the
  fused **field** trainer additionally requires `advRt.tag === "off" ||
  fusedAdvOk` (`src/main.ts:3405`) because pass B needs the fused adversary's
  extGrads buffer at construction. So hashgrid + adversary ⇒ the ENTIRE learn
  step (D, G, field) is tfjs autograd. Measured in the piece comment:
  ~24 fps / 40 ms learn vs 60 fps / 0.8 ms fused. User measured 92.7 ms at
  110k particles / train B 800.
- HUD label `(cpu·tfjs)` at `src/main.ts:3232` — label text says cpu; backend
  is forced webgpu in main.ts. Whether the tfjs learn path actually hits CPU
  sync is part of the audit (workstream A).
- Directional collapse is the OPEN item from
  `agent_notes/2026-08-06_214641_KST_angle_disc_diagonal_regression.md`:
  soft-angle/ANGLE pays G for laminar fields (constant world-frame force is
  hard to predict on the pair observer; magnitude is unpunished once |F|≫τ);
  standalone adversary pieces run `ZERO_FIELD_LOSS` so nothing makes laminar
  expensive. Proposed there: small fused isotropy/curl pressure (W_ISO).
- Dock persistence: `nffa.dock.v2` in `src/index.tsx`; explicit URL knobs win
  over storage; piece index is NOT URL-addressable yet
  (`agent_notes/2026-08-06_145416_KST_dock_localstorage.md`).

## User's hand-tuned settings (screenshot 2026-08-17, look better "to start")

particles 110000 · train B 800 · max vel 70.3 · drive 0.53×clip · respawn 3.4%
· border WRAP · trails 0.42 · stroke CURL length 6.5 · blend A/B 0.22 · target
FORCE · loss ANGLE · soft τ 0.050 · tuple 1·POINT · reward 0.025 · G lr 3.1e-4
· D lr 1.2e-4 (D/G 0.38×) · guesses K 8 · relax ε 0.22 · color VEL.
Even with these: collapses to one direction over time.

## Workstreams

- **A — Fused audit (subagent, read-only):** table of every GALLERY piece →
  which trainer path it takes at defaults + why (gates); what fusing hashgrid
  backward for the adversary requires concretely; what `(cpu·tfjs)` really
  runs on.
- **B — Export/share (subagent, implements):** dock "COPY LINK" (+ JSON
  export). Design suggestion: single `?dock=<base64url(v2 blob)>` param
  parsed with the same validation as storage restore; precedence explicit
  knob params > ?dock > localStorage > piece defaults; add piece
  addressability.
- **C — Collapse fix (subagent, diagnose→prototype):** make laminar/constant
  direction expensive; candidates: batch-mean-direction penalty
  ‖mean(unit(F))‖², weak curl/isotropy pressure, magnitude term. Evidence via
  soak (tools/soak_adversary.mjs, tools/smoke.mjs).

## Results

- **A — audit: DONE.** 17/18 pieces fully fused; only the default (hashgrid)
  piece falls back, one gate knocks out D+G+field together. "(cpu·tfjs)" HUD
  label was a hardcoded lie (backend is webgpu; dispatch-bound + sync stalls)
  — fixed in main tree at `src/main.ts` (~:3232). Full audit:
  `agent_notes/2026-08-17_120215_KST_fused_kernel_audit.md`. Fusion agent
  (worktree) implementing the ~60-line codegen plan; pixel-gate footguns
  filed as a separate task chip.
- **B — export/share: DONE, merged in main tree, NOT committed.** SHARE
  section in dock: COPY LINK (`?dock=<base64url(v2 blob)>`) + COPY JSON.
  Precedence: knob params > ?dock > localStorage > piece defaults. Links
  honor their own recipe (RecipePolicy "blob"); storage keeps piece recipe.
  Piece identity by name, index repaired loudly. Verified: build, smoke, 27
  round-trip checks, live ingestion on real Metal adapter (loop adopted
  k=8/point from a link). Fixed in passing: useRef(initialDock()) re-ran
  ingestion every render. Filed: tfjs-core blanks page on stray `%` in URL.
- **C — collapse: DIAGNOSED + prototype merged (default-off).** Mechanism
  PROVEN: soft-angle's ψτ(0) north pole sits √2 from every equatorial
  prediction ⇒ zeroing the target pays 3.15× more than any varied field ⇒
  payoff maximum IS the laminar/dead field (no energy anchor opposes it).
  Point preset: near-constant at init + tanh saturation freeze (46% of
  domain both-components saturated). D-loss/G-residual charts are provably
  the same scalar (fused: stats[0]≡stats[1] when finite) — HUD could never
  have shown this; replace one with the R1 order parameter later.
  Fix measured (pair, seed 1234, 1000-2000 steps): POLAR+NEMATIC 0.05 drops
  R1 0.98→0.057 (point: 0.95→0.10); polar alone is escapable by ±F₀
  counter-streaming sheets (R2 0.95) — nematic companion required.
  Merged into main tree: `directionOrderLoss` (src/core/losses/isotropy.ts),
  `GamePressure` sum type + κ + dispatcher (main.ts), `?advPolar= ?advNematic=
  ?advPolarTau=` knobs (+ DOCK_OVERRIDE_PARAMS in index.tsx),
  `tools/collapse_probe.ts` instrument. fusedAdvOk refuses declared pressure
  (loud, never silently dropped). Verified: build + real-adapter boot with
  `?advPolar=0.05&advNematic=0.05` → `pressure=anti-collapse` adopted, no
  page errors. Full note: `agent_notes/2026-08-17_120215_KST_collapse_fix.md`
  (worktree copy synced here).

- **HASHGRID FUSION: DONE, MERGED into main tree (not committed).** The
  default piece now trains fully fused (D + G reward + field). Live A/B on
  Metal: 24.6 fps / learn 39.6 ms tfjs → 60 fps / rollout 0.01 + optim
  0.98 ms. Gaps closed in adversary_wgsl.ts (dEncOff scratch, hashgrid
  bwdCall arm, gather-side grid block in fieldGrad — zero atomics);
  emitBwdStore now takes explicit dEncMode "seed"|"accumulate" (lane
  hazard fixed as a type). New gate: tools/train_wta_hashgrid_test.ts —
  6 configs vs live tfjs at cos = 1.0000000 with GRID slice + support
  asserted separately, extGrad seam proven additive (not double-counted)
  against a nonzero field loss. Merged here and re-verified IN THIS TREE:
  hashgrid suite, train_wta_test, train_types_test, build, real-adapter
  boot (default piece logs FUSED + fused field trainer batchCap=16384;
  ?advPolar still routes to tfjs loudly). fusedAdvOk hand-reconciled:
  pressure clause kept, hashgrid exclusion dropped. Pixel critic still
  refuses hashgrid (out of scope, now loud). Handoff:
  `agent_notes/2026-08-17_120215_KST_hashgrid_adversary_fusion.md`.
  Known loose end from that note: one soak flake (1/4 runs failed two
  warm-up-timed percentile gates at SOAK_WARMUP_MS=4000; rerun both keys
  ×5 at 8 s warmup to attribute).

Agent worktrees under .claude/worktrees/ (a4861fda…= collapse,
a02bfdcb…= fusion) are fully merged but left in place as source-of-truth
until the tree is committed.

## Round 2 (2026-08-17 afternoon): "fix them all" — all four agents landed

- **PRESSURE FUSED + DEFAULT ON (merged).** `directionOrderLoss` fused into
  the adversary codegen for raw/fourier/hashgrid; adversary pieces bake
  polar=nematic=0.05; `?advPolar=0&advNematic=0` disables. Live A/B (default
  piece, 45 s, Metal): R1 0.88→0.999 laminar without pressure vs 0.001–0.11
  WITH — streaks become vortices at the same 60 fps. Parity: 71 checks,
  cos=1.0000000 (tools/train_wta_pressure_test.ts), degenerate cases pinned
  closed-form. Byte-identity: 1080 pressure-off blocks unchanged; overhead
  off=0, on=+0.6–5.8 µs. stats[0] renamed discLoss→payoffUngated (NOT
  recycled for R1 — it is the nonfinite-payoff canary; four moment slots
  appended instead). HUD: R1 chart replaces the redundant G-residual chart;
  √2 reference line on payoff (soft-angle only). NOTE the √2 line is
  ambiguous alone: √2 + R1→1 = collapse; √2 + R1→0 = healthy isotropy D
  can't predict (transient) — read the two together. λ=0.05 shipped; the
  fused term bites harder than the tfjs prototype (different sampling
  measure) — λ=0.01 also meets target and removes the √2 warm-up; first
  knob if pieces read sluggish. Detail:
  agent_notes/2026-08-17_pressure_fusion_hud.md.
- **Pixel gates + TS (merged):** PixelCriticPlan sum type; game+critic
  throws; six silencing gates named; ?train=tfjs pixel crash explained
  loudly; two AdversarySpec narrowing errors fixed; @types/react/dom/node
  installed (bundle byte-identical). agent_notes/2026-08-17_pixel_gate_ts_fixes.md.
- **%-URL guard (merged):** src/url_guard.ts first-import bootstrap;
  per-param κ; drops only undecodable params, loudly; 28 unit checks + live
  gate (tools/percent_url_gate.mjs — needs a server on :8799 or URL arg).
  agent_notes/2026-08-17_percent_url_guard.md.
- **CONCURRENT-SESSION COLLISION (recovered):** another live session's
  uncommitted post-c342cb7 work (fillForceGrid parallel pixel-critic pass +
  pixel piece dims G=8 retune + cost probe + docs) was partially clobbered
  by a wholesale cp during the pixel merge; recovered byte-exact from
  dist/index.2d3ace14.js.map sourcesContent and re-applied. Rule now in
  memory: git diff HEAD before any wholesale copy; patch --forward, not cp.
- **Soak flake:** see item 3 below — attribution done pre-pressure. Post-
  pressure re-run of the matrix pending (√2-burst hypothesis); then apply
  the gate fix (hi >= lo + not-FLAT). Note soak_adversary defaults to
  http://localhost:1234 — serve dist there or pass a URL arg.
- Worktrees for all four round-2 agents removed after merge+verify
  (pressure agent's pending removal once soak re-run concludes).

## Next

1. Consider baking user's 2026-08-17 hand-tuned settings (above) into a new
   gallery piece once they share a COPY LINK. (User decision.)
3. ~~Soak flake attribution~~ DONE (agent_notes/2026-08-17_soak_flake_attribution.md):
   hashgrid 6/9 fail vs pair4 0/8 (Fisher p=0.0068); warm-up irrelevant; NOT a
   pipeline-compile tail (60 fps from first sample). Mechanism: the two span
   gates parse toExponential(1) HUD strings and assert STRICT spanHi > spanLo;
   hashgrid's RAW surprise metric degenerates (p98 ≯ p2) in multi-second
   bursts pegged at 1.3–1.4 ≈ √2 — 29.1% of frames vs 0% on pair4 — while
   pair4 passes by one rounding bucket. The exact in-app FLAT test passed in
   every failing run. HYPOTHESIS (unverified): the √2-pegged RAW bursts are
   the north-pole collapse dynamics surfacing in the soak — re-run the matrix
   AFTER pressure defaults land; if bursts vanish that confirms it. Then apply
   the proposed gate fix regardless (hi >= lo + not-FLAT instead of strict >,
   since strict > on 2-sig-fig strings is ill-formed) — proposed in the note,
   deliberately NOT applied yet; also filed as a chip.
4. Swap the redundant G-residual HUD chart for the R1 direction-order stat,
   and/or draw a √2 reference line on the payoff chart: a soft-angle payoff
   parked near √2 ≈ 1.414 IS the north-pole collapse signature (read 1.362
   at baseline collapse; 1.23–1.32 through the swirl run's flat phase) —
   zero-plumbing tripwire.

## Late lesson from the 2000-step POLAR+NEMATIC+SWIRL run (2026-08-17)

Stacking the Okubo–Weiss swirl term DOES get vortex dominance (OW −0.94)
but it crushed ‖F‖ 0.035→0.005 by step 1300, and while the field was that
small the payoff sat at √2 for ~1000 steps — the swirl pressure pushed the
targets INTO the north-pole regime and G collected the collapse bonus the
whole time. Final AC 0.051 vs 0.157 for POLAR+NEMATIC alone.
**Generalized: ANY term that shrinks ‖F‖ feeds the pole exploit.** A swirl
term needs a much smaller λ plus a two-sided rms‖y‖ anchor. Ship
recommendation unchanged: POLAR+NEMATIC alone at 0.05 (R1 ≤ 0.06, AC
monotone, payoff healthy 0.48–0.63).

---

## Workstream B — export/share button

**Status:** implemented, verified, NOT committed.

### What shipped

A `SHARE` section at the top of the dock's collapsible region (not in the
sticky header — collapsing exists to uncover the artwork, and a share row
pinned above it would be the one chrome that never goes away on a phone):

- **COPY LINK** → `location.origin + pathname + "?dock=<base64url(JSON)>"`,
  written with `navigator.clipboard.writeText`, flashing `COPIED ✓` / `FAILED ✗`
  for 1.5 s (`COPY_FLASH_MS`).
- **JSON** (secondary, smaller) → the same blob pretty-printed, for bug reports.

### Files touched

- `src/share.ts` — **new**, and deliberately dependency-free (no React, no tfjs,
  no GALLERY) so the round-trip is testable outside a GPU: `encodeDockParam` /
  `decodeDockParam` (UTF-8 → base64url, typed `DecodedParam` failure) and
  `resolveSharedPiece` (piece identity, typed `SharedPiece`).
- `src/index.tsx` — `SharedDock` (= the v2 blob + `pieceName`), `dockToShareUrl`
  / `dockToShareJson` / `sharedDock`, the `parseDockParam` κ, `readQuery`,
  `warnSharedPiece`, `initialDock` (replaces `loadPersistedDock`),
  `loadStoredDock`, `RecipePolicy` + `restoredRecipe`, `copyView`,
  `writeClipboard`, and the share section markup.
- `src/ui.css` — `.share-secondary`, `.tui-chip[data-flash="copied"|"failed"]`.

### Precedence (implemented)

    explicit knob params (?gLR, ?advLoss, …) > ?dock= > localStorage > piece defaults

The knob params win **coarsely**: their presence suppresses *both* restore
sources and hands the decision to main.ts's κ, which is the only thing that can
apply them to the running loop. Half-restoring a dock underneath them would put
the sliders and the artwork in different states. `?dock=` present but suppressed
warns by name (`… explicit knob params (gLR) outrank a shared dock`). Every
rejection is a `console.warn` naming the reason, then a fall-through — that warn
IS the typed-error path.

### No drift between save and share

The App now builds **one** `PersistedDock` per render; the localStorage effect
and both share buttons serialize that same object. There is one serializer, so
a stored session and a copied link cannot describe different settings.

### Deviation from the brief (deliberate, one item)

The brief said to run `?dock=` through the *same* validation as the storage
restore. It does — same function, same guards, same clamps — but that function
now takes a `RecipePolicy` (`"piece"` | `"blob"`), a thin dispatcher with one
handler each:

- **`"piece"` (localStorage, unchanged behaviour):** the gallery piece's baked
  observer/target/loss/K/ε wins, and a stored reward of 0 with the game on is
  treated as v1 poison and replaced by the piece's own reward.
- **`"blob"` (share links, new):** the link's own recipe wins.

Why: under `"piece"`, the settings this feature exists to share are literally
unrepresentable. The user's hand-tuned state recorded above — tuple `1·POINT`,
K 8, ε 0.22, τ 0.050 on a piece baked as PAIR / K 4 / ε 0.05 — would come back
silently as PAIR / 4 / 0.05, i.e. a link that lies about itself. The storage
policy exists because a *saved* blob is a side effect of clicking around that
nobody chose; a `?dock=` link is an explicit act carrying an explicit recipe,
exactly like `?advLoss=` / `?advTau=` / `?advK=`, which already outrank both
storage and the piece defaults. Same reasoning applies to a shared reward of 0
("run the game, feed the HUD, do not steer"), matching main.ts's `floatParam`
note on an explicit `?advWeight=0`.

Two things stay piece-owned under **both** policies:

- `adversaryKind` — the adversary's *existence* is piece identity, not a dial
  (startLoop resolves it from GALLERY + `?adv=`; no dock override can switch a
  game on). Adopting a blob's kind would draw a full adversary panel over a loop
  that reports no telemetry.
- the `k` floor — `createAdversary` **throws** on a wta with `k < 2`
  (`src/core/gan/adversary.ts:1483`), so a hand-edited link must not be able to
  reach it. `kFloor` follows the piece's kind.

Additionally, `parsePersistedDock` now calls `objectiveDims(encoding, target,
loss)` before returning. That is the one cross-field invariant the per-field
guards cannot see and the one `startLoop` itself throws on (post-velocity is
point-only). Verified: a link with `post-velocity` + a pair observer is rejected
as invalid input instead of blanking the page.

### Piece identity

The shared blob carries `pieceName` alongside `runtime.piece`. On ingest, if
they disagree the **name wins**, loudly. GALLERY is append-only so an index is
usually stable, but a name is the identity convention the codebase already
commits to (`DEFAULT_PIECE_NAME`). An unknown name falls back to the index with
a warning that it may be a different artwork.

### Bug found and fixed in passing

`useRef(initialDock())` re-evaluates its argument on **every render** even
though only the first result is kept. Invisible while ingestion was a silent
localStorage read; with URL ingestion it re-decoded the param and re-logged
every `[dock]` diagnostic on each render — five times a second once the
telemetry poll starts (confirmed: the diagnostics appeared twice in a 4 s
window with the loop stalled). Now a `useState` lazy initializer.

### Verification

1. **`npm run build`** (`parcel build --no-scope-hoist`) — passes.
   (`yarn` is not on PATH on this box; the package script is identical.)
2. **`node tools/smoke.mjs http://localhost:8798/index.html 8000 …`** against
   `dist/` served by `python3 -m http.server` — no page errors from app code.
   smoke.mjs forces `forceFallbackAdapter`, and this box has **no software
   WebGPU adapter**, so `PROBE {adapter:"null", warning:true}`: the app
   correctly shows the "This needs WebGPU" notice. The residual console lines
   are tfjs's backend-init failure under that condition plus a `/favicon.ico`
   404 — both pre-existing and reproducible without this change.
3. **Round-trip unit gate** — `bun share_roundtrip_test.ts` (scratchpad),
   importing the real `src/share.ts`, 27 checks PASS: deep identity through
   blob → param → blob including float precision (`3.1e-4`, `70.3`, `0.034`)
   and the non-ASCII `·` in piece names; the param needs no percent-encoding
   and survives a real `URLSearchParams` round trip (the actual regression
   base64url prevents: plain base64's `+` decodes to a SPACE); and six
   corruption classes (truncated, single flipped char, non-base64, non-JSON,
   empty, invalid UTF-8 bytes) all take the typed-invalid path with a reason.
4. **Live ingestion, real WebGPU adapter.** This box *does* have a hardware
   adapter — it only looks adapter-less to smoke.mjs because that harness
   forces the software fallback. Dropping those flags, headless Chrome runs the
   piece at ~60 fps. Loaded a `?dock=` link whose `pieceName` says
   "Adversary · Pair WTA K=4" (index 8) while `runtime.piece` says 3:
   - console: the rename warning, then `[dock] adopted ?dock= share link`;
   - dock shows piece "Adversary · Pair WTA K=4", border BOUNCE, tuple 1·POINT,
     loss ANGLE, τ 0.075, K 8, ε 0.22, reward 0.025, particles 20 000, max vel
     70.3, trails 0.42, stroke CURL 6.5 — every dial from the blob;
   - the **loop** adopted it too: `[adversary] FUSED wta encoding=point k=8`
     and the HUD reads `adv wta k=8 ε=0.22 · force · soft-angle`, i.e. the
     blob-policy recipe reaches the GPU trainer, not just the sliders;
   - clicking COPY LINK flashes `COPIED ✓` and produces a link that decodes
     back to the same state with the piece index **repaired** to 8; COPY JSON
     produces the pretty blob; both revert to idle after 1.5 s.
   - Headless Chrome denies clipboard-write even with `overridePermissions`, so
     the rejection path was exercised for real first: `FAILED ✗` +
     `[dock] clipboard write failed — NotAllowedError…`. The success path was
     then measured with the OS clipboard stubbed at that single boundary.
5. **Precedence matrix**, each in a fresh browser context (empty localStorage):

   | case | result |
   |---|---|
   | `?dock=<valid>&gLR=0.002` | warns, ignores the link, opens on piece defaults |
   | `?dock=` piece index 999 | warns "failed v2 dock validation", falls through |
   | `?dock=` post-velocity + pair observer | warns, falls through (would have crashed startLoop) |
   | `?dock=<valid>` | adopts, resolves by name |

### Open item found (NOT fixed, out of scope — background task filed)

`?dock=%%%not-base64%%%` blanks the page. The throw is **not** in the dock
path: it is `URIError: URI malformed` from tfjs-core's module-level
`populateURLFlags`, which decodes the query string at **import** time, before
one line of app code runs. Any stray `%` anywhere in the query does this, with
or without this feature. `readQuery` in `src/index.tsx` now guards the dock's
own parsing (defence in depth — that layer must not be the second place it
breaks) and documents the measurement. Realistic corrupted share links cannot
hit it: base64url contains no `%`.
