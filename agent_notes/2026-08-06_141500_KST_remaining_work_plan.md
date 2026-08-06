# Remaining work plan — field arch, fused cover, AD IR

**Status:** Phase A + Phase C **done**; Phase B (IR→production cutover) scoped, not started  
**Updated:** 2026-08-06

## Phase A results

- [x] Galaxy / Spiral pieces → `SPIRAL_FIELD_LOSS` (fused)
- [x] Vortex → `CENTER_FIELD_LOSS` (`W_CENTER`)
- [x] IR `coverTerm` / `coverLoss` + `tools/cover_oracle_test.ts` PASS
- [x] `field_loss_routing_test` cover + center markers PASS
- [x] Cover fused via `COVER_FIELD_LOSS` / `W_COVER` / `COVER_SAMPLES: 256`

## Phase C results (was “optional later” — completed)

- [x] True single-head `FieldSpec.kind: "vector"` (advect + train; no idle B)
- [x] `W_CENTER` fused + Vortex / Galaxy / Trails parity
- [x] Fourier+SIREN combo (`ARCH.fourierSiren`, dock + gallery Cover piece)
- [x] IR `wCenter` in `buildSample`; `wCover` reserved + batch `coverLoss`
- [x] Softmin cover — **deferred** (hard min remains the v1 contract)

## Phase B — AD IR as production compiler (NOT “done”; real project)

**Why not production today:** deliberate oracle role after hand-fused types hit cos=1.0; gaps (hashgrid gather, batch-coupled cover, adversary/pixel games, Adam/scratch/multi-WG scaffolding still in `train_wgsl.ts`).

**Cutover checklist (kill criteria):**

1. IR expresses all loss seeds used in production (chaos/iso/div/spiral/cover/center)
2. IR emit covers head VJP + BPTT for raw/fourier/siren (hashgrid may stay hand or extend IR)
3. `FusedTrainer` builds pass A/B **from IR emit** (or shared emitter), templates deleted or thin wrappers
4. Metal parity: cos≥0.999 vs tfjs fixtures for every modelType + Cover
5. Adversary + pixel-disc still compose (extGrads seam unchanged)
6. Gallery soak: no NaN, Cover/Galaxy form filament, chaos pieces unchanged FPS class

**Estimate:** multi-session; do not half-cutover.

## Non-goals this session

- Replacing `train_wgsl.ts` with IR emit as live path
- Softmin cover / live arch dock for dual-head chaos
- Renaming `HelmholtzField`
