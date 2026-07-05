# Experimental & Dead Code Inventory

This document catalogs code that is currently **not imported or used by the live app** (src/index.tsx → src/main.ts → src/renderers.ts).

**Status legend**:
- **KEEP**: Reusable core components; earmarked for future lanes (e.g., Lane 2, 3).
- **QUARANTINE**: Speculative or exploratory; save for reference, move to archive branch if needed.
- **DELETE**: Clearly superseded or orthogonal; safe to remove without impact.
- **REWRITE**: Concept is sound but implementation is incomplete or uses deprecated APIs.

---

## trashPanda: Linear Algebra & Kernels (Keep)

**Path**: `src/trashPanda/`

### linalg.ts (KEEP)
```typescript
export function normalize(tensor: tf.Tensor)
```
- Simple vector normalization.
- May be reused in Lane 3 (GAN objectives, e.g., normalizing force covariance for isotropy loss).

### utils/kernels.ts (KEEP)
```typescript
export function gaussianKernel(M: number, std?: number, sym?: boolean): tf.Tensor
```
- Generates Gaussian window function.
- Use case: spectral loss windowing (Lane 3), or power-spectrum analysis.
- Note: Comment says "not yet in use" but the implementation is clean and the concept is sound.

### types.ts (KEEP)
```typescript
export interface Module {
  predict(x: tf.Tensor): tf.Tensor;
  trainableWeights: tf.Variable[];
}
```
- Generic neural module interface.
- Used by ScoreNetwork (below) and may be used by HelmholtzField (Lane 2).

---

## trashPanda: Score Network (Keep)

**Path**: `src/trashPanda/models/ScoreNetwork.ts`

### ScoreNetwork class (KEEP)
```typescript
export class ScoreNetwork implements Module {
  net: tf.Sequential;
  constructor({ hiddenUnits?: number[] });
  predict(x: tf.Tensor): tf.Tensor;
  get trainableWeights(): tf.Variable[];
}
```
- A 2D→2D MLP with tanh activation, trained for score matching.
- **Reuse in Lane 2**: Helmholtz potential φ (the smooth gradient component).
- Already well-integrated; no changes needed.

---

## trashPanda: Vision / NLP Models (Quarantine)

**Path**: `src/trashPanda/models/`

All of the following are **vision transformers, NLP embeddings, or auxiliary layers** that were exploratory tangents. **Not imported by main.ts. Not used in the live engine.**

| File | Purpose | Status | Reason |
|------|---------|--------|--------|
| clipTransformer.ts | CLIP (vision-language) variant | QUARANTINE | Vision + text matching; unrelated to particle physics. |
| mobilevit.ts | MobileViT (efficient vision transformer) | QUARANTINE | Image feature extraction; not applicable to 2D force fields. |
| rotaryEmbeddingTransformer.ts | Rotary position embeddings (RoPE) | QUARANTINE | NLP/sequence model component; force field models don't need sequence structure. |
| FastTransformer.ts | Fast attention variant | QUARANTINE | General transformer utility; not active. |

### Embeddings (Quarantine)
**Path**: `src/trashPanda/embeddings/`
- **bpe.ts**: Byte-pair encoding (text tokenization). Unrelated to particles.
- **geometricEmbeddings.ts**: Learned geometric position encodings. Unused.
- **rotaryEmbeddings.ts**: Rotary position encoding. Unused in live app.

### Layers (Quarantine)
**Path**: `src/trashPanda/layers/`
- **MultiheadAttention.ts**: Attention layer, useful for transformers, not for MLPs.
- **Windowed_attention.ts**: Windowed attention, same.

### Training Code (Quarantine)
**Path**: `src/trashPanda/train/`
- **gpt.ts**: GPT-style language model training. Unrelated.
- **fastTransformer.ts**: Training harness for FastTransformer. Unrelated.

### Experimental (Quarantine)
**Path**: `src/trashPanda/experimental/`
- **maxEmbeddingTransformer.ts**: Speculative transformer variant.
- **pointEmbeddings.ts**: Unused position encoding.
- **test.ts**: Ad-hoc tests.

**Action**: Move to a separate `archive/trashPanda-vision-nlp/` folder (or delete after archiving). These are intellectual artifacts; the force-field domain does not need vision or language modeling.

---

## agentSets: Multi-Agent Transformers (Delete)

**Path**: `src/agentSets/`

A collection of transformer-based multi-agent environment experiments:
- **set1.ts**: Generic agent set.
- **set_test.ts**: Test harness.
- **set_position_transformer.ts**: Transformer over agent positions.
- **max_embedding.ts**: Embedding-based agent reward.
- **rewards.ts**: Reward definitions.
- **transformer_depth.ts**: Depth-based transformer variant.

**Status**: **DELETE**

**Reason**: 
- These model multi-agent dynamics (swarms, game theory).
- The current live engine is a single unified particle system, not a competitive/cooperative agent ensemble.
- No imports from main.ts.
- The Helmholtz field (Lane 2) and GAN objectives (Lane 3) operate on a single force field, not agent policies.

**Migration path**: If multi-agent gameplay is desired in future, design from scratch; these are too tightly coupled to old architecture.

---

## quickDraw: tfjs→GPU Zero-Copy Renderer (Quarantine → Rewrite in Lane 1)

**Path**: `src/quickDraw/`

**Files**:
- main.ts (~270 lines): High-level commentary and example API.
- index.ts (~50 lines): Stub exports.

**Concept**: 
- A high-level GPU drawing API for generative artists (drawCircle, drawTriangle) backed by tfjs tensors.
- Intended to keep data on GPU, avoid CPU readback.
- Uses WebGL + twgl.js.

**Status**: **QUARANTINE → REWRITE**

**Reason**:
- The concept is sound and directly addresses the Lane 1 goal (zero-copy GPU rendering).
- Current implementation is incomplete (mostly pseudocode).
- WebGL is older; Lane 1 requires WebGPU for better interop with tfjs backends.
- Main.ts does not import quickDraw; instead, it calls pos.arraySync() and renders via Canvas2D (the exact bottleneck Lane 1 aims to fix).

**Action in Lane 1**: 
- Read quickDraw/main.ts for inspiration and API design philosophy.
- Rewrite using WebGPU instanced rendering (src/gpu/dataToGPU.ts, src/renderers/gpu-instanced.ts).
- Deprecate quickDraw folder after Lane 1 ships.

---

## draw: WebGPU Stub & Canvas2D (Quarantine → Keep Canvas2D)

**Path**: `src/draw/`

### draw_webgpu.ts (QUARANTINE)
- Attempts to draw agents using @swissgl/swissgl2 library.
- **Problem**: @swissgl/swissgl2 is not published; import fails.
- **Status**: Broken stub. Do not use.
- **Action**: Delete or archive. Lane 1 will provide a working WebGPU renderer.

### draw_webgl.ts (QUARANTINE)
- WebGL-based drawing harness.
- Incomplete; no active integration.
- **Action**: Delete unless needed for fallback rendering.

### draw_canvas2d.ts (AUDIT NEEDED)
- Canvas2D drawing functions.
- **Check**: Does src/renderers.ts use it? 
  - *No direct import found in renderers.ts.*
  - Might be dead code or leftover from old refactor.
- **Action**: If unused, delete. If used, mark as KEEP.

### colors.ts (AUDIT NEEDED)
- Color utilities.
- **Check**: Do any active renderers import it?
  - *Not found in renderers.ts imports.*
- **Action**: Audit and delete if orphaned.

---

## physics/updateParticles.ts (Delete)

**Path**: `src/physics/updateParticles.ts` (~276 lines)

**Functions**:
- `newParticles()`: Initialize particle positions.
- `newField()`: Initialize force field.
- `updateParticles2()`: Old physics integration step.
- `randomReset()`: Respawn particles.
- `clipField()`: Clip force magnitude.
- Etc.

**Status**: **DELETE**

**Reason**:
- Main.ts has its own inline implementations of particle initialization, physics stepping (physicsForward), and reset (randomReset).
- This file duplicates that logic with slightly different signatures.
- No imports from main.ts.
- If referenced from old/dead code, deletion won't break the live app.

**Action**:
- Before deleting, grep the codebase for imports: `grep -r "from.*physics/updateParticles" src/`
- If none found, delete.

---

## types/all.ts (Audit)

**Path**: `src/types/all.ts`

**Status**: **AUDIT NEEDED**

**Reason**: 
- Contains type definitions, possibly for old agent-set / vision models.
- Check what types are actually imported by main.ts.
- If only dead code uses it, delete.
- If any live code depends on it, keep.

**Command**:
```bash
grep -r "from.*types/all" src/
grep -r "AgentBatch\|AgentSetConfig" src/  # if types/all.ts exports these
```

---

## utils: Audit & Prune

**Path**: `src/utils/`

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| math.ts | Math utilities (sqrt, clamp, etc.) | ? | Audit usage. |
| color.ts | Color space conversions (Lab, RGB). | ? | Audit usage; may be needed by Lane 3 (spectral visualization). |
| tensor_utils.ts | Tensor reshaping, tiling. | ? | Audit usage. |
| traced.ts | Debugging/tracing utilities. | ? | Audit usage. |
| assert.ts | Assertion helpers. | ? | Audit usage. |
| env.ts | Environment config. | ? | Check if used by main.ts. |

**Action**: For each, grep for imports:
```bash
grep -r "from.*utils/math\|import.*math from" src/ | grep -v utils/math.ts
```
Then categorize as KEEP, QUARANTINE, or DELETE.

---

## Summary: Cleanup Checklist

- [ ] **Lane 0**: Confirm main is ancestor; no cleanup needed.
- [ ] **Lane 1**: Create src/gpu/ and src/renderers/gpu-instanced.ts (new files).
- [ ] **Lane 2**: Create src/models/HelmholtzField.ts (new file using ScoreNetwork).
- [ ] **Lane 3**: Create src/objectives/ganLosses.ts (new file, may use linalg/kernels).
- [ ] **Lane 4**: Update src/index.tsx and src/main.ts for UI controls (mods to existing).
- [ ] **Lane 5 (Triage)**:
  - [ ] Delete: agentSets/
  - [ ] Delete: physics/updateParticles.ts
  - [ ] Delete: draw/draw_webgpu.ts (broken import)
  - [ ] Archive/Delete: quickDraw/ (after Lane 1 ships)
  - [ ] Archive/Delete: trashPanda/embeddings/
  - [ ] Archive/Delete: trashPanda/models/clip*, mobilevit, rotaryEmbedding*
  - [ ] Archive/Delete: trashPanda/layers/
  - [ ] Archive/Delete: trashPanda/train/
  - [ ] Archive/Delete: trashPanda/experimental/
  - [ ] Audit & decide: draw/draw_canvas2d.ts, draw/colors.ts
  - [ ] Audit & decide: types/all.ts
  - [ ] Audit & decide: utils/

---

## Archival Strategy

Rather than delete, consider preserving exploratory work:
1. Create a branch `archive/pre-helmholtz-cleanup`.
2. Tag as `v0-with-exploratory-code`.
3. Delete from main, but keep tag for reference.

This allows future researchers to understand the exploration history without cluttering the live codebase.

---

## Notes

- **Why so much NLP/vision stuff?** These files suggest previous experiments with multi-modal learning (e.g., CLIP for guiding particle motion via text) or vision-based feedback. The pivot to pure physics + GAN objectives made these tangents obsolete.
- **Helmholtz decomposition insight**: The cleanest formulation for learned force fields. Once deployed, the decomposition can be visualized separately (potential heatmap + vorticity colorwheel), enabling deeper interpretability than a black-box MLP.
