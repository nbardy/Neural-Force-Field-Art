# Configurable field architecture

**Status:** shipped  
**Goal:** Make force-field architecture orthogonal to loss/renderer; replace
weak `mlpWide` (256→2); expose encoding / depth / SIREN / hashgrid as selectable
options; rewire Galaxy / Spiral Cover onto real nets.

## Shipped

### `src/core/field/arch.ts`
- Declarative `FieldArch`: encoding (`raw|fourier|hashgrid`), activation
  (`selu|sin`), `hiddenUnits`, `heads: 1|2`, grid/fourier/siren knobs.
- Named `ARCH` presets: `mlp256`, `mlpDeep`, `mlpShallow`, `fourier`,
  `fourierWide`, `siren`, `hashgrid`, plus dual-head chaos/adversary presets.
- `createFieldFromArch` → `HelmholtzField` (runtime; name still legacy).
- `ARCH_DOCK_PRESETS` for the model dock.

### `HelmholtzField`
- `heads?: 1 | 2` — single-head forces = head A only; α forced to 0; second
  head still allocated for fused WGSL dual-head layout.

### `ArtPieceConfig`
- `fieldArch`, `archEditable`
- `fieldLoss` omitted ⇒ **skip fused trainer** (keeps tfjs `computeLoss` for
  Spiral Cover / Galaxy so chaos default cannot replace aesthetic losses)

### Gallery
- Spiral / Vortex / Galaxy / Spiral Cover → `fieldArch` (single-head), editable
- Arch × look variants collapsed into dock axes (see gallery recipe cleanup note)
- Neural Field · Max Chaos → dual dock (SIREN/Fourier/HashGrid no longer separate rows)
- Chaos / adversary pieces → `createFieldFromArch(ARCH.dual*)`
- Removed unused sigmoid `mlpShallow/Deep/Wide` factories

### UI
- Model section: arch summary + preset chips (restarts loop)

## Verified
- `tsc --noEmit` clean
- Parcel rebuild OK

## Open
- [x] Fused spiral-cover loss (`COVER_FIELD_LOSS` / `W_COVER`)
- [x] True single-head WGSL kind (`FieldSpec.kind: "vector"`)
- Fourier+SIREN combo not supported
