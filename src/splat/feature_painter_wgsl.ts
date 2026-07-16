/**
 * Tiled 2D Feature Painter raster shaders.
 *
 * A splat owns three vec32 latent banks: base, local-x, and local-y. At a
 * pixel it emits `base + ux * localX + uy * localY`, where (ux, uy) is the
 * point in a conic-normalized local frame. The 32D field is alpha composited
 * and decoded to RGB by Feature32Colorizer. Local-coordinate derivatives into
 * geometry are intentionally stopped in this first renderer; geometry still
 * receives the exact alpha-compositing gradient, while the local banks learn
 * mark appearance. This keeps the first live path stable and leaves a clearly
 * isolated upgrade point for the full appearance Jacobian.
 */
import {
  ALPHA_THRESHOLD, DERIVED_STRIDE, EPS, MAX_ALPHA, PARAM_STRIDE, TILE,
  TRANSMITTANCE_CUTOFF, type RasterConfig, resolveDims,
} from "./raster_wgsl";

export const FEATURE_CHANNELS = 32;
export const FEATURE_GROUPS = FEATURE_CHANNELS / 4;
export const FEATURE_BANKS = 3; // base, local-x, local-y
export const FEATURE_STRIDE = FEATURE_CHANNELS * FEATURE_BANKS;

const f = (x: number) => /[.eE]/.test(String(x)) ? String(x) : `${x}.0`;
const u = (x: number) => `${x >>> 0}u`;

function common(cfg: RasterConfig) {
  const d = resolveDims(cfg);
  return { d, hw: d.H * d.W, p: `
const FEATURE_GROUPS : u32 = ${u(FEATURE_GROUPS)};
const FEATURE_STRIDE : u32 = ${u(FEATURE_STRIDE)};
fn featureBase(g : u32, bank : u32, group : u32) -> u32 {
  return (g * ${u(FEATURE_BANKS)} + bank) * FEATURE_GROUPS + group;
}
fn fixadd(dst : ptr<storage, array<atomic<i32>>, read_write>, index : u32, v : f32) {
  atomicAdd(&(*dst)[index], i32(clamp(round(v * ${f(d.gradScale)}), -2.14e9, 2.14e9)));
}
` };
}

/** Feature composite. It reuses the RGB raster's proven prep/binning layout. */
export function featureForwardShader(cfg: RasterConfig): string {
  const { d, hw, p } = common(cfg);
  return /* wgsl */ `
${p}
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> derived : array<f32>;
@group(0) @binding(3) var<storage, read> splatFeatures : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> imageFeatures : array<f32>;
@group(0) @binding(5) var<storage, read_write> tileStop : array<u32>;

var<workgroup> shIds : array<u32, ${u(d.cap)}>;
var<workgroup> shMaxStop : atomic<u32>;
fn nextPow2(x : u32) -> u32 { var v = max(x, 1u) - 1u; v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u; return v + 1u; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${u(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${u(d.cap)});
  let start = tileId * ${u(d.cap)};
  let sortN = nextPow2(count);
  for (var i = tid; i < sortN; i += 256u) { shIds[i] = select(0xffffffffu, binnedIds[start + i], i < count); }
  if (tid == 0u) { atomicStore(&shMaxStop, 0u); }
  workgroupBarrier();
  var k = 2u;
  loop { if (k > sortN) { break; } var j = k >> 1u;
    loop { if (j == 0u) { break; } let pairs = sortN >> 1u;
      for (var pair = tid; pair < pairs; pair += 256u) { let pos = 2u * j * (pair / j) + (pair % j); let ixj = pos + j; let asc = (pos & k) == 0u; let va = shIds[pos]; let vb = shIds[ixj]; if ((va > vb) == asc) { shIds[pos] = vb; shIds[ixj] = va; } }
      workgroupBarrier(); j >>= 1u;
    } k <<= 1u;
  }
  for (var i = tid; i < count; i += 256u) { binnedIds[start + i] = shIds[i]; }
  workgroupBarrier();
  let tileX = tileId % ${u(d.tilesX)}; let tileY = tileId / ${u(d.tilesX)};
  let x = tileX * ${u(TILE)} + (tid % ${u(TILE)}); let y = tileY * ${u(TILE)} + (tid / ${u(TILE)});
  var localStop = 0u;
  if (x < ${u(d.W)} && y < ${u(d.H)}) {
    let pxc = f32(x) + 0.5; let pyc = f32(y) + 0.5;
    var acc : array<vec4f, ${u(FEATURE_GROUPS)}>;
    for (var group = 0u; group < FEATURE_GROUPS; group++) { acc[group] = vec4f(0.0); }
    var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let g = shIds[i]; let db = g * ${u(DERIVED_STRIDE)};
      let dx = pxc - derived[db]; let dy = pyc - derived[db + 1u];
      let a = derived[db + 2u]; let b = derived[db + 3u]; let c = derived[db + 4u];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[db + 8u] * exp(power); let alpha = min(${f(MAX_ALPHA)}, raw);
      if (alpha < ${f(ALPHA_THRESHOLD)}) { continue; }
      // The conic gives a stable normalized local frame without a second raw
      // parameter buffer in the hot pass. The full rotated-frame Jacobian is
      // deliberately deferred with the stopped appearance-geometry gradient.
      let ux = clamp(dx * sqrt(max(a, 1e-8)), -3.0, 3.0);
      let uy = clamp(dy * sqrt(max(c, 1e-8)), -3.0, 3.0);
      let w = T * alpha;
      for (var group = 0u; group < FEATURE_GROUPS; group++) {
        let z = splatFeatures[featureBase(g, 0u, group)] + ux * splatFeatures[featureBase(g, 1u, group)] + uy * splatFeatures[featureBase(g, 2u, group)];
        acc[group] += w * z;
      }
      T *= 1.0 - alpha;
      if (T < ${f(TRANSMITTANCE_CUTOFF)}) { break; }
    }
    let pixel = y * ${u(d.W)} + x;
    for (var group = 0u; group < FEATURE_GROUPS; group++) {
      let v = acc[group]; let channel = group * 4u;
      imageFeatures[(channel + 0u) * ${u(hw)} + pixel] = v.x;
      imageFeatures[(channel + 1u) * ${u(hw)} + pixel] = v.y;
      imageFeatures[(channel + 2u) * ${u(hw)} + pixel] = v.z;
      imageFeatures[(channel + 3u) * ${u(hw)} + pixel] = v.w;
    }
  }
  atomicMax(&shMaxStop, localStop); workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&shMaxStop); }
}`;
}

/** Reverse alpha recurrence with fixed-point feature gradients. */
export function featureBackwardShader(cfg: RasterConfig): string {
  const { d, p } = common(cfg);
  return /* wgsl */ `
${p}
@group(0) @binding(0) var<storage, read> imageFeatureGrad : array<f32>;
@group(0) @binding(1) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read> binnedIds : array<u32>;
@group(0) @binding(3) var<storage, read> tileStop : array<u32>;
@group(0) @binding(4) var<storage, read> derived : array<f32>;
@group(0) @binding(5) var<storage, read> splatFeatures : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> accGeom : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> accFeature : array<atomic<i32>>;

var<workgroup> shIds : array<u32, ${u(d.cap)}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x; if (tileId >= ${u(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${u(d.cap)}); let stopc = min(count, tileStop[tileId]); let start = tileId * ${u(d.cap)};
  for (var i = tid; i < stopc; i += 256u) { shIds[i] = binnedIds[start + i]; }
  workgroupBarrier();
  let tileX = tileId % ${u(d.tilesX)}; let tileY = tileId / ${u(d.tilesX)};
  let x = tileX * ${u(TILE)} + (tid % ${u(TILE)}); let y = tileY * ${u(TILE)} + (tid / ${u(TILE)});
  if (x >= ${u(d.W)} || y >= ${u(d.H)}) { return; }
  let pxc = f32(x) + 0.5; let pyc = f32(y) + 0.5; let pixel = y * ${u(d.W)} + x;
  var go : array<vec4f, ${u(FEATURE_GROUPS)}>;
  for (var group = 0u; group < FEATURE_GROUPS; group++) { let ch = group * 4u; go[group] = vec4f(imageFeatureGrad[(ch + 0u) * ${u(d.H * d.W)} + pixel], imageFeatureGrad[(ch + 1u) * ${u(d.H * d.W)} + pixel], imageFeatureGrad[(ch + 2u) * ${u(d.H * d.W)} + pixel], imageFeatureGrad[(ch + 3u) * ${u(d.H * d.W)} + pixel]); }
  var T = 1.0; var endi = stopc;
  for (var i = 0u; i < stopc; i++) { let g = shIds[i]; let db = g * ${u(DERIVED_STRIDE)}; let dx = pxc-derived[db]; let dy = pyc-derived[db+1u]; let a=derived[db+2u]; let b=derived[db+3u]; let c=derived[db+4u]; let power=-0.5*(a*dx*dx+2.0*b*dx*dy+c*dy*dy); if (power > 0.0) { continue; } let alpha=min(${f(MAX_ALPHA)}, derived[db+8u]*exp(power)); if (alpha < ${f(ALPHA_THRESHOLD)}) { continue; } T *= 1.0-alpha; if (T < ${f(TRANSMITTANCE_CUTOFF)}) { endi=i+1u; break; } }
  var Tcur = T; var gT = 0.0;
  for (var ii = i32(endi)-1; ii >= 0; ii--) {
    let g = shIds[u32(ii)]; let db = g * ${u(DERIVED_STRIDE)}; let dx=pxc-derived[db]; let dy=pyc-derived[db+1u]; let a=derived[db+2u]; let b=derived[db+3u]; let c=derived[db+4u]; let power=-0.5*(a*dx*dx+2.0*b*dx*dy+c*dy*dy); if (power > 0.0) { continue; } let op=derived[db+8u]; let raw=op*exp(power); let alpha=min(${f(MAX_ALPHA)},raw); if(alpha<${f(ALPHA_THRESHOLD)}) { continue; }
    let denom=max(1.0-alpha,${f(EPS)}); let Tprev=Tcur/denom;
    let ux=clamp(dx*sqrt(max(a,1e-8)),-3.0,3.0); let uy=clamp(dy*sqrt(max(c,1e-8)),-3.0,3.0);
    var dotgz = 0.0;
    for (var group=0u; group<FEATURE_GROUPS; group++) { let z=splatFeatures[featureBase(g,0u,group)]+ux*splatFeatures[featureBase(g,1u,group)]+uy*splatFeatures[featureBase(g,2u,group)]; dotgz += dot(go[group],z); let gf=go[group]*(Tprev*alpha); let base=featureBase(g,0u,group); fixadd(&accFeature, base+0u, gf.x); fixadd(&accFeature, base+1u, gf.y); fixadd(&accFeature, base+2u, gf.z); fixadd(&accFeature, base+3u, gf.w); fixadd(&accFeature, base+FEATURE_GROUPS, gf.x*ux); fixadd(&accFeature, base+FEATURE_GROUPS+1u, gf.y*ux); fixadd(&accFeature, base+FEATURE_GROUPS+2u, gf.z*ux); fixadd(&accFeature, base+FEATURE_GROUPS+3u, gf.w*ux); fixadd(&accFeature, base+2u*FEATURE_GROUPS, gf.x*uy); fixadd(&accFeature, base+2u*FEATURE_GROUPS+1u, gf.y*uy); fixadd(&accFeature, base+2u*FEATURE_GROUPS+2u, gf.z*uy); fixadd(&accFeature, base+2u*FEATURE_GROUPS+3u, gf.w*uy); }
    let gAlpha=Tprev*(dotgz-gT); let gate=select(0.0,1.0,raw<${f(MAX_ALPHA)}); let gRaw=gAlpha*gate; let gPower=gRaw*raw; let gdx=gPower*(-(a*dx+b*dy)); let gdy=gPower*(-(b*dx+c*dy));
    fixadd(&accGeom,db+2u,gPower*(-0.5)*dx*dx); fixadd(&accGeom,db+3u,gPower*(-1.0)*dx*dy); fixadd(&accGeom,db+4u,gPower*(-0.5)*dy*dy); fixadd(&accGeom,db+0u,-gdx); fixadd(&accGeom,db+1u,-gdy); fixadd(&accGeom,db+8u,gRaw*(raw/max(op,${f(EPS)})));
    gT=alpha*dotgz+(1.0-alpha)*gT; Tcur=Tprev;
  }
}`;
}

export function featureChainShader(cfg: RasterConfig): string {
  const { d } = common(cfg);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read_write> grad : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) { let i=gid.x; if (i >= ${u(d.G * FEATURE_STRIDE)}) { return; } grad[i]=f32(acc[i])*${f(1/d.gradScale)}; }
`;
}
