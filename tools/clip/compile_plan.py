# tools/clip/compile_plan.py — κ for the fused WGSL MobileCLIP-S0 vision encoder.
#
# Run:  uv run --with onnx --with numpy python tools/clip/compile_plan.py
#
# Canonicalizes the ONNX graph (518 nodes of Conv/decomposed-GELU/attention
# plumbing) into a typed execution plan of FOUR step kinds — conv / se /
# attention / head — plus one packed f32 weights blob. The WGSL side
# (src/clip/vision_wgsl.ts) generates one specialized shader per step from
# this plan; it never inspects ONNX. House rules apply: every node must be
# consumed by a recognized idiom or compilation THROWS — no silent fallbacks,
# no default: swallowing unknown ops.
#
# Weight packing (all segments padded to 16B / 4-float boundaries):
#   pointwise 1x1 g=1 : [Cout,Cin,1,1] stored TRANSPOSED [Cin][Cout] so the
#                       kernel vec4-loads 4 consecutive couts per input chan
#   depthwise g=C     : [C,1,k,k] flattened [C][k*k]
#   general grouped   : [Cout,cpg,k,k] as-is
#   attention         : BN folded to scale/shift; qkv [Cin][3C] as-is (column
#                       o = part*C + head*32 + d matches the kernel's qkv
#                       buffer layout exactly); proj [Cin][Cout] as-is
#   layer_scale       : [C,1,1] → [C]
#
# Outputs (models/mobileclip_s0/, gitignored, regenerable):
#   plan.json    — slots (sizes in floats) + steps (typed, offsets in floats)
#   weights.bin  — one packed f32 blob (bind whole as read-only storage)
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODEL_DIR = Path("models/mobileclip_s0")
model = onnx.load(str(MODEL_DIR / "vision_model.onnx"))
graph = model.graph

# --train : emit the training plan (plan_train.json + weights_train.bin) for the
# hand-written WGSL backward (dL/dpixels, weights FROZEN — no dW). Differences
# from the inference plan (see docs/clip_backward_spec.md §1):
#   - NO slot reuse: every activation gets a unique slot (saved for backward).
#   - GELU is SPLIT: the conv/se emits act:"none" into its own (pre-activation)
#     slot, followed by a standalone {kind:"gelu"} elementwise step. Both keep a
#     real ONNX ref tensor, so per-step forward verification still bisects.
#   - Pointwise convs ALSO pack the untransposed [Cout][Cin] weights (wOffT) that
#     dX = Wᵀ·dY needs; layer-scale γ is folded into that transposed copy.
#   - A `backward:[...]` reverse step list + mirrored grad slots are emitted.
TRAIN = "--train" in sys.argv

inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
consts = {}
for n in graph.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    if default is None:
        raise KeyError(f"{node.name}: missing attr {name}")
    return default


# Attention-internal plumbing ops: shape bookkeeping that a hand-written
# kernel makes vanish. They are only legal between recognized anchors; the
# matcher checks every anchor's wiring so plumbing can't hide semantics.
PLUMBING = {"Constant", "Shape", "Gather", "Unsqueeze", "Concat", "Slice",
            "Squeeze", "Reshape", "Transpose", "Split", "Mul"}

# Shape/Gather/Unsqueeze/Concat/Slice only ever compute shape tuples here —
# with every shape static they are dead code, and keeping them would inflate
# slot refcounts (Shape "consumes" its data tensor). Drop them globally.
DEAD = {"Constant", "Shape", "Gather", "Unsqueeze", "Concat", "Slice"}
nodes = [n for n in graph.node if n.op_type not in DEAD]

# ---------------------------------------------------------------------------
# Weight packer
# ---------------------------------------------------------------------------
packed: list[np.ndarray] = []
packed_len = 0


def pack(arr: np.ndarray) -> int:
    """Append f32 array to the blob at a 4-float boundary; return float offset."""
    global packed_len
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    off = (packed_len + 3) & ~3
    if off != packed_len:
        packed.append(np.zeros(off - packed_len, dtype=np.float32))
    packed.append(a)
    packed_len = off + a.size
    return off


# ---------------------------------------------------------------------------
# Slot allocator — tensor name → buffer slot, freed at last use.
# Consumer counts come from the raw graph so fused idioms can't leak slots.
# ---------------------------------------------------------------------------
consumers: dict[str, int] = {}
for n in nodes:
    for i in n.input:
        if i and i not in inits and i not in consts:
            consumers[i] = consumers.get(i, 0) + 1

slot_sizes: list[int] = []          # floats
free_slots: list[int] = []
tensor_slot: dict[str, int] = {}
tensor_refs: dict[str, int] = {}


def alloc(name: str, floats: int) -> int:
    if free_slots and not TRAIN:
        # take the smallest free slot that fits, else grow the largest
        fit = [s for s in free_slots if slot_sizes[s] >= floats]
        s = min(fit, key=lambda s: slot_sizes[s]) if fit else max(free_slots, key=lambda s: slot_sizes[s])
        free_slots.remove(s)
        slot_sizes[s] = max(slot_sizes[s], floats)
    else:
        # TRAIN: never reuse — every activation keeps a unique slot for backward.
        s = len(slot_sizes)
        slot_sizes.append(floats)
    tensor_slot[name] = s
    tensor_refs[name] = consumers.get(name, 0)
    return s


def use(name: str) -> int:
    """Read a tensor's slot, releasing it after its last consumer (inference
    only — in TRAIN mode nothing is freed so activations survive for backward)."""
    s = tensor_slot[name]
    tensor_refs[name] -= 1
    if tensor_refs[name] == 0 and not TRAIN:
        free_slots.append(s)
    return s


# ---------------------------------------------------------------------------
# Shape tracking: (C, H, W) per tensor name, batch=1 canonical.
# ---------------------------------------------------------------------------
shapes: dict[str, tuple] = {"pixel_values": (3, 256, 256)}
INPUT = "pixel_values"
alloc(INPUT, 3 * 256 * 256)
# PIN the input slot (phantom +1 consumer): run() must be repeatable — the
# live loop re-submits every frame. Without this the allocator hands slot 0
# to step 1's output, and the SECOND forward reads last frame's activations
# as the image (bit-identical garbage; cost a debugging session — see
# tools/clip/fused_test.ts history).
tensor_refs[INPUT] = consumers.get(INPUT, 0) + 1

steps: list[dict] = []
scratch_qkv: int | None = None
scratch_attn: int | None = None
cur = 0  # cursor into nodes


def peek(k=0):
    return nodes[cur + k] if cur + k < len(nodes) else None


def op(k=0):
    n = peek(k)
    return n.op_type if n is not None else None


def expect(cond, msg):
    if not cond:
        n = peek()
        raise SystemExit(f"compile_plan: {msg} (at node {cur}: {n.op_type} {n.name if n else ''})")


def match_gelu(src_name):
    """Div,Erf,Add,Mul,Mul decomposition of x·0.5·(1+erf(x/√2)). Returns output name."""
    global cur
    seq = [op(i) for i in range(5)]
    if seq != ["Div", "Erf", "Add", "Mul", "Mul"]:
        return None
    div = peek()
    if div.input[0] != src_name:
        return None
    expect(abs(float(consts[div.input[1]]) - math.sqrt(2)) < 1e-6, "GELU Div const != sqrt(2)")
    out = peek(4).output[0]
    cur += 5
    return out


def conv_geometry(node):
    w = inits[node.input[1]]
    b = inits[node.input[2]] if len(node.input) > 2 else None
    expect(b is not None, f"conv {node.name} missing bias")
    groups = attr(node, "group", 1)
    k = attr(node, "kernel_shape")[0]
    stride = attr(node, "strides")[0]
    pads = attr(node, "pads")
    cin = shapes[node.input[0]][0]
    cout = w.shape[0]
    expect(w.shape[1] == cin // groups, f"conv {node.name} weight/group mismatch")
    expect(pads[0] == pads[1] == pads[2] == pads[3], f"conv {node.name} asymmetric pads")
    return w, b, groups, k, stride, pads[0], cin, cout


def emit_conv(node, act="none", layer_scale=None, residual_name=None):
    """One canonical conv step; variant is decided by (groups, k) — the WGSL
    side type-dispatches on it. Returns output tensor name."""
    w, b, groups, k, stride, pad, cin, cout = conv_geometry(node)
    _, h, wd = shapes[node.input[0]]
    oh, ow = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    if groups == 1 and k == 1:
        variant, wpack = "pointwise", w.reshape(cout, cin).T  # → [Cin][Cout]
    elif groups == cin and cout == cin:
        variant, wpack = "depthwise", w.reshape(cout, k * k)
    else:
        variant, wpack = "general", w
    # dst is allocated BEFORE inputs are released: a slot still being read
    # can never be handed out as the output (same-dispatch read/write race).
    out_name = node.output[0] if layer_scale is None else layer_scale["out"]
    dst = alloc(out_name, cout * oh * ow)
    src = use(node.input[0])
    residual = use(residual_name) if residual_name is not None else None
    shapes[out_name] = (cout, oh, ow)
    wOff = pack(wpack)
    bOff = pack(b)
    layerScaleOff = pack(layer_scale["gamma"]) if layer_scale else None
    # TRAIN: dX = Wᵀ·dY reuses the pointwise kernel reading the OTHER orientation
    # [Cout][Cin]. Fold the layer-scale γ (a per-output-channel scale on dY that
    # the kernel would otherwise apply on load) directly into the packed row —
    # exact algebra, one fewer runtime scale (docs/clip_backward_spec.md §2 pw_bwd).
    wOffT = None
    if TRAIN and variant == "pointwise":
        wt = np.ascontiguousarray(wpack.T)              # [Cout][Cin]
        if layer_scale is not None:
            wt = wt * layer_scale["gamma"].reshape(-1)[:, None]
        wOffT = pack(np.ascontiguousarray(wt))
    steps.append({
        "kind": "conv", "variant": variant, "name": node.name,
        "cin": cin, "cout": cout, "k": k, "stride": stride, "pad": pad,
        "groups": groups, "h": h, "w": wd, "outH": oh, "outW": ow,
        "src": src, "dst": dst, "wOff": wOff, "bOff": bOff, "wOffT": wOffT,
        "act": act, "residual": residual,
        "layerScaleOff": layerScaleOff,
        "ref": out_name,
    })
    return out_name


def emit_gelu(pre_name: str, post_name: str, c: int, h: int, w: int) -> str:
    """TRAIN split-GELU: standalone elementwise step reading the pre-activation
    (Conv/SE output, saved for gelu-backward) and writing the activated tensor.
    Both `pre_name` and `post_name` are real ONNX tensors → both verifiable."""
    dst = alloc(post_name, c * h * w)
    src = use(pre_name)
    shapes[post_name] = (c, h, w)
    steps.append({
        "kind": "gelu", "name": post_name + ":gelu",
        "src": src, "dst": dst, "n": c * h * w, "c": c, "h": h, "w": w,
        "ref": post_name,
    })
    return post_name


def try_layer_scale_residual():
    """Mul(init [C,1,1]) then Add — the RepMixer/attention block tail."""
    if op() != "Mul" or op(1) != "Add":
        return None
    mul = peek()
    gamma_name = next((i for i in mul.input if i in inits), None)
    if gamma_name is None or "layer_scale" not in gamma_name:
        return None
    add = peek(1)
    expect(mul.output[0] in add.input, "layer_scale Mul not feeding Add")
    res_name = next(i for i in add.input if i != mul.output[0])
    return {"gamma": inits[gamma_name].reshape(-1), "res": res_name, "out": add.output[0]}


def match_se(src_name, c, h, w):
    """SE gate: GAP → fc1 → Relu → fc2 → Sigmoid → Mul. GAP is either
    ReduceMean or the Pad+AveragePool spelling (identical on our static 8×8)."""
    global cur
    if op() == "ReduceMean":
        cur += 1
    elif op() == "Pad" and op(1) == "AveragePool":
        expect(h == attr(peek(1), "kernel_shape")[0], "AveragePool != global on static shape")
        cur += 2
    else:
        expect(False, "SE: expected ReduceMean or Pad+AveragePool")
    expect(op() == "Conv" and op(1) == "Relu" and op(2) == "Conv" and op(3) == "Sigmoid",
           "SE: expected Conv,Relu,Conv,Sigmoid")
    fc1, fc2 = peek(), peek(2)
    cur += 4
    if op() == "Reshape":  # conv_exp spelling reshapes [B,C] back to [B,C,1,1]
        cur += 1
    expect(op() == "Mul", "SE: expected final Mul")
    mul = peek()
    expect(src_name in mul.input, "SE Mul does not scale its own input")
    cur += 1
    se_pre = mul.output[0]                 # SE Mul output = pre-GELU (real ONNX)
    w1, b1 = inits[fc1.input[1]], inits[fc1.input[2]]
    w2, b2 = inits[fc2.input[1]], inits[fc2.input[2]]
    cmid = w1.shape[0]
    g = match_gelu(se_pre)                 # advances cur past the GELU decomp if present
    if TRAIN:
        act, se_out = "none", se_pre       # keep se pre-activation; split GELU below
    else:
        act, se_out = ("gelu", g) if g is not None else ("none", se_pre)
    dst = alloc(se_out, c * h * w)  # before use(): no src/dst aliasing
    src = use(src_name)
    shapes[se_out] = (c, h, w)
    steps.append({
        "kind": "se", "name": mul.name, "c": c, "cmid": cmid, "h": h, "w": w,
        "src": src, "dst": dst,
        "w1Off": pack(w1.reshape(cmid, c)), "b1Off": pack(b1),
        "w2Off": pack(w2.reshape(c, cmid)), "b2Off": pack(b2),
        "act": act, "ref": se_out,
    })
    if TRAIN and g is not None:
        return emit_gelu(se_pre, g, c, h, w)
    return se_out


def match_attention():
    """BatchNormalization anchor → one fused attention step (3 dispatches at
    runtime). Verifies: prenorm wiring, qkv/proj anchors, q-scale const,
    softmax axis, residual = BN input, trailing layer_scale."""
    global cur
    bn = peek()
    x_name = bn.input[0]
    c, h, w = shapes[x_name]
    n_tok = h * w
    gamma, beta, mean, var = (inits[bn.input[i]] for i in (1, 2, 3, 4))
    eps = attr(bn, "epsilon", 1e-5)
    bn_scale = gamma / np.sqrt(var + eps)
    bn_shift = beta - mean * bn_scale
    cur += 1

    qkv_w = proj_w = proj_b = None
    scale_const = None
    softmax_seen = False
    matmul_count = 0
    while True:
        node = peek()
        expect(node is not None, "attention: ran off graph")
        if node.op_type == "MatMul":
            matmul_count += 1
            init_in = next((i for i in node.input if i in inits), None)
            if init_in is not None:
                if qkv_w is None:
                    qkv_w = inits[init_in]
                else:
                    proj_w = inits[init_in]
        elif node.op_type == "Softmax":
            expect(attr(node, "axis") == 3, "attention: softmax axis != 3")
            softmax_seen = True
        elif node.op_type == "Add":
            init_in = next((i for i in node.input if i in inits), None)
            if init_in is not None and proj_w is not None and proj_b is None:
                proj_b = inits[init_in]
            else:
                break  # the residual Add — handled below
        elif node.op_type == "Mul":
            cnst = next((i for i in node.input if i in consts), None)
            gamma_in = next((i for i in node.input if i in inits and "layer_scale" in i), None)
            if cnst is not None:
                scale_const = float(consts[cnst])
            elif gamma_in is not None:
                break  # layer_scale Mul — handled below
        else:
            expect(node.op_type in PLUMBING, f"attention: unexpected op {node.op_type}")
        cur += 1

    ls = try_layer_scale_residual()
    expect(ls is not None, "attention: expected layer_scale Mul+Add tail")
    expect(ls["res"] == x_name, "attention: residual is not the pre-norm input")
    cur += 2
    expect(qkv_w is not None and proj_w is not None and proj_b is not None, "attention: anchors missing")
    expect(softmax_seen and matmul_count == 4, "attention: wrong MatMul/Softmax count")
    heads, hd = 16, 32
    expect(qkv_w.shape == (c, 3 * c) and proj_w.shape == (c, c), "attention: weight shapes")
    expect(abs(scale_const - hd ** -0.5) < 1e-6, f"attention: q-scale {scale_const} != 1/sqrt({hd})")

    out_name = ls["out"]
    dst = alloc(out_name, c * h * w)  # before use(): no src/dst aliasing
    shapes[out_name] = (c, h, w)
    # Two dedicated scratch slots: qkv [3C][nTok] channel-planar (channel
    # o = part*C + head*hd + d), attnout [C][nTok] channel-planar. In inference
    # they are shared by all (sequential) attention steps; in TRAIN each block
    # gets its own pair so qkv is saved for attn_core-backward.
    global scratch_qkv, scratch_attn
    if TRAIN:
        sq = len(slot_sizes); slot_sizes.append(3 * c * n_tok)
        sa = len(slot_sizes); slot_sizes.append(n_tok * c)
    else:
        if scratch_qkv is None:
            scratch_qkv = len(slot_sizes)
            slot_sizes.append(3 * c * n_tok)
            scratch_attn = len(slot_sizes)
            slot_sizes.append(n_tok * c)
        sq, sa = scratch_qkv, scratch_attn

    # The two attention matmuls ARE pointwise convs over nTok "pixels", so
    # they reuse the tiled pointwise kernel. Everything affine is folded into
    # the qkv weights at compile time (exact algebra, zero runtime cost):
    #   BN prenorm:  y = Σ_ci (x_ci·s_ci + t_ci)·W[ci,o]
    #              = Σ_ci x_ci·(s_ci·W[ci,o])  +  Σ_ci t_ci·W[ci,o]  (bias)
    #   q pre-scale: scale the q-part columns of W and bias by 1/sqrt(hd).
    qkv_folded = qkv_w * bn_scale[:, None]
    qkv_bias = bn_shift @ qkv_w
    qkv_folded[:, :c] *= scale_const
    qkv_bias[:c] *= scale_const
    x_slot_qkv = use(x_name)   # x has 2 graph consumers (BN + residual Add):
    x_slot_res = use(x_name)   # one use() per emitted reader keeps counts honest
    assert x_slot_qkv == x_slot_res
    qkv_wOff = pack(qkv_folded)
    qkv_bOff = pack(qkv_bias)
    # TRAIN transposed copies (see emit_conv): qkv unfused, proj folds γ.
    qkv_wOffT = pack(np.ascontiguousarray(qkv_folded.T)) if TRAIN else None
    steps.append({
        "kind": "conv", "variant": "pointwise", "name": bn.name + ":qkv",
        "cin": c, "cout": 3 * c, "k": 1, "stride": 1, "pad": 0, "groups": 1,
        "h": h, "w": w, "outH": h, "outW": w,
        "src": x_slot_qkv, "dst": sq,
        "wOff": qkv_wOff, "bOff": qkv_bOff, "wOffT": qkv_wOffT,
        "act": "none", "residual": None, "layerScaleOff": None,
        "ref": None,   # layout differs from any ONNX tensor — verified via block output
    })
    steps.append({
        "kind": "attn_core", "name": bn.name + ":core",
        "c": c, "heads": heads, "hd": hd, "nTok": n_tok,
        "src": sq, "dst": sa,
        "ref": None,
    })
    proj_wOff = pack(proj_w)
    proj_bOff = pack(proj_b)
    proj_wOffT = pack(np.ascontiguousarray(proj_w.T * ls["gamma"].reshape(-1)[:, None])) if TRAIN else None
    steps.append({
        "kind": "conv", "variant": "pointwise", "name": bn.name + ":proj",
        "cin": c, "cout": c, "k": 1, "stride": 1, "pad": 0, "groups": 1,
        "h": h, "w": w, "outH": h, "outW": w,
        "src": sa, "dst": dst,
        "wOff": proj_wOff, "bOff": proj_bOff, "wOffT": proj_wOffT,
        "act": "none", "residual": x_slot_res,
        "layerScaleOff": pack(ls["gamma"]),
        "ref": out_name,
    })
    return out_name


def match_head(src_name):
    global cur
    expect(op() == "ReduceMean" and op(1) == "MatMul", "head: expected ReduceMean,MatMul")
    rm, mm = peek(), peek(1)
    expect(list(attr(rm, "axes")) == [-2, -1] and attr(rm, "keepdims") == 0, "head: pool axes")
    w = inits[next(i for i in mm.input if i in inits)]
    c, h, wd = shapes[src_name]
    expect(w.shape[0] == c, "head: proj rows != channels")
    cur += 2
    out_name = mm.output[0]
    dst = alloc(out_name, int(w.shape[1]))  # before use(): no src/dst aliasing
    src = use(src_name)
    steps.append({
        "kind": "head", "name": mm.name, "cin": c, "cout": int(w.shape[1]),
        "h": h, "w": wd, "src": src, "dst": dst, "wOff": pack(w),
        "ref": out_name,
    })
    return out_name


# ---------------------------------------------------------------------------
# The spine walk — thin dispatcher, one matcher per idiom.
# ---------------------------------------------------------------------------
while cur < len(nodes):
    node = peek()
    if node.op_type == "Conv":
        cur += 1
        src_name = node.input[0]
        out_name = node.output[0]
        # SE directly after a conv (downsample lkb / conv_exp)?
        if op() in ("ReduceMean", "Pad") and peek().input[0] == out_name:
            emit_conv(node, act="none")
            cshape = shapes[out_name]
            match_se(out_name, *cshape)
            continue
        # trailing GELU?
        save = cur
        g = match_gelu(out_name)
        if g is not None:
            if TRAIN:
                # split: conv act:none into its own pre-activation slot (ref =
                # the Conv output), then a standalone GELU step (ref = the Mul_1
                # output) — both real ONNX tensors, both per-step verifiable.
                emit_conv(node, act="none")
                cc, hh, ww = shapes[out_name]
                emit_gelu(out_name, g, cc, hh, ww)
                continue
            emit_conv(node, act="gelu")
            # rename: emit_conv registered node.output[0]; the GELU output is
            # the real product — remap bookkeeping to the gelu name.
            s = tensor_slot.pop(out_name)
            tensor_slot[g] = s
            tensor_refs[g] = consumers.get(g, 0)
            if tensor_refs[g] == 0:
                free_slots.append(s)
            shapes[g] = shapes[out_name]
            steps[-1]["ref"] = g
            continue
        cur = save
        # layer_scale + residual tail (fc2 of a RepMixer/ConvFFN block)?
        ls = try_layer_scale_residual()
        if ls is not None:
            cur += 2
            emit_conv(node, act="none",
                      layer_scale={"gamma": ls["gamma"], "out": ls["out"]},
                      residual_name=ls["res"])
            continue
        emit_conv(node, act="none")
        continue
    if node.op_type == "BatchNormalization":
        match_attention()
        continue
    if node.op_type == "ReduceMean":
        match_head(node.input[0])
        continue
    expect(False, f"unmatched op {node.op_type}")

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
weights = np.concatenate(packed)
total_floats = int(weights.size)
macs = 0
for s in steps:
    if s["kind"] == "conv":
        macs += s["outH"] * s["outW"] * s["cout"] * (s["cin"] // s["groups"]) * s["k"] * s["k"]
    elif s["kind"] == "se":
        macs += 2 * s["c"] * s["cmid"]
    elif s["kind"] == "attn_core":
        macs += 2 * s["nTok"] * s["nTok"] * s["c"]
    elif s["kind"] == "head":
        macs += s["cin"] * s["cout"]

embed_slot = steps[-1]["dst"]
embed_dim = steps[-1]["cout"]
plan = {
    "model": "mobileclip_s0_vision",
    "inputSlot": 0, "inputShape": [3, 256, 256],
    "outputSlot": embed_slot, "embedDim": embed_dim,
    "slots": slot_sizes,
    "weightsFloats": total_floats,
    "steps": steps,
}

# ---------------------------------------------------------------------------
# Backward (TRAIN only) — the reverse step list for dL/dpixels, weights FROZEN.
# Grad slots MIRROR the activation slots: grad(actSlot s) = nAct + s. Backward
# runs strictly reverse-forward, so grad[T] is fully accumulated (every forward
# consumer of T is a later forward step = an earlier backward entry) before T's
# producer reads it. The FIRST writer of a grad slot overwrites; later writers
# ADD (`accumulate: true`). No global zero-fill (spec §1).
# ---------------------------------------------------------------------------
if TRAIN:
    n_act = len(slot_sizes)
    slot_sizes.extend(slot_sizes[:n_act])          # mirror grad slots [nAct, 2*nAct)

    def gslot(s: int) -> int:
        return n_act + s

    backward: list[dict] = []
    grad_written: set[int] = set()

    def emit_back(e: dict) -> None:
        e["accumulate"] = e["dX"] in grad_written   # already written this pass?
        grad_written.add(e["dX"])
        backward.append(e)

    # L = -cos(embed, text): reads saved embed + a text uniform, writes grad[embed].
    emit_back({"kind": "loss_bwd", "name": "loss",
               "embed": embed_slot, "dX": gslot(embed_slot), "dim": int(embed_dim)})

    for i in range(len(steps) - 1, -1, -1):
        s = steps[i]
        k = s["kind"]
        if k == "conv" and s["variant"] == "pointwise":
            if s["residual"] is not None:
                # out = res + γ⊙conv → dRes = dOut (γ folded into wOffT for dConv)
                emit_back({"kind": "residual_bwd", "name": s["name"] + ":resbwd",
                           "dY": gslot(s["dst"]), "dX": gslot(s["residual"]),
                           "n": s["cout"] * s["outH"] * s["outW"]})
            # dX = Wᵀ·dY : the pointwise kernel with reduction over forward-cout,
            # producing forward-cin outputs, reading the transposed weights.
            emit_back({"kind": "pw_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "dX": gslot(s["src"]),
                       "wOffT": s["wOffT"],
                       "cin": s["cout"], "cout": s["cin"],
                       "outH": s["outH"], "outW": s["outW"]})
        elif k == "conv":                            # depthwise / general (gather)
            emit_back({"kind": "spatial_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "dX": gslot(s["src"]), "wOff": s["wOff"],
                       "cin": s["cin"], "cout": s["cout"], "k": s["k"],
                       "stride": s["stride"], "pad": s["pad"], "groups": s["groups"],
                       "h": s["h"], "w": s["w"], "outH": s["outH"], "outW": s["outW"]})
        elif k == "gelu":
            emit_back({"kind": "gelu_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "pre": s["src"], "dX": gslot(s["src"]),
                       "n": s["n"]})
        elif k == "se":
            emit_back({"kind": "se_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "savedSrc": s["src"], "dX": gslot(s["src"]),
                       "c": s["c"], "cmid": s["cmid"], "h": s["h"], "w": s["w"],
                       "w1Off": s["w1Off"], "b1Off": s["b1Off"],
                       "w2Off": s["w2Off"], "b2Off": s["b2Off"]})
        elif k == "attn_core":
            emit_back({"kind": "attn_core_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "savedQkv": s["src"], "dX": gslot(s["src"]),
                       "c": s["c"], "heads": s["heads"], "hd": s["hd"], "nTok": s["nTok"]})
        elif k == "head":
            emit_back({"kind": "head_bwd", "name": s["name"] + ":bwd",
                       "dY": gslot(s["dst"]), "dX": gslot(s["src"]), "wOff": s["wOff"],
                       "cin": s["cin"], "cout": s["cout"], "h": s["h"], "w": s["w"]})
        else:
            raise SystemExit(f"compile_plan: backward — unhandled step kind {k}")

    plan["backward"] = backward
    plan["nActSlots"] = n_act
    plan["inputGradSlot"] = gslot(0)
    plan["embedSlot"] = embed_slot
    plan["embedGradSlot"] = gslot(embed_slot)
    plan["textDim"] = int(embed_dim)

plan_name = "plan_train.json" if TRAIN else "plan.json"
weights_name = "weights_train.bin" if TRAIN else "weights.bin"
(MODEL_DIR / plan_name).write_text(json.dumps(plan, indent=1))
weights.tofile(MODEL_DIR / weights_name)

kinds = {}
for s in steps:
    key = s["kind"] + (":" + s["variant"] if s["kind"] == "conv" else "")
    kinds[key] = kinds.get(key, 0) + 1
print(f"mode: {'TRAIN' if TRAIN else 'inference'}  → {plan_name} / {weights_name}")
print(f"steps: {len(steps)}  {kinds}")
print(f"slots: {len(slot_sizes)}  ({'act+grad' if TRAIN else 'reused'})")
print(f"weights: {total_floats} floats = {total_floats*4/1e6:.1f} MB   MACs/forward: {macs/1e9:.3f} G")
if TRAIN:
    bkinds: dict = {}
    for e in plan["backward"]:
        bkinds[e["kind"]] = bkinds.get(e["kind"], 0) + 1
    acc = sum(1 for e in plan["backward"] if e["accumulate"])
    print(f"backward: {len(plan['backward'])} entries {bkinds}  (accumulate={acc})")
for i, s in enumerate(steps):
    extra = ""
    if s["kind"] == "conv":
        extra = f"{s['variant']:9s} {s['cin']}→{s['cout']} k{s['k']} s{s['stride']} g{s['groups']} {s['h']}x{s['w']}→{s['outH']}x{s['outW']} act={s['act']}" + (" +res" if s["residual"] is not None else "")
    elif s["kind"] == "se":
        extra = f"c{s['c']} mid{s['cmid']} {s['h']}x{s['w']} act={s['act']}"
    elif s["kind"] == "attn_core":
        extra = f"c{s['c']} heads{s['heads']} n{s['nTok']}"
    elif s["kind"] == "head":
        extra = f"{s['cin']}→{s['cout']}"
    print(f"  [{i:3d}] {s['kind']:9s} slot{s['src']}→{s['dst']}  {extra}")
