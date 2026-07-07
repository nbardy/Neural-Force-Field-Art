# tools/clip/dump_refs.py — per-step golden activations for the WGSL port.
#
# Run:  uv run --with onnx --with numpy --with onnxruntime python tools/clip/dump_refs.py
#
# Re-exports the ONNX model with every plan step's `ref` tensor as an extra
# graph output, runs ORT CPU on the SAME deterministic fixture input that
# tools/clip/onnx_forward.mjs wrote, and saves one raw f32 .bin per step.
# tools/clip/fused_test.ts compares each WGSL step against these — the first
# mismatching step IS the broken kernel (bisection for free).
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

MODEL_DIR = Path("models/mobileclip_s0")
# PLAN=plan_train.json → dump train-plan refs (split-GELU steps, unique slots)
# into refs_train/ so the inference refs/ stay intact (spec §3, gate 3).
PLAN = os.environ.get("PLAN", "plan.json")
REF_DIR = MODEL_DIR / ("refs_train" if "train" in PLAN else "refs")
REF_DIR.mkdir(exist_ok=True)

plan = json.loads((MODEL_DIR / PLAN).read_text())
# ref=None steps (attention internals) have no ONNX tensor in our layout —
# they are covered by the next ref'd step (the block output).
refs = [s["ref"] for s in plan["steps"] if s["ref"] is not None]

model = onnx.load(str(MODEL_DIR / "vision_model.onnx"))
existing = {o.name for o in model.graph.output}
for name in refs:
    if name not in existing:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        existing.add(name)

sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CPUExecutionProvider"],
)

input_bin = MODEL_DIR / "fixtures" / "input_1x3x256x256.f32.bin"
x = np.fromfile(input_bin, dtype=np.float32).reshape(1, 3, 256, 256)
outputs = dict(zip(refs, sess.run(refs, {"pixel_values": x})))

manifest = []
for i, step in enumerate(plan["steps"]):
    if step["ref"] is None:
        manifest.append({"step": i, "ref": None, "shape": None, "file": None})
        continue
    arr = np.ascontiguousarray(outputs[step["ref"]], dtype=np.float32)
    fname = f"step_{i:03d}.bin"
    arr.tofile(REF_DIR / fname)
    manifest.append({"step": i, "ref": step["ref"], "shape": list(arr.shape), "file": fname})

(REF_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
total = sum(np.prod(m["shape"]) for m in manifest if m["shape"] is not None) * 4 / 1e6
print(f"wrote {len(manifest)} refs ({total:.0f} MB) → {REF_DIR}")
final_ref = plan["steps"][-1]["ref"]
print(f"final embedding first 8: {outputs[final_ref].reshape(-1)[:8]}")
