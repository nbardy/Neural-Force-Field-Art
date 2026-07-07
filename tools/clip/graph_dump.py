# tools/clip/graph_dump.py — MobileCLIP-S0 ONNX graph inventory.
#
# Run:  uv run --with onnx --with numpy python tools/clip/graph_dump.py [model.onnx]
#
# Dumps the vision encoder's op-level structure to plan the fused WGSL kernel
# set (tools/clip/README.md): op-type histogram, per-node shapes (via shape
# inference), initializer (weight) inventory, and the linearized node list.
# Writes JSON next to the model: <model>.graph.json
import json
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import shape_inference

model_path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/mobileclip_s0/vision_model.onnx")
model = onnx.load(str(model_path))
inferred = shape_inference.infer_shapes(model)
graph = inferred.graph


def dims(vi):
    t = vi.type.tensor_type
    return [d.dim_value if d.HasField("dim_value") else str(d.dim_param) for d in t.shape.dim]


value_shapes = {vi.name: dims(vi) for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
init_shapes = {init.name: list(init.dims) for init in graph.initializer}

nodes = []
for n in graph.node:
    nodes.append(
        {
            "op": n.op_type,
            "name": n.name,
            "inputs": [
                {"name": i, "shape": value_shapes.get(i) or init_shapes.get(i), "init": i in init_shapes}
                for i in n.input
                if i
            ],
            "outputs": [{"name": o, "shape": value_shapes.get(o)} for o in n.output],
            "attrs": {a.name: str(onnx.helper.get_attribute_value(a))[:120] for a in n.attribute},
        }
    )

histogram = Counter(n["op"] for n in nodes)
total_params = sum(
    int(__import__("math").prod(s or [0])) for s in init_shapes.values()
)

report = {
    "model": str(model_path),
    "graph_inputs": [{"name": i.name, "shape": dims(i)} for i in graph.input],
    "graph_outputs": [{"name": o.name, "shape": dims(o)} for o in graph.output],
    "node_count": len(nodes),
    "op_histogram": dict(histogram.most_common()),
    "initializer_count": len(init_shapes),
    "total_params": total_params,
    "nodes": nodes,
}

out = model_path.with_suffix(".graph.json")
out.write_text(json.dumps(report, indent=1))

print(f"model: {model_path}")
print(f"inputs: {report['graph_inputs']}")
print(f"outputs: {report['graph_outputs']}")
print(f"nodes: {len(nodes)}  initializers: {len(init_shapes)}  params: {total_params/1e6:.2f}M")
print("op histogram:")
for op, c in histogram.most_common():
    print(f"  {op:24s} {c}")
print(f"wrote {out}")
