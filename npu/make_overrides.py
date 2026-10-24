"""v68 HTP cannot multiply two 16-bit activation tensors. For an A16 build, run every
activation-by-activation MatMul (attention q.k^T and probs.v) in 8 bits; everything else stays 16.

Data-movement ops (Transpose, Reshape, Squeeze, Split, ...) inherit their input's encoding and
ignore overrides, so walk back through them and also override the compute op that produced the
tensor.
    python npu/make_overrides.py npu/build/gen.onnx > npu/build/gen_overrides.json"""
import json
import sys

import onnx

MOVE = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Split", "Gather", "Slice", "Identity"}
g = onnx.load(sys.argv[1]).graph
init = {i.name for i in g.initializer} | {n.output[0] for n in g.node if n.op_type == "Constant"}
producer = {o: n for n in g.node for o in n.output}
names = set()
for n in g.node:
    if n.op_type != "MatMul" or set(n.input) & init:
        continue
    names.update(n.output)
    for t in n.input:
        while True:
            names.add(t)
            p = producer.get(t)
            if p is None or p.op_type not in MOVE:
                break
            t = p.input[0]
print(json.dumps({"activation_encodings": {t: [{"bitwidth": 8, "dtype": "int"}] for t in sorted(names)},
                  "param_encodings": {}}, indent=1))
print(f"{len(names)} tensors forced to 8 bits", file=sys.stderr)
