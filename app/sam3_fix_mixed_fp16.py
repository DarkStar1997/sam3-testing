#!/usr/bin/env python
"""Insert Casts to fix mixed F32/F16 elementwise inputs in the w16 ONNX."""
import onnx
from onnx import TensorProto, helper, shape_inference

import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/sam3_grounding_w16.onnx"
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT

m = onnx.load(PATH, load_external_data=True)

for it in range(6):
    m.graph.value_info.clear()
    m = shape_inference.infer_shapes(m)
    dt = {}
    for i in m.graph.initializer:
        dt[i.name] = i.data_type
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dt[vi.name] = vi.type.tensor_type.elem_type

    cast_map = {}
    n_fixed = 0
    new_nodes = []
    for n in m.graph.node:
        fl = {dt.get(o) for o in n.input if dt.get(o) in (F16, F32)}
        if len(fl) == 2:  # mixed float inputs
            n_fixed += 1
            new_inputs = []
            for o in n.input:
                if dt.get(o) == F32:
                    if o not in cast_map:
                        cast_map[o] = o + "__c16"
                        new_nodes.append(helper.make_node(
                            "Cast", [o], [cast_map[o]], to=F16,
                            name="cast_fix_" + o))
                    new_inputs.append(cast_map[o])
                else:
                    new_inputs.append(o)
            del n.input[:]
            n.input.extend(new_inputs)
        new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    print(f"round {it}: fixed {n_fixed} mixed nodes")
    if n_fixed == 0:
        break
else:
    raise SystemExit("did not converge")

onnx.save(m, PATH, save_as_external_data=True,
          location=PATH.split("/")[-1] + ".data", size_threshold=1024 * 1024)
print("saved", PATH)

import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(PATH, sess_options=so, providers=["CPUExecutionProvider"])
print("ORT OK:", [(i.name, i.type) for i in s.get_inputs()])
