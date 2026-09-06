#!/usr/bin/env python
"""Transfer bs1 calibration ranges onto a re-exported named graph via op-sequence alignment.

The re-export (b8 / dynamic-K / static-K multi) is the same math as bs1
(validated row-wise at export time), but modelopt's deterministic renumbering
assigns different val_N names. Align the two graphs' op-type sequences with
difflib and map outputs positionally. Fallbacks: identity names, then
Gather/Expand producer-consumer range inheritance (covers replicated-vision
tensors), then +-3.0 fill (needs --force).
"""
import difflib
import pickle
import sys

import numpy as np
import onnx

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
A_NAMED = argv[0] if len(argv) > 0 else "/workspace/out/sam3_grounding_w16_named.onnx"  # bs1 (ranges source)
B_NAMED = argv[1] if len(argv) > 1 else "/workspace/out/sam3_grounding_b8_w16_named.onnx"  # target
PKL_IN = argv[2] if len(argv) > 2 else "/workspace/out/sam3_fp8_ranges.pkl"
PKL_OUT = argv[3] if len(argv) > 3 else "/workspace/out/sam3_fp8_ranges_b8.pkl"
print(f"source named: {A_NAMED}\ntarget named: {B_NAMED}\nout pkl: {PKL_OUT}")

a = onnx.load(A_NAMED, load_external_data=False)
b = onnx.load(B_NAMED, load_external_data=False)
an, bn = list(a.graph.node), list(b.graph.node)

# --- anchor-based alignment -------------------------------------------------
# Pure op-sequence difflib fails on structurally divergent re-exports (the
# dynamic-K graph inserts ~500 Shape/Gather/Unsqueeze nodes, breaking global
# resync). Weight-bearing nodes (MatMul/Gemm/Conv) whose weight initializer
# kept its torch name (NOT renamed val_N by modelopt's constant pass) are
# stable anchors; align anchor sequences with difflib, then map the node
# segments between consecutive matched anchors with local op-type difflib.
import re

_VAL = re.compile(r"^val_\d+$")
ia = {i.name for i in a.graph.initializer}
ib = {i.name for i in b.graph.initializer}
shared_inits = ia & ib
WEIGHT_OPS = {"MatMul", "Gemm", "Conv"}


def anchor_key(node):
    if node.op_type not in WEIGHT_OPS:
        return None
    w = tuple(sorted(i for i in node.input
                     if i in shared_inits and not _VAL.match(i)))
    return (node.op_type, w) if w else None


a_anchor = [(i, k) for i, k in
            ((i, anchor_key(n)) for i, n in enumerate(an)) if k]
b_anchor = [(i, k) for i, k in
            ((i, anchor_key(n)) for i, n in enumerate(bn)) if k]
a_keys = [str(k) for _, k in a_anchor]
b_keys = [str(k) for _, k in b_anchor]
print(f"anchors a={len(a_anchor)} b={len(b_anchor)} shared_inits={len(shared_inits)}")

name_map = {}


def map_nodes(na, nb):
    if len(na.output) == len(nb.output):
        for oa, ob in zip(na.output, nb.output):
            if oa and ob:
                name_map[ob] = oa


def map_segment(a_lo, a_hi, b_lo, b_hi):
    seg_a = an[a_lo:a_hi]
    seg_b = bn[b_lo:b_hi]
    if not seg_a or not seg_b:
        return
    sm = difflib.SequenceMatcher(None, [n.op_type for n in seg_a],
                                 [n.op_type for n in seg_b], autojunk=False)
    for bl in sm.get_matching_blocks():
        for k in range(bl.size):
            map_nodes(seg_a[bl.a + k], seg_b[bl.b + k])


sm_anchor = difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False)
ab_blocks = [bl for bl in sm_anchor.get_matching_blocks() if bl.size]
print(f"anchor alignment {sum(bl.size for bl in ab_blocks)}/{len(a_anchor)}")
prev_a = prev_b = 0
for bl in ab_blocks:
    # segment before this matched anchor run
    map_segment(prev_a, a_anchor[bl.a][0], prev_b, b_anchor[bl.b][0])
    for k in range(bl.size):
        map_nodes(an[a_anchor[bl.a + k][0]], bn[b_anchor[bl.b + k][0]])
    # inside a run of identical anchors, pair segments between consecutive
    # anchors too (guards against duplicated weights shifting by one)
    for k in range(bl.size - 1):
        map_segment(a_anchor[bl.a + k][0] + 1, a_anchor[bl.a + k + 1][0],
                    b_anchor[bl.b + k][0] + 1, b_anchor[bl.b + k + 1][0])
    prev_a = a_anchor[bl.a + bl.size - 1][0] + 1
    prev_b = b_anchor[bl.b + bl.size - 1][0] + 1
map_segment(prev_a, len(an), prev_b, len(bn))

print(f"name_map entries: {len(name_map)}")

with open(PKL_IN, "rb") as fh:
    td = pickle.load(fh)
src = td.data
remapped = {}
miss = []
for bname, aname in name_map.items():
    if aname in src:
        remapped[bname] = src[aname]
    else:
        miss.append((bname, aname))
print(f"mappable b8 tensors: {len(remapped)}, in-map but not in pkl: {len(miss)}")

# coverage of the b8 quantizable set
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

qset = set(QLinearOpsRegistry) | set(QDQRegistry)
qset |= {"BatchNormalization", "ConvTranspose", "LayerNormalization", "LRN", "HardSwish"}
qset -= {
    "ArgMax", "Concat", "EmbedLayerNormalization", "Gather", "GatherElements",
    "GatherND", "InstanceNormalization", "LeakyRelu", "Pad", "Relu",
    "Reshape", "Slice", "Sigmoid", "Softmax", "Split", "Squeeze",
    "Transpose", "Unsqueeze", "Where",
}
init = {i.name for i in b.graph.initializer}
needed = set()
for n in bn:
    if n.op_type in qset:
        for t in list(n.input) + list(n.output):
            if t and t not in init:
                needed.add(t)
covered = needed & set(remapped)
print(f"b8 qset tensors: {len(needed)}, covered by remap: {len(covered)} "
      f"({100.0 * len(covered) / len(needed):.2f}%)")
missing = sorted(needed - set(remapped))

# identity fallback: tensors that kept the same torch-export name in both graphs
ident = [t for t in missing if t in src]
for t in ident:
    remapped[t] = src[t]
missing = [t for t in missing if t not in remapped]
print(f"identity fallback recovered {len(ident)}; still missing {len(missing)}")

# producer-consumer inheritance: outputs of pure movement/replication ops
# (Gather/Expand/shape ops/Concat) inherit the union of their inputs' ranges.
# Covers replicated-vision tensors (img_ids gather) without recalibration.
if missing:
    MOVE_OPS = {"Gather", "Expand", "Reshape", "Transpose", "Unsqueeze",
                "Squeeze", "Slice", "Flatten", "Identity", "Concat"}
    producer = {}
    for n in b.graph.node:
        for o in n.output:
            if o:
                producer[o] = n
    from onnxruntime.quantization.calibrate import TensorData

    def _union(td_a, td_b):
        return TensorData(
            lowest=np.minimum(td_a.lowest.astype(np.float32), td_b.lowest.astype(np.float32)).astype(np.float16),
            highest=np.maximum(td_a.highest.astype(np.float32), td_b.highest.astype(np.float32)).astype(np.float16),
        )

    progressed = True
    rounds = 0
    while missing and progressed and rounds < 6:
        progressed = False
        rounds += 1
        still = []
        for t in missing:
            n = producer.get(t)
            if n is None or n.op_type not in MOVE_OPS:
                still.append(t)
                continue
            parts = [remapped[i] for i in n.input if i in remapped]
            if not parts:
                still.append(t)
                continue
            acc = parts[0]
            for p in parts[1:]:
                acc = _union(acc, p)
            remapped[t] = acc
            progressed = True
        missing = still
        if progressed:
            print(f"inheritance round {rounds}: still missing {len(missing)}")

if missing and "--force" not in sys.argv:
    print("MISSING (first 20):", missing[:20])
    sys.exit(1)
if missing:
    # conservative stand-in: amax 3.0 symmetric for unresolvable tensors
    from onnxruntime.quantization.calibrate import TensorData

    for t in missing:
        remapped[t] = TensorData(lowest=np.array([-3.0], dtype=np.float16),
                                  highest=np.array([3.0], dtype=np.float16))
    print(f"filled {len(missing)} with +-3.0 fallback range: {missing[:10]} ...")

out = type(td)(td.calibration_method, remapped)
with open(PKL_OUT, "wb") as fh:
    pickle.dump(out, fh, protocol=4)
print(f"saved {PKL_OUT} ({len(remapped)} ranges)")
