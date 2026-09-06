#!/usr/bin/env python
"""Rebuild a clean true-fp16 ONNX: fresh shape inference + sane externalization."""
import sys

import onnx
from onnx import shape_inference

SRC = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/sam3_grounding.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out/sam3_grounding_w16.onnx"

print("loading fp32 model with external data ...")
m = onnx.load(SRC, load_external_data=True)
print("converting to fp16 ...")
from onnxconverter_common import float16
m16 = float16.convert_float_to_float16(
    m, keep_io_types=True, disable_shape_infer=True
)
del m
print("clearing stale value_info, re-running shape inference ...")
m16.graph.value_info.clear()
m16 = shape_inference.infer_shapes(m16)  # weights ~1.7GB in-memory, under 2GB proto limit
print("saving ...")
onnx.save(m16, DST, save_as_external_data=True,
          location=DST.split("/")[-1] + ".data", size_threshold=1024 * 1024)
del m16
print("saved", DST)
