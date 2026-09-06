#!/usr/bin/env python
"""Run modelopt's deterministic preprocess (naming + constant dedup) on a w16 ONNX."""
import gc
import os
import sys

W16 = sys.argv[1]
NAMED = sys.argv[2]
TMP_OUT = sys.argv[3] if len(sys.argv) > 3 else W16.replace("_w16.onnx", "_fp8.onnx")

from modelopt.onnx.quantization.quantize import _preprocess_onnx

res = _preprocess_onnx(W16, True, TMP_OUT, True, None, None, None, False, "fp8", None)
named = res[0]
del res
gc.collect()
if os.path.abspath(named) != os.path.abspath(NAMED):
    os.replace(named, NAMED)
    for ext in (".data", "_data"):
        s = named + ext
        if os.path.exists(s):
            os.replace(s, NAMED + ext)
print("named ->", NAMED)
