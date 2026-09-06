#!/usr/bin/env python
"""Patch NVIDIA modelopt 0.46.0 site-packages to reduce peak RAM during
ONNX FP8 static quantization of large graphs (the SAM3 grounding graph is
~5.5k nodes / 3.3GB and stock modelopt OOMs a 20GB container).

Applies two idempotent patches (marker: LA-MEMPATCH):
  1. modelopt/onnx/quantization/fp8.py  — release the fp32 proto + gs graph
     references before `quantize_static()` inserts Q/DQ nodes (~9GB freed;
     both are reassigned from the call's return value afterwards).
  2. modelopt/onnx/quantization/quantize.py — release the dead reference to
     the preprocessed model before the int8/fp8 quantize func runs (~4.5GB).

Usage: python sam3_patch_modelopt.py [--check]
  --check: only report patch status, exit 1 if unpatched.
"""
import argparse
import ast
import sys
from pathlib import Path

MARKER = "LA-MEMPATCH"

PATCHES = {
    "modelopt/onnx/quantization/fp8.py": {
        "anchor": "        quantize_static(\n",
        "insert": "        onnx_model = None  # LA-MEMPATCH: free fp32 proto+graph "
                  "before QDQ insert (~9GB)\n"
                  "        graph = None  # LA-MEMPATCH\n"
                  "        import gc as _gc  # LA-MEMPATCH\n"
                  "        _gc.collect()  # LA-MEMPATCH\n",
    },
    "modelopt/onnx/quantization/quantize.py": {
        "anchor": "        quantize_func = quantize_int8 if quantize_mode == \"int8\" else quantize_fp8\n",
        "insert": "        onnx_model = None  # LA-MEMPATCH: dead ref from preprocess, "
                  "frees ~4.5GB\n"
                  "        import gc as _gc  # LA-MEMPATCH\n"
                  "        _gc.collect()  # LA-MEMPATCH\n",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        import modelopt
    except ImportError:
        sys.exit("modelopt not installed in this environment")

    base = Path(modelopt.__file__).parent
    dirty = False
    for rel, p in PATCHES.items():
        f = base / rel
        src = f.read_text()
        if MARKER in src:
            print(f"[ok] {rel} already patched")
            continue
        if p["anchor"] not in src:
            sys.exit(f"[fail] {rel}: anchor not found - modelopt version "
                     "differs from 0.46.0; patch manually around "
                     f"{p['anchor'].strip()!r}")
        if args.check:
            print(f"[missing] {rel} unpatched")
            dirty = True
            continue
        patched = src.replace(p["anchor"], p["insert"] + p["anchor"], 1)
        ast.parse(patched)  # never write a broken file
        f.write_text(patched)
        print(f"[patched] {rel}")
        dirty = True

    sys.exit(1 if (args.check and dirty) else 0)


if __name__ == "__main__":
    main()
