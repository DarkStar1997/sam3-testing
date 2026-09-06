#!/usr/bin/env python
"""Standalone MinMax calibration-range computation for SAM3 FP8 quantization.

ORT keeps every tapped activation live during an augmented calibration run
(~21GB for this model at 1008^2), so this computes ranges chunk-by-chunk, each
chunk in a FRESH subprocess (byte-budgeted tap sets, per-chunk checkpointing,
auto-split on failure, resume-safe). Result is bit-identical to a one-shot
calibration over the same feeds.
"""
import argparse
import gc
import json
import os
import pickle
import subprocess
import sys


class _Reader:
    def __init__(self, feeds):
        self._it = iter(feeds)

    def get_next(self):
        return next(self._it, None)


def _setup_patches():
    from modelopt.onnx.quantization.ort_patching import patch_ort_modules

    patch_ort_modules(False)

    import onnxruntime as ort

    _real_so = ort.SessionOptions

    def _lean_so():
        so = _real_so()
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.intra_op_num_threads = 4
        return so

    ort.SessionOptions = _lean_so


def _feeds(refs, tile=1):
    import numpy as np

    feeds = []
    for ref in refs:
        f = np.load(ref)
        px = f["pixel_values"].astype(np.float32)
        tok = f["text_tokens"].astype(np.int64)
        ids = f["img_ids"].astype(np.int64) if "img_ids" in f else \
            np.zeros(len(tok), dtype=np.int64)
        if tile > 1:
            px = np.repeat(px, tile, axis=0)
            tok = np.repeat(tok, tile, axis=0)
            ids = np.repeat(ids, tile, axis=0)
        feed = {"pixel_values": px, "text_tokens": tok}
        if ids.shape == tok.shape[:1]:
            feed["img_ids"] = ids  # dynamic-K graphs take ids as an input
        feeds.append(feed)
        del f
    return feeds


def plan(args):
    import onnx
    from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

    # Replicate configure_ort()'s effective quantizable op-type set:
    # registry union + modelopt's custom QDQ entries - removed copy/shape ops.
    qset = set(QLinearOpsRegistry) | set(QDQRegistry)
    qset |= {"BatchNormalization", "ConvTranspose", "LayerNormalization", "LRN", "HardSwish"}
    qset -= {
        "ArgMax", "Concat", "EmbedLayerNormalization", "Gather", "GatherElements",
        "GatherND", "InstanceNormalization", "LeakyRelu", "Pad", "Relu", "Reshape",
        "Slice", "Sigmoid", "Softmax", "Split", "Squeeze", "Transpose", "Unsqueeze",
        "Where",
    }

    if not os.path.exists(args.named):
        print("running modelopt preprocess (naming + constant duplication) ...", flush=True)
        from modelopt.onnx.quantization.quantize import _preprocess_onnx

        res = _preprocess_onnx(
            args.onnx, True,
            "/workspace/out/sam3_grounding_fp8.onnx",
            True, None, None, None, False, "fp8", None,
        )
        named_path = res[0]
        del res
        gc.collect()
        if os.path.abspath(named_path) != os.path.abspath(args.named):
            os.replace(named_path, args.named)
            for ext in (".data", "_data"):
                src = named_path + ext
                if os.path.exists(src):
                    os.replace(src, args.named + ext)
    print(f"named model: {args.named}", flush=True)

    m = onnx.load(args.named, load_external_data=False)
    vi = {v.name: v for v in m.graph.value_info}
    vi.update({o.name: o for o in m.graph.output})
    vi.update({i.name: i for i in m.graph.input})
    init_names = {i.name for i in m.graph.initializer}

    def _tbytes(tname):
        v = vi.get(tname)
        if v is None or not v.type.HasField("tensor_type"):
            return 4 * 1024 * 1024
        tt = v.type.tensor_type
        if tt.elem_type not in (1, 10):  # FLOAT, FLOAT16
            return 0
        n = 1
        for d in tt.shape.dim:
            # symbolic dims (dynamic K) counted at the concrete calib K
            n *= (d.dim_value if d.HasField("dim_value") else args.sym_dim)
        return n * (2 if tt.elem_type == 10 else 4)

    costed = []
    for node in m.graph.node:
        if node.op_type in qset:
            b = sum(_tbytes(t) for t in list(node.input) + list(node.output)
                    if t not in init_names)
            costed.append((node.name, b))
    del m
    costed.sort(key=lambda x: -x[1])
    total = sum(b for _, b in costed)
    print(f"{len(costed)} quantizable nodes, {total/1e9:.2f}GB tapped bytes", flush=True)

    budget = args.budget_gb * 1e9
    chunks, cur, cur_b = [], [], 0.0
    for name, b in costed:
        if cur and (cur_b + b > budget or len(cur) >= args.max_nodes):
            chunks.append(cur)
            cur, cur_b = [], 0.0
        cur.append(name)
        cur_b += b
    if cur:
        chunks.append(cur)
    with open(args.chunks_file, "w") as fh:
        json.dump(chunks, fh)
    print(f"{len(chunks)} chunks (budget {args.budget_gb}GB, max {args.max_nodes} nodes)", flush=True)
    return chunks


def run_chunk(args):
    with open(args.chunks_file) as fh:
        chunks = json.load(fh)
    names = chunks[args.idx]
    _setup_patches()
    import onnxruntime.quantization.calibrate as _calib_mod
    from onnxruntime.quantization.calibrate import CalibrationMethod

    aug_path = f"/tmp/sam3_aug_{args.idx}.onnx"
    calib = _calib_mod.create_calibrator(
        args.named, names,
        augmented_model_path=aug_path,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
    )
    calib.model = None
    calib.augment_model = None
    calib.augmented_model = None
    gc.collect()
    tile = 1
    import onnx as _onnx

    _m = _onnx.load(args.named, load_external_data=False)
    for _inp in _m.graph.input:
        if _inp.name == "pixel_values":
            tile = int(_inp.type.tensor_type.shape.dim[0].dim_value or 1)
    del _m
    if tile > 1:
        print(f"tiling feeds x{tile} to match static batch", flush=True)
    calib.collect_data(_Reader(_feeds(args.refs, tile)))
    td = calib.compute_data()
    with open(args.part, "wb") as fh:
        pickle.dump(td.data, fh, protocol=4)
    print(f"chunk {args.idx}: {len(td.data)} ranges -> {args.part}", flush=True)


def merge(args, chunks):
    from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData

    all_ranges = {}
    if os.path.exists(args.out):
        with open(args.out, "rb") as fh:
            prev = pickle.load(fh)
        all_ranges = prev.data
        print(f"resuming with {len(all_ranges)} existing ranges", flush=True)

    pending = list(range(len(chunks)))
    while pending:
        idx = pending.pop(0)
        part = args.part_pattern % idx
        if os.path.exists(part):
            with open(part, "rb") as fh:
                all_ranges.update(pickle.load(fh))
            print(f"chunk {idx}: merged from checkpoint ({len(all_ranges)} total)", flush=True)
            _save(args, CalibrationMethod, TensorsData, all_ranges)
            continue
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        cmd = [sys.executable, os.path.abspath(__file__), "child",
               "--chunks-file", args.chunks_file, "--idx", str(idx),
               "--named", args.named, "--part", part]
        cmd += ["--refs"] + args.refs
        rc = subprocess.run(cmd, env=env).returncode
        if rc == 0 and os.path.exists(part):
            with open(part, "rb") as fh:
                all_ranges.update(pickle.load(fh))
            _save(args, CalibrationMethod, TensorsData, all_ranges)
            print(f"chunk {idx}: done ({len(all_ranges)} total ranges)", flush=True)
        else:
            nodes = chunks[idx]
            if len(nodes) == 1:
                print(f"chunk {idx}: FAILED single node {nodes[0]}, skipping", flush=True)
                continue
            mid = len(nodes) // 2
            chunks[idx] = nodes[:mid]
            chunks.insert(idx + 1, nodes[mid:])
            with open(args.chunks_file, "w") as fh:
                json.dump(chunks, fh)
            pending = [idx, idx + 1] + [i + 1 if i > idx else i for i in pending]
            print(f"chunk {idx}: failed (rc={rc}), split into two; {len(chunks)} chunks now",
                  flush=True)

    print(f"ALL DONE: {len(all_ranges)} tensor ranges -> {args.out}", flush=True)


def _save(args, CalibrationMethod, TensorsData, all_ranges):
    ranges = TensorsData(CalibrationMethod.MinMax, all_ranges)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(ranges, fh, protocol=4)
    os.replace(tmp, args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("child")
    p.add_argument("--chunks-file", required=True)
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--named", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--refs", nargs="+", required=True)

    p = sub.add_parser("run")
    p.add_argument("--onnx", default="/workspace/out/sam3_grounding_w16.onnx")
    p.add_argument("--named", default="/workspace/out/sam3_grounding_w16_named.onnx")
    p.add_argument("--out", default="/workspace/out/sam3_fp8_ranges.pkl")
    p.add_argument("--refs", nargs="+", default=[
        "/workspace/out/sam3_ref_ball60_fp16.npz",
        "/workspace/out/sam3_export_ref.npz",
    ])
    p.add_argument("--chunks-file", default="/tmp/sam3_chunks.json")
    p.add_argument("--budget-gb", type=float, default=1.5)
    p.add_argument("--max-nodes", type=int, default=120)
    p.add_argument("--sym-dim", type=int, default=3,
                   help="concrete value for symbolic dims in size estimates")

    args = ap.parse_args()
    _stem = args.chunks_file.rsplit(".", 1)[0] if hasattr(args, "chunks_file") else "/tmp/sam3_chunks"
    args.part_pattern = _stem + "_part_%d.pkl"

    if args.mode == "child":
        run_chunk(args)
    else:
        chunks = plan(args)
        merge(args, chunks)


if __name__ == "__main__":
    main()
