#!/usr/bin/env python
"""Quantize SAM3 grounding ONNX to FP8 (Q/DQ) via NVIDIA modelopt."""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/workspace/out/sam3_grounding.onnx")
    ap.add_argument("--out", default="/workspace/out/sam3_grounding_fp8.onnx")
    ap.add_argument("--refs", nargs="+", default=[
        "/workspace/out/sam3_ref_ball60_fp16.npz",
        "/workspace/out/sam3_export_ref.npz",
    ])
    ap.add_argument("--calib-size", type=int, default=0,
                    help="override pixel input edge for calibration (0=keep 1008)")
    ap.add_argument("--ranges", default="/workspace/out/sam3_fp8_ranges.pkl",
                    help="precomputed MinMax ranges (see sam3_calib_ranges.py)")
    args = ap.parse_args()

    import os

    import numpy as np
    import torch

    # True-fp16-weight copy of the model (halves ORT/calibration memory).
    fp16_onnx = args.onnx if args.onnx.endswith("_w16.onnx") else \
        args.onnx.replace(".onnx", "_w16.onnx")
    if not os.path.exists(fp16_onnx):
        print("converting weights to real fp16 ...")
        import onnx
        from onnxconverter_common import float16

        os.makedirs(os.path.dirname(fp16_onnx) or ".", exist_ok=True)
        m = onnx.load(args.onnx, load_external_data=True)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True,
                                               disable_shape_infer=True)
        onnx.save(m16, fp16_onnx, save_as_external_data=True,
                  location=os.path.basename(fp16_onnx) + ".data", size_threshold=0)
        del m, m16
        print(f"saved {fp16_onnx}")

    from modelopt.onnx.quantization import quantize
    from modelopt.onnx.quantization import ort_utils
    import onnxruntime as _ort

    _real_so = _ort.SessionOptions

    def _lean_so():
        so = _real_so()
        so.enable_mem_pattern = False   # avoid arena prealloc over 17k tapped outputs
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 4
        return so

    ort_utils.ort.SessionOptions = _lean_so  # patches module-wide (all call sites)

    # Inject precomputed calibration ranges: quantize_static would otherwise build
    # a calibrator (model copy + augmented copy + session) on top of modelopt's own
    # model copies -> over the 20GB container cap. Ranges computed separately in
    # sam3_calib_ranges.py.
    import pickle

    import onnxruntime.quantization.calibrate as _ort_calibrate

    class _FakeCalibrator:
        def __init__(self, ranges):
            self._ranges = ranges

        def collect_data(self, reader):
            while reader.get_next() is not None:
                pass

        def compute_data(self):
            return self._ranges

    with open(args.ranges, "rb") as fh:
        _ranges = pickle.load(fh)
    _orig_create_calibrator = _ort_calibrate.create_calibrator

    def _create_calibrator_from_ranges(*a, **k):
        return _FakeCalibrator(_ranges)

    _ort_calibrate.create_calibrator = _create_calibrator_from_ranges
    # modelopt's configure_ort() -> patch_ort_modules() RE-stomps
    # calibrate.create_calibrator with ort_patching._create_calibrator_with_extra_options
    # inside quantize() (ort_utils.py:626, ort_patching.py:1758). Patch that symbol
    # too so the stomping binds our fake: no real calibrator, no augmented model,
    # no ORT session is ever created.
    import modelopt.onnx.quantization.ort_patching as _mo_ort_patching

    _mo_ort_patching._create_calibrator_with_extra_options = _create_calibrator_from_ranges
    print(f"injected calibration ranges from {args.ranges}")

    px_all, tok_all, ids_all = [], [], []
    for ref in args.refs:
        f = np.load(ref, allow_pickle=True)
        px_all.append(torch.from_numpy(f["pixel_values"].astype(np.float32)))
        tok_all.append(f["text_tokens"].astype(np.int64))
        ids_all.append(f["img_ids"].astype(np.int64) if "img_ids" in f
                       else np.zeros(len(f["text_tokens"]), dtype=np.int64))
    px = torch.cat(px_all)
    if args.calib_size and px.shape[-1] != args.calib_size:
        px = torch.nn.functional.interpolate(
            px, size=(args.calib_size, args.calib_size), mode="bilinear",
            align_corners=False)
        print(f"calib images resized to {tuple(px.shape)}")

    # tile the calib set to the model's static batch dim (modelopt splits
    # calibration_data into per-batch sections -> needs #samples == batch)
    import onnx as _onnx

    _m = _onnx.load(fp16_onnx, load_external_data=False)
    _dims = {}
    for _inp in _m.graph.input:
        _dims[_inp.name] = [d.dim_value or d.dim_param for d in _inp.type.tensor_type.shape.dim]
    del _m
    nb = _dims["pixel_values"][0]
    if isinstance(nb, int) and nb and px.shape[0] != nb:
        reps = -(-nb // px.shape[0])
        px = px.repeat(reps, 1, 1, 1)[:nb]
        tok_all = tok_all * reps
        ids_all = ids_all * reps
        print(f"tiled calibration set to {nb} sample(s)")
    tok = np.concatenate(tok_all)[:nb] if isinstance(nb, int) else np.concatenate(tok_all)
    ids = np.concatenate(ids_all)[:len(tok)]
    conc_k = tok.shape[0]  # concrete K substituted for symbolic dims

    calib = {
        "pixel_values": px.numpy(),
        "text_tokens": tok,
    }
    print(f"calibration set: {px.shape[0]} sample(s), K={conc_k}")

    t0 = time.time()
    overrides = None
    if args.calib_size:
        overrides = f"pixel_values:1x3x{args.calib_size}x{args.calib_size}"

    # derive calibration_shapes from the actual model input dims (batch may
    # be >1 or symbolic -> substitute the concrete K of the calib set)
    def _dim_str(d):
        return str(d) if isinstance(d, int) else str(conc_k)

    if "img_ids" in _dims:
        calib["img_ids"] = ids
    _calib_shapes = ",".join(
        f"{n}:{'x'.join(_dim_str(d) for d in _dims[n])}"
        for n in ("pixel_values", "text_tokens", "img_ids") if n in _dims
    )
    print(f"calibration_shapes: {_calib_shapes}")

    quantize(
        onnx_path=fp16_onnx,
        quantize_mode="fp8",
        output_path=args.out,
        calibration_data=calib,
        calibration_shapes=_calib_shapes,
        calibration_method="max",
        calibration_eps=["cpu"],
        override_shapes=overrides,
        disable_mha_qdq=True,  # skips the 17k-tap extended-model run (host OOM)
        high_precision_dtype="fp16",
        use_external_data_format=True,
        verbosity="info",
    )
    print(f"fp8 quantized ({time.time() - t0:.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()
