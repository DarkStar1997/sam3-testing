#!/usr/bin/env python
"""Build a TensorRT engine from the SAM3 grounding ONNX (Phase 2, step 2).

Static shapes: pixel_values (1,3,1008,1008) fp32, text_tokens (1,32) int64.
FP16 tactic selection; weights stored fp16 where safe (strongly-typed off).
"""
import argparse
import time

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/workspace/out/sam3_grounding.onnx")
    ap.add_argument("--engine", default="/workspace/out/sam3_grounding_fp16.engine")
    ap.add_argument("--strongly-typed", action="store_true",
                    help="use STRONGLY_TYPED network (TRT 11: honors graph "
                         "casts; needed for mixed-precision graphs)")
    ap.add_argument("--kmin", type=int, default=1,
                    help="dynamic-K profile: min K")
    ap.add_argument("--kopt", type=int, default=3,
                    help="dynamic-K profile: optimal K")
    ap.add_argument("--kmax", type=int, default=8,
                    help="dynamic-K profile: max K")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        if args.strongly_typed
        else 0
    )
    parser = trt.OnnxParser(network, logger)

    t0 = time.time()
    ok = parser.parse_from_file(args.onnx)
    if not ok:
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        raise SystemExit(1)
    print(f"parsed onnx ({time.time() - t0:.1f}s): {len(network)} tensors? inputs:")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input  {t.name} shape={tuple(t.shape)} dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name} shape={tuple(t.shape)} dtype={t.dtype}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 20 << 30)

    # optimization profile for dynamic-K inputs (dims == -1 after parse)
    dyn_inputs = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if any(d == -1 for d in t.shape):
            dyn_inputs.append(t)
    if dyn_inputs:
        prof = builder.create_optimization_profile()
        for t in dyn_inputs:
            smin = tuple(args.kmin if d == -1 else d for d in t.shape)
            sopt = tuple(args.kopt if d == -1 else d for d in t.shape)
            smax = tuple(args.kmax if d == -1 else d for d in t.shape)
            prof.set_shape(t.name, smin, sopt, smax)
            print(f"  profile {t.name}: min={smin} opt={sopt} max={smax}")
        cfg.add_optimization_profile(prof)

    t0 = time.time()
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise SystemExit("engine build failed")
    with open(args.engine, "wb") as f:
        f.write(plan)
    print(f"engine built ({time.time() - t0:.1f}s) -> {args.engine}")


if __name__ == "__main__":
    main()
