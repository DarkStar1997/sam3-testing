#!/usr/bin/env python
"""Run SAM3 grounding TRT engine: parity vs torch ref npz + latency.

Supports both the 2-input static engines (bs1/b8) and the 3-input dynamic-K
engine (pixel_values, text_tokens (K,32), img_ids (K,)). With a dynamic-K
engine, --k-sweep times every K in 1..8 using rows from --tokens-npz.
"""
import argparse

import numpy as np
import tensorrt as trt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postproc_row(logits, boxes, masks, presence, row, orig_hw, thresh=0.5):
    probs = (sigmoid(logits[row]) * sigmoid(presence[row])).squeeze(-1)  # (Q,)
    idx = np.nonzero(probs > thresh)[0]
    h, w = orig_hw
    m = masks[row, idx][:, None]
    import torch

    mt = torch.from_numpy(m)
    mt = torch.nn.functional.interpolate(
        mt, (h, w), mode="bilinear", align_corners=False).sigmoid().numpy()
    mb = mt[:, 0] > 0.5
    b = boxes[row, idx]
    b = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                  b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    b = b * np.array([w, h, w, h], dtype=np.float32)
    return idx, probs[idx], b, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/workspace/out/sam3_grounding_fp32.engine")
    ap.add_argument("--ref", default="/workspace/out/sam3_export_ref.npz")
    ap.add_argument("--orig-hw", default="480,640")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tokens-npz", default="",
                    help="dynamic-K engines: tokens npz for the K sweep")
    ap.add_argument("--k-sweep", action="store_true")
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=True)
    h, w = (int(x) for x in args.orig_hw.split(","))
    dynamic = "img_ids" in ref

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    import torch

    pix = torch.from_numpy(ref["pixel_values"].astype(np.float32)).cuda()
    tok = torch.from_numpy(ref["text_tokens"].astype(np.int64)).cuda()
    ids = (torch.from_numpy(ref["img_ids"].astype(np.int64)).cuda()
           if dynamic else None)

    ctx.set_input_shape("pixel_values", tuple(pix.shape))
    ctx.set_input_shape("text_tokens", tuple(tok.shape))
    if dynamic:
        ctx.set_input_shape("img_ids", tuple(ids.shape))

    outs = {}
    for name in engine:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
    names = list(outs)
    print({n: tuple(t.shape) for n, t in outs.items()})

    def bind():
        ctx.set_tensor_address("pixel_values", pix.data_ptr())
        ctx.set_tensor_address("text_tokens", tok.data_ptr())
        if dynamic:
            ctx.set_tensor_address("img_ids", ids.data_ptr())
        for n in names:
            ctx.set_tensor_address(n, outs[n].data_ptr())

    def run():
        bind()
        t = torch.cuda.Event(True)
        t2 = torch.cuda.Event(True)
        t.record()
        ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        t2.record()
        torch.cuda.synchronize()
        assert ok
        return t.elapsed_time(t2)

    ms = run()  # warmup
    print(f"warmup {ms:.1f}ms")

    # parity: engine rows vs torch ref rows
    tl = {n: outs[n].cpu().numpy() for n in names}
    rows = tl["pred_logits"].shape[0]
    prompts = ([str(p) for p in ref["prompts"]] if "prompts" in ref
               else list(range(rows)))
    for n in names:
        r = ref[n].astype(np.float64)
        d = np.abs(tl[n].astype(np.float64) - r)
        print(f"{n}: max_abs_diff {d.max():.5f} mean {d.mean():.6f}")

    for row in range(rows):
        idx_t, sc_t, bx_t, mb_t = postproc_row(
            tl["pred_logits"], tl["pred_boxes"], tl["pred_masks"],
            tl["presence"], row, (h, w))
        idx_r, sc_r, bx_r, mb_r = postproc_row(
            ref["pred_logits"], ref["pred_boxes"], ref["pred_masks"],
            ref["presence"], row, (h, w))
        print(f"row {row} '{prompts[row]}': ref dets {len(idx_r)} "
              f"scores {np.round(sc_r, 3).tolist()} | trt dets {len(idx_t)} "
              f"scores {np.round(sc_t, 3).tolist()}")
        for i in idx_t:
            if i in idx_r:
                a = mb_t[list(idx_t).index(i)].astype(bool)
                b = mb_r[list(idx_r).index(i)].astype(bool)
                inter = (a & b).sum()
                union = (a | b).sum()
                iou = inter / union if union else 1.0
                print(f"  query {i}: IoU {iou:.4f} area {a.sum()}/{b.sum()}")

    times = np.array([run() for _ in range(args.iters)])
    print(f"latency K={rows}: mean {times.mean():.1f}ms "
          f"median {np.median(times):.1f}ms min {times.min():.1f}ms "
          f"-> {1000.0 / np.median(times):.2f} FPS")

    if args.k_sweep and dynamic and args.tokens_npz:
        tn = np.load(args.tokens_npz, allow_pickle=True)
        rows_all = tn["text_tokens" if "text_tokens" in tn else "tokens"]
        rows_all = rows_all.astype(np.int64)
        for k in range(1, min(9, len(rows_all) + 1)):
            tok = torch.from_numpy(np.ascontiguousarray(rows_all[:k])).cuda()
            ids = torch.zeros(k, dtype=torch.int64).cuda()
            ctx.set_input_shape("text_tokens", tuple(tok.shape))
            ctx.set_input_shape("img_ids", tuple(ids.shape))
            for name in names:
                shape = tuple(ctx.get_tensor_shape(name))
                outs[name] = torch.empty(shape, dtype=torch.float32,
                                         device="cuda")
            run()
            t = np.array([run() for _ in range(10)])
            print(f"K={k}: median {np.median(t):.1f}ms "
                  f"-> {1000.0 / np.median(t):.2f} FPS")


if __name__ == "__main__":
    main()
