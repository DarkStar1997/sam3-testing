#!/usr/bin/env python3
"""Probe prompt variants against the SAM3 dynamic-K TRT engine.

Scores each prompt on frames spread over a video and reports max score,
hit count above threshold and best frame, so you can pick the right
prompts before running the full pipeline.

Usage:
  python sam3_probe_prompts.py --video X.mp4 --prompts "a,b,c" \
      --engine E --points 12 --thresh 0.5
"""
import argparse

import cv2
import numpy as np
import torch
import tensorrt as trt


def enc(tok, p):
    ids = list(tok.encode(p))[:32]
    return ids + [0] * (32 - len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/workspace/out/sam3_grounding_fp8_dynk.engine")
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompts", required=True, help="comma-separated")
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="exclusive; 0 = video end")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    assert len(prompts) <= 8, "engine profile supports K<=8"

    import pkg_resources
    from sam3.model.tokenizer_ve import SimpleTokenizer
    tok = SimpleTokenizer(
        bpe_path=pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"),
        context_length=32)

    toks = np.array([enc(tok, p) for p in prompts], dtype=np.int64)
    K = len(prompts)

    cap = cv2.VideoCapture(args.video)
    N = int(cap.get(7))
    lo, hi = args.start, (args.end or N)
    span = hi - lo
    idxs = sorted(set(lo + i * span // max(args.points, 1)
                     for i in range(args.points)))[:args.points]

    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, "rb") as f:
        eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ex = eng.create_execution_context()
    ex.set_input_shape("text_tokens", (K, 32))
    ex.set_input_shape("img_ids", (K,))
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    dev = {}
    for n in names:
        shp = tuple(ex.get_tensor_shape(n))
        dt = torch.int64 if eng.get_tensor_dtype(n) == trt.int64 else torch.float32
        dev[n] = torch.empty(shp, dtype=dt, device="cuda")
        ex.set_tensor_address(n, dev[n].data_ptr())
    dev["text_tokens"].copy_(torch.from_numpy(toks).cuda())
    dev["img_ids"].copy_(torch.zeros(K, dtype=torch.int64).cuda())
    s = torch.cuda.Stream()

    hits = {p: 0 for p in prompts}
    best = {p: (0.0, -1) for p in prompts}
    for fi in idxs:
        cap.set(1, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        inp = cv2.cvtColor(cv2.resize(fr, (1008, 1008)), cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - 0.5) / 0.5
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1))
        dev["pixel_values"].copy_(torch.from_numpy(inp).cuda().unsqueeze(0))
        ex.execute_async_v3(s.cuda_stream)
        torch.cuda.synchronize()
        probs = (torch.sigmoid(dev["pred_logits"][:, :, 0].float())
                 * torch.sigmoid(dev["presence"][:, :1].float())).cpu().numpy()
        row = []
        for i, p in enumerate(prompts):
            m = float(probs[i].max())
            n_hit = int((probs[i] >= args.thresh).sum())
            hits[p] += 1 if n_hit else 0
            if m > best[p][0]:
                best[p] = (m, fi)
            row.append(f"{p}:{m:.2f}/{n_hit}")
        print(f"frame {fi}: " + " ".join(row), flush=True)
    print("--- SUMMARY (prompt: hits/probed, best score @ frame) ---")
    for p in prompts:
        print(f"{p}: {hits[p]}/{len(idxs)} best {best[p][0]:.2f} @ {best[p][1]}")


if __name__ == "__main__":
    main()
