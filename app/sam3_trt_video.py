#!/usr/bin/env python
"""SAM3.1 grounding TRT engine as a per-frame detector on a video clip.

Pure TRT loop (no torch model load): text tokens are taken from a reference
npz captured at export time. --batch N runs a batched engine (static N) with
a producer thread for decode/preprocess overlap; the tail batch is padded by
repeating the last frame.
"""
import argparse
import faulthandler
import json
import queue
import signal
import threading
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

faulthandler.register(signal.SIGUSR1, all_threads=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/workspace/out/sam3_grounding_fp8.engine")
    ap.add_argument("--video", default="/tmp/football.mp4")
    ap.add_argument("--ref", default="/workspace/out/sam3_ref_ball60_fp16.npz",
                    help="npz supplying the text_tokens for the prompt")
    ap.add_argument("--out", default="/workspace/out/sam3_fp8_football_ball.mp4")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    B = args.batch
    tok_np = np.load(args.ref)["text_tokens"].astype(np.int64)

    logger = trt.Logger(trt.Logger.WARNING)
    t0 = time.time()
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    print(f"engine loaded {time.time() - t0:.1f}s")

    tok = torch.from_numpy(np.repeat(tok_np, B, axis=0)).cuda()
    pix = torch.empty((B, 3, 1008, 1008), dtype=torch.float32, device="cuda")
    outs = {}
    for name in engine:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(engine.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
    names = list(outs)
    assert tuple(engine.get_tensor_shape("pixel_values"))[0] == B, \
        f"engine batch {engine.get_tensor_shape('pixel_values')[0]} != --batch {B}"

    def run():
        ctx.set_tensor_address("pixel_values", pix.data_ptr())
        ctx.set_tensor_address("text_tokens", tok.data_ptr())
        for n in names:
            ctx.set_tensor_address(n, outs[n].data_ptr())
        ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert ok

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (vw, vh))
    print(f"video {vw}x{vh} @ {fps:.1f}fps, {total} frames, batch {B} -> {args.out}")

    mean_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # producer: decode + preprocess, bounded queue of (frame, inp)
    q = queue.Queue(maxsize=2 * B)

    def produce():
        n = 0
        try:
            while n < total:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % args.stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(rgb, (1008, 1008), interpolation=cv2.INTER_LINEAR)
                    inp = inp.astype(np.float32) / 255.0
                    inp = (inp - mean_rgb) / mean_rgb
                    q.put((frame, np.ascontiguousarray(inp.transpose(2, 0, 1))))
                n += 1
        except Exception as e:  # never leave the consumer blocked on q.get()
            print(f"producer thread died: {e!r}", flush=True)
        finally:
            q.put(None)

    th = threading.Thread(target=produce, daemon=True)
    th.start()

    ev0 = torch.cuda.Event(True)
    ev1 = torch.cuda.Event(True)
    n_proc = n_det = 0
    batch_ms = []
    t_start = time.time()
    last_box = None

    drained = False
    while not drained:
        items = []
        while len(items) < B:
            it = q.get()
            if it is None:
                drained = True
                break
            items.append(it)
        if not items:
            break
        pad = B - len(items)
        for i in range(B - len(items)):
            items.append(items[-1])  # pad with last frame

        for i, (_, inp) in enumerate(items):
            pix[i].copy_(torch.from_numpy(inp).cuda())

        ev0.record()
        run()
        ev1.record()
        torch.cuda.synchronize()
        batch_ms.append(ev0.elapsed_time(ev1))

        # GPU-side scoring: probs (B,Q), argmax per row
        logits, presence = outs["pred_logits"], outs["presence"]
        probs = torch.sigmoid(logits) * torch.sigmoid(presence)[:, None, :]  # (B,Q,1)
        probs = probs[:, :, 0]
        qs = torch.argmax(probs, dim=1).cpu().numpy()
        scores = probs.max(dim=1).values.cpu().numpy()

        real = B - pad
        for i in range(real):
            frame = items[i][0]
            score = float(scores[i])
            if score >= args.thresh:
                q_ = int(qs[i])
                b = outs["pred_boxes"][i, q_].cpu().numpy()
                x1 = (b[0] - b[2] / 2) * vw
                y1 = (b[1] - b[3] / 2) * vh
                x2 = (b[0] + b[2] / 2) * vw
                y2 = (b[1] + b[3] / 2) * vh
                last_box = (x1, y1, x2, y2)
                n_det += 1
                mlog = outs["pred_masks"][i, q_].cpu().numpy()
                mlog = cv2.resize(mlog, (vw, vh), interpolation=cv2.INTER_LINEAR)
                mb = sigmoid(mlog) > 0.5
                color = np.array([0, 220, 0], dtype=np.uint8)
                frame[mb] = (0.45 * color + 0.55 * frame[mb]).astype(np.uint8)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"ball {score:.2f}",
                            (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                            cv2.LINE_AA)
            elif last_box is not None:
                x1, y1, x2, y2 = last_box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 220, 220), 1)
                cv2.putText(frame, f"lost ({score:.2f})",
                            (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1,
                            cv2.LINE_AA)
            writer.write(frame)
            n_proc += 1
        if n_proc % 100 < B:
            print(f"[{n_proc}/{total}] det {n_det} "
                  f"batch_med {np.median(batch_ms[-100:]):.1f}ms", flush=True)

    writer.release()
    cap.release()
    wall = time.time() - t_start
    batch_ms = np.array(batch_ms)
    per_frame = batch_ms / B
    stats = {
        "frames_processed": n_proc,
        "frames_detected": n_det,
        "coverage_pct": 100.0 * n_det / max(n_proc, 1),
        "batch": B,
        "batch_ms_mean": float(batch_ms.mean()),
        "infer_ms_per_frame": float(per_frame.mean()),
        "infer_fps": 1000.0 / float(per_frame.mean()),
        "end_to_end_fps": n_proc / wall,
        "wall_s": wall,
        "engine": args.engine,
        "thresh": args.thresh,
        "out": args.out,
    }
    print("SUMMARY " + json.dumps(stats, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
