#!/usr/bin/env python
"""SAM3.1 dynamic-K FP8 TRT engine: multi-class per-frame detector on a video.

One engine pass per frame covers K text-prompt rows (shared vision trunk).
Token rows come from a tokens npz (rows of (32,) int64 + optional 'prompts').
Draw styles by prompt substring:
  *ball*   -> green mask overlay + box for every query >= thresh
              (NMS-deduped, so multiple balls are drawn)
  *goal*   -> magenta boxes for every query >= thresh
  *field*/*pitch*/*area* -> translucent blue union mask of queries
              >= thresh (drawn first, under everything else)
  other    -> cyan boxes for every query >= thresh
Producer thread (decode/preprocess) + async writer thread (bounded FIFO queue).
"""
import argparse
import faulthandler
import json
import queue
import signal
import threading
import time

import av
import cv2
import numpy as np
import tensorrt as trt
import torch
from fractions import Fraction

faulthandler.register(signal.SIGUSR1, all_threads=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, iou_thr=0.5):
    """Greedy NMS over xyxy pixel boxes. Returns kept indices, score order."""
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            xx1 = max(boxes[i][0], boxes[j][0])
            yy1 = max(boxes[i][1], boxes[j][1])
            xx2 = min(boxes[i][2], boxes[j][2])
            yy2 = min(boxes[i][3], boxes[j][3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            a = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            b = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            if inter / max(a + b - inter, 1e-9) > iou_thr:
                suppressed[j] = True
    return keep


def style_for(prompt):
    p = prompt.lower()
    if "ball" in p:
        return ("ball", (0, 255, 0))
    if "goal" in p:
        return ("goal", (255, 0, 255))
    if "field" in p or "pitch" in p or "area" in p:
        return ("area", (255, 0, 0))
    return ("box", (255, 200, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="/workspace/out/sam3_grounding_fp8_dynk.engine")
    ap.add_argument("--video", default="/tmp/football.mp4")
    ap.add_argument("--tokens", default="/workspace/out/sam3_tokens.npz",
                    help="npz with text_tokens (K,32) int64 + prompts")
    ap.add_argument("--rows", default="",
                    help="comma-separated row indices to use (default: all)")
    ap.add_argument("--labels", default="",
                    help="comma-separated display labels, one per row "
                         "(default: prompts from npz)")
    ap.add_argument("--out", default="/workspace/out/sam3_fp8_multi.mp4")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-boxes", type=int, default=48)
    ap.add_argument("--ball-iou", type=float, default=0.5,
                    help="NMS IoU threshold for ball-class queries")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    f = np.load(args.tokens, allow_pickle=True)
    tok_all = f["text_tokens" if "text_tokens" in f else "tokens"].astype(np.int64)
    prompts = ([str(p) for p in f["prompts"]] if "prompts" in f
               else [f"row{i}" for i in range(len(tok_all))])
    if args.rows:
        idx = [int(i) for i in args.rows.split(",")]
    else:
        idx = list(range(len(tok_all)))
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        assert len(labels) == len(idx)
    else:
        labels = [prompts[i] for i in idx]
    K = len(idx)
    styles = [style_for(p) for p in labels]
    ball_rows = [k for k, s in enumerate(styles) if s[0] == "ball"]
    print(f"K={K} classes: " + ", ".join(
        f"{l}[{s[0]}]" for l, s in zip(labels, styles)))

    tok_np = tok_all[idx]
    ids_np = np.zeros(K, dtype=np.int64)

    logger = trt.Logger(trt.Logger.WARNING)
    t0 = time.time()
    with open(args.engine, "rb") as fh, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(fh.read())
    ctx = engine.create_execution_context()
    print(f"engine loaded {time.time() - t0:.1f}s")

    pix = torch.empty((1, 3, 1008, 1008), dtype=torch.float32, device="cuda")
    tok = torch.from_numpy(tok_np).cuda()
    ids = torch.from_numpy(ids_np).cuda()
    ctx.set_input_shape("pixel_values", (1, 3, 1008, 1008))
    ctx.set_input_shape("text_tokens", tuple(tok_np.shape))
    ctx.set_input_shape("img_ids", tuple(ids_np.shape))
    outs = {}
    for name in engine:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
    print({n: tuple(t.shape) for n, t in outs.items()})

    def run():
        ctx.set_tensor_address("pixel_values", pix.data_ptr())
        ctx.set_tensor_address("text_tokens", tok.data_ptr())
        ctx.set_tensor_address("img_ids", ids.data_ptr())
        for n, t in outs.items():
            ctx.set_tensor_address(n, t.data_ptr())
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
    ew, eh = vw & ~1, vh & ~1  # yuv420p needs even dims
    container = av.open(args.out, mode="w")
    stream = container.add_stream(
        "libx264", rate=Fraction(fps).limit_denominator(1001))
    stream.width, stream.height = ew, eh
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "veryfast"}
    print(f"video {vw}x{vh} @ {fps:.1f}fps, {total} frames -> {args.out}")

    mean_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # producer: decode + preprocess
    pq = queue.Queue(maxsize=4)

    def produce():
        n = 0
        try:
            while n < total:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % args.stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(rgb, (1008, 1008),
                                     interpolation=cv2.INTER_LINEAR)
                    inp = inp.astype(np.float32) / 255.0
                    inp = (inp - mean_rgb) / mean_rgb
                    pq.put((frame, np.ascontiguousarray(
                        inp.transpose(2, 0, 1))))
                n += 1
        except Exception as e:  # never leave the consumer blocked on pq.get()
            print(f"producer thread died: {e!r}", flush=True)
        finally:
            pq.put(None)

    # async writer: bounded FIFO, keeps draw+encode off the inference thread
    wq = queue.Queue(maxsize=64)

    def write_loop():
        while True:
            it = wq.get()
            if it is None:
                return
            f = av.VideoFrame.from_ndarray(it[:eh, :ew], format="bgr24")
            for pkt in stream.encode(f):
                container.mux(pkt)

    threading.Thread(target=produce, daemon=True).start()
    wt = threading.Thread(target=write_loop, daemon=True)
    wt.start()

    ev0 = torch.cuda.Event(True)
    ev1 = torch.cuda.Event(True)
    n_proc = 0
    infer_ms = []
    t_start = time.time()
    last_ball = None
    cls_frames = [0] * K
    cls_dets = [0] * K
    ball_det = 0
    area_pct = []

    drained = False
    while not drained:
        it = pq.get()
        if it is None:
            drained = True  # sentinel consumed: outer loop must exit
        items = [] if it is None else [it]
        if not items:
            break

        frame, inp = items[0]
        pix[0].copy_(torch.from_numpy(inp).cuda())

        ev0.record()
        run()
        ev1.record()
        torch.cuda.synchronize()
        infer_ms.append(ev0.elapsed_time(ev1))

        # GPU-side scoring: probs (K,Q)
        logits, presence = outs["pred_logits"], outs["presence"]
        probs = torch.sigmoid(logits)[:, :, 0] * torch.sigmoid(presence)[:, :1]
        pk = probs.cpu().numpy()  # (K,Q) small
        boxes_k = outs["pred_boxes"].cpu().numpy()  # (K,Q,4) cxcywh

        # draw order: area regions first (underneath), then balls/goals/boxes
        draw_order = sorted(range(K),
                            key=lambda k: styles[k][0] != "area")
        for k in draw_order:
            kind, color = styles[k]
            if kind == "area":
                sel = np.where(pk[k] >= args.thresh)[0][:args.max_boxes]
                if len(sel):
                    mlog_all = outs["pred_masks"][k, sel].cpu().numpy()
                    union = np.zeros((vh, vw), dtype=bool)
                    for j in range(len(sel)):
                        mlog = cv2.resize(mlog_all[j], (vw, vh),
                                          interpolation=cv2.INTER_LINEAR)
                        union |= sigmoid(mlog) > 0.5
                    if union.any():
                        blend = frame[union].astype(np.float32) * 0.7 + \
                            np.float32(color) * 0.3
                        frame[union] = blend.astype(np.uint8)
                        cnts, _ = cv2.findContours(
                            union.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame, cnts, -1, color, 2)
                        cls_frames[k] += 1
                        cls_dets[k] += len(sel)
                        area_pct.append(100.0 * union.sum() / (vw * vh))
                        cv2.putText(frame, f"{labels[k]} {area_pct[-1]:.0f}%",
                                    (10, 25 + 22 * k),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                                    cv2.LINE_AA)
            elif kind == "ball":
                sel = np.where(pk[k] >= args.thresh)[0][:args.max_boxes]
                if len(sel):
                    pboxes = []
                    for q_ in sel:
                        b = boxes_k[k, q_]
                        pboxes.append(((b[0] - b[2] / 2) * vw,
                                       (b[1] - b[3] / 2) * vh,
                                       (b[0] + b[2] / 2) * vw,
                                       (b[1] + b[3] / 2) * vh))
                    keep = nms(np.array(pboxes), pk[k][sel], args.ball_iou)
                    mlog_all = outs["pred_masks"][k, sel].cpu().numpy()
                    best = None
                    for j in keep:
                        q_ = sel[j]
                        score = float(pk[k][q_])
                        x1, y1, x2, y2 = pboxes[j]
                        if best is None or score > best[0]:
                            best = (score, x1, y1, x2, y2)
                        mlog = cv2.resize(mlog_all[j], (vw, vh),
                                          interpolation=cv2.INTER_LINEAR)
                        mb = sigmoid(mlog) > 0.5
                        overlay = np.zeros_like(frame)
                        overlay[mb] = color
                        frame[mb] = cv2.addWeighted(overlay, 0.45, frame,
                                                    0.55, 0)[mb]
                        cv2.rectangle(frame, (int(x1), int(y1)),
                                      (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{labels[k]} {score:.2f}",
                                    (int(x1), max(15, int(y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                    cv2.LINE_AA)
                    last_ball = best[1:]
                    ball_det += 1
                    cls_frames[k] += 1
                    cls_dets[k] += len(keep)
                    cv2.putText(frame, f"{labels[k]} x{len(keep)}",
                                (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2, cv2.LINE_AA)
                elif last_ball is not None:
                    x1, y1, x2, y2 = last_ball
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 220, 220), 1)
                    cv2.putText(frame, "lost",
                                (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220),
                                1, cv2.LINE_AA)
            else:
                sel = np.where(pk[k] >= args.thresh)[0][:args.max_boxes]
                for q_ in sel:
                    b = boxes_k[k, q_]
                    x1 = int((b[0] - b[2] / 2) * vw)
                    y1 = int((b[1] - b[3] / 2) * vh)
                    x2 = int((b[0] + b[2] / 2) * vw)
                    y2 = int((b[1] + b[3] / 2) * vh)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if len(sel):
                    cls_frames[k] += 1
                    cls_dets[k] += len(sel)
                    cv2.putText(frame,
                                f"{labels[k]} x{len(sel)}",
                                (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2, cv2.LINE_AA)

        wq.put(frame)
        n_proc += 1
        if n_proc % 100 == 0:
            print(f"[{n_proc}/{total}] infer_med "
                  f"{np.median(infer_ms[-100:]):.1f}ms "
                  f"ball={ball_det}", flush=True)

    wq.put(None)
    wt.join()  # wall time includes the writer drain
    for pkt in stream.encode():  # flush
        container.mux(pkt)
    container.close()
    cap.release()
    wall = time.time() - t_start
    infer_ms = np.array(infer_ms)
    per_cls = {
        labels[k]: {
            "frames_with_dets": cls_frames[k],
            "total_dets": cls_dets[k],
            "mean_dets_when_present":
                round(cls_dets[k] / cls_frames[k], 2) if cls_frames[k] else 0.0,
        } for k in range(K)
    }
    stats = {
        "frames_processed": n_proc,
        "infer_ms_mean": float(infer_ms.mean()),
        "infer_ms_median": float(np.median(infer_ms)),
        "infer_fps": 1000.0 / float(infer_ms.mean()),
        "end_to_end_fps": n_proc / wall,
        "wall_s": wall,
        "ball_coverage_pct": 100.0 * ball_det / max(n_proc, 1),
        "ball_dets_total": sum(cls_dets[k] for k in ball_rows),
        "field_area_pct_mean": (float(np.mean(area_pct))
                                if area_pct else 0.0),
        "field_frames": len(area_pct),
        "thresh": args.thresh,
        "classes": per_cls,
        "engine": args.engine,
        "out": args.out,
    }
    print("SUMMARY " + json.dumps(stats, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
