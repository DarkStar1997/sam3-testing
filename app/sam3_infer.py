#!/usr/bin/env python
"""SAM 3 / SAM 3.1 inference CLI (image text-prompt segmentation, video tracking)."""
import argparse
import json
import sys
import time

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.float().detach().cpu().numpy()
    return np.asarray(x)


def run_image(args):
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"model_load: {time.time() - t0:.1f}s")

    image = Image.open(args.image).convert("RGB")
    t0 = time.time()
    state = processor.set_image(image)
    prefill = time.time() - t0
    t0 = time.time()
    out = processor.set_text_prompt(state=state, prompt=args.text)
    decode = time.time() - t0

    masks = _to_numpy(out["masks"])
    boxes = _to_numpy(out["boxes"])
    scores = _to_numpy(out["scores"])
    print(f"prefill: {prefill:.2f}s  decode: {decode:.2f}s  objects: {len(scores)}")
    print(f"masks{masks.shape} boxes{boxes.shape} -> boxes raw: {boxes.tolist()}")

    import cv2
    vis = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(len(scores)):
        m = masks[i].astype(bool)
        if m.shape != vis.shape[:2]:
            continue
        color = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255)][i % 4]
        layer = vis.copy()
        layer[m] = color
        vis = cv2.addWeighted(layer, 0.5, vis, 0.5, 0)
    ok = cv2.imwrite(args.out, vis)
    print(f"saved overlay: {args.out} ({ok})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "prompt": args.text,
                    "scores": scores.tolist(),
                    "boxes": boxes.tolist(),
                    "prefill_s": prefill,
                    "decode_s": decode,
                },
                f,
                indent=2,
            )


def run_video(args):
    import cv2
    import os
    from sam3 import build_sam3_predictor
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity

    # upstream bug @660a5e9: base start_session passes offload_state_to_cpu,
    # but multiplex init_state drops it; re-inject into the state dict (the
    # deep base stores this flag in inference_state and propagation honors it)
    _orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state

    def _init_state_compat(self, *a, **kw):
        offload_state = kw.pop("offload_state_to_cpu", False)
        state = _orig_init_state(self, *a, **kw)
        if offload_state and isinstance(state, dict):
            state["offload_state_to_cpu"] = True
            state["storage_device"] = torch.device("cpu")
        return state

    Sam3MultiplexTrackingWithInteractivity.init_state = _init_state_compat

    # free the reserved pool before each batched grounding pass (its transient
    # peak is ~9-16GB on top of the live state)
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector

    _orig_gcb = Sam3MultiplexDetector._process_grounding_chunk_batched

    def _gcb_compat(self, *a, **kw):
        torch.cuda.empty_cache()
        return _orig_gcb(self, *a, **kw)

    Sam3MultiplexDetector._process_grounding_chunk_batched = _gcb_compat

    t0 = time.time()
    predictor = build_sam3_predictor(
        version=args.version,
        compile=False,
        async_loading_frames=False,
        use_fa3=False,
        max_num_objects=args.max_objects,
        multiplex_count=args.multiplex_count,
    )
    print(f"model_load: {time.time() - t0:.1f}s")

    # multiplex predictor takes a frame directory; extract mp4 -> JPEGs
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if args.max_frames:
        frames = frames[: args.max_frames]
    n = len(frames)
    print(f"video: {n} frames {w}x{h}@{fps:.1f}")

    frame_dir = args.frames_dir or "/tmp/sam3_frames"
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(frame_dir, f"{i:05d}.jpg"), frame)

    r = predictor.handle_request(
        dict(
            type="start_session",
            resource_path=frame_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
    )
    sid = r["session_id"]
    print(f"session: {sid}")

    r = predictor.handle_request(
        dict(
            type="add_prompt",
            session_id=sid,
            frame_index=args.frame,
            text=args.text,
        )
    )
    print(f"add_prompt keys: {list(r.keys()) if isinstance(r, dict) else type(r)}")
    if isinstance(r, dict) and isinstance(r.get("outputs"), dict):
        o = r["outputs"]
        ids = o.get("out_obj_ids")
        scores = o.get("out_scores") or o.get("scores")
        print(f"prompt detection: ids={ids} scores={scores if not isinstance(scores, torch.Tensor) else scores.float().cpu().numpy().round(3).tolist()}")

    per_frame = {}
    t0 = time.time()
    n_tracked = 0
    for resp in predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=sid,
            propagation_direction=args.direction,
        )
    ):
        fi = resp.get("frame_index")
        if fi is None:
            continue
        n_tracked += 1
        outputs = resp.get("outputs", {}) or {}
        obj_ids = outputs.get("out_obj_ids", [])
        bm = outputs.get("out_binary_masks")
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy().tolist()
        if isinstance(bm, torch.Tensor):
            bm = bm.cpu().numpy()
        per_frame[fi] = (obj_ids, bm)
        if n_tracked == 1:
            print(
                f"first yield: frame={fi} objs={obj_ids} masks shape={None if bm is None else bm.shape}"
            )
    wall = time.time() - t0
    print(f"propagate: {n_tracked} frames in {wall:.1f}s")

    palette = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (0, 0, 255)]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, frame in enumerate(frames):
        vis = frame
        if i in per_frame:
            obj_ids, bm = per_frame[i]
            if bm is not None:
                if bm.ndim == 4:
                    bm = bm.squeeze(1)
                for j in range(len(obj_ids)):
                    m = bm[j].astype(bool)
                    if m.shape == vis.shape[:2]:
                        color = palette[int(obj_ids[j]) % len(palette)]
                        layer = vis.copy()
                        layer[m] = color
                        vis = cv2.addWeighted(layer, 0.5, vis, 0.5, 0)
        writer.write(vis)
    writer.release()
    print(f"saved: {args.out}")

    predictor.handle_request(dict(type="close_session", session_id=sid))
    n_with_objs = sum(1 for v in per_frame.values() if len(v[0]) > 0)
    areas = [
        int(bm[j].sum())
        for obj_ids, bm in per_frame.values()
        if bm is not None and len(obj_ids)
        for j in range(len(obj_ids))
    ]
    print(
        json.dumps(
            {
                "frames_total": n,
                "frames_tracked": n_tracked,
                "frames_with_objects": n_with_objs,
                "mask_area_px_median": sorted(areas)[len(areas) // 2] if areas else 0,
                "propagate_wall_s": round(wall, 2),
                "inference_fps": round(n_tracked / wall, 2) if wall else None,
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("image", help="text-prompt image segmentation")
    pi.add_argument("--image", required=True)
    pi.add_argument("--text", required=True)
    pi.add_argument("--out", default="out_sam3.png")
    pi.add_argument("--json", default=None)
    pi.set_defaults(fn=run_image)

    pv = sub.add_parser("video", help="SAM 3.1 multiplex video tracking")
    pv.add_argument("--video", required=True)
    pv.add_argument("--text", required=True)
    pv.add_argument("--frame", type=int, default=0, help="prompt frame index")
    pv.add_argument("--out", default="out_sam3_video.mp4")
    pv.add_argument("--version", default="sam3.1", choices=["sam3", "sam3.1"])
    pv.add_argument("--max-frames", type=int, default=None)
    pv.add_argument("--frames-dir", default=None, help="reuse existing frame dir")
    pv.add_argument("--max-objects", type=int, default=16)
    pv.add_argument("--multiplex-count", type=int, default=16)
    pv.add_argument(
        "--direction", default="forward", choices=["forward", "backward", "both"]
    )
    pv.set_defaults(fn=run_video)

    args = p.parse_args()
    print(f"numpy {np.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        args.fn(args)


if __name__ == "__main__":
    main()
