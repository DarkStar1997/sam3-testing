#!/usr/bin/env python
"""Debug: where does SAM3.1 multiplex memory accumulate? Short run + tensor audit."""
import os
import sys
import time

import numpy as np
import torch
import cv2

from sam3 import build_sam3_predictor
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity

_orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state


def _init_state_compat(self, *a, **kw):
    offload_state = kw.pop("offload_state_to_cpu", False)
    state = _orig_init_state(self, *a, **kw)
    if offload_state and isinstance(state, dict):
        state["offload_state_to_cpu"] = True
        state["storage_device"] = torch.device("cpu")
    self._last_state = state
    return state


Sam3MultiplexTrackingWithInteractivity.init_state = _init_state_compat

# free reserved pool before each batched grounding pass (its transient peak is ~15-16GB)
from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector  # noqa: E402

_orig_gcb = Sam3MultiplexDetector._process_grounding_chunk_batched


def _gcb_compat(self, *a, **kw):
    torch.cuda.empty_cache()
    print(
        f"  [grounding chunk] pre: alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB",
        flush=True,
    )
    out = _orig_gcb(self, *a, **kw)
    print(
        f"  [grounding chunk] post: alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB",
        flush=True,
    )
    return out


Sam3MultiplexDetector._process_grounding_chunk_batched = _gcb_compat

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/football-sample05.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 90

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
frames = frames[:N]
h, w = frames[0].shape[:2]
print(f"{len(frames)} frames {w}x{h}")

frame_dir = "/tmp/frames"
os.makedirs(frame_dir, exist_ok=True)
for i, f in enumerate(frames):
    cv2.imwrite(os.path.join(frame_dir, f"{i:05d}.jpg"), f)

predictor = build_sam3_predictor(
    version="sam3.1", compile=False, async_loading_frames=False, use_fa3=False
)
r = predictor.handle_request(
    dict(
        type="start_session",
        resource_path=frame_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
)
sid = r["session_id"]
predictor.handle_request(
    dict(type="add_prompt", session_id=sid, frame_index=60, text="soccer ball")
)

print(f"after add_prompt: alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB")

t0 = time.time()
try:
    for i, resp in enumerate(
        predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid))
    ):
        if i % 20 == 0:
            print(
                f"[{i}] alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB "
                f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB t={time.time()-t0:.1f}s"
            )
    print(f"done: alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB wall={time.time()-t0:.1f}s")
except torch.OutOfMemoryError as e:
    print(f"OOM at iter {i}: {e}")
    print("=== memory summary (top) ===")
    for line in torch.cuda.memory_summary().splitlines():
        if any(k in line for k in ("Large pool", "Small pool", "Segment", "blocks", "----", "Sum", "reserved", "allocated")):
            print(line)
    import traceback
    traceback.print_exc()

state = getattr(predictor.model, "_last_state", None)
if state is None:
    print("no state handle")
    sys.exit(0)

def audit(obj, path, acc, depth=0):
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        key = (path, str(obj.device), tuple(obj.shape))
        acc[key] = acc.get(key, 0) + obj.numel() * obj.element_size()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            audit(v, f"{path}[{k!r:.40}]", acc, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            audit(v, f"{path}[{j}]", acc, depth + 1)

acc = {}
audit(state, "state", acc)
gpu = [(p, s, sh, b) for (p, dv, sh), b in acc.items() if "cuda" in dv]
cpu_total = sum(b for (p, dv, sh), b in acc.items() if "cpu" in dv)
gpu_total = sum(b for (p, dv, sh), b in acc.items() if "cuda" in dv)
print(f"\nstate GPU tensors: {gpu_total/2**30:.2f}GiB | CPU tensors: {cpu_total/2**30:.2f}GiB")
for p, dv, sh, b in sorted(gpu, key=lambda x: -x[3])[:15]:
    print(f"  {b/2**20:8.1f}MiB {dv} {sh} {p[:110]}")
