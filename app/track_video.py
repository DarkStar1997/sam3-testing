import argparse
import re
import statistics
import time

import cv2
from PIL import Image

from locateanything_worker import LocateAnythingWorker

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
parser.add_argument("--categories", default="soccer ball,player")
args = parser.parse_args()
CATEGORIES = [c.strip() for c in args.categories.split(",")]

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_frames > 0:
    total = min(total, args.max_frames)

writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
worker = LocateAnythingWorker("/opt/LocateAnything-3B")

BALL_COLORS = {"ball": (0, 0, 255), "soccer ball": (0, 0, 255)}
PLAYER_COLOR = (255, 200, 0)

def detect_boxes(frame_rgb: Image.Image) -> list[tuple[str, tuple[int, int, int, int]]]:
    res = worker.detect(frame_rgb, CATEGORIES,
                        generation_mode=args.mode, max_new_tokens=8192, verbose=False)
    out = []
    label = None
    token = re.compile(r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
    for m in token.finditer(res["answer"]):
        if m.group(1) is not None:
            if m.group(1).strip() and not m.group(1).strip().isdigit():
                label = m.group(1).strip()
        else:
            x1, y1, x2, y2 = (int(m.group(i)) for i in range(2, 6))
            box = (int(x1 / 1000 * W), int(y1 / 1000 * H), int(x2 / 1000 * W), int(y2 / 1000 * H))
            out.append((label or "player", box))
    return out

def draw(frame, boxes, stale: int):
    for label, (x1, y1, x2, y2) in boxes:
        is_ball = "ball" in label.lower()
        color = BALL_COLORS.get(label.lower(), (0, 0, 255) if is_ball else PLAYER_COLOR)
        thickness = 3 if is_ball else 2
        if stale:
            color = tuple(int(c * 0.55) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 if is_ball else 0.35, color, 1)
    return frame

last_boxes: list = []
last_inference_boxes: list = []
times = []
n_inferred = 0
t_start = time.perf_counter()
idx = 0
written = 0

while True:
    ok, frame = cap.read()
    if not ok or written >= total:
        break

    if idx % args.stride == 0:
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        last_inference_boxes = detect_boxes(rgb)
        times.append(time.perf_counter() - t0)
        n_inferred += 1
        last_boxes = last_inference_boxes
        stale = 0
        n_ball = sum(1 for l, _ in last_boxes if "ball" in l.lower())
        print(f"[{idx+1}/{total}] infer {times[-1]:.2f}s  boxes={len(last_boxes)} (ball={n_ball})",
              flush=True)
    else:
        stale = idx % args.stride

    draw(frame, last_boxes, stale)
    writer.write(frame)
    written += 1
    idx += 1

cap.release()
writer.release()
wall = time.perf_counter() - t_start

print("=== SUMMARY ===")
print(f"frames_written={written} detections_run={n_inferred} stride={args.stride} mode={args.mode}")
print(f"inference: mean={statistics.mean(times):.3f}s median={statistics.median(times):.3f}s "
      f"max={max(times):.3f}s -> INFERENCE FPS={1/statistics.mean(times):.2f}")
print(f"wall={wall:.1f}s end_to_end_FPS={written/wall:.2f} "
      f"effective_detection_rate={n_inferred/(written/fps):.2f} detections/sec_of_video")
