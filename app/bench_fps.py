import argparse
import math
import statistics
import time

from PIL import Image, ImageDraw

from locateanything_worker import LocateAnythingWorker

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
parser.add_argument("--frames", type=int, default=30)
args = parser.parse_args()

W, H, N = args.width, args.height, args.frames
scale = W / 1280

def make_frame(i: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (40, 120, 60))
    d = ImageDraw.Draw(img)
    x, y = int((120 + i * 34) * scale), int(H / 2 + 60 * math.sin(i / 3))
    r = max(8, int(28 * scale))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 140, 0), outline=(0, 0, 0), width=3)
    return img

frames = [make_frame(i) for i in range(N)]
worker = LocateAnythingWorker("/opt/LocateAnything-3B")

warm = worker.point(frames[0], "the ball", generation_mode=args.mode, verbose=False)
print(f"mode={args.mode} res={W}x{H} warmup: {warm['answer'][:80]}")

times = []
for f in frames:
    t0 = time.perf_counter()
    r = worker.point(f, "the ball", generation_mode=args.mode, verbose=False)
    times.append(time.perf_counter() - t0)

print(f"frames={N} mode={args.mode} res={W}x{H}")
print(f"mean={statistics.mean(times):.3f}s  median={statistics.median(times):.3f}s  "
      f"min={min(times):.3f}s  max={max(times):.3f}s")
print(f"FPS: mean={1/statistics.mean(times):.2f}  median={1/statistics.median(times):.2f}")
