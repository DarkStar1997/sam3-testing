#!/bin/bash
set -e
python -c "from flash_attn import flash_attn_varlen_func; import torch; print('flash_attn OK | torch', torch.__version__)"
python bench_fps.py --mode hybrid --width 1280 --height 720 --frames 30 2>/dev/null | grep FPS
for BS in 4 8 16; do
  S=$(date +%s.%N)
  python /opt/LocateAnything-3B/batch_infer.py --model /opt/LocateAnything-3B --attn la_flash --vision-attn auto --scheduler pipeline --batch-size $BS --requests /host/test/requests.jsonl --out /host/test/out_laf_bs$BS.jsonl > /dev/null 2>&1
  E=$(date +%s.%N)
  python -c "print('la_flash bs=$BS wall=%.1fs FPS=%.2f' % ($E-$S, 128/($E-$S)))"
done
