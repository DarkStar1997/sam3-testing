# sam3-testing — SAM 3 text-grounded detection & segmentation on TensorRT

Turns Facebook's [SAM 3](https://github.com/facebookresearch/sam3) image
grounding model into a **TensorRT FP8 engine with a dynamic label-count axis
(K = 1..8 text prompts per pass)**: one GPU pass segments/detects any mix of
open-vocabulary labels — e.g. *soccer ball*, *player*, *goalpost*, *soccer
field* — with masks and boxes.

No training is involved anywhere in this repo: the pipeline is
export → fp16 weight conversion → activation calibration → FP8 quantization
(Q/DQ) → TensorRT engine build. Everything is scripted and reproducible.

> **Note:** this repo is a general GPU-inference workbench — alongside the
> SAM3 pipeline it also carries the NVIDIA **LocateAnything-3B** deployment
> (`Dockerfile`, `app/locateanything_worker.py`, `app/infer.py`,
> `app/track_video.py`, `app/bench_fps.py`) and the flash-attn source-wheel
> builder (`Dockerfile.wheel`, wheel published at
> `ghcr.io/darkstar1997/flash-attn-wheel:torch2.14-cu130-py312`). The SAM3
> chain lives in `app/sam3_*.py` + `Dockerfile.sam3` / `Dockerfile.trt`.

Benchmarks (RTX 4090, 720p football video):

| Setup | Infer | End-to-end | Notes |
|---|---|---|---|
| torch bf16 (autocast) | ~560 ms | — | reference, `sam3_infer.py` |
| TRT FP16 engine | 39.5 ms (25.3 FPS) | — | single label |
| TRT FP8 engine | 34.2 ms (29.3 FPS) | 24.8 FPS | single label, masks IoU ≥ 0.998 vs FP16 |
| TRT FP8 dynamic-K, K=4 | 46.4 ms (21.4 FPS) | 19.4 FPS | ball + player + person + goalpost |
| TRT FP8 dynamic-K, K=5 | 50.6 ms (19.7 FPS) | 13.5 FPS | + soccer-field region mask |

Marginal cost per extra label ≈ 4–5 ms (vision trunk is computed once and
gathered K times). Compare: LocateAnything-3B on the same video runs at
4.05 FPS end-to-end (single class); the torch SAM3 video tracker at 6.94 FPS.

## Repo layout

```
app/sam3_export_onnx.py     torch model -> ONNX (static, --batch, or --texts dynamic-K)
app/sam3_make_tokens.py     text prompts -> token npz (CLIP BPE tokenizer, no model load)
app/sam3_make_w16.py        fp32 ONNX -> fp16-weights ONNX (halves calibration RAM)
app/sam3_fix_mixed_fp16.py  fixes leftover mixed-precision Mul nodes from the converter
app/sam3_make_named.py      modelopt `_preprocess_onnx` step standalone (deterministic names)
app/sam3_calib_ranges.py    chunked activation-range calibration (OOM-safe, resumable)
app/sam3_fp8_quant.py       modelopt FP8 Q/DQ quantization w/ injected ranges + RAM patches
app/sam3_patch_modelopt.py  idempotent site-packages patches (modelopt 0.46.0 peak RAM)
app/sam3_trt_build.py       ONNX -> TRT engine (strongly-typed, optimization profile)
app/sam3_trt_infer.py       parity vs reference npz + latency + K-sweep
app/sam3_trt_video_multi.py multi-class video pipeline (H.264, async writer, NMS, masks)
app/sam3_trt_video.py       single-class video pipeline
app/sam3_infer.py           torch baseline (image + video tracker, needs checkpoints)
Dockerfile.trt              engine-builder + inference env (no checkpoints)
Dockerfile.sam3             same + SAM3 checkpoints baked into HF cache
```

## Artifacts on ghcr.io

Model artifacts live in scratch-based images under
`ghcr.io/darkstar1997/sam3-models` (no shell; extract with `docker cp`):

| Tag | Contents | Size |
|---|---|---|
| `fp8-dynk-onnx` | `sam3_grounding_fp8_dynk.onnx` + external data — **build an engine from this** | 1.7 GB |
| `dynk-fp32-onnx` | fp32 export + external data (start of the full chain) | 3.4 GB |
| `dynk-w16` | fp16-weights ONNX (for FP16 engines / re-quantization) | 3.3 GB |
| `refs` | parity refs (`sam3_dynk_ref.npz`, `sam3_ref_ball60_fp16.npz`), token npzs, `sam3_fp8_ranges_dynk.pkl` | 0.2 GB |
| `engine-rtx4090` | prebuilt FP8 engine for RTX 4090 (sm_89), TRT 11.2 / CUDA 13 | 1.1 GB |

TRT engines are **compiled for a specific GPU (and TRT version)** — that is
the whole point of the Quickstart below. Copying the 4090 engine to another
GPU will not work; rebuild instead.

---

## Quickstart: build an engine for YOUR GPU

Requirements: NVIDIA GPU (Ampere or newer for FP16; **Ada/Blackwell —
sm_89, sm_90, sm_100, sm_120 — for FP8**), recent driver, Docker with
`nvidia-container-toolkit`. If your GPU is Ampere (A100 / RTX 30xx), skip
FP8 and build the strongly-typed engine directly from the `dynk-w16` ONNX —
it runs as FP16 (39.5 ms on a 4090) and needs no calibration.

```bash
git clone https://github.com/DarkStar1997/sam3-testing && cd sam3-testing

# 1. environment (torch 2.14 + cu13 + TRT 11.2 + modelopt + sam3 pkg)
docker build -t sam3-trt -f Dockerfile.trt .

# 2. pull the FP8 ONNX + reference data and extract them
docker pull ghcr.io/darkstar1997/sam3-models:fp8-dynk-onnx
docker pull ghcr.io/darkstar1997/sam3-models:refs
mkdir -p models
docker create --name tmp1 ghcr.io/darkstar1997/sam3-models:fp8-dynk-onnx
docker cp tmp1:/models/. models/
docker rm tmp1
docker create --name tmp2 ghcr.io/darkstar1997/sam3-models:refs
docker cp tmp2:/models/. models/
docker rm tmp2

# 3. build the engine ON YOUR GPU (~1-2 min; auto-detects the dynamic K axis)
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace sam3-trt \
    python sam3_trt_build.py \
    --onnx models/sam3_grounding_fp8_dynk.onnx \
    --engine models/sam3_grounding_fp8_dynk.engine \
    --strongly-typed --kmin 1 --kopt 3 --kmax 8

# 4. parity + latency on your GPU (uses the reference npz from :refs)
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace sam3-trt \
    python sam3_trt_infer.py \
    --engine models/sam3_grounding_fp8_dynk.engine \
    --ref models/sam3_dynk_ref.npz --orig-hw 360,640 \
    --tokens-npz models/sam3_tokens.npz --k-sweep

# 5. multi-class video (H.264 output, per-class boxes/masks, JSON stats)
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace sam3-trt \
    python sam3_trt_video_multi.py \
    --video /workspace/your.mp4 \
    --tokens models/sam3_tokens_field.npz --rows 0,1,2,3,4 \
    --out /workspace/out/annotated.mp4 \
    --json-out /workspace/out/annotated.json
```

Expectations for step 4: 2 cats detected on the cats.jpg reference row
(mask IoU ≥ 0.99 vs the torch model, raw logits differ — FP8 — but
post-sigmoid masks match), and a K-sweep where K=1 ≈ 34 ms on a 4090 with
+4–5 ms per added label.

### Labels

The engine is label-agnostic: text prompts are runtime inputs. Make your own
token file and run any open-vocabulary labels (K ≤ 8 per pass):

```bash
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace sam3-trt \
    python sam3_make_tokens.py --out models/tokens.npz \
    "soccer ball" "player" "referee" "goalpost"
```

Drawing styles are chosen by prompt substring: `*ball*` → green masks +
NMS-deduped multi-instance boxes; `*goal*` → magenta boxes; `*field*/*pitch*/*area*`
→ translucent region mask (union of queries, drawn underneath everything);
anything else → cyan boxes.

---

## Full chain: checkpoint → FP8 engine

Run this when you need a different input resolution, K range, or want to
re-quantize. Everything runs inside the `sam3-trt` image above. The chain
reproduces exactly the published `fp8-dynk-onnx` artifact.

```bash
# 0. checkpoints into the HF cache layout (~7 GB, research license)
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace \
    -e HF_HOME=/workspace/hf-cache sam3-trt \
    python -m huggingface_hub.commands.huggingface_cli download \
    facebook/sam3 config.json sam3.pt
# (or build Dockerfile.sam3 with a BuildKit hf_token secret to bake them in)

# 1. token npz for your labels
python app/sam3_make_tokens.py --out models/sam3_tokens_field.npz \
    "soccer ball" "player" "person" "goalpost" "soccer field"

# 2. dynamic-K export (TF32 is force-disabled inside: batch-1 vs batch-K
#    conv kernels diverge otherwise and row parity fails)
python app/sam3_export_onnx.py --out models/sam3_grounding_dynk.onnx \
    --image <some.jpg> --texts "a cat,soccer ball,person" \
    --dynamic --kmin 1 --kmax 8 --ref-npz models/sam3_dynk_ref.npz

# 3. fp16 weights (halves every later stage's RAM)
python app/sam3_make_w16.py models/sam3_grounding_dynk.onnx \
    models/sam3_grounding_dynk_w16.onnx

# 4. the fp16 converter leaves a few genuinely-mixed Mul nodes; fix them
python app/sam3_fix_mixed_fp16.py models/sam3_grounding_dynk_w16.onnx

# 5. deterministic node names for calibration (modelopt preprocess)
python app/sam3_make_named.py models/sam3_grounding_dynk_w16.onnx \
    models/sam3_grounding_dynk_w16_named.onnx /tmp/named_tmp.onnx

# 6. chunked activation calibration. Stock modelopt taps ALL ~17k tensors
#    in one ORT session and needs >20 GB RAM; this runs node-chunks in
#    fresh subprocesses under a tapped-bytes budget, resumable, and
#    auto-splits chunks that OOM. ~18 min / ~8 GB peak on this model.
python app/sam3_calib_ranges.py run \
    --onnx models/sam3_grounding_dynk_w16.onnx \
    --named models/sam3_grounding_dynk_w16_named.onnx \
    --out models/sam3_fp8_ranges_dynk.pkl \
    --refs models/sam3_dynk_ref.npz models/sam3_ref_ball60_fp16.npz \
    --chunks-file /tmp/sam3_dynk_chunks.json

# 7. modelopt RAM patches (already applied if you use Dockerfile.trt)
python app/sam3_patch_modelopt.py --check

# 8. FP8 Q/DQ quantization, injecting the precomputed ranges (~5 min,
#    ~13 GB peak; calibrates on CPU — onnxruntime-gpu EPs need CUDA 12)
python app/sam3_fp8_quant.py \
    --onnx models/sam3_grounding_dynk_w16.onnx \
    --out models/sam3_grounding_fp8_dynk.onnx \
    --ranges models/sam3_fp8_ranges_dynk.pkl \
    --refs models/sam3_dynk_ref.npz

# 9. engine build + verification: same as Quickstart steps 3-4
```

## Gotchas learned the hard way

- **numpy**: the `sam3` pip package needs `numpy<2`; the tensorrt/torch
  system site-packages are happy with 2.x. The venv shadow pin must be
  re-applied after any pip install that drags numpy back in.
- **External data suffixes differ**: torch.onnx.export writes `.onnx.data`,
  modelopt `_preprocess_onnx` writes `.onnx_data`. Both must travel with
  their proto file. `onnx.save` external data *appends* to existing files —
  rewriting to the same name grows the file (harmless but surprising).
- **modelopt peak RAM**: three separate multi-GB hogs (all-tensor taps,
  proto + graph copies during QDQ insert, fp16 post-processing). The
  chunked calibrator + `sam3_patch_modelopt.py` + fp16 weights keep the
  whole chain under ~13 GB.
- **Strongly-typed networks**: TRT 11 removed the FP16/INT64 builder flags;
  precision comes from graph casts (that's what the w16+QDQ chain produces)
  and `1 << STRONGLY_TYPED` in `sam3_trt_build.py`.
- **FP8 hardware**: Q/DQ FP8 engines require sm_89+ (RTX 40xx/L40S) or
  Blackwell. On Ampere use the `dynk-w16` ONNX directly (FP16 engine).
- **Engines are not portable**: rebuild per GPU (and per TRT version).

## Licenses

- Scripts in this repo: MIT.
- SAM3 code (pinned `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`) and the
  derived ONNX/engines: Meta **SAM License, non-commercial research use** —
  checkpoints stay on Hugging Face (`facebook/sam3`, `facebook/sam3.1`) and
  are NOT redistributed here.
- flash-attn wheel image, PyAV, TensorRT, modelopt: their respective
  licenses.
