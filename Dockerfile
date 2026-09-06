# syntax=docker/dockerfile:1.7
# Custom flash-attn wheel (torch 2.14 + cu130, archs 80;90;100;120) from ghcr
FROM ghcr.io/darkstar1997/flash-attn-wheel:torch2.14-cu130-py312 AS fa-wheel

FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_DIR=/opt/LocateAnything-3B \
    PYTHONPATH=/opt/LocateAnything-3B \
    HF_HOME=/tmp/hf

RUN apt-get update && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

COPY app/ /workspace/

RUN pip install --no-cache-dir huggingface_hub

# Bake model weights into the image (token passed as BuildKit secret, never stored in layers)
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    hf download nvidia/LocateAnything-3B --local-dir /opt/LocateAnything-3B \
    && rm -rf /opt/LocateAnything-3B/.cache /tmp/hf

COPY --from=fa-wheel /wheels/ /tmp/wheels/
RUN pip install --no-cache-dir \
        transformers==4.57.1 \
        tokenizers==0.22.0 \
        opencv-python-headless==4.11.0.86 \
        Pillow==11.1.0 \
        peft \
        accelerate \
        huggingface_hub \
        lmdb==1.7.5 \
        eva-decord==0.6.1 \
        /tmp/wheels/flash_attn-*.whl \
    && rm -rf /tmp/wheels

CMD ["python", "infer.py", "--help"]
