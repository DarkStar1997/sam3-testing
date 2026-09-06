#!/usr/bin/env python
"""Export SAM3 image grounding graph to ONNX (Phase 2, step 1).

Covers: vision trunk+FPN necks, CLIP-style text tower (token ids in),
prompt-aware transformer encoder, query decoder, dot-prod scoring,
segmentation head. Static shapes:
  pixel_values (1,3,1008,1008) float32
  text_tokens  (1,32) int64  (0 = pad)
Outputs:
  pred_logits (1,Q,1), pred_boxes cxcywh (1,Q,4),
  pred_masks logits (1,Q,288,288), presence (1,Q,1)

Post-processing (sigmoid, presence-mult, threshold, box scaling, mask
interpolation to original resolution) stays on the host side.

--autocast fp16|bf16 wraps compute in cuda autocast so the ONNX carries
explicit half casts -> feed to TRT STRONGLY_TYPED builder for a mixed-
precision engine.
"""
import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out/sam3_grounding.onnx")
    ap.add_argument("--image", default="/workspace/test/cats.jpg")
    ap.add_argument("--prompt", default="a cat")
    ap.add_argument("--ref-npz", default="/workspace/out/sam3_export_ref.npz")
    ap.add_argument("--no-export", action="store_true", help="only run ref pass")
    ap.add_argument("--autocast", default="none", choices=["none", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--texts",
        default=None,
        help="comma-separated prompts -> multi-label graph: pixel batch 1, "
        "K text rows, vision trunk shared via img_ids=zeros(K) gather. "
        "Overrides --prompt/--batch.",
    )
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="with --texts: export K dim as dynamic (torch.export Dim, "
        "--kmin..--kmax) instead of static K",
    )
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=8)
    args = ap.parse_args()

    import sam3.model.vitdet as vitdet_mod
    import sam3.perflib.fused as fused_mod

    def plain_addmm_act(activation, linear, mat1):
        # upstream casts everything to bf16 + aten._addmm_activation (cuBLASLt
        # fused gemm+gelu) -> unusable for fp32 tracing and ONNX; replace with
        # the mathematically identical plain path
        y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        raise ValueError(f"unexpected activation {activation}")

    fused_mod.addmm_act = plain_addmm_act
    vitdet_mod.addmm_act = plain_addmm_act

    # geometry encoder does scale.pin_memory().to(...) on a 4-elem tensor even
    # with an empty box prompt; aten._pin_memory is NYI for torch.export
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    # decoder _get_rpb_matrix receives feat_size=(spatial_shapes[0,0],
    # spatial_shapes[0,1]) i.e. 0-dim TENSORS -> data-dependent guards under
    # torch.export. Capture real (H,W) on the first eager call, then freeze.
    from sam3.model.decoder import TransformerDecoder

    _orig_rpb = TransformerDecoder._get_rpb_matrix
    _frozen_hw = {}

    def frozen_rpb(self, reference_boxes, feat_size):
        if "hw" not in _frozen_hw:
            _frozen_hw["hw"] = tuple(int(x) for x in feat_size)
            print(f"rpb frozen feat_size={_frozen_hw['hw']}")
        hw = _frozen_hw["hw"]
        if self.compilable_cord_cache is None:
            self.compilable_cord_cache = self._get_coords(
                hw[0], hw[1], reference_boxes.device
            )
            self.compilable_stored_size = hw
        return _orig_rpb(self, reference_boxes, hw)

    TransformerDecoder._get_rpb_matrix = frozen_rpb

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import FindStage

    t0 = time.time()
    # exact-fp32 ref pass: cudnn conv TF32 is on by default and batch-1 vs
    # batch-K convs pick different kernels -> TF32 noise amplified through
    # the trunk drowns out the row-parity signal
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_sam3_image_model()
    model.eval()
    print(f"model_load {time.time() - t0:.1f}s")

    lang = model.backbone.language_backbone
    ctx = lang.context_length
    ac_dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.autocast
    ]

    class Graph(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            n = args.batch
            ids = torch.arange(n, device="cuda")
            self.find_stage = FindStage(
                img_ids=ids,
                text_ids=ids,
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

        def forward(self, pixel_values, text_tokens):
            m = self.m
            if ac_dtype is not None:
                cm = torch.autocast("cuda", dtype=ac_dtype)
            else:
                import contextlib

                cm = contextlib.nullcontext()
            with cm:
                lb = m.backbone.language_backbone
                _, text_memory = lb.encoder(text_tokens)  # (1, ctx, W)
                lang_feats = lb.resizer(text_memory.transpose(0, 1))  # (ctx, 1, d)
                lang_mask = text_tokens == 0  # (1, ctx) True = pad
                backbone_out = m.backbone.forward_image(pixel_values)
                # text-only grounding: geometry/visual prompts are always
                # empty here; their 0-size concat breaks TRT parsing
                prompt = lang_feats
                prompt_mask = lang_mask
                backbone_out, encoder_out, _ = m._run_encoder(
                    backbone_out, self.find_stage, prompt, prompt_mask
                )
                out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
                out, hs = m._run_decoder(
                    memory=out["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )
                m._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=self.find_stage.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )
                return (
                    out["pred_logits"].float(),
                    out["pred_boxes"].float(),
                    out["pred_masks"].float(),
                    out["presence_logit_dec"].float(),
                )

    # inputs
    from PIL import Image
    from torchvision.transforms import v2

    tr = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    pix1 = tr(v2.functional.to_image(img)).cuda().unsqueeze(0).contiguous()

    if args.texts:
        prompts = [p.strip() for p in args.texts.split(",") if p.strip()]
        kn = len(prompts)
        print(f"multi-label mode: K={kn} prompts={prompts} dynamic={args.dynamic}")

        class GraphMulti(torch.nn.Module):
            """Shared vision trunk + K text branches.

            img_ids (K,) int64 zeros replicates the single image's features
            to K rows (native Gather in _get_img_feats); text tower runs on
            the (K,32) token batch. text_ids is unused on this path (lang
            feats are built directly from the token batch).
            """

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values, text_tokens, img_ids):
                m = self.m
                if ac_dtype is not None:
                    cm = torch.autocast("cuda", dtype=ac_dtype)
                else:
                    import contextlib

                    cm = contextlib.nullcontext()
                with cm:
                    lb = m.backbone.language_backbone
                    _, text_memory = lb.encoder(text_tokens)  # (K, ctx, W)
                    lang_feats = lb.resizer(text_memory.transpose(0, 1))  # (ctx, K, d)
                    lang_mask = text_tokens == 0  # (K, ctx) True = pad
                    backbone_out = m.backbone.forward_image(pixel_values)
                    prompt = lang_feats
                    prompt_mask = lang_mask
                    find_stage = FindStage(
                        img_ids=img_ids,
                        text_ids=img_ids,  # unused on this path
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    )
                    backbone_out, encoder_out, _ = m._run_encoder(
                        backbone_out, find_stage, prompt, prompt_mask
                    )
                    out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
                    out, hs = m._run_decoder(
                        memory=out["encoder_hidden_states"],
                        pos_embed=encoder_out["pos_embed"],
                        src_mask=encoder_out["padding_mask"],
                        out=out,
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        encoder_out=encoder_out,
                    )
                    m._run_segmentation_heads(
                        out=out,
                        backbone_out=backbone_out,
                        img_ids=img_ids,
                        vis_feat_sizes=encoder_out["vis_feat_sizes"],
                        encoder_hidden_states=out["encoder_hidden_states"],
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        hs=hs,
                    )
                    return (
                        out["pred_logits"].float(),
                        out["pred_boxes"].float(),
                        out["pred_masks"].float(),
                        out["presence_logit_dec"].float(),
                    )

        g = GraphMulti(model).eval()
        pix = pix1  # batch stays 1; trunk shared
        tokens = lang.tokenizer(prompts, context_length=ctx).cuda()  # (K,32)
        img_ids = torch.zeros(kn, dtype=torch.long, device="cuda")
        print(
            f"pix {tuple(pix.shape)} {pix.dtype}  tokens {tuple(tokens.shape)} "
            f"{tokens.dtype}  img_ids {tuple(img_ids.shape)}"
        )
        with torch.no_grad():
            logits, boxes, masks, presence = g(pix, tokens, img_ids)
        print(
            f"ref pass: logits {tuple(logits.shape)} boxes {tuple(boxes.shape)} "
            f"masks {tuple(masks.shape)} presence {tuple(presence.shape)}"
        )

        # row parity: K-row run vs per-prompt K=1 runs (same eager model)
        maxd = [0.0, 0.0, 0.0]
        for i, p in enumerate(prompts):
            tok1 = lang.tokenizer([p], context_length=ctx).cuda()
            id1 = torch.zeros(1, dtype=torch.long, device="cuda")
            with torch.no_grad():
                l1, b1, m1, p1 = g(pix1, tok1, id1)
            dl = (l1[0, :, 0] - logits[i, :, 0]).abs().max().item()
            db = (b1[0] - boxes[i]).abs().max().item()
            dm = (m1[0] - masks[i]).abs().max().item()
            dp = abs(p1.flatten()[0].item() - presence[i].flatten()[0].item())
            print(f"  row {i} '{p}': dlogits={dl:.5f} dboxes={db:.5f} "
                  f"dmasks={dm:.5f} dpresence={dp:.5f}")
            maxd = [max(maxd[0], dl), max(maxd[1], db), max(maxd[2], dm)]
        if maxd[0] > 0.01 or maxd[1] > 0.005 or maxd[2] > 0.1:
            raise SystemExit(f"row parity FAILED {maxd}")
        print(f"row parity OK (max diffs logits={maxd[0]:.5f} boxes={maxd[1]:.5f} "
              f"masks={maxd[2]:.5f})")

        np.savez(
            args.ref_npz,
            pixel_values=pix.cpu().numpy(),
            text_tokens=tokens.cpu().numpy().astype(np.int64),
            img_ids=img_ids.cpu().numpy().astype(np.int64),
            pred_logits=logits.float().cpu().numpy(),
            pred_boxes=boxes.float().cpu().numpy(),
            pred_masks=masks.float().cpu().numpy(),
            presence=presence.float().cpu().numpy(),
            prompts=np.array(prompts),
        )
        print(f"saved reference npz: {args.ref_npz}")
        if args.no_export:
            return

        t0 = time.time()
        in_names = ["pixel_values", "text_tokens", "img_ids"]
        out_names = ["pred_logits", "pred_boxes", "pred_masks", "presence"]
        with torch.no_grad():
            if args.dynamic:
                from torch.export import Dim

                kdim = Dim("K", min=args.kmin, max=args.kmax)
                ep = torch.export.export(
                    g,
                    (pix, tokens, img_ids),
                    dynamic_shapes=({}, {0: kdim}, {0: kdim}),
                )
                print(f"torch.export ({time.time() - t0:.1f}s)")
                torch.onnx.export(
                    ep,
                    (pix, tokens, img_ids),
                    args.out,
                    input_names=in_names,
                    output_names=out_names,
                    dynamo=True,
                )
            else:
                torch.onnx.export(
                    g,
                    (pix, tokens, img_ids),
                    args.out,
                    input_names=in_names,
                    output_names=out_names,
                    dynamo=True,
                )
        print(f"onnx export: {time.time() - t0:.1f}s -> {args.out}")
        return

    pix = pix1.repeat(args.batch, 1, 1, 1).contiguous()
    tokens = lang.tokenizer([args.prompt] * args.batch, context_length=ctx).cuda()
    print(
        f"pix {tuple(pix.shape)} {pix.dtype}  tokens {tuple(tokens.shape)} "
        f"{tokens.dtype} {tokens[0].tolist()}"
    )

    g = Graph(model).eval()
    with torch.no_grad():
        logits, boxes, masks, presence = g(pix, tokens)
    print(
        f"ref pass: logits {tuple(logits.shape)} boxes {tuple(boxes.shape)} "
        f"masks {tuple(masks.shape)} presence {tuple(presence.shape)}"
    )
    print(f"logits[0,:5,0] {logits[0, :5, 0].tolist()}")

    np.savez(
        args.ref_npz,
        pixel_values=pix.cpu().numpy(),
        text_tokens=tokens.cpu().numpy().astype(np.int64),
        pred_logits=logits.float().cpu().numpy(),
        pred_boxes=boxes.float().cpu().numpy(),
        pred_masks=masks.float().cpu().numpy(),
        presence=presence.float().cpu().numpy(),
    )
    print(f"saved reference npz: {args.ref_npz}")
    if args.no_export:
        return

    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            g,
            (pix, tokens),
            args.out,
            input_names=["pixel_values", "text_tokens"],
            output_names=["pred_logits", "pred_boxes", "pred_masks", "presence"],
            dynamo=True,
        )
    print(f"onnx export: {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
