#!/usr/bin/env python
"""Regenerate the SAM3 text-token npz for the dynk TRT engine.

Standalone: uses the CLIP BPE tokenizer shipped with sam3, no model load.
Usage: python sam3_make_tokens.py --out tokens.npz "prompt1" "prompt2" ...
"""
import argparse

import numpy as np
import pkg_resources

from sam3.model.tokenizer_ve import SimpleTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out/sam3_tokens.npz")
    ap.add_argument("prompts", nargs="+")
    args = ap.parse_args()

    bpe = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tok = SimpleTokenizer(bpe_path=bpe, context_length=32)
    rows = []
    for p in args.prompts:
        ids = tok(p).squeeze(0).numpy().astype(np.int64)
        assert ids.shape == (32,), (p, ids.shape)
        rows.append(ids)
    arr = np.stack(rows)
    np.savez(args.out, tokens=arr, prompts=np.array(args.prompts))
    print(f"saved {args.out}: tokens {arr.shape} for {args.prompts}")


if __name__ == "__main__":
    main()
