#!/usr/bin/env python3
"""zwt eval harness — perplexity + hook into lm-eval-harness.

The purpose is to turn a zwt training run into an evaluatable artifact
without re-implementing the evaluation stack ourselves. After
`zwt_export_hf` produces a Llama-layout safetensors checkpoint, this
script:

  1. Loads it as a HuggingFace CausalLM.
  2. Computes mean negative-log-likelihood / perplexity on a tokenized
     text file (one document per line, or a single blob).
  3. Prints the lm-evaluation-harness command line to copy-paste for
     zero-shot downstream evals (HellaSwag, PIQA, ARC-e/c, Winogrande,
     SciQ, OBQA, BoolQ, LAMBADA — the OLMo-2 standard set).

We stop short of invoking lm-eval-harness ourselves because its dep
surface is large (datasets, accelerate, numerous task YAMLs) and every
user has a different pinned version. The printed command is the
contract; zwt's job is just to hand a loadable checkpoint.

Run:
    python3 zwt/scripts/eval_harness.py \
        --ckpt-dir /path/to/exported_hf/ \
        --eval-tokens data/eval_tokens.bin \
        --batch 8 --seq 2048
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    print(f"eval_harness: requires torch + transformers ({exc})", file=sys.stderr)
    print("  pip install torch transformers safetensors", file=sys.stderr)
    sys.exit(2)


def load_tokens(path: str) -> torch.Tensor:
    """Load tokens. Supports .npy (int32 or int64) and .bin (raw int32).

    A .bin file is a flat stream of int32 ids — the format zwt_pretrain
    consumes. .npy is the typical prepare_data output.
    """
    p = Path(path)
    if p.suffix == ".npy":
        import numpy as np
        arr = np.load(str(p), mmap_mode="r")
        return torch.from_numpy(arr.astype("int64", copy=False))
    # Raw int32 bin.
    size = p.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"{path}: not a multiple of 4 bytes")
    import numpy as np
    arr = np.memmap(str(p), dtype=np.int32, mode="r")
    return torch.from_numpy(np.asarray(arr, dtype=np.int64))


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seq: int, batch: int, device) -> dict:
    """Sliding-window perplexity over a flat token stream.

    Chunks ids into non-overlapping [batch, seq] windows and averages the
    per-token cross-entropy. First token of each window has no label, so
    we shift labels by one like a standard LM eval.
    """
    total_tokens = ids.numel()
    window = batch * seq
    n_windows = total_tokens // window
    if n_windows == 0:
        raise ValueError(f"eval tokens ({total_tokens}) < window ({window})")

    nll_sum = 0.0
    tok_count = 0
    t0 = time.perf_counter()
    for i in range(n_windows):
        chunk = ids[i * window : (i + 1) * window].reshape(batch, seq).to(device)
        logits = model(chunk).logits.float()           # [B, S, V]
        # Shift: predict token t+1 from the logits at position t.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_sum   += loss.item()
        tok_count += shift_labels.numel()
    dt = time.perf_counter() - t0
    mean_nll = nll_sum / tok_count
    return dict(
        nll=mean_nll,
        perplexity=math.exp(mean_nll),
        bits_per_byte=mean_nll / math.log(2),  # with a BPE tokenizer this is
                                               # only approximate unless you
                                               # know bytes/token; still a
                                               # useful relative signal.
        tokens=tok_count,
        windows=n_windows,
        seconds=dt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="directory produced by zwt_export_hf")
    ap.add_argument("--eval-tokens", required=True,
                    help=".npy or .bin tokens from prepare_data")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq",   type=int, default=2048)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    print(f"[load] {args.ckpt_dir}  dtype={args.dtype}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_dir, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    # Tokenizer isn't required for perplexity over raw token ids, but loading
    # it verifies the exported tokenizer_config.json is compatible.
    try:
        AutoTokenizer.from_pretrained(args.ckpt_dir, local_files_only=True)
    except Exception as exc:
        print(f"[warn] tokenizer load failed: {exc}")

    print(f"[load-tokens] {args.eval_tokens}")
    ids = load_tokens(args.eval_tokens)
    print(f"  total tokens = {ids.numel():,}")

    r = perplexity(model, ids, args.seq, args.batch, device)
    print("---")
    print(f"nll          = {r['nll']:.4f}")
    print(f"perplexity   = {r['perplexity']:.3f}")
    print(f"bits/byte    ≈ {r['bits_per_byte']:.4f}  (tokenizer-dependent)")
    print(f"tokens eval'd= {r['tokens']:,}  over {r['windows']} windows")
    print(f"eval walltime= {r['seconds']:.1f}s")

    print("\n--- zero-shot (lm-evaluation-harness) ---")
    tasks = "hellaswag,piqa,arc_easy,arc_challenge,winogrande,sciq,openbookqa,boolq,lambada_openai"
    print("lm-eval \\")
    print(f"    --model hf \\")
    print(f"    --model_args pretrained={args.ckpt_dir},dtype={args.dtype} \\")
    print(f"    --tasks {tasks} \\")
    print( "    --batch_size 8 \\")
    print( "    --num_fewshot 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
