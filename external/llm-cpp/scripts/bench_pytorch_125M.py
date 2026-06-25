#!/usr/bin/env python3
"""scripts/bench_pytorch_125M.py

Vanilla PyTorch baseline for 125M-class decode on Mac CPU/MPS. Apples-to-
apples comparison against our C++ build/bench_chat:
   * same model shape (d_model=768, n_layers=12, n_heads=12, vocab=50257)
   * same prompt_len + decode_len
   * same hardware (Mac CPU or MPS)
   * same dtype (fp32 by default; bf16/fp16 via --dtype)

Uses ai2-olmo-core's Transformer when available (already pip-installed in
.bench_venv). Falls back to a plain nn.TransformerEncoder if the import
fails. The fallback is honest: prints "FALLBACK" so the comparison
disclaimer is loud.

Output: JSON to --output with TTFT / TPOT / tok-per-s matching
bench_chat.cpp's schema, so scripts/bench_compare.sh's diff logic can
read it the same way.

Usage:
   .bench_venv/bin/python scripts/bench_pytorch_125M.py \\
       --d-model 768 --n-layers 12 --n-heads 12 --vocab 50257 \\
       --device mps --dtype fp32 \\
       --prompt-len 128 --decode-len 256 --warmup 3 --iters 5 \\
       --output results/bench_pytorch_mps.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def select_device(pref: str) -> torch.device:
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_dtype(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16,
            "fp16": torch.float16}[s]


def device_sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def build_model(args, device: torch.device, dtype: torch.dtype):
    """Try olmo-core; fall back to nn.TransformerEncoder."""
    note = ""
    try:
        from olmo_core.nn.transformer import TransformerConfig
        cfg = TransformerConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            vocab_size=args.vocab,
            qk_norm=True,
            rope_theta=500000,
        )
        model = cfg.build(device=device)
        note = "olmo_core_transformer"
        print(f"[pytorch] using olmo_core TransformerConfig", flush=True)
    except Exception as e:
        print(f"[pytorch] olmo_core unavailable ({type(e).__name__}: {e})",
              flush=True)
        print(f"[pytorch] FALLBACK: nn.TransformerEncoder", flush=True)
        d_ffn = args.d_model * 4
        # SiLU-ish via GELU; not exact OLMo-2 SwiGLU but close enough for
        # tok/s comparison.
        layer = torch.nn.TransformerEncoderLayer(
            d_model=args.d_model, nhead=args.n_heads, dim_feedforward=d_ffn,
            dropout=0.0, batch_first=True, norm_first=True,
            activation="gelu",
        )
        backbone = torch.nn.TransformerEncoder(layer, args.n_layers)
        class Fallback(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(args.vocab, args.d_model)
                self.backbone = backbone
                self.norm = torch.nn.LayerNorm(args.d_model)
                self.lm_head = torch.nn.Linear(args.d_model, args.vocab, bias=False)
            def forward(self, x):
                h = self.embed(x)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    x.size(1), device=x.device)
                h = self.backbone(h, mask=mask, is_causal=True)
                h = self.norm(h)
                return self.lm_head(h)
        model = Fallback().to(device)
        note = "fallback_nn_transformer"
    model = model.to(dtype)
    return model, note


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--decode-len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="")
    args = p.parse_args()

    if args.device == "auto":
        args.device = ("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = select_device(args.device)
    dtype  = select_dtype(args.dtype)
    torch.manual_seed(args.seed)

    print(f"[pytorch] device={device} dtype={dtype} d_model={args.d_model} "
          f"n_layers={args.n_layers} n_heads={args.n_heads}", flush=True)

    model, note = build_model(args, device, dtype)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[pytorch] params: {n_params/1e6:.1f}M", flush=True)
    model.eval()

    # Same deterministic prompt pattern bench_chat uses: token 0 everywhere
    # except first slot. Greedy decode (argmax) — no RNG variance.
    prompt = torch.zeros((args.batch, args.prompt_len), dtype=torch.int64,
                         device=device)
    prompt[:, 0] = 50256  # GPT-2 EOS

    ttft_ms_all = []
    tpot_ms_all = []
    step_ms_all = []

    def run_once(measure: bool):
        device_sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(prompt)
            next_tok = logits[:, -1, :].argmax(-1)
            device_sync(device)
            t_prefill = time.perf_counter()
            step_ms = []
            ctx = prompt
            for _ in range(args.decode_len):
                ctx = torch.cat([ctx, next_tok.unsqueeze(-1)], dim=1)
                s0 = time.perf_counter()
                logits = model(ctx)
                next_tok = logits[:, -1, :].argmax(-1)
                device_sync(device)
                s1 = time.perf_counter()
                step_ms.append((s1 - s0) * 1000.0)
            t_end = time.perf_counter()
        if measure:
            ttft_ms_all.append((t_prefill - t0) * 1000.0)
            total_decode_ms = (t_end - t_prefill) * 1000.0
            tpot_ms_all.append(total_decode_ms / args.decode_len)
            step_ms_all.extend(step_ms)

    for _ in range(args.warmup): run_once(False)
    for _ in range(args.iters):  run_once(True)

    def p(xs, q): return statistics.quantiles(xs, n=100)[q-1] if len(xs) > 1 else xs[0]
    ttft_mean = statistics.mean(ttft_ms_all)
    tpot_mean = statistics.mean(tpot_ms_all)
    tpot_p50  = statistics.median(step_ms_all)
    tpot_p99  = p(step_ms_all, 99) if len(step_ms_all) >= 100 else max(step_ms_all)
    tok_s     = 1000.0 * args.batch / tpot_mean

    print(f"\n[pytorch] device={device} batch={args.batch} "
          f"prompt_len={args.prompt_len} decode_len={args.decode_len}", flush=True)
    print(f"TTFT mean: {ttft_mean:.3f} ms", flush=True)
    print(f"TPOT mean: {tpot_mean:.3f} ms (p50={tpot_p50:.3f}, p99={tpot_p99:.3f})",
          flush=True)
    print(f"Throughput: {tok_s:,.1f} tok/s", flush=True)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "device": args.device, "dtype": args.dtype, "note": note,
            "batch": args.batch,
            "prompt_len": args.prompt_len, "decode_len": args.decode_len,
            "warmup": args.warmup, "iters": args.iters,
            "ttft_ms_mean": ttft_mean,
            "tpot_ms_mean": tpot_mean,
            "tpot_ms_p50": tpot_p50,
            "tpot_ms_p99": tpot_p99,
            "throughput_tok_per_s": tok_s,
            "params_M": n_params / 1e6,
        }, open(args.output, "w"), indent=2)
        print(f"[pytorch] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
