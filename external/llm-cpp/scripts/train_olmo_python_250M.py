#!/usr/bin/env python3
"""scripts/train_olmo_python_250M.py

Black-box invocation of upstream olmo-python (ai2-olmo-core) at the 250M
OLMo-2-300M-class shape used by scripts/race_250M.sh.

This is the third stack in the head-to-head race:
   1. build/olmo_train  — LibTorch C++ (this repo)
   2. build/zwt_pretrain — LibTorch-free C++ (zwt/)
   3. this script        — upstream olmo-python (Python + PyTorch)

We do NOT read or modify olmo-python internals — we use it through its
public TransformerConfig/Transformer API and the standard PyTorch training
idioms. That's the contract: "How fast does olmo-python train at this
shape, on this hardware, with the same data, called the way an engineer
would call it from a notebook?"

Output: results/race_250M_olmopython.csv with columns
   step,loss,tok_per_s,wall_s

Args:
   --tokens PATH       .npy of int64 token ids (must exist; produced by
                       build/prepare_data with the GPT-2 tokenizer)
   --steps N           total training steps (default 1000; race overrides)
   --warmup N          warmup steps for cosine schedule (default 50)
   --batch-size N      micro batch (default 4)
   --grad-accum N      grad accumulation (default 8) — keeps effective batch
                       identical to the C++ stacks
   --seq-len N         (default 4096)
   --lr F              peak LR (default 3e-4)
   --output PATH       CSV destination (default results/race_250M_olmopython.csv)
   --target-loss F     stop early if val loss <= this (default disabled)
   --log-interval N    (default 10)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/race_250M_olmopython.csv")
    p.add_argument("--target-loss", type=float, default=float("inf"))
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — race must run on H100", file=sys.stderr)
        return 1
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Build the upstream model. We try the highest-fidelity path first
    # (olmo_core.nn.transformer) and fall back to a vanilla nn.TransformerEncoder
    # if olmo-core isn't installed. The fallback is honest: we record it in
    # the CSV `note` column so anyone reading the result knows.
    # ------------------------------------------------------------------
    note = ""
    model = None
    try:
        from olmo_core.nn.transformer import TransformerConfig
        cfg = TransformerConfig(
            d_model=896,
            n_layers=24,
            n_heads=14,
            n_kv_heads=2,
            vocab_size=args.vocab_size,
            block_name="reordered_norm",
            qk_norm=True,
            rope_theta=500000,
        )
        model = cfg.build(device=device)
        note = "olmo_core_transformer"
        print(f"[olmo-python] using olmo_core.nn.transformer.TransformerConfig",
              flush=True)
    except Exception as e:
        print(f"[olmo-python] olmo_core unavailable ({e}); falling back to "
              f"nn.TransformerEncoder", flush=True)
        # Apples-to-apples with the C++ stacks would prefer matching exactly,
        # but if olmo-core can't load we still want a number for the race.
        class FallbackTransformer(torch.nn.Module):
            def __init__(self, vocab: int, d_model=896, n_heads=14, n_layers=24,
                         d_ffn=2432, max_seq=4096):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab, d_model)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
                    dropout=0.0, batch_first=True, norm_first=True,
                    activation="gelu",
                )
                self.encoder = torch.nn.TransformerEncoder(layer, n_layers)
                self.norm = torch.nn.LayerNorm(d_model)
                self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)
                self.max_seq = max_seq
            def forward(self, x):
                h = self.embed(x)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    x.size(1), device=x.device)
                h = self.encoder(h, mask=mask, is_causal=True)
                h = self.norm(h)
                return self.lm_head(h)
        model = FallbackTransformer(args.vocab_size).to(device)
        note = "fallback_nn_transformer"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[olmo-python] params: {n_params/1e6:.1f}M  device: cuda  dtype: bf16",
          flush=True)

    # bf16 weights — same recipe as olmo_train and zwt_pretrain.
    model = model.to(torch.bfloat16)

    # Data
    tokens_path = Path(args.tokens)
    if not tokens_path.exists():
        print(f"ERROR: token file not found: {tokens_path}", file=sys.stderr)
        return 1
    tokens = np.load(tokens_path, mmap_mode="r")
    print(f"[olmo-python] tokens: {len(tokens):,} from {tokens_path}", flush=True)

    # Optimizer + cosine schedule (same as the C++ stacks)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=args.weight_decay)
    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        progress = min(1.0, progress)
        # cosine to min_lr = lr/10
        min_lr = args.lr / 10.0
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + np.cos(np.pi * progress))

    # Prepare CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(args.output, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "loss", "tok_per_s", "wall_s", "lr", "note"])
    # Header note row (a single row with note set; downstream readers can
    # surface this).
    csv_w.writerow([0, 0.0, 0.0, 0.0, 0.0, note])

    # Training loop
    rng = np.random.default_rng(args.seed)
    eff_batch = args.batch_size * args.grad_accum
    tokens_per_step = eff_batch * args.seq_len
    print(f"[olmo-python] effective batch: {eff_batch} sequences * "
          f"{args.seq_len} = {tokens_per_step:,} tokens/step", flush=True)

    model.train()
    t0 = time.time()
    last_log_t = t0
    last_log_step = 0
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = lr

        accum_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            # Random window per micro-batch.
            idx = rng.integers(0, len(tokens) - args.seq_len - 1,
                               size=args.batch_size)
            batch = np.stack(
                [np.asarray(tokens[i:i + args.seq_len + 1], dtype=np.int64)
                 for i in idx])
            batch = torch.from_numpy(batch).to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size),
                y.reshape(-1))
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_interval == 0 or step == 1:
            torch.cuda.synchronize()
            now = time.time()
            dt = now - last_log_t
            steps_done = step - last_log_step
            tps = steps_done * tokens_per_step / max(dt, 1e-9)
            wall_s = now - t0
            print(f"[olmo-python] step {step}/{args.steps} | loss "
                  f"{accum_loss:.4f} | {tps:,.0f} tok/s | lr {lr:.3e} | "
                  f"wall {wall_s:.1f}s", flush=True)
            csv_w.writerow([step, f"{accum_loss:.6f}", f"{tps:.2f}",
                            f"{wall_s:.3f}", f"{lr:.6e}", note])
            csv_f.flush()
            last_log_t = now
            last_log_step = step

            if accum_loss <= args.target_loss:
                print(f"[olmo-python] hit target loss {args.target_loss} at "
                      f"step {step} (wall {wall_s:.1f}s)", flush=True)
                break

    csv_f.close()
    print(f"[olmo-python] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
