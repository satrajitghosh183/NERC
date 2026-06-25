#!/usr/bin/env python3
"""PyTorch baseline for zwt — same model, same config, same step count.

Purpose: end-to-end tok/s comparison on the *same* hardware. The architecture
this builds is the OLMo-2 / Llama-3 family (RMSNorm + RoPE + SwiGLU + GQA)
and is parameterized from a zwt .conf INI file so the shape is identical.

Differences from zwt by design:
  * PyTorch eager path, bf16 storage + fp32 master grads via AdamW.
  * F.scaled_dot_product_attention for the attention kernel (uses FA-2 when
    available, same as what the industry calls "tuned PyTorch").
  * Torch's default fused AdamW (foreach=True). No muon/lion variants.

Run:
    python3 zwt/scripts/pt_baseline.py --config conf/owt_1B_h100.conf \
        --steps 50 --warmup 5

Output is one line summarising tok/s averaged over the timed window. Use
that alongside `zwt_pretrain <conf> --dry-run` + a wall-clock measurement
for the three-way benchmark (#11).

This script intentionally has no external deps beyond torch. If torch isn't
importable the script exits with a clear message — PyTorch is not a zwt
build dependency.
"""
from __future__ import annotations

import argparse
import configparser
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    print(f"pt_baseline: PyTorch is not installed ({exc})", file=sys.stderr)
    print("  pip install torch", file=sys.stderr)
    sys.exit(2)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ffn: int
    n_layers: int
    max_seq: int
    rope_base: float
    norm_eps: float
    tie_embeddings: bool


def _parse_int(s: str) -> int:
    s = s.strip()
    return int(s, 0)  # supports 0x... 0o... decimal


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


def load_cfg(path: str) -> tuple[ModelCfg, dict]:
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.read(path)
    m = cp["model"]
    model = ModelCfg(
        vocab_size=_parse_int(m["vocab_size"]),
        d_model=_parse_int(m["d_model"]),
        n_heads=_parse_int(m["n_heads"]),
        n_kv_heads=_parse_int(m.get("n_kv_heads", m["n_heads"])),
        head_dim=_parse_int(m["head_dim"]),
        d_ffn=_parse_int(m["d_ffn"]),
        n_layers=_parse_int(m["n_layers"]),
        max_seq=_parse_int(m["max_seq"]),
        rope_base=float(m.get("rope_base", "10000")),
        norm_eps=float(m.get("norm_eps", "1e-5")),
        tie_embeddings=_parse_bool(m.get("tie_embeddings", "false")),
    )
    d = cp["data"]
    data = dict(
        seq_len=_parse_int(d["seq_len"]),
        batch_size=_parse_int(d["batch_size"]),
        grad_accum=_parse_int(d.get("grad_accum", "1")),
    )
    return model, data


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rstd * self.w


def build_rope(S: int, D: int, base: float, device, dtype=torch.float32):
    half = D // 2
    freqs = base ** (-torch.arange(0, half, device=device, dtype=dtype) * 2 / D)
    t = torch.arange(S, device=device, dtype=dtype)
    angles = torch.outer(t, freqs)                  # [S, half]
    return torch.cos(angles), torch.sin(angles)     # [S, half] each


def apply_rope(x, cos, sin):
    # x: [B, S, H, D] — rotate (lo, hi) halves.
    half = x.shape[-1] // 2
    lo, hi = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(2)             # [1,S,1,half]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.h = cfg.n_heads
        self.hkv = cfg.n_kv_heads
        self.d = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, self.h * self.d, bias=False)
        self.k = nn.Linear(cfg.d_model, self.hkv * self.d, bias=False)
        self.v = nn.Linear(cfg.d_model, self.hkv * self.d, bias=False)
        self.o = nn.Linear(self.h * self.d, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.h,   self.d)
        k = self.k(x).view(B, S, self.hkv, self.d)
        v = self.v(x).view(B, S, self.hkv, self.d)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)  # [B, H,   S, D]
        k = k.transpose(1, 2)  # [B, Hkv, S, D]
        v = v.transpose(1, 2)
        # Replicate k/v out to full head count. enable_gqa arg in newer torch
        # versions lets SDPA consume [B,Hkv,S,D] directly; older ones need the
        # explicit repeat_interleave.
        if self.h != self.hkv:
            g = self.h // self.hkv
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, S, self.h * self.d)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        # Fused gate+up, matches zwt's single-GEMM layout.
        self.gate_up = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.down    = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.at = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mp = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x, cos, sin):
        x = x + self.at(self.n1(x), cos, sin)
        x = x + self.mp(self.n2(x))
        return x


class OLMo2(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids, cos, sin):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        if self.lm_head is None:
            return x @ self.emb.weight.T
        return self.lm_head(x)


# -----------------------------------------------------------------------------
# Bench loop
# -----------------------------------------------------------------------------
def _ddp_init():
    """Initialize torch.distributed if launched under torchrun.

    torchrun sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    If none are present we run single-rank (the original code path).
    Returns (rank, local_rank, world_size, ddp_active).
    """
    import torch.distributed as dist
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, False
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return 0, 0, 1, False
    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps",  type=int, default=50)
    ap.add_argument("--batch",  type=int, default=0, help="override cfg.batch_size")
    ap.add_argument("--seq",    type=int, default=0, help="override cfg.seq_len")
    ap.add_argument("--grad-accum", type=int, default=0,
                    help="override cfg.grad_accum (default: read from .conf)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="grad clipping threshold; pass 0 to disable. Default 1.0 "
                         "matches zwt's default; set 0 if you want pure speed and "
                         "the comparison happens to be the unclipped regime.")
    ap.add_argument("--dtype",  default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("pt_baseline: CUDA not available — numbers would be meaningless",
              file=sys.stderr)
        return 2

    # cuDNN's SDPA backend in recent PyTorch sometimes fails to build a
    # valid execution plan for our GQA + is_causal shape ("No valid
    # execution plans built"). Disable cuDNN SDPA and let torch fall
    # through to Flash / mem-efficient / math. Flash-attn is what tuned-
    # PyTorch users actually run anyway, so this also makes the bench
    # more representative.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    rank, local_rank, world_size, ddp_active = _ddp_init()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    is_rank0 = (rank == 0)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    cfg, data = load_cfg(args.config)
    B  = args.batch      if args.batch      > 0 else data["batch_size"]
    S  = args.seq        if args.seq        > 0 else data["seq_len"]
    GA = args.grad_accum if args.grad_accum > 0 else data.get("grad_accum", 1)

    torch.manual_seed(0xC0DEBA5E + rank)
    model = OLMo2(cfg).to(device=device, dtype=dtype)

    # DDP wrapping happens BEFORE torch.compile so the compile sees the DDP
    # graph. gradient_as_bucket_view=True elides the extra grad copy.
    #
    # NOTE: we deliberately DO NOT pass static_graph=True. PyTorch's DDP
    # `static_graph` mode is incompatible with `model.no_sync()` over a
    # grad_accum window — the first backward inside no_sync() doesn't
    # register the autograd hooks static_graph expects to be invariant,
    # and you get:
    #   RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED at
    #     "torch/csrc/distributed/c10d/reducer.cpp":1705
    # Without static_graph, DDP rebuilds its bucket map per step (small
    # overhead) but `no_sync()` semantics work — that's the fair-comparison
    # contract against zwt's "one allreduce per opt_step."
    if ddp_active:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank],
                    gradient_as_bucket_view=True)

    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                            fused=True)

    cos, sin = build_rope(S, cfg.head_dim, cfg.rope_base, device)
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)

    ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    inv_ga = 1.0 / float(GA)

    # Inner closure: one micro-batch fwd+bwd. `sync_grads` controls whether
    # DDP fires its allreduce on this backward. We pass sync_grads=False
    # for accum micro-batches 0..GA-2 (via model.no_sync()) and True for
    # the last one — matching zwt's design of one allreduce per optimizer
    # step. Without no_sync(), DDP fires N allreduces per step which is a
    # different (and unfair) host-overhead profile.
    def micro_batch(sync_grads: bool):
        if ddp_active and not sync_grads:
            ctx = model.no_sync()
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            logits = model(ids, cos, sin)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                                   tgt.reshape(-1))
            (loss * inv_ga).backward()
        return loss

    def opt_step():
        opt.zero_grad(set_to_none=True)
        for i in range(GA):
            micro_batch(sync_grads=(i == GA - 1))
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    # Warmup.
    for _ in range(args.warmup):
        opt_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # Tokens-per-second across ALL ranks: B * S * GA * world_size * steps / dt.
    # This is the headline benchmark figure; rank 0 is the only one that
    # prints it.
    toks = B * S * GA * world_size * args.steps
    tps  = toks / dt
    if is_rank0:
        print(f"pt_baseline: config={args.config} dtype={args.dtype} "
              f"compile={args.compile} world_size={world_size}")
        print(f"  B={B} S={S} grad_accum={GA} steps={args.steps} "
              f"dt={dt:.3f}s tok/s={tps:,.0f} ms/step={dt*1000/args.steps:.2f}")

    if ddp_active:
        import torch.distributed as dist
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
