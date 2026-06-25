#!/usr/bin/env python3
"""olmo-python bench leg — invoke AI2's actual OLMo-core Transformer.

This is the head-to-head leg: we don't reimplement OLMo-core's model in
PyTorch like pt_baseline.py does. We instantiate the SAME class their
own training scripts use (`olmo_core.nn.transformer.TransformerConfig
.olmo2_1B_v2`), wrap it in plain DDP, and time fwd+bwd+AdamW on synthetic
data for N steps. tok/s is what gets printed.

Why this is fair vs pt_baseline.py:
 * pt_baseline.py is a from-scratch reimplementation of the OLMo-2 / Llama-3
   architecture in PyTorch — useful for "tuned eager" and "torch.compile"
   comparisons but not "AI2's actual production stack."
 * This script uses olmo_core's Transformer module, their attention backends,
   their fused kernels — whatever ships with olmo-python is what runs.

What we control to keep the comparison apples-to-apples with zwt:
 * seq_len, micro-batch, grad_accum, dtype, world_size — read from the same
   .conf file zwt does.
 * AdamW with matching betas/eps/weight_decay.
 * Plain DDP (NCCL), not FSDP/HSDP — matches zwt's data-parallel design.
 * model.no_sync() across grad_accum micro-batches — matches zwt's
   "allreduce once per optimizer step" contract.

Run via torchrun for multi-GPU, or directly for single-GPU:

  python3 zwt/scripts/run_olmo_python.py --config zwt/conf/owt_1B_2xh100.conf \
      --warmup 5 --steps 50

  torchrun --nproc-per-node=2 zwt/scripts/run_olmo_python.py \
      --config zwt/conf/owt_1B_2xh100.conf --warmup 5 --steps 50

Output line (parsed by bench_threeway.sh):

  olmo_python: ...
    B=2 S=2048 grad_accum=16 steps=50 dt=12.345s tok/s=53,123 ms/step=246.90
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
import time

try:
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError as exc:
    print(f"olmo_python: torch import failed ({exc})", file=sys.stderr)
    sys.exit(2)

try:
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.config import DType
except ImportError as exc:
    print(f"olmo_python: import olmo_core failed ({exc})", file=sys.stderr)
    print("  pip install -e ./olmo-python", file=sys.stderr)
    sys.exit(2)


def _parse_int(s):
    return int(s.strip(), 0)


def load_zwt_conf(path):
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.read(path)
    m = cp["model"]
    d = cp["data"]
    return dict(
        vocab_size=_parse_int(m["vocab_size"]),
        seq_len   =_parse_int(d["seq_len"]),
        batch_size=_parse_int(d["batch_size"]),
        grad_accum=_parse_int(d.get("grad_accum", "1")),
    )


def init_dist():
    """Read torchrun env vars; init NCCL group when world_size > 1."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, False
    world = int(os.environ["WORLD_SIZE"])
    if world <= 1:
        return 0, 0, 1, False
    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="zwt .conf — only B/S/grad_accum/vocab are read")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps",  type=int, default=50)
    ap.add_argument("--batch",  type=int, default=0, help="override conf batch_size")
    ap.add_argument("--seq",    type=int, default=0, help="override conf seq_len")
    ap.add_argument("--grad-accum", type=int, default=0, help="override conf grad_accum")
    ap.add_argument("--dtype",  default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("olmo_python: CUDA not available", file=sys.stderr)
        return 2

    # Pre-empt cuDNN SDPA "No valid execution plans built" — same issue
    # that bites pt_baseline.py on GQA + is_causal. Let torch fall through
    # to Flash / mem-efficient.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    rank, local_rank, world, ddp_active = init_dist()
    is_rank0 = (rank == 0)
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dtype  = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

    cfg = load_zwt_conf(args.config)
    B  = args.batch      if args.batch      > 0 else cfg["batch_size"]
    S  = args.seq        if args.seq        > 0 else cfg["seq_len"]
    GA = args.grad_accum if args.grad_accum > 0 else cfg["grad_accum"]
    V  = cfg["vocab_size"]

    # Build the actual AI2 model. olmo2_1B_v2 is what their OLMo2-1B.py uses
    # in production. Other knobs (rope_theta=500_000, qk_norm=True,
    # hidden_size_multiplier=1.5, n_layers=16) are inherited from their
    # default — we don't override them because the bench question is "their
    # 1B vs ours, head to head."
    torch.manual_seed(0xC0DEBA5E + rank)
    transformer_cfg = TransformerConfig.olmo2_1B_v2(vocab_size=V, dtype=DType.bfloat16)
    model = transformer_cfg.build()
    model = model.to(device=device, dtype=dtype)
    # OLMo-core's transformer initializes its own weights via init_weights;
    # build() handles it. If running into init issues, call
    # model.init_weights(...) here.

    if ddp_active:
        model = DDP(model, device_ids=[local_rank],
                    static_graph=True, gradient_as_bucket_view=True)

    # Match zwt's AdamW hyperparameters from the .conf — we want to compare
    # framework overhead, not optimizer choice.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                            fused=True)

    # Synthetic ids — same shape every step, regenerated to keep cache
    # behavior representative without actually touching disk.
    ids = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    tgt = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

    inv_ga = 1.0 / float(GA)

    def micro_batch(sync_grads: bool):
        if ddp_active and not sync_grads:
            ctx = model.no_sync()
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            # OLMo-core's Transformer.forward signature: forward(input_ids, ...)
            # Returns logits [B, S, V] in most configurations.
            logits = model(input_ids=ids)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits)
            loss = F.cross_entropy(
                logits.reshape(-1, V).float(),
                tgt.reshape(-1),
            )
            (loss * inv_ga).backward()

    def opt_step():
        opt.zero_grad(set_to_none=True)
        for i in range(GA):
            micro_batch(sync_grads=(i == GA - 1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(args.warmup):
        opt_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    toks = B * S * GA * world * args.steps
    tps  = toks / dt
    if is_rank0:
        print(f"olmo_python: config={args.config} dtype={args.dtype} "
              f"world_size={world}")
        print(f"  B={B} S={S} grad_accum={GA} steps={args.steps} "
              f"dt={dt:.3f}s tok/s={tps:,.0f} ms/step={dt*1000/args.steps:.2f}")

    if ddp_active:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
