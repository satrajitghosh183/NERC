#!/usr/bin/env python3
"""
OLMo-core Python Training Instrumentation

Runs a training loop with the OLMo-core Python model and instruments every phase:
  - Seed initialization (prints and records the seed)
  - Weight initialization timing
  - Per-step breakdown: data_loading, forward, backward, optimizer_step
  - Memory usage tracking
  - RNG state snapshots

Produces a JSON report matching the C++ profiler output format, so you can
directly compare where time is spent in Python vs C++.

Usage:
    # Quick comparison run (matches C++ defaults)
    python3 scripts/instrument_olmo_core.py --seed 42 --steps 50

    # Full instrumentation with specific model
    python3 scripts/instrument_olmo_core.py --seed 42 --model 30M --device mps --steps 100

    # Compare with C++ run:
    ./build/olmo_train --train --seed 42 --profile --steps 50 --device mps
    python3 scripts/instrument_olmo_core.py --seed 42 --steps 50 --device mps
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add olmo-python to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "olmo-python" / "src"))

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)


# ─── Seed Management (mirrors C++ seed_all and OLMo-core's seed_all) ───────

def seed_all(seed: Optional[int] = None) -> int:
    """
    Seed all RNG sources. If seed is None, generate one randomly.
    Prints the seed prominently (matching C++ output format).

    This mirrors:
    - OLMo-core's seed_all() from src/olmo_core/utils.py
    - Our C++ seed_all() from src/seed.cpp
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        explicit = False
    else:
        explicit = True

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  SEED: {seed:<37}║")
    print(f"║  Mode: {'explicit (--seed)':<37}║" if explicit else
          f"║  Mode: {'auto-generated':<37}║")
    print(f"╚══════════════════════════════════════════════╝\n")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


# ─── Profiler (matches C++ ProfileScope output) ────────────────────────────

@dataclass
class TimingStats:
    count: int = 0
    total_us: float = 0.0
    min_us: float = float('inf')
    max_us: float = 0.0
    sum_sq_us: float = 0.0

    @property
    def mean_us(self):
        return self.total_us / self.count if self.count > 0 else 0.0

    @property
    def std_us(self):
        if self.count < 2:
            return 0.0
        mean = self.mean_us
        return (self.sum_sq_us / self.count - mean * mean) ** 0.5

    @property
    def total_ms(self):
        return self.total_us / 1000.0

    @property
    def mean_ms(self):
        return self.mean_us / 1000.0


class Profiler:
    def __init__(self):
        self.stats = defaultdict(TimingStats)
        self._active = {}

    def start(self, name: str):
        self._active[name] = time.perf_counter()

    def stop(self, name: str):
        end = time.perf_counter()
        start = self._active.pop(name, None)
        if start is None:
            return
        us = (end - start) * 1e6
        s = self.stats[name]
        s.count += 1
        s.total_us += us
        s.min_us = min(s.min_us, us)
        s.max_us = max(s.max_us, us)
        s.sum_sq_us += us * us

    def scope(self, name: str):
        return ProfileScope(name, self)

    def report(self, title="Profile Report"):
        if not self.stats:
            print("[Profiler] No data collected.")
            return

        sorted_stats = sorted(self.stats.items(), key=lambda x: -x[1].total_us)
        total_wall_us = sum(s.total_us for _, s in sorted_stats)

        print(f"\n┌─────────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ {title:<78}│")
        print(f"├──────────────────────┬────────┬──────────┬──────────┬──────────┬───────────────┤")
        print(f"│ Region               │ Calls  │ Total ms │ Mean ms  │ Std ms   │ % of Total    │")
        print(f"├──────────────────────┼────────┼──────────┼──────────┼──────────┼───────────────┤")

        for name, s in sorted_stats:
            pct = (s.total_us / total_wall_us * 100) if total_wall_us > 0 else 0
            bar = '#' * int(pct / 5)
            print(f"│ {name[:21]:<21}│ {s.count:>6} │ {s.total_ms:>8.1f} │ "
                  f"{s.mean_ms:>8.2f} │ {s.std_us/1000:>8.2f} │ {pct:>5.1f}% {bar:<7}│")

        print(f"├──────────────────────┴────────┴──────────┴──────────┴──────────┴───────────────┤")
        print(f"│ Total wall time: {total_wall_us/1000:>10.1f} ms{' ':49}│")
        print(f"└─────────────────────────────────────────────────────────────────────────────────┘\n")

    def to_dict(self):
        return {name: {
            "count": s.count,
            "total_ms": s.total_ms,
            "mean_ms": s.mean_ms,
            "std_ms": s.std_us / 1000,
        } for name, s in self.stats.items()}


class ProfileScope:
    def __init__(self, name, profiler):
        self.name = name
        self.profiler = profiler

    def __enter__(self):
        self.profiler.start(self.name)
        return self

    def __exit__(self, *args):
        self.profiler.stop(self.name)


# ─── RNG State Inspector ───────────────────────────────────────────────────

def print_rng_state_summary(seed: int, explicit: bool):
    print(f"[RNG] State summary:")
    print(f"  Original seed: {seed} ({'explicit' if explicit else 'auto'})")
    print(f"  Python random state hash: {hash(str(random.getstate())[:100])}")
    print(f"  NumPy RNG: PCG64, seed={seed}")
    print(f"  torch CPU generator seed: {torch.initial_seed()}")
    if torch.cuda.is_available():
        print(f"  torch CUDA devices: {torch.cuda.device_count()}")
    print()


# ─── Model Configuration ──────────────────────────────────────────────────

MODEL_CONFIGS = {
    "30M": {"d_model": 256, "n_layers": 4, "n_heads": 8, "vocab_size": 50257},
    "100M": {"d_model": 768, "n_layers": 12, "n_heads": 12, "vocab_size": 50257},
    "1B": {"d_model": 2048, "n_layers": 16, "n_heads": 16, "vocab_size": 50257},
}


def build_model(model_name: str, device: torch.device):
    """Build a model matching the C++ configuration."""
    config = MODEL_CONFIGS[model_name]

    try:
        from olmo_core.nn.transformer import TransformerConfig
        # Use OLMo-core's factory if available
        factory_name = f"olmo2_{model_name}"
        if hasattr(TransformerConfig, factory_name):
            tc = getattr(TransformerConfig, factory_name)(vocab_size=config["vocab_size"])
            model = tc.build(init_device="cpu")
            model = model.to(device)
            return model, "olmo_core"
    except (ImportError, AttributeError):
        pass

    # Fallback: simple transformer matching C++ architecture
    print(f"  Using simple PyTorch transformer (OLMo-core not available)")
    from torch import nn

    class SimpleTransformer(nn.Module):
        def __init__(self, d_model, n_layers, n_heads, vocab_size):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.embed_norm = nn.RMSNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=int(d_model * 8 / 3 * 1.5),
                activation='silu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.lm_head_norm = nn.RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.vocab_size = vocab_size

        def forward(self, input_ids, labels=None):
            h = self.embed_norm(self.embed(input_ids))
            mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1),
                                                                   device=input_ids.device)
            h = self.transformer(h, mask=mask, is_causal=True)
            logits = self.lm_head(self.lm_head_norm(h))
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size), labels.view(-1))
                return type('Output', (), {'loss': loss, 'logits': logits})()
            return type('Output', (), {'loss': None, 'logits': logits})()

    model = SimpleTransformer(**config).to(device)
    return model, "simple"


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OLMo-core Python Training Instrumentation")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: auto)")
    parser.add_argument("--model", default="30M", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Seed ──
    actual_seed = seed_all(args.seed)

    # ── Build model ──
    profiler = Profiler()

    with profiler.scope("model_init"):
        model, model_type = build_model(args.model, device)
        model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params/1e6:.1f}M params, backend={model_type})")

    # ── Optimizer ──
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print(f"Optimizer: {args.optimizer} (lr={args.lr})")
    print(f"Training: {args.steps} steps, batch_size={args.batch_size}, seq_len={args.seq_len}")

    # ── Sync function (for accurate GPU timing) ──
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # ── Training Loop ──
    total_tokens = 0
    vocab_size = MODEL_CONFIGS[args.model]["vocab_size"]

    print(f"\nStarting training...\n")

    sync()
    train_start = time.perf_counter()

    for step in range(args.steps):
        with profiler.scope("step_total"):

            # Data loading
            with profiler.scope("data_loading"):
                input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                                          device=device)
                labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                                       device=device)

            # Forward
            with profiler.scope("forward"):
                if model_type == "olmo_core":
                    out = model(input_ids, labels=labels)
                    loss = out.loss
                else:
                    out = model(input_ids, labels=labels)
                    loss = out.loss

            # Backward
            with profiler.scope("backward"):
                loss.backward()

            # Optimizer
            with profiler.scope("optimizer_step"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_tokens += args.batch_size * args.seq_len

            if step % 10 == 0:
                sync()
                elapsed = time.perf_counter() - train_start
                tok_per_s = total_tokens / elapsed if elapsed > 0 else 0
                print(f"Step {step}/{args.steps}  loss: {loss.item():.4f}  "
                      f"tok/s: {int(tok_per_s)}  elapsed: {elapsed:.1f}s")

    sync()
    total_time = time.perf_counter() - train_start

    # ── Summary ──
    print(f"\n=== Training Summary ===")
    print(f"  Steps: {args.steps}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {total_time:.1f}s")
    print(f"  Throughput: {int(total_tokens / total_time)} tok/s")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"========================")

    # ── Profile Report ──
    profiler.report("Python Training Profile")

    # ── RNG State ──
    print_rng_state_summary(actual_seed, args.seed is not None)

    # ── Memory ──
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"[Memory] CUDA peak allocated: {peak_mem:.1f} MB")
    elif device.type == "mps":
        try:
            cur_mem = torch.mps.current_allocated_memory() / 1e6
            print(f"[Memory] MPS current allocated: {cur_mem:.1f} MB")
        except AttributeError:
            print("[Memory] MPS memory tracking not available in this PyTorch version")

    # ── Save JSON report ──
    if args.output:
        report = {
            "framework": f"OLMo-core Python ({model_type})",
            "model": args.model,
            "params": n_params,
            "device": str(device),
            "seed": actual_seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "total_tokens": total_tokens,
            "wall_time_s": total_time,
            "throughput_tok_s": total_tokens / total_time,
            "final_loss": loss.item(),
            "profile": profiler.to_dict(),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
