#!/usr/bin/env python3
"""
Python PyTorch benchmark — 125M transformer, matches C++ benchmark_125M.conf exactly.

Architecture: d_model=768, n_layers=12, n_heads=12, vocab_size=50257
Training:     batch=8, seq_len=512, lr=3e-4, AdamW, BF16 autocast, 100 steps

Usage:
    python scripts/py_benchmark_125M.py --data-path data/tokens.npy --steps 100
    python scripts/py_benchmark_125M.py --steps 100  # random data
"""

import argparse
import math
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Model: matches C++ FusedTransformer with fused QKV, RMSNorm, SwiGLU
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=500000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, seq_len):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class FusedAttention(nn.Module):
    """Single QKV projection + RoPE + SDPA — matches C++ FusedAttention."""
    def __init__(self, d_model, n_heads, head_dim, rope):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope = rope
        # Fused QKV: one GEMM instead of 3
        total = n_heads * head_dim * 3  # Q + K + V
        self.w_qkv = nn.Linear(d_model, total, bias=False)
        self.w_out = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_size = n_heads * head_dim
        self.kv_size = n_heads * head_dim

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.w_qkv(x)
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:self.q_size + self.kv_size]
        v = qkv[:, :, self.q_size + self.kv_size:]

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.w_out(out)


class SwiGLUFFN(nn.Module):
    """Fused gate+up SwiGLU FFN — matches C++ FeedForward."""
    def __init__(self, d_model, hidden_mult=1.5):
        super().__init__()
        # Match OLMo2 hidden size: round to multiple of 256
        hidden = int(d_model * hidden_mult * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        # Fused gate_up: one GEMM for both gate and up projections
        self.w_gate_up = nn.Linear(d_model, hidden * 2, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gu = self.w_gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, rope):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FusedAttention(d_model, n_heads, head_dim, rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, head_dim, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        rope = RotaryEmbedding(head_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, head_dim, rope)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class NumpyTokenDataset:
    def __init__(self, path, seq_len, device):
        if not HAS_NUMPY:
            raise RuntimeError("numpy is required for --data-path. Install with: pip install numpy")
        arr = np.load(path)
        self.tokens = torch.from_numpy(arr.astype(np.int64)).to(device)
        self.seq_len = seq_len
        self.num_chunks = (len(self.tokens) - 1) // seq_len
        self.indices = torch.randperm(self.num_chunks, device=device)
        self.cursor = 0

    def get_batch(self, batch_size):
        if self.cursor + batch_size > self.num_chunks:
            self.indices = torch.randperm(self.num_chunks, device=self.tokens.device)
            self.cursor = 0
        offsets = self.indices[self.cursor:self.cursor + batch_size] * self.seq_len
        self.cursor += batch_size
        # Pure GPU gather — zero CPU involvement
        rng = torch.arange(self.seq_len, device=self.tokens.device)
        input_idx = offsets.unsqueeze(1) + rng.unsqueeze(0)
        label_idx = input_idx + 1
        inputs = self.tokens[input_idx.reshape(-1)].reshape(batch_size, self.seq_len)
        labels = self.tokens[label_idx.reshape(-1)].reshape(batch_size, self.seq_len)
        return inputs, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Architecture — exact match to C++ benchmark_125M.conf
    VOCAB_SIZE = 50257
    D_MODEL = 768
    N_LAYERS = 12
    N_HEADS = 12
    HEAD_DIM = D_MODEL // N_HEADS  # 64
    SEQ_LEN = args.seq_len
    BATCH_SIZE = args.batch_size

    print(f"=== Python PyTorch Benchmark ===")
    print(f"Model: {D_MODEL}d, {N_LAYERS}L, {N_HEADS}H, vocab={VOCAB_SIZE}")
    print(f"Training: batch={BATCH_SIZE}, seq_len={SEQ_LEN}, steps={args.steps}")
    print(f"Optimizer: AdamW (lr={args.lr}, wd=0.01)")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    model = Transformer(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, max_seq_len=SEQ_LEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    dataset = None
    if args.data_path:
        dataset = NumpyTokenDataset(args.data_path, SEQ_LEN, device)
        print(f"Dataset: {args.data_path} ({dataset.num_chunks} chunks, GPU-resident)")
    else:
        print("Dataset: random (no file specified)")
    print()

    # Warmup
    model.train()
    for _ in range(3):
        if dataset:
            inp, lab = dataset.get_batch(BATCH_SIZE)
        else:
            inp = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
            lab = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(inp, lab)
        loss.backward()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    print("Warmup complete, starting timed run...")

    # Timed training
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0

    for step in range(args.steps):
        step_t0 = time.perf_counter()

        # LR schedule: cosine warmup
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()

        if dataset:
            inp, lab = dataset.get_batch(BATCH_SIZE)
        else:
            inp = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
            lab = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(inp, lab)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += BATCH_SIZE * SEQ_LEN

        if step % 10 == 0:
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_t0) * 1000
            elapsed = time.perf_counter() - t0
            tok_s = total_tokens / elapsed if elapsed > 0 else 0
            print(f"Step {step}/{args.steps}  loss: {loss.item():.4f}  lr: {lr:.6f}  step_ms: {step_ms:.0f}  tok/s: {tok_s:.0f}")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    avg_step_ms = total_time / args.steps * 1000

    print()
    print(f"=== Python Training Summary ===")
    print(f"  Steps: {args.steps}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {total_time:.2f}s")
    print(f"  Avg step: {avg_step_ms:.1f}ms")
    print(f"  Throughput: {total_tokens / total_time:.0f} tok/s")
    print(f"===============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python PyTorch 125M benchmark")
    parser.add_argument("--data-path", type=str, default=None, help="Path to .npy token file")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()
    train(args)
