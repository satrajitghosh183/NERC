#!/usr/bin/env python3
"""scripts/race/python_inference_baseline.py

Minimal PyTorch generation loop for the Python inference race. Loads a
checkpoint, builds a model matching scripts/race/configs/race_250m_python.yaml,
runs greedy generation from a prompt, prints the same `[N tokens, T tok/s]`
trailer the C++ chat tool prints so the analyzer can compare apples-to-apples.

Why "minimal": the user's race compares pure C++ implementation against
the standard PyTorch path. This file IS the standard PyTorch path. No
vllm, no HF transformers, no speculative — straight `model(input_ids)`
in a loop with a KV cache.

If you'd rather measure vllm or HF transformers, swap this out via
PY_INFER_CMD env var in 07_infer_python.sh.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model ──────────────────────────────────────────────────────────────
# Matches race_250m_python.yaml exactly: SwiGLU FFN, RMSNorm, RoPE,
# tied embeddings, no QK-norm, no GQA, no MTP.

@dataclass
class ModelConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    vocab_size: int = 50257
    max_seq_len: int = 1024
    ffn_hidden: int = 2816
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    weight_tying: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, head_dim]. cos/sin: [S, head_dim/2].
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cb = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sb = sin.view(1, 1, sin.shape[0], sin.shape[1])
    return torch.cat([first * cb - second * sb, first * sb + second * cb], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.w_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=None):
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # RoPE cos/sin slice for these positions.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            past_k, past_v = kv_cache[layer_idx]
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            kv_cache[layer_idx] = (k, v)
        # Causal SDPA. PyTorch ≥ 2.0 dispatches to FlashAttention on CUDA.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(S > 1))
        return self.w_o(out.transpose(1, 2).reshape(B, S, -1))


class FFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.w_down    = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gu = self.w_gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn      = Attention(cfg)
        self.ffn_norm  = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.ffn       = FFN(cfg)

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=None):
        h = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, layer_idx)
        return h + self.ffn(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        if cfg.weight_tying:
            self.lm_head = lambda h: F.linear(h, self.embed.weight)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[list] = None,
                pos_start: int = 0):
        x = self.embed(input_ids)
        S = input_ids.shape[1]
        cos, sin = build_rope_cache(
            self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads,
            self.cfg.rope_theta, x.device, x.dtype)
        cos = cos[pos_start : pos_start + S]
        sin = sin[pos_start : pos_start + S]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, kv_cache, i)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── BPE tokenizer (vocab.json + merges.txt) ────────────────────────────
# Minimal GPT-2 BPE encoder. Matches the format the C++ side reads.

def load_bpe(vocab_path: str, merges_path: str):
    encoder = json.load(open(vocab_path))
    decoder = {v: k for k, v in encoder.items()}
    merges = []
    with open(merges_path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if line:
                a, b = line.split(" ", 1)
                merges.append((a, b))
    bpe_ranks = {pair: i for i, pair in enumerate(merges)}
    return encoder, decoder, bpe_ranks


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def encode_text(text: str, encoder, bpe_ranks, b2u):
    out = []
    for token in text.split():
        token = " " + token if out else token
        token = "".join(b2u[b] for b in token.encode("utf-8"))
        # Apply BPE.
        word = tuple(token)
        pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
        while pairs:
            bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
            if bigram not in bpe_ranks:
                break
            a, b = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    new_word.append(a + b); i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = tuple(new_word)
            pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
        for piece in word:
            out.append(encoder[piece])
    return out


def decode_tokens(ids, decoder, u2b):
    text = "".join(decoder.get(i, "") for i in ids)
    return bytes([u2b[c] for c in text if c in u2b]).decode("utf-8", errors="replace")


# ── Generation loop ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-vocab", required=True)
    ap.add_argument("--tokenizer-merges", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    # Build model and load weights.
    cfg = ModelConfig()
    model = Transformer(cfg).to(device).to(torch.bfloat16)
    state = torch.load(args.checkpoint, map_location=device)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model.eval()

    encoder, decoder, bpe_ranks = load_bpe(args.tokenizer_vocab, args.tokenizer_merges)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    prompt_ids = encode_text(args.prompt, encoder, bpe_ranks, b2u)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kv_cache = [(None, None) for _ in range(cfg.n_layers)]

    # Prefill.
    _ = model(input_ids, kv_cache=kv_cache, pos_start=0)
    pos = input_ids.shape[1]

    # Decode loop (greedy).
    out_ids = list(prompt_ids)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        last = input_ids[:, -1:] if input_ids.shape[1] == 1 else None
        # Use last-token-only forward via KV cache.
        last = torch.tensor([[out_ids[-1]]], dtype=torch.long, device=device)
        logits = model(last, kv_cache=kv_cache, pos_start=pos)
        next_id = int(logits[0, -1].argmax().item())
        out_ids.append(next_id)
        pos += 1
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    tok_per_s = args.max_new_tokens / elapsed

    text = decode_tokens(out_ids, decoder, u2b)
    print(text)
    # Trailer format matches the C++ chat tool's so the analyzer can grep both.
    print(f"\n[{args.max_new_tokens} tokens, {tok_per_s:.1f} tok/s]")


if __name__ == "__main__":
    main()
