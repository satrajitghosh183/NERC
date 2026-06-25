#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// All ops assume tensors are contiguous and live on the same device.
// Shapes must match. In-place versions update the first argument.

// y = x + bias (broadcast over the last dim or none if shapes match)
void add_bias(Tensor& y, const Tensor& bias);

// grad_bias += sum over all rows of grad_y.
// grad_y is [rows, cols] bf16; grad_bias is [cols] fp32 (master gradient).
void bias_backward(const Tensor& grad_y, Tensor& grad_bias);

// Transpose between head-major [B,H,S,D] and token-major [B,S,H,D] layouts.
// Allocates into `out` which must already be shaped correctly.
void transpose_bshd_to_bhsd(const Tensor& in, Tensor& out);
void transpose_bhsd_to_bshd(const Tensor& in, Tensor& out);

// GQA head replication on a head-major [B, Hkv, S, D] tensor.
//   in:  [B, Hkv, S, D]
//   out: [B, H,   S, D]   where H % Hkv == 0 and group = H / Hkv.
// Each KV head is replicated `group` times, interleaved so head h of the
// output corresponds to kv-head (h / group) of the input. This is the
// forward-pass "broadcast K/V into full head count" op used by an MHA
// attention kernel when the model uses Grouped-Query Attention.
void repeat_kv_heads(const Tensor& in, Tensor& out);

// Backward of repeat_kv_heads: reduce-sum each group of `group` adjacent heads.
//   in:  [B, H,   S, D]
//   out: [B, Hkv, S, D]
// For each (b, kv, s, d), out[b, kv, s, d] = sum over g in [0, group)
// of in[b, kv*group + g, s, d]. The out tensor is overwritten.
void reduce_kv_heads_sum(const Tensor& in, Tensor& out);

// y += x * alpha
void axpy(Tensor& y, const Tensor& x, float alpha);

// y = alpha * y
void scale(Tensor& y, float alpha);

// Fused residual add (out = a + b).
void add(Tensor& out, const Tensor& a, const Tensor& b);

// y = dropout_scale * mask * x (mask stored in-place via 1-bit bools)
// Forward returns mask; backward re-applies it.
void dropout(Tensor& y, const Tensor& x, Tensor& mask_u8, float p, uint64_t seed);

// SiLU gate: out = silu(gate) * up. Fused — two reads, one write per element.
void silu_mul(Tensor& out, const Tensor& gate, const Tensor& up);

// Backward: given grad_out, gate, up produce grad_gate, grad_up.
void silu_mul_backward(const Tensor& grad_out, const Tensor& gate, const Tensor& up,
                       Tensor& grad_gate, Tensor& grad_up);

// Fused gate+up SwiGLU where gate and up are the two halves of a single
// combined output tensor [N, 2H] produced by one GEMM. Per row, the first
// H elements are gate, the next H elements are up.
//   combined: [N, 2H] bf16 (or f32 on CPU test path)
//   out:      [N, H]  bf16
// Per row: out[n, i] = silu(combined[n, i]) * combined[n, H + i].
void silu_mul_gated(Tensor& out, const Tensor& combined);

// Backward of silu_mul_gated. Writes into the two halves of grad_combined.
//   grad_out:      [N, H]  bf16
//   combined:      [N, 2H] bf16 (saved from forward)
//   grad_combined: [N, 2H] bf16 — gate-half and up-half both filled.
void silu_mul_gated_backward(const Tensor& grad_out, const Tensor& combined,
                             Tensor& grad_combined);

}  // namespace zwt::ops
