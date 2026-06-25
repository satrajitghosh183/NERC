#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// Rotary Position Embedding (half-split convention, used by Llama/GPT-NeoX).
//
// Build the cos/sin table once at model init:
//   tab = rope_build_table(max_seq, head_dim, base=10000, device)
// Returns a tensor of shape [max_seq, head_dim] where the first half holds
// cos values and the second half holds sin values. Stored as fp32.
//
// Layout of `table` (one contiguous buffer):
//   [max_seq, head_dim]   where table[s, i]         = cos(s * theta_{i})
//                                table[s, i + D/2]  = sin(s * theta_{i})
// theta_i = base^(-2i/D) for i in [0, D/2).
Tensor rope_build_table(int64_t max_seq, int64_t head_dim, float base,
                        Device device);

// Apply RoPE in-place to x[:, :, :, :].
//   x:     [B, S, H, D] bf16 (or f32 on CPU test path)
//   table: [max_seq, D] fp32; only rows [0, S) are read.
void rope_apply(Tensor& x, const Tensor& table);

// Backward: rotate grad_x by -theta in-place (transpose of forward rotation).
void rope_apply_backward(Tensor& grad_x, const Tensor& table);

}  // namespace zwt::ops
