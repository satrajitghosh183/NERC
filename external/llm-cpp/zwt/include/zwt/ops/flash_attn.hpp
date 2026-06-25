#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// FlashAttention-2-style tiled attention. The entry-point shape and
// semantics match sdpa/sdpa_backward (see ops/attn.hpp); the difference is
// algorithmic:
//
//   sdpa: materializes the full [B*H, Sq, Sk] scores tensor in HBM.
//         Memory complexity: O(BH * Sq * Sk).
//   flash: tiles Q into blocks Br x D and K/V into blocks Bc x D, runs
//         online softmax with running (m, l) statistics, accumulates into
//         the output tile without ever writing an Sq x Sk matrix to HBM.
//         Memory complexity for intermediates: O(BH * Sq) (logsumexp only).
//
// Forward additionally returns the logsumexp tensor M[B, H, Sq] which the
// backward reuses to avoid recomputing the softmax max and denominator.
//
// CPU path: reference implementation of the tiling + online softmax. This
// is the gold reference that the CUDA kernel (to be written) validates
// against. CPU is slower than plain sdpa on CPU (no memory-bandwidth wall)
// but matches numerically within bf16 tolerances.
//
// CUDA path: drops through to sdpa for now — the FA-2 kernel itself is the
// next artifact and lives in zwt/src/ops/flash_attn.cu.
//
// Shape inputs: q [B,H,Sq,D], k [B,Hkv,Sk,D], v [B,Hkv,Sk,D]. GQA caller
// must pre-expand K/V to H heads (same contract as sdpa).
//
// Output: out [B,H,Sq,D] and lse [B,H,Sq] (fp32). is_causal masks the upper
// triangle. softmax_scale < 0 defaults to 1/sqrt(D).
void flash_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     Tensor& out, Tensor& lse,
                     bool is_causal = true, float softmax_scale = -1.f);

void flash_attention_backward(const Tensor& grad_out,
                              const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& out, const Tensor& lse,
                              Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
                              bool is_causal = true,
                              float softmax_scale = -1.f);

}  // namespace zwt::ops
