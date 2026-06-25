#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// Scaled dot-product attention. This is a thin dispatch:
//
//   * On CUDA with CUDNN_FRONTEND / FlashAttention available, calls into
//     that. (We build on cuDNN's fused attention operator for portability;
//     teams that want FA2/FA3 can swap this impl behind the same signature.)
//   * On CUDA without cuDNN fused attention, a hand-written tiled kernel
//     (rope_mha.cu) at sm_80+ — bf16 inputs, fp32 accumulate, causal mask.
//   * On CPU, an f32 reference — test-only.
//
// Input layout: [B, H, S, D] for Q and [B, Hkv, S, D] for K, V (GQA ok).
// Output: [B, H, S, D].
//
// If `is_causal` is true, the upper triangle is masked.
// If `softmax_scale` is negative, it defaults to 1/sqrt(D).
void sdpa(const Tensor& q, const Tensor& k, const Tensor& v,
          Tensor& out, bool is_causal = true, float softmax_scale = -1.f);

void sdpa_backward(const Tensor& grad_out,
                   const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& out,
                   Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
                   bool is_causal = true, float softmax_scale = -1.f);

}  // namespace zwt::ops
