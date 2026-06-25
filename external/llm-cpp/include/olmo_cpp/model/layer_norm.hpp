#pragma once
/**
 * include/olmo_cpp/model/layer_norm.hpp
 *
 * Declares the two normalisation modules used inside transformer
 * blocks:
 *
 *   - RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
 *     (no mean-subtraction; just rescales every row to unit RMS,
 *     then a learned per-channel gain.) THE default in OLMo, LLaMA,
 *     Gemma. See kernels/rms_norm.cu for a much longer pedagogical
 *     description.
 *
 *   - LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 *     (the original; subtracts mean, more parameters, slightly slower).
 *
 * RMSNorm exposes a `forward_add` variant that fuses the preceding
 * residual add with the norm — see kernels/rms_norm.cu's residual
 * variant.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/layer_norm.cpp : implementation.
 *   - src/model/block.cpp / fused_block.cpp / block_variants.cpp :
 *     RMSNorm appears inside every transformer block (pre-attn norm,
 *     pre-FFN norm, optional QK-norm).
 *
 * --- Role in training pipeline ---
 *   Foundational. Each forward pass invokes RMSNorm 2N+ times
 *   (N=number of layers).
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
/// OLMo uses elementwise_affine (weight), no bias
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);

  torch::Tensor forward(torch::Tensor x);

  /// Fused: returns residual + rms_norm(x) * weight
  torch::Tensor forward_add(torch::Tensor x, torch::Tensor residual);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
};

TORCH_MODULE(RMSNorm);

}  // namespace olmo_cpp
