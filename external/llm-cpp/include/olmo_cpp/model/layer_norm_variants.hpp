#pragma once
/**
 * include/olmo_cpp/model/layer_norm_variants.hpp
 *
 * Variants of the normalisation layer beyond the canonical RMSNorm
 * declared in layer_norm.hpp:
 *
 *   - **L2Norm**: divides each row by its L2 norm (no eps), then
 *     applies a learned gain. Equivalent to projecting hidden states
 *     onto the unit sphere — used in nGPT-style configs.
 *
 *   - **FusedRMSNorm**: declarative wrapper that always dispatches
 *     to the fused kernel via get_backend().rms_norm(...) instead of
 *     ATen's at::rms_norm; useful when you want to force the fused
 *     path even on tiny tensors.
 *
 *   - **LayerNormImpl**: the original BERT/GPT-2 LayerNorm with
 *     mean subtraction. Off the hot path in OLMo/LLaMA-style models
 *     but kept for HF checkpoint compatibility.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/layer_norm_variants.cpp : implementations.
 *   - src/model/block_variants.cpp : selects the variant based on
 *     cfg.layer_norm_type.
 *
 * --- Role in training pipeline ---
 *   Selected per-config; not all are on the hot path of every run.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
class LayerNormImpl : public torch::nn::Module {
 public:
  LayerNormImpl(int64_t size, double eps = 1e-5, bool elementwise_affine = true, bool bias = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_, bias_;
  double eps_;
  bool elementwise_affine_, has_bias_;
  int64_t size_;
};
TORCH_MODULE(LayerNorm);

/// L2Norm: y = x / ||x||_2 * weight
class L2NormImpl : public torch::nn::Module {
 public:
  L2NormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
};
TORCH_MODULE(L2Norm);

/// FusedRMSNorm: same as RMSNorm but with a hint for fused kernels
/// When CUDA kernels are available, uses fused implementation
class FusedRMSNormImpl : public torch::nn::Module {
 public:
  FusedRMSNormImpl(int64_t size, double eps = 1e-6, bool elementwise_affine = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor weight_;
  double eps_;
  bool elementwise_affine_;
  int64_t size_;
};
TORCH_MODULE(FusedRMSNorm);

}  // namespace olmo_cpp
