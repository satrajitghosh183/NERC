/**
 * src/model/layer_norm_variants.cpp
 *
 * Alternate normalization layers that the model can pick via
 * TransformerConfig::LayerNormType. Three are defined here:
 *   - LayerNorm  : full mean-subtracting LayerNorm (with optional bias).
 *   - L2Norm     : x / ||x||_2 * weight, used by some experimental heads.
 *   - FusedRMSNorm: an explicit pure-ATen RMSNorm with FP32 internal accum,
 *                  selected when the user wants the deterministic ATen path
 *                  rather than the backend-dispatched fused kernel.
 * The standard RMSNorm path lives in layer_norm.cpp; this file is for the
 * branches that get compiled in for ablation, debugging, or non-OLMo
 * architectures (the structural / DC-MRE variants).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/layer_norm_variants.hpp: declarations of the three
 *     Impl classes implemented here.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp / src/config.cpp: parse layer_norm_type strings
 *     ("layer_norm", "l2_norm", "fused_rms_norm") into the enum that selects
 *     these variants for the transformer stack.
 *   - Direct callers not located via quick grep (selection happens via the
 *     TransformerConfig::LayerNormType enum used by block constructors).
 *
 * --- Role in training pipeline ---
 *   Each variant runs at every block boundary in place of RMSNorm when the
 *   config selects it. Used during forward + backward for every token. The
 *   FusedRMSNorm class is also a useful FP32-reference implementation for
 *   correctness-checking the custom CUDA RMSNorm kernels in kernels/.
 */
#include "olmo_cpp/model/layer_norm_variants.hpp"

#include <cmath>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// LayerNorm — classic Ba/Kiros/Hinton 2016 normalization.
// y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
// We implement it via ATen ops (no fused kernel) since this branch is rarely
// selected and exists mostly for ablation against RMSNorm.
// ---------------------------------------------------------------------------

/// LayerNorm constructor. `bias` is independent of `elementwise_affine`:
/// if affine is off, neither weight nor bias is allocated; if affine is on,
/// weight is always created and bias is created only when `bias` is true.
LayerNormImpl::LayerNormImpl(int64_t size, double eps, bool elementwise_affine, bool bias)
    : eps_(eps),
      elementwise_affine_(elementwise_affine),
      has_bias_(bias),
      size_(size) {
  if (elementwise_affine_) {
    // gamma initialized to 1 so the layer is identity at step 0.
    weight_ = register_parameter("weight", torch::ones(size));
    if (has_bias_) {
      // beta initialized to 0 — same reason.
      bias_ = register_parameter("bias", torch::zeros(size));
    }
  }
}

/// Forward pass over the last dimension.
/// x: [..., size] -> [..., size].
torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
  // Reduce along the last axis only; keepdim=true so broadcasting works.
  auto mean = x.mean(-1, /*keepdim=*/true);
  // Biased variance (population variance, divide by N), the LayerNorm convention.
  auto var = x.var(-1, /*unbiased=*/false, /*keepdim=*/true);
  // Subtract mean, divide by std. eps inside the sqrt prevents NaNs when var=0.
  auto x_norm = (x - mean) / torch::sqrt(var + eps_);
  if (elementwise_affine_ && weight_.defined()) {
    // Per-feature gain.
    x_norm = x_norm * weight_;
    if (has_bias_ && bias_.defined()) {
      // Per-feature shift.
      x_norm = x_norm + bias_;
    }
  }
  return x_norm;
}

// ---------------------------------------------------------------------------
// L2Norm — pure direction normalization. y = x / ||x||_2 * gamma.
// Used in some experimental output heads (e.g. sphere-projected logits) and
// as a regularizer in DC-MRE structural variants. Differs from RMSNorm by
// a factor of sqrt(d): RMSNorm divides by RMS, L2Norm divides by total
// vector length without per-feature averaging.
// ---------------------------------------------------------------------------

/// Construct an L2Norm. eps is a clamp floor on the denominator (not added
/// inside a sqrt), which prevents division by tiny norms.
L2NormImpl::L2NormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine_) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

/// Forward pass: project onto the unit sphere then optional per-feature gain.
torch::Tensor L2NormImpl::forward(torch::Tensor x) {
  // ord=2, last dim, keepdim=true so the broadcast division below has the
  // right shape for arbitrary leading batch dimensions.
  auto norm = torch::norm(x, 2, /*dim=*/-1, /*keepdim=*/true);
  // clamp_min ensures we never divide by something smaller than eps (this is
  // the L2Norm convention; RMSNorm puts eps inside an rsqrt instead).
  auto x_norm = x / torch::clamp_min(norm, eps_);
  if (elementwise_affine_ && weight_.defined()) {
    x_norm = x_norm * weight_;
  }
  return x_norm;
}

// ---------------------------------------------------------------------------
// FusedRMSNorm — explicit ATen RMSNorm with FP32 internal accumulation.
//
// "Fused" here is aspirational: the name is a hint that this layer should
// be replaced by a single fused kernel when one is available. The fallback
// implemented below mimics the canonical numerically-stable recipe used by
// HuggingFace LLaMA: keep the squared-mean reduction in FP32 even when the
// input is bf16/fp16, then cast the rescaled output back. The non-fused
// RMSNorm path in layer_norm.cpp routes through the backend dispatcher and
// can pick a true CUDA fused kernel; this class is the fallback / reference.
// ---------------------------------------------------------------------------

/// Construct a FusedRMSNorm. Same parameters and initialization as RMSNormImpl.
FusedRMSNormImpl::FusedRMSNormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine), size_(size) {
  if (elementwise_affine_) {
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

/// Forward pass: y = x * rsqrt(mean(x^2) + eps) * weight, with FP32 reduction.
torch::Tensor FusedRMSNormImpl::forward(torch::Tensor x) {
  // RMS normalization: x * rsqrt(mean(x^2) + eps).
  // Only upcast to FP32 when input is a reduced-precision dtype — the FP32
  // fast path avoids two redundant casts.
  if (x.dtype() == torch::kFloat32) {
    // mean(x^2) along last dim with keepdim so we can broadcast back.
    auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
    // rsqrt is the reciprocal square root: faster + more stable than 1/sqrt.
    auto x_norm = x * torch::rsqrt(variance + eps_);
    if (elementwise_affine_ && weight_.defined()) {
      x_norm = x_norm * weight_;
    }
    return x_norm;
  }

  // Reduced precision path: compute the reduction in FP32 to avoid catastrophic
  // accumulation error. Inputs typically come in bf16 during AMP training.
  auto input_dtype = x.dtype();
  auto x_fp32 = x.to(torch::kFloat32);
  auto variance = x_fp32.pow(2).mean(-1, /*keepdim=*/true);
  auto x_norm = x_fp32 * torch::rsqrt(variance + eps_);
  // Cast back to the original dtype before applying the (also-bf16) weight.
  x_norm = x_norm.to(input_dtype);

  if (elementwise_affine_ && weight_.defined()) {
    x_norm = x_norm * weight_;
  }
  return x_norm;
}

}  // namespace olmo_cpp
