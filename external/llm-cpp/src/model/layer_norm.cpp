/**
 * src/model/layer_norm.cpp
 *
 * Implements RMSNorm (Root Mean Square LayerNorm) — the normalization variant
 * used by LLaMA, OLMo, and most modern decoder-only LLMs. RMSNorm drops the
 * mean-subtraction step of LayerNorm, normalizes only by the RMS of the last
 * dimension, and applies a learned per-feature gain. Removing the mean is
 * cheaper, more numerically stable in low precision, and empirically shows
 * no quality loss versus full LayerNorm in pre-norm transformer stacks.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/layer_norm.hpp: RMSNormImpl declaration (weight tensor,
 *     eps, the forward / forward_add API).
 *   - olmo_cpp/backend/backend.hpp: get_backend().rms_norm() and rms_norm_add()
 *     dispatch to the active backend (default ATen, SIMD on CPU, fused CUDA
 *     kernels in kernels/rms_norm.cu).
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp / fused_block.cpp: TransformerBlock holds two
 *     RMSNorms (attention_norm_, feed_forward_norm_); forward_add fuses the
 *     post-sublayer residual add with the next pre-norm.
 *   - src/model/lm_head.cpp: optional final RMSNorm before the unembedding.
 *   - src/model/mtp_head.cpp: each multi-token-prediction head ends in RMSNorm.
 *   - src/model/transformer.cpp: top-level model uses RMSNorm via the blocks.
 *
 * --- Role in training pipeline ---
 *   Called on every forward pass at the entry to each sublayer (pre-norm)
 *   and once at the end of the stack. Backward gradients re-enter through
 *   the same kernel. The fused forward_add path matters because it avoids a
 *   full d_model-wide read+write between the residual add and the next norm,
 *   which on H100 is roughly 5-10% of per-step time at the small batch sizes
 *   used during DDP head-to-head benchmarking.
 */
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

/// Construct an RMSNorm over a feature dimension of `size`.
/// eps: numerical floor inside the rsqrt; OLMo defaults to 1e-6.
/// elementwise_affine: if true, register a learnable per-feature gain
/// initialized to ones (matches PyTorch nn.LayerNorm convention).
RMSNormImpl::RMSNormImpl(int64_t size, double eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine) {
    // Register as a parameter so it shows up in named_parameters() and gets
    // gradients; initial value 1.0 makes the norm an identity at step 0.
    weight_ = register_parameter("weight", torch::ones(size));
  }
}

/// Forward pass: y = x * rsqrt(mean(x^2) + eps) * weight.
/// x shape: [..., size]. Output: same shape, normalized along the last dim.
torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
  // Pass an undefined tensor when affine is off so the backend can skip the
  // weight multiply rather than allocating an ones-tensor.
  torch::Tensor w = (elementwise_affine_ && weight_.defined()) ? weight_ : torch::Tensor();
  return get_backend().rms_norm(x, w, eps_);
}

/// Fused: y = residual + rms_norm(x) * weight.
/// Used in pre-norm blocks where the previous sublayer's output (`x`) is
/// added back into the residual stream while we already have it in registers
/// — saves one full HBM read+write versus calling forward then add separately.
torch::Tensor RMSNormImpl::forward_add(torch::Tensor x, torch::Tensor residual) {
  torch::Tensor w = (elementwise_affine_ && weight_.defined()) ? weight_ : torch::Tensor();
  return get_backend().rms_norm_add(x, residual, w, eps_);
}

}  // namespace olmo_cpp
