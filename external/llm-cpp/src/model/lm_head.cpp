/**
 * src/model/lm_head.cpp
 *
 * Implements the language-model output head: an optional final RMSNorm
 * followed by a Linear "unembedding" projection from the residual stream
 * (d_model) to the vocabulary logits (vocab_size). This module sits at the
 * top of the transformer stack and produces the per-token logits that feed
 * into cross-entropy loss during training and into sampling during inference.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/lm_head.hpp: LMHeadImpl declaration plus the public
 *     w_out() accessor used by µP zero-init and weight-sharing logic.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp: registers lm_head_ = LMHead(d_model,
 *     vocab_size, true, layer_norm_eps); calls lm_head_(h) at line 164 for
 *     the main path and lm_head_(mtp_h) at lines 193 and 228 for the
 *     multi-token-prediction heads (LM head is shared across MTP positions).
 *   - src/model/fused_transformer.cpp: same usage in the fused variant
 *     (lines 16, 89, 158, 181, 215 — including µP truncated-normal init of
 *     w_out()->weight).
 *
 * --- Role in training pipeline ---
 *   Final stage of the forward pass. Its weight matrix is the largest dense
 *   tensor in most LLMs (d_model * vocab_size), so its GEMM dominates the
 *   tail of every step. It is also the parameter zeroed by µP init
 *   (zero_init_lm_head) so the model starts at a uniform output distribution.
 */
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/backend/cublas_direct.hpp"

namespace olmo_cpp {

/// Construct the LM head.
/// d_model   : residual-stream feature width (input).
/// vocab_size: number of token classes (output).
/// use_norm  : if true, prepend a final RMSNorm so the unembedding sees a
///             well-conditioned input — strongly recommended for stability.
/// eps       : eps passed through to the RMSNorm.
LMHeadImpl::LMHeadImpl(int64_t d_model, int64_t vocab_size, bool use_norm, double eps)
    : w_out_(register_module("w_out", torch::nn::Linear(torch::nn::LinearOptions(d_model, vocab_size).bias(false)))) {
  if (use_norm) {
    // Build the optional norm; register_module wires it into the parameter
    // tree so it appears under "norm.*" in named_parameters().
    norm_ = RMSNorm(d_model, eps);
    register_module("norm", *norm_);
  }
}

/// Forward pass.
/// x  : [B, S, d_model] hidden states from the last transformer block.
/// out: [B, S, vocab_size] logits ready for cross-entropy loss.
torch::Tensor LMHeadImpl::forward(torch::Tensor x) {
  if (norm_) {
    // Pre-projection RMSNorm. std::optional<RMSNorm> stores a TORCH_MODULE
    // holder; deref with *norm_ to get the holder, then operator() to call.
    x = (*norm_)(x);
  }
  // L (cuBLASLt direct): the LM head is the biggest GEMM in the model
  // ([B*S, d_model] × [d_model, vocab]); bypass the ATen dispatcher.
  return fast_linear(x, w_out_->weight, torch::Tensor());
}

}  // namespace olmo_cpp
