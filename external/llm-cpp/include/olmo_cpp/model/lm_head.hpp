#pragma once
/**
 * include/olmo_cpp/model/lm_head.hpp
 *
 * ─── What the "LM head" is ──────────────────────────────────────────
 *
 * The very last operation of the transformer's forward pass turns
 * the d_model-wide hidden vector for each position into a
 * vocab_size-wide vector of "logits" — one score per possible next
 * token. Softmax over those gives the next-token probability
 * distribution.
 *
 *     logits = lm_head( final_norm( h ) )       // [B, S, vocab_size]
 *
 * `LMHead` is just an optional final RMSNorm followed by a Linear
 * projection D -> vocab_size. The Linear has no bias by default
 * (matches LLaMA / OLMo). Some configs tie the LM head's weight to
 * the embedding matrix to save parameters; this implementation does
 * NOT tie them (the lm_head has its own learnable W).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/layer_norm.hpp : RMSNorm prefix.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/lm_head.cpp : implementation.
 *   - src/model/transformer.cpp / fused_transformer.cpp : the LMHead
 *     is the very last submodule in the forward path.
 *
 * --- Role in training pipeline ---
 *   Foundational. One forward call per microbatch produces the
 *   logits the loss is computed against.
 */

#include "olmo_cpp/model/layer_norm.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// LM Head: optional RMSNorm + Linear to vocab
class LMHeadImpl : public torch::nn::Module {
 public:
  LMHeadImpl(int64_t d_model, int64_t vocab_size, bool use_norm = true, double eps = 1e-6);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear w_out() const { return w_out_; }

  // A3 — accessors that let callers do the LM head's pieces separately
  // so the fused LM-head + CE kernel can take pre-norm'd input + the
  // raw weight, skipping logits materialization entirely.
  torch::Tensor apply_norm(torch::Tensor x) {
    return norm_ ? (*norm_)(x) : x;
  }
  torch::Tensor weight() const { return w_out_->weight; }
  bool has_norm() const { return norm_.has_value(); }

 private:
  std::optional<RMSNorm> norm_;
  torch::nn::Linear w_out_;
};

TORCH_MODULE(LMHead);

}  // namespace olmo_cpp
