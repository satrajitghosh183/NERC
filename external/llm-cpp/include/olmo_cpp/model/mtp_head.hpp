#pragma once
/**
 * include/olmo_cpp/model/mtp_head.hpp
 *
 * ─── What "MTP" is ──────────────────────────────────────────────────
 *
 * MTP = **M**ulti-**T**oken **P**rediction. Instead of training the
 * model to predict only the next token, MTP adds K extra "heads"
 * that each predict the next 1, 2, ..., K tokens at the same time.
 * The auxiliary loss forces the residual stream to encode
 * longer-range planning information.
 *
 *     loss_total = ce(logits, t+1)
 *                + λ · sum_k ce(mtp_head_k(h), t+1+k)
 *
 * It also enables fast speculative decoding at inference time (the
 * MTP heads provide cheap drafts, see src/generate/speculative_decode.cpp).
 *
 * Each MTP head is structurally identical to the main lm_head: a
 * RMSNorm + Linear D->vocab. They share the input residual stream
 * but have independent weights.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/layer_norm.hpp : RMSNorm prefix.
 *   - olmo_cpp/model/lm_head.hpp    : Linear D->vocab projection.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/mtp_head.cpp : implementation.
 *   - src/model/transformer.cpp : when cfg.num_mtp_heads > 0, builds
 *     a ModuleList of MTPHeads and calls them in forward().
 *
 * --- Role in training pipeline ---
 *   Optional. Off in the quickstart conf (num_mtp_heads=0).
 */

#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// Single MTP prediction head: Linear projection + RMSNorm
/// Shares the LM head's output projection with the main model.
///
/// For head k predicting token at position t+k+1:
///   logits_k = shared_lm_head( norm_k( proj_k(h_t) ) )
///
/// During training, the loss target is labels shifted by k positions.
class MTPHeadImpl : public torch::nn::Module {
 public:
  MTPHeadImpl(int64_t d_model, double eps = 1e-6);

  /// Project hidden states through this head's transform.
  /// Returns transformed hidden states (caller applies shared LM head).
  torch::Tensor forward(torch::Tensor hidden_states);

 private:
  torch::nn::Linear proj_;
  RMSNorm norm_;
};

TORCH_MODULE(MTPHead);

}  // namespace olmo_cpp
