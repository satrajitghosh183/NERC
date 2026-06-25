/**
 * src/model/mtp_head.cpp
 *
 * Implements one Multi-Token-Prediction (MTP) head. MTP, used in DeepSeek-V3
 * and similar models, augments next-token prediction by also predicting
 * tokens at positions t+2, t+3, ..., t+K from the same final hidden state.
 * Each future-position k gets its own small head: a linear projection plus
 * an RMSNorm. After this transform, the resulting hidden vector is fed
 * through the *shared* main LM head (see transformer.cpp call sites) so
 * each MTP position uses the same unembedding weights as the main path.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/mtp_head.hpp: MTPHeadImpl declaration; depends on
 *     RMSNorm and LMHead to clarify the shared-head contract.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp: lines 45, 96, 191, 226 — pushes one MTPHead
 *     per future position into mtp_heads_ (a torch::nn::ModuleList) and
 *     invokes them per training step to compute auxiliary losses.
 *   - src/model/fused_transformer.cpp: lines 44, 92, 179, 213 — same usage
 *     in the fused-attention transformer variant.
 *
 * --- Role in training pipeline ---
 *   When mtp_num_heads > 0 in TransformerConfig, the model adds one MTPHead
 *   per future-token offset. Each head runs every forward pass and produces
 *   an extra cross-entropy loss term against labels shifted by k positions.
 *   The auxiliary losses are scaled and added to the main NTP loss; backward
 *   propagates through the projections and the shared LM head as usual.
 */
#include "olmo_cpp/model/mtp_head.hpp"

namespace olmo_cpp {

/// Construct an MTP head.
/// d_model: residual stream width — the projection is square (D x D) so that
/// the output remains compatible with the shared LM head.
/// eps    : RMSNorm epsilon, normally taken from cfg.layer_norm_eps.
MTPHeadImpl::MTPHeadImpl(int64_t d_model, double eps)
    : proj_(register_module("proj",
          torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false)))),
      // Per-head RMSNorm so each future-position prediction is normalized
      // independently — allows the heads to specialize without leaking scale
      // changes back into the shared LM head's input distribution.
      norm_(register_module("norm", RMSNorm(d_model, eps))) {}

/// Forward pass for one MTP position.
/// hidden_states: [B, S, d_model] from the last transformer block.
/// returns      : [B, S, d_model] transformed for this future offset; the
///                caller still has to apply the shared LM head to get logits.
torch::Tensor MTPHeadImpl::forward(torch::Tensor hidden_states) {
  // Apply head-specific projection then per-head normalization. Order is
  // proj -> norm so the norm sees the projected representation (this matches
  // the DeepSeek-V3 recipe).
  return norm_(proj_(hidden_states));
}

}  // namespace olmo_cpp
