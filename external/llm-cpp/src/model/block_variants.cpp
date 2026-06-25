/**
 * src/model/block_variants.cpp
 *
 * ─── What this file is ──────────────────────────────────────────────
 *
 * The "reordered-norm" block in block.cpp is the production default,
 * but research often wants to try alternative wirings of attention,
 * FFN, and norms. This file collects those alternatives:
 *
 *   - **PeriNormBlock**          (norm wrapped *around* each sublayer
 *                                  on both sides — was the LLaMA-1
 *                                  recipe; tends to be more stable
 *                                  with deeper models).
 *   - **LayerNormScaledBlock**   (block.cpp + a learnable scalar
 *                                  multiplying each sublayer output —
 *                                  inspired by ReZero).
 *   - **NormalizedNGPTBlock**    (the "normalised GPT" block from the
 *                                  nGPT paper — every weight matrix
 *                                  kept on the unit sphere).
 *   - **MoEReorderedNormBlock**  (same skeleton as ReorderedNorm but
 *                                  the FFN sublayer is replaced by an
 *                                  MoE layer — see src/model/moe/).
 *   - **MoEHybridReorderedNorm** (alternates dense FFN and MoE FFN
 *                                  every K layers — cheaper than
 *                                  all-MoE without losing much
 *                                  capacity).
 *
 * All of them implement the same forward signature as the canonical
 * Block so transformer.cpp can swap them in based on cfg.block_type.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/block_variants.hpp : declarations of the block
 *     variant classes.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp / fused_transformer.cpp: select one
 *     variant at construction time based on cfg.block_type.
 *
 * --- Role in training pipeline ---
 *   Off the hot path unless explicitly selected via the .conf.
 */
#include "olmo_cpp/model/block_variants.hpp"

namespace olmo_cpp {

// =============================================================================
// PeriNormBlock
// =============================================================================

PeriNormBlockImpl::PeriNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      norm1_(register_module("norm1", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      norm2_(register_module("norm2", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size(), false))) {}

torch::Tensor PeriNormBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  // PeriNorm: norm wraps the residual connection
  auto h = norm1_(x + attention_(x, rope_bufs, start_pos, layer_cache));
  return norm2_(h + feed_forward_(h));
}

// =============================================================================
// LayerNormScaledBlock
// =============================================================================

LayerNormScaledBlockImpl::LayerNormScaledBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      norm1_(register_module("norm1", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      norm2_(register_module("norm2", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size(), false))) {
  // Learned scaling factors, initialized to 1.0
  scale_attn_ = register_parameter("scale_attn", torch::ones({1}));
  scale_ff_ = register_parameter("scale_ff", torch::ones({1}));
}

torch::Tensor LayerNormScaledBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto h = x + scale_attn_ * norm1_(attention_(x, rope_bufs, start_pos, layer_cache));
  return h + scale_ff_ * norm2_(feed_forward_(h));
}

// =============================================================================
// NormalizedBlock (nGPT)
// =============================================================================

NormalizedBlockImpl::NormalizedBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      feed_forward_(register_module("feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size(), false))),
      d_model_(cfg.d_model) {
  // Learnable interpolation factors, initialized to small values
  alpha_attn_ = register_parameter("alpha_attn",
      torch::full({1}, 0.01));
  alpha_ff_ = register_parameter("alpha_ff",
      torch::full({1}, 0.01));
}

torch::Tensor NormalizedBlockImpl::normalize(torch::Tensor x) {
  // L2-normalize along the last dimension (model dim)
  return torch::nn::functional::normalize(x,
      torch::nn::functional::NormalizeFuncOptions().dim(-1));
}

torch::Tensor NormalizedBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  // nGPT: interpolate on the unit hypersphere
  auto attn_out = attention_(x, rope_bufs, start_pos, layer_cache);
  auto h = normalize((1.0 - alpha_attn_) * x + alpha_attn_ * attn_out);

  auto ff_out = feed_forward_(h);
  return normalize((1.0 - alpha_ff_) * h + alpha_ff_ * ff_out);
}

// =============================================================================
// MoEReorderedNormBlock
// =============================================================================

MoEReorderedNormBlockImpl::MoEReorderedNormBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      moe_norm_(register_module("moe_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      moe_(register_module("moe", MoELayer(
          cfg.d_model, cfg.get_moe_hidden_size(), cfg.moe_num_experts,
          cfg.moe_top_k, cfg.moe_dropless, cfg.moe_capacity_factor, false))),
      config_(cfg) {}

MoEReorderedNormBlockImpl::BlockOutput MoEReorderedNormBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  // ReorderedNorm pattern: norm after sublayer, not before
  auto h = x + attention_norm_(attention_(x, rope_bufs, start_pos, layer_cache));
  auto moe_out = moe_->forward(h, config_.moe_zloss_weight, config_.moe_lb_loss_weight);
  auto out = h + moe_norm_(moe_out.hidden_states);
  return {out, moe_out.aux_loss};
}

// =============================================================================
// MoEHybridReorderedNormBlock
// =============================================================================

MoEHybridReorderedNormBlockImpl::MoEHybridReorderedNormBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      ff_norm_(register_module("ff_norm",
          RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      use_moe_(block_idx % cfg.moe_hybrid_interval == 0),
      config_(cfg) {
  if (use_moe_) {
    moe_ = MoELayer(cfg.d_model, cfg.get_moe_hidden_size(), cfg.moe_num_experts,
                     cfg.moe_top_k, cfg.moe_dropless, cfg.moe_capacity_factor, false);
    register_module("moe", moe_.value());
  } else {
    feed_forward_ = FeedForward(cfg.d_model, cfg.get_hidden_size(), false);
    register_module("feed_forward", feed_forward_.value());
  }
}

MoEHybridReorderedNormBlockImpl::BlockOutput
MoEHybridReorderedNormBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto h = x + attention_norm_(attention_(x, rope_bufs, start_pos, layer_cache));

  torch::Tensor aux_loss = torch::zeros({1}, x.options());
  torch::Tensor out;

  if (use_moe_) {
    auto moe_out = (*moe_)->forward(h, config_.moe_zloss_weight, config_.moe_lb_loss_weight);
    out = h + ff_norm_(moe_out.hidden_states);
    aux_loss = moe_out.aux_loss;
  } else {
    out = h + ff_norm_((*feed_forward_)(h));
  }

  return {out, aux_loss};
}

}  // namespace olmo_cpp
