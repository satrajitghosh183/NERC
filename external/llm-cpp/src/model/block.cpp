/**
 * src/model/block.cpp
 *
 * ─── What a "transformer block" is ──────────────────────────────────
 *
 * The transformer's stack is just N copies of the same Block. Each
 * Block does two things in series, with a residual connection around
 * each:
 *
 *     h  = h + attention( rms_norm(h) )           // attention sublayer
 *     h  = h + feed_forward( rms_norm(h) )        // FFN sublayer
 *
 * "Pre-norm + residual" is the de-facto standard for stable training
 * in transformers >> 100M parameters (the alternative is "post-norm"
 * which used to fail at scale).
 *
 * This file implements the "reordered-norm" variant used in OLMo-2
 * and LLaMA-2/3. The norm is computed BEFORE the sublayer (attention
 * or FFN) so that the residual stream `h` keeps a clean,
 * unnormalised history all the way through the network — gradients
 * flow more smoothly that way.
 *
 * Sister files:
 *   - block_variants.cpp: PeriNorm, LayerNormScaled, NormalizedNGPT,
 *     hybrid MoE blocks — alternative ways to wire up the same two
 *     sublayers.
 *   - fused_block.cpp: same topology, but uses FusedAttention and a
 *     fused FFN with merged gate+up projections.
 *
 * The pieces:
 *   attention_       : an Attention module (multi-head SDPA + RoPE)
 *   feed_forward_    : a SwiGLU feed-forward (W1, W2, W3 linear)
 *   attention_norm_  : RMSNorm applied before attention (with fused
 *                      residual-add via forward_add)
 *   feed_forward_norm_: RMSNorm applied before the FFN
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/block.hpp     : own class declaration.
 *   - olmo_cpp/backend/backend.hpp : get_backend() for fused norm/silu_mul
 *     ops; begin_scope()/end_scope() drive the per-block arena allocator
 *     so block-local intermediates can be freed in one shot.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp: TransformerImpl::forward() walks its
 *     ModuleList of blocks and calls each one in sequence.
 *
 * --- Role in training pipeline ---
 *   The unit of repetition inside the model. For a 4-layer 30M model,
 *   the forward pass runs this 4 times sequentially per microbatch.
 *   For a 32-layer 7B model it runs 32 times.
 */
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

ReorderedNormTransformerBlockImpl::ReorderedNormTransformerBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", Attention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_norm_(register_module("feed_forward_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward", FeedForward(cfg.d_model, cfg.get_hidden_size(), false))) {
  // I-5 / T-6: AttentionImpl reads cfg.use_float8 itself; FeedForwardImpl
  // doesn't take cfg, so we wire its FP8 toggle from here.
  if (cfg.use_float8) feed_forward_->enable_float8(true);
}

torch::Tensor ReorderedNormTransformerBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto& backend = get_backend();
  backend.begin_scope();  // Arena scope for block intermediates
  auto h = attention_norm_->forward_add(
      attention_(x, rope_bufs, start_pos, layer_cache), x);
  auto out = feed_forward_norm_->forward_add(feed_forward_(h), h);
  backend.end_scope();    // Free all scratch within this block
  return out;
}

torch::Tensor ReorderedNormTransformerBlockImpl::forward_paged(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    int64_t start_pos,
    IPagedKVCache* paged,
    int64_t layer_idx) {
  auto& backend = get_backend();
  backend.begin_scope();
  auto attn_out = attention_->forward_paged(x, rope_bufs, start_pos, paged, layer_idx);
  auto h = attention_norm_->forward_add(attn_out, x);
  auto out = feed_forward_norm_->forward_add(feed_forward_(h), h);
  backend.end_scope();
  return out;
}

torch::Tensor ReorderedNormTransformerBlockImpl::forward_with_mask(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    torch::Tensor attn_mask) {
  auto& backend = get_backend();
  backend.begin_scope();
  auto attn_out = attention_->forward_with_mask(x, rope_bufs, attn_mask);
  auto h = attention_norm_->forward_add(attn_out, x);
  auto out = feed_forward_norm_->forward_add(feed_forward_(h), h);
  backend.end_scope();
  return out;
}

}  // namespace olmo_cpp
