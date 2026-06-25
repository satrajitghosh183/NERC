/**
 * src/model/fused_block.cpp
 *
 * The "fused" sister of block.cpp. Same algorithm — pre-norm
 * attention + residual, then pre-norm FFN + residual — but using:
 *
 *   - FusedAttention (single QKV matmul instead of three separate
 *     Q/K/V Linears, see fused_attention.cpp), and
 *   - FeedForward(use_fused_gate_up=true) (single 2H-wide W1 matmul
 *     instead of separate gate and up Linears, see feed_forward.cpp).
 *
 * The math is identical to a regular Block; only the matmul launch
 * count is reduced. With cudaGraphs disabled, each kernel launch
 * costs ~5-10 µs — saving 4 launches per block per fwd is a few
 * percent of step time on a 4-layer model and more on big ones.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/fused_block.hpp : class declaration.
 *   - olmo_cpp/backend/backend.hpp   : get_backend() for fused norms.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/fused_transformer.cpp: FusedTransformer instantiates
 *     N of these and walks its ModuleList in forward().
 *
 * --- Role in training pipeline ---
 *   The unit of repetition inside FusedTransformer. The quickstart's
 *   3060 conf has fused=1 so this file is on the hot path.
 */
#include "olmo_cpp/model/fused_block.hpp"
#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

FusedTransformerBlockImpl::FusedTransformerBlockImpl(
    const TransformerConfig& cfg, int64_t block_idx)
    : attention_(register_module("attention", FusedAttention(cfg, block_idx))),
      attention_norm_(register_module("attention_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_norm_(register_module("feed_forward_norm", RMSNorm(cfg.d_model, cfg.layer_norm_eps))),
      feed_forward_(register_module("feed_forward",
          FeedForward(cfg.d_model, cfg.get_hidden_size(), /*bias=*/false, /*use_fused_gate_up=*/true))) {}

torch::Tensor FusedTransformerBlockImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto& backend = get_backend();
  backend.begin_scope();

  // Fused norm+residual: h = x + rms_norm(attention(x))
  auto h = attention_norm_->forward_add(attention_(x, rope_bufs, start_pos, layer_cache), x);

  // Fused norm+residual: out = h + rms_norm(ffn(h))
  auto out = feed_forward_norm_->forward_add(feed_forward_(h), h);

  backend.end_scope();
  return out;
}

}  // namespace olmo_cpp
