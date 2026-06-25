#pragma once

/**
 * include/olmo_cpp/model/block.hpp
 *
 * One transformer layer in the "reordered norm" arrangement used by
 * OLMo-core. The layer pattern is:
 *   h   = x + attention_norm( attention(x) )
 *   out = h + feed_forward_norm( feed_forward(h) )
 * i.e. RMSNorm wraps the *output* of each sublayer rather than its input
 * (the standard pre-norm pattern). See `block.cpp` for the exact forward.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig (d_model, layer_norm_eps,
 *     hidden_size for FFN sizing)
 *   - olmo_cpp/model/attention.hpp: Attention sublayer
 *   - olmo_cpp/model/feed_forward.hpp: FeedForward (SwiGLU MLP) sublayer
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache for incremental decoding
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm with fused forward_add
 *   - olmo_cpp/model/rope.hpp: RoPEBuffers passed through to attention
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp: TransformerImpl pushes one
 *     ReorderedNormTransformerBlock per layer into its ModuleList; calls
 *     each block's forward in the layer loop
 *
 * --- Role in training pipeline ---
 *   This is the per-layer compute unit of the standard (un-fused) model.
 *   Each forward through the model does n_layers calls to this block's
 *   forward, sandwiched between embedding and LM head.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// ReorderedNormTransformerBlock: h = x + attn_norm(attn(x)), out = h + ff_norm(ff(h))
class ReorderedNormTransformerBlockImpl : public torch::nn::Module {
 public:
  /// Construct block; sub-modules are registered with names "attention",
  /// "attention_norm", "feed_forward_norm", "feed_forward" so checkpoint
  /// load/save round-trips with the upstream OLMo-core naming.
  ReorderedNormTransformerBlockImpl(const TransformerConfig& cfg, int64_t block_idx);

  /// Run attention + FFN with the reordered-norm residual pattern. Pass
  /// rope_bufs / start_pos / layer_cache straight through to the attention
  /// sublayer (see Attention::forward for shape contract).
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

  /// Paged-KV variant: same reordered-norm residual pattern, but attention's
  /// per-layer K/V append/materialize goes through `paged` at `layer_idx`.
  torch::Tensor forward_paged(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs,
      int64_t start_pos,
      IPagedKVCache* paged,
      int64_t layer_idx);

  /// Tree-attention forward (item 8.1). Attention uses caller-supplied mask.
  torch::Tensor forward_with_mask(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs,
      torch::Tensor attn_mask);

 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// RMSNorm applied to the attention output before residual add.
  RMSNorm attention_norm_;
  /// RMSNorm applied to the FFN output before residual add.
  RMSNorm feed_forward_norm_;
  /// SwiGLU FeedForward sublayer.
  FeedForward feed_forward_;
};

/// Module-holder macro for ReorderedNormTransformerBlockImpl.
TORCH_MODULE(ReorderedNormTransformerBlock);

}  // namespace olmo_cpp
