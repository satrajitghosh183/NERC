#pragma once

/**
 * include/olmo_cpp/model/transformer.hpp
 *
 * The standard (un-fused) Transformer language model. Owns the embedding
 * (plain or DC-MRE multi-resolution), an embedding RMSNorm, a stack of
 * ReorderedNormTransformerBlock layers, a tied LM head, and an optional
 * Multi-Token Prediction (MTP) head list. Provides three forward paths:
 * full forward (loss or logits), backbone-only (returns hidden states),
 * and MTP draft (logits per MTP head) used by speculative decoding.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig (model + MTP + activation
 *     checkpoint settings)
 *   - olmo_cpp/model/block.hpp: ReorderedNormTransformerBlock layer
 *   - olmo_cpp/model/kv_cache.hpp: KVCache for inference
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm for the embedding norm
 *   - olmo_cpp/model/lm_head.hpp: LMHead (final norm + linear, optional
 *     tied weights)
 *   - olmo_cpp/model/mtp_head.hpp: MTPHead for multi-token prediction
 *   - olmo_cpp/nn/multi_res_embedding.hpp: DC-MRE multi-resolution embed
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: builds Transformer when [optimization].fused == 0
 *   - src/train.cpp: drives the training loop, calls forward(input_ids,
 *     labels) to obtain loss
 *   - src/eval/lm_evaluator.cpp: per-token-loss evaluation
 *   - src/generate/speculative_decode.cpp: uses forward_backbone +
 *     forward_mtp_draft for verification of speculated tokens
 *   - tools/chat.cpp, tools/convert_hf.cpp, tools/dump_params.cpp
 *
 * --- Role in training pipeline ---
 *   The whole un-fused training stack (loss path) lives in this module's
 *   forward(). Used as the apples-to-apples baseline against OLMo-core's
 *   Python Transformer.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/mtp_head.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/nn/multi_res_embedding.hpp"
#include <torch/torch.h>
#include <optional>
#include <vector>

namespace olmo_cpp {

/// Standard transformer module. Holds the full backbone + LM head + MTP
/// heads and dispatches them in forward().
class TransformerImpl : public torch::nn::Module {
 public:
  /// Construct backbone, LM head, MTP heads, and embeddings according to
  /// the config. After construction, init_weights() must be called to
  /// apply the truncated-normal initialization.
  explicit TransformerImpl(const TransformerConfig& cfg);

  /// Forward pass. When kv_cache is provided, uses incremental decoding:
  /// only the new tokens are processed, and K/V are cached for future steps.
  ///
  /// When labels are provided and MTP heads are active, computes:
  ///   loss = main_ce_loss + mtp_weight * mean(mtp_ce_losses)
  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100,
      KVCache* kv_cache = nullptr);

  /// MTP draft: given hidden state at the last position, returns logits from
  /// each MTP head. Used for speculative decoding.
  /// Returns vector of [vocab_size] logit tensors, one per MTP head.
  std::vector<torch::Tensor> forward_mtp_draft(torch::Tensor hidden_state);

  /// Get the backbone hidden states (no LM head). Used by speculative decode
  /// to verify candidate tokens.
  torch::Tensor forward_backbone(
      torch::Tensor input_ids,
      KVCache* kv_cache = nullptr);

  /// Paged-KV variant of forward_backbone. Threads an IPagedKVCache through
  /// the block stack via ReorderedNormTransformerBlock::forward_paged.
  /// Inference path only — no activation checkpointing.
  torch::Tensor forward_backbone_paged(
      torch::Tensor input_ids,
      IPagedKVCache* paged);

  /// Paged-KV variant of forward. Inference path only: returns logits over the
  /// vocab. No labels / no MTP loss path here — the legacy `forward` covers
  /// training. Decode call sites should prefer this once they construct a
  /// PagedKVCache via `make_paged_kv_cache`.
  torch::Tensor forward_paged(
      torch::Tensor input_ids,
      IPagedKVCache* paged);

  /// Tree-attention forward (item 8.1 wiring). input_ids: [1, N] flat tree.
  /// attn_mask: [N, N] bool tensor where mask[i, j] = true iff j is an
  /// ancestor of i (DraftTree::flatten output). Returns logits [1, N, V].
  /// No KV cache: each verify is a fresh forward.
  torch::Tensor forward_tree(torch::Tensor input_ids,
                              torch::Tensor attn_mask);

  /// Apply truncated-normal init to all parameters with model-specific
  /// scaling (separate stds for embeddings vs. blocks vs. LM head).
  /// gen: optional torch generator for reproducibility.
  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  // Returns a reference to the cached per-layer RoPE buffers. Recomputed only
  // on cache miss (seq_len grows or dtype changes). Returning by const ref
  // avoids a per-forward vector copy + tensor refcount churn.
  const std::vector<RoPEBuffers>& get_rope_buffers(
      int64_t seq_len, torch::Device device,
      torch::Dtype dtype = torch::kFloat32);

  /// Number of transformer layers.
  int64_t n_layers() const { return config_.n_layers; }
  /// Number of MTP heads (0 disables MTP).
  int64_t num_mtp_heads() const { return config_.num_mtp_heads; }
  /// Convenience predicate.
  bool has_mtp() const { return config_.num_mtp_heads > 0; }

  /// Apply LM head to hidden states → logits
  torch::Tensor apply_lm_head(torch::Tensor hidden_states) { return lm_head_(hidden_states); }

  /// LM head's unembedding matrix [V, H]. Used by the fused LM-head +
  /// Gumbel-max sampler (fast-inference [6]) which computes the GEMV
  /// itself instead of going through apply_lm_head + sample.
  const torch::Tensor& lm_head_weight() const { return lm_head_->w_out()->weight; }

 private:
  // Embedding: either plain or multi-resolution (DC-MRE)
  /// Plain torch::nn::Embedding (active when use_multi_res_ is false).
  torch::nn::Embedding embeddings_{nullptr};
  /// DC-MRE multi-resolution embedding (active when use_multi_res_ is true).
  MultiResEmbedding multi_res_embed_{nullptr};  // used when config.use_multi_res
  /// Selects between the two embedding paths above.
  bool use_multi_res_ = false;

  /// RMSNorm applied to the embedding output before the first block.
  std::optional<RMSNorm> embedding_norm_;
  /// Holds n_layers ReorderedNormTransformerBlock modules.
  torch::nn::ModuleList blocks_;
  /// Final norm + tied linear projection to vocab logits.
  LMHead lm_head_;
  /// Optional pre-block embedding multiplier (scales weights at init time).
  std::optional<double> embed_scale_;
  /// Cached config (also queried in forward to read MTP weights etc.).
  TransformerConfig config_;

  // Multi-Token Prediction heads
  /// Holds num_mtp_heads MTPHead modules; empty when MTP is disabled.
  torch::nn::ModuleList mtp_heads_;

  // Cached RoPE — recomputed only when seq_len grows or dtype changes
  /// Last allocated RoPE table length (in tokens).
  int64_t cached_rope_len_ = 0;
  /// Activation dtype used for the cached RoPE tables.
  torch::Dtype cached_rope_dtype_ = torch::kFloat32;
  /// Per-layer RoPE buffers (currently shared across layers via refcount).
  std::vector<RoPEBuffers> cached_rope_bufs_;
};

/// Module-holder macro: `Transformer` is a shared-ptr around the impl.
TORCH_MODULE(Transformer);

}  // namespace olmo_cpp
