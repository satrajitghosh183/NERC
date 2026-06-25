#pragma once

/**
 * include/olmo_cpp/model/block_variants.hpp
 *
 * Catalog of alternative per-layer block architectures explored as
 * ablations against the default ReorderedNormTransformerBlock. Five
 * variants are provided: PeriNorm (norm wraps the residual add),
 * LayerNormScaled (per-sublayer learned scalar gain), Normalized (nGPT-
 * style spherical interpolation), and two MoE blocks (full MoE and a
 * hybrid that interleaves dense FFN with MoE every k layers).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig (block dims + MoE knobs)
 *   - olmo_cpp/model/attention.hpp: shared Attention sublayer
 *   - olmo_cpp/model/feed_forward.hpp: FeedForward (used by dense variants)
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache for inference reuse
 *   - olmo_cpp/model/rope.hpp: RoPEBuffers passthrough
 *   - olmo_cpp/model/moe/moe.hpp: MoELayer for the MoE block variants
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. These variants are
 *   provided as building blocks for ablation models; the standard model
 *   path uses ReorderedNormTransformerBlock from block.hpp.
 *
 * --- Role in training pipeline ---
 *   Off the bench path. Used when running variant configurations to
 *   compare alternative residual / normalization patterns and MoE.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/rope.hpp"
#include "olmo_cpp/model/moe/moe.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// PeriNorm block: applies normalization periodically
/// h = norm(x + attn(x)), out = norm(h + ff(h))
class PeriNormBlockImpl : public torch::nn::Module {
 public:
  /// Construct attention, two RMSNorms, and dense FeedForward sublayers.
  PeriNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  /// PeriNorm forward — see file-level note for residual pattern.
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// Post-attention norm (`norm1_`) and post-FFN norm (`norm2_`).
  RMSNorm norm1_, norm2_;
  /// Dense SwiGLU FFN sublayer.
  FeedForward feed_forward_;
};
/// Module holder for PeriNormBlockImpl.
TORCH_MODULE(PeriNormBlock);

/// LayerNormScaled block: pre-norm with learned scaling factor
/// h = x + scale_attn * norm1(attn(x)), out = h + scale_ff * norm2(ff(h))
class LayerNormScaledBlockImpl : public torch::nn::Module {
 public:
  /// Build sublayers and learned per-sublayer scalar gains (init=1).
  LayerNormScaledBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  /// Apply the learned-gain residual pattern. See file note for formula.
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// Pre-attention and pre-FFN RMSNorms.
  RMSNorm norm1_, norm2_;
  /// Dense FFN sublayer.
  FeedForward feed_forward_;
  /// Per-sublayer scalar gain parameters (shape [1]).
  torch::Tensor scale_attn_, scale_ff_;
};
/// Module holder for LayerNormScaledBlockImpl.
TORCH_MODULE(LayerNormScaledBlock);

/// nGPT Normalized block: all vectors live on the unit hypersphere
/// Normalizes all intermediate representations to unit norm
class NormalizedBlockImpl : public torch::nn::Module {
 public:
  /// Build sublayers and small init=0.01 alpha interpolation factors.
  NormalizedBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  /// Spherical-interpolation forward: keeps activations unit-norm.
  torch::Tensor forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                        std::optional<int64_t> start_pos = std::nullopt,
                        LayerKVCache* layer_cache = nullptr);
 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// Dense FFN sublayer.
  FeedForward feed_forward_;
  /// Per-sublayer learned interpolation coefficients on the unit sphere.
  torch::Tensor alpha_attn_, alpha_ff_;  // learnable interpolation factors
  /// Cached d_model for norm sanity (currently unused at runtime).
  int64_t d_model_;
  /// L2-normalize x along the model dimension; helper used in forward.
  torch::Tensor normalize(torch::Tensor x);
};
/// Module holder for NormalizedBlockImpl.
TORCH_MODULE(NormalizedBlock);

/// MoE Reordered Norm block: replaces FFN with MoE layer
class MoEReorderedNormBlockImpl : public torch::nn::Module {
 public:
  /// Build attention, two RMSNorms and the MoELayer per the config knobs.
  MoEReorderedNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  /// MoE blocks return both the new hidden and an auxiliary load-balance /
  /// z-loss term that the training loop adds to the main loss.
  struct BlockOutput {
    torch::Tensor hidden_states;
    torch::Tensor aux_loss;
  };
  /// Forward returning (hidden, aux_loss).
  BlockOutput forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                      std::optional<int64_t> start_pos = std::nullopt,
                      LayerKVCache* layer_cache = nullptr);
 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// Norm wrapping attention output and norm wrapping MoE output.
  RMSNorm attention_norm_, moe_norm_;
  /// Mixture-of-experts FFN.
  MoELayer moe_;
  /// Cached config (held by value for MoE loss weights).
  TransformerConfig config_;
};
/// Module holder for MoEReorderedNormBlockImpl.
TORCH_MODULE(MoEReorderedNormBlock);

/// MoE Hybrid block: alternates between dense FFN and MoE
class MoEHybridReorderedNormBlockImpl : public torch::nn::Module {
 public:
  /// At construction, decides whether this layer is MoE or dense based on
  /// `block_idx % moe_hybrid_interval == 0`.
  MoEHybridReorderedNormBlockImpl(const TransformerConfig& cfg, int64_t block_idx);
  /// Same (hidden, aux_loss) tuple as the pure MoE block; aux is zero
  /// when this layer is dense.
  struct BlockOutput {
    torch::Tensor hidden_states;
    torch::Tensor aux_loss;
  };
  /// Forward returning (hidden, aux_loss).
  BlockOutput forward(torch::Tensor x, const RoPEBuffers* rope_bufs = nullptr,
                      std::optional<int64_t> start_pos = std::nullopt,
                      LayerKVCache* layer_cache = nullptr);
 private:
  /// Self-attention sublayer.
  Attention attention_;
  /// Pre-FFN/MoE RMSNorms.
  RMSNorm attention_norm_, ff_norm_;
  /// Whether this particular layer instance is MoE (vs. dense FFN).
  bool use_moe_;
  /// Optional MoELayer (active iff use_moe_).
  std::optional<MoELayer> moe_;
  /// Optional dense FeedForward (active iff !use_moe_).
  std::optional<FeedForward> feed_forward_;
  /// Cached config for MoE loss weight access.
  TransformerConfig config_;
};
/// Module holder for MoEHybridReorderedNormBlockImpl.
TORCH_MODULE(MoEHybridReorderedNormBlock);

}  // namespace olmo_cpp
