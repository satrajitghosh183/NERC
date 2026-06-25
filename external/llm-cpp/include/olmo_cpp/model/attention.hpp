#pragma once

/**
 * include/olmo_cpp/model/attention.hpp
 *
 * Multi-head / grouped-query self-attention module used by the baseline
 * (un-fused) Transformer. Wraps four linear projections (Q, K, V, output),
 * optional QK-norm (per-head or per-tensor), rotary positional embeddings
 * (RoPE), and a sliding-window causal mask. Forward dispatches to ATen's
 * scaled_dot_product_attention so on CUDA it transparently uses
 * FlashAttention when available.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig (d_model, n_heads, n_kv_heads,
 *     head_dim, RoPE theta, sliding window size, qk-norm flags)
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache for incremental decoding
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm for optional QK-norm
 *   - olmo_cpp/model/rope.hpp: RotaryEmbedding + RoPEBuffers (precomputed
 *     sin/cos tables)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp: ReorderedNormTransformerBlockImpl owns one
 *     Attention as the attention sublayer of each block
 *   - src/model/block_variants.cpp: PeriNormBlock, LayerNormScaledBlock,
 *     NormalizedBlock, and the MoE block variants all use this Attention
 *
 * --- Role in training pipeline ---
 *   This is the un-fused reference implementation of self-attention used by
 *   the standard Transformer. It is the unit of attention compute called
 *   once per layer per forward; on the bench path FusedAttention replaces
 *   it but the math is identical.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/rope.hpp"
#include "olmo_cpp/float8/float8.hpp"
#include <torch/torch.h>
#include <optional>
#include <memory>

namespace olmo_cpp {

/// Standard (un-fused) multi-head self-attention with optional GQA, QK-norm,
/// RoPE and sliding window. Registered with the LibTorch module system via
/// TORCH_MODULE(Attention) below.
class AttentionImpl : public torch::nn::Module {
 public:
  /// Build all linear projections, optional q/k norms, and the RoPE module
  /// from the model config. layer_idx is currently unused but kept for
  /// future per-layer parameter overrides.
  AttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// Forward pass with optional KV cache for incremental decoding.
  /// x: [B, S, d_model]. Returns [B, S, d_model].
  /// rope_bufs: precomputed sin/cos for this layer (nullptr disables RoPE).
  /// start_pos: token offset of the first row of x (used for cache append).
  /// layer_cache: when non-null, K/V are appended to the cache and the full
  /// cached K/V are used as keys/values.
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

  /// Paged-KV variant of forward. Mirrors `forward` exactly except that the
  /// per-layer K/V append/materialize goes through `paged` at `layer_idx`
  /// instead of a LayerKVCache. Currently still uses ATen SDPA on the
  /// materialized K/V views — the dedicated paged-attention decode kernel
  /// (kernels/paged_attention.cu) can be wired in once dtype handling and
  /// batched-q support are finalised. Decode-only call site; expects
  /// `paged != nullptr`.
  torch::Tensor forward_paged(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs,
      int64_t start_pos,
      IPagedKVCache* paged,
      int64_t layer_idx);

  /// Tree-attention forward (item 8.1). Same as forward() but uses
  /// `attn_mask` as the additive SDPA mask instead of the built-in
  /// causal / sliding-window logic. No KV cache; verifies the whole
  /// tree in one shot.
  torch::Tensor forward_with_mask(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs,
      torch::Tensor attn_mask);

 private:
  /// Q projection: [d_model] -> [n_heads * head_dim].
  torch::nn::Linear w_q_;
  /// K projection: [d_model] -> [n_kv_heads * head_dim] (GQA-aware).
  torch::nn::Linear w_k_;
  /// V projection: [d_model] -> [n_kv_heads * head_dim] (GQA-aware).
  torch::nn::Linear w_v_;
  /// Output projection: [n_heads * head_dim] -> [d_model].
  torch::nn::Linear w_out_;
  /// Optional QK-norm on queries (RMSNorm), per-head or per-tensor.
  std::optional<RMSNorm> q_norm_;
  /// Optional QK-norm on keys (RMSNorm), per-head or per-tensor.
  std::optional<RMSNorm> k_norm_;
  /// Rotary positional embeddings; populated unconditionally in ctor.
  std::optional<RotaryEmbedding> rope_;
  /// Number of query heads (full multi-head count).
  int64_t n_heads_;
  /// Number of key/value heads for grouped-query attention.
  int64_t n_kv_heads_;
  /// Per-head dimensionality (d_model / n_heads in the simple case).
  int64_t head_dim_;
  /// Repetition factor for GQA: each KV head is shared by this many Q heads.
  int64_t n_heads_rep_;  // n_heads / n_kv_heads for GQA repeat
  /// If true, QK-norm is applied per-head after reshape; else pre-reshape.
  bool use_head_qk_norm_;
  /// Sliding window size in tokens; -1 disables (full causal attention).
  int64_t sliding_window_size_;  // -1 = no window (full attention)

  /// Cached query length used to build the last attention mask.
  // Cached sliding window mask — reused when (S, full_S) unchanged
  int64_t cached_mask_S_ = 0;
  /// Cached key length used to build the last attention mask.
  int64_t cached_mask_full_S_ = 0;
  /// Cached additive (-inf / 0) sliding-window mask, dtype-matched to acts.
  torch::Tensor cached_attn_mask_;

  /// FP8 emulation (I-5 / T-6). When cfg.use_float8 is true, each Q/K/V/out
  /// projection routes through float8_linear_emulated using these per-
  /// Linear scale trackers. Held by unique_ptr so the per-call updates
  /// (mutating amax_history) work with the otherwise-const attention call
  /// pattern. unique_ptr-empty when FP8 is disabled.
  bool use_float8_ = false;
  std::unique_ptr<Float8ScaleState> fp8_qx_, fp8_kx_, fp8_vx_, fp8_ox_;
  std::unique_ptr<Float8ScaleState> fp8_qw_, fp8_kw_, fp8_vw_, fp8_ow_;

  /// A4 — packed-QKV weight cache. torch::cat({w_q, w_k, w_v}) was
  /// rebuilt on every forward; for 125M shapes that's ~3.5 MB of bf16
  /// alloc + copy per layer per forward (~3% of step time). Cache the
  /// result and invalidate via the underlying weights' version counters.
  /// Autograd flows through correctly because we rebuild a fresh cat
  /// node (with up-to-date saved-variable references) whenever any
  /// source weight's version increments — i.e. after every optimizer
  /// step. Within a step the same node is reused across forwards.
  torch::Tensor cached_w_packed_;
  uint32_t cached_w_packed_v_q_ = 0;
  uint32_t cached_w_packed_v_k_ = 0;
  uint32_t cached_w_packed_v_v_ = 0;
  bool     cached_w_packed_valid_ = false;
  torch::Tensor packed_qkv_weight();
};

/// Holder type macro from LibTorch — defines `Attention` as a shared-ptr
/// wrapper around AttentionImpl so it composes cleanly with ModuleList etc.
TORCH_MODULE(Attention);

}  // namespace olmo_cpp
