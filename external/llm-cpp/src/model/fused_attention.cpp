/**
 * src/model/fused_attention.cpp
 *
 * ─── What "attention" is ────────────────────────────────────────────
 *
 * Self-attention is the operation that lets every token in the
 * sequence look at every other token. Three projections of the same
 * input compete:
 *
 *     Q = X · W_q    (queries:  "what am I looking for?")
 *     K = X · W_k    (keys:     "what do I have to offer?")
 *     V = X · W_v    (values:   "what should I broadcast if you pick me?")
 *
 * Then each query computes a softmax-weighted sum of values, where
 * the weights are dot-products with all keys (scaled by 1/sqrt(d_k)
 * for stability):
 *
 *     attn(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V
 *
 * In multi-head attention you split D = head_dim · n_heads and run
 * the operation n_heads times in parallel on different sub-vectors.
 *
 * GQA (Grouped-Query Attention): n_kv_heads < n_heads — Q has more
 * heads than K/V. K and V are repeated across query-head groups.
 * Saves KV cache memory at inference time with negligible quality loss.
 *
 * ─── What "fused" means here ────────────────────────────────────────
 *
 * Plain `Attention` (in attention.cpp) has THREE separate Linear
 * layers for Q, K, V. Each is a separate matmul kernel launch. When
 * the input is the same X for all three, you can concat their weight
 * matrices into a single [D, q_size + 2·kv_size] Linear and do ONE
 * matmul, then split the output. That's what FusedAttention does.
 *
 * The math is identical; the only difference is fewer launches and
 * better memory locality. Net effect: a few percent faster training
 * step on small models, more on large ones (where launch overhead
 * dominates short kernels).
 *
 * After QKV: this file applies optional QK-RMSNorm (a stability
 * trick — normalising Q and K before the dot product), then RoPE
 * (rotary position embedding via the kernel in kernels/rope.cu),
 * then ATen's scaled_dot_product_attention which itself dispatches
 * to FlashAttention on CUDA.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/fused_attention.hpp : own class declaration.
 *   - olmo_cpp/profiler.hpp              : ProfileScope around the
 *                                           SDPA call when profile=1.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/fused_block.cpp: FusedTransformerBlock instantiates a
 *     FusedAttention and calls forward each microbatch.
 *
 * --- Role in training pipeline ---
 *   Half of every transformer block in the FusedTransformer variant.
 *   The other half is the FFN (feed_forward.cpp).
 */
#include "olmo_cpp/model/fused_attention.hpp"
#include "olmo_cpp/backend/fused_qkv_rope.hpp"   // H3: fused QKV+RoPE kernel
#include "olmo_cpp/backend/cublas_direct.hpp"    // H3: fast_linear (cuBLASLt direct)
#include "olmo_cpp/profiler.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <limits>

namespace olmo_cpp {

FusedAttentionImpl::FusedAttentionImpl(const TransformerConfig& cfg, int64_t /*layer_idx*/)
    : n_heads_(cfg.n_heads),
      n_kv_heads_(cfg.get_n_kv_heads()),
      head_dim_(cfg.get_head_dim()),
      n_heads_rep_(cfg.n_heads / cfg.get_n_kv_heads()),
      q_size_(cfg.n_heads * cfg.get_head_dim()),
      kv_size_(cfg.get_n_kv_heads() * cfg.get_head_dim()),
      use_head_qk_norm_(cfg.use_head_qk_norm),
      sliding_window_size_(cfg.sliding_window_size) {

  // Fused QKV: single weight matrix [d_model, q_size + 2 * kv_size]
  // This halves kernel launches and improves memory locality
  int64_t total_out = q_size_ + 2 * kv_size_;
  w_qkv_ = register_module("w_qkv", torch::nn::Linear(
      torch::nn::LinearOptions(cfg.d_model, total_out).bias(false)));

  w_out_ = register_module("w_out", torch::nn::Linear(
      torch::nn::LinearOptions(q_size_, cfg.d_model).bias(false)));

  if (cfg.use_qk_norm) {
    if (cfg.use_head_qk_norm) {
      q_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
    } else {
      q_norm_ = RMSNorm(q_size_, cfg.layer_norm_eps);
      k_norm_ = RMSNorm(kv_size_, cfg.layer_norm_eps);
    }
    register_module("q_norm", q_norm_.value());
    register_module("k_norm", k_norm_.value());
  }

  rope_ = RotaryEmbedding(cfg.get_head_dim(), cfg.rope_theta);
  register_module("rope", rope_.value());
}

torch::Tensor FusedAttentionImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {

  auto B = x.size(0);
  auto S = x.size(1);

  // ── H3: fully fused QKV+RoPE kernel + cuBLASLt-direct out-proj ──
  // Mirrors AttentionImpl's hot path. One launch produces q/k/v in head-major
  // layout with RoPE applied (replaces the ATen Linear + split + reshape +
  // separate rope below), and the output projection skips the ATen dispatcher
  // via fast_linear. Only when on CUDA, no QK-norm, RoPE active, no QKV bias,
  // and not in sliding-window mode (that path keeps its cached mask below).
  const bool can_use_fused_qkv =
      x.is_cuda() && !q_norm_ && !k_norm_ && rope_bufs != nullptr &&
      !w_qkv_->bias.defined() && sliding_window_size_ <= 0;
  if (can_use_fused_qkv) {
    const int64_t s_offset = start_pos.value_or(0);
    auto cos_full = rope_bufs->pos_cos.narrow(0, s_offset, S);
    auto sin_full = rope_bufs->pos_sin.narrow(0, s_offset, S);
    auto cos_half = cos_full.narrow(-1, 0, head_dim_ / 2);
    auto sin_half = sin_full.narrow(-1, 0, head_dim_ / 2);
    auto out = fused_qkv_rope_autograd(x, w_qkv_->weight, cos_half, sin_half,
                                       n_heads_, n_kv_heads_, head_dim_);
    auto q = std::get<0>(out);  // [B, n_q,  S, head_dim] with RoPE applied
    auto k = std::get<1>(out);
    auto v = std::get<2>(out);
    if (layer_cache) {
      auto [full_k, full_v] = layer_cache->update(k, v);
      k = full_k;
      v = full_v;
    }
    if (n_heads_rep_ > 1) {
      auto kS = k.size(2);
      k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_})
           .reshape({B, n_heads_, kS, head_dim_});
      v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_})
           .reshape({B, n_heads_, kS, head_dim_});
    }
    bool is_causal_fused =
        (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
    auto attn_out =
        at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal_fused);
    attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
    return fast_linear(attn_out, w_out_->weight,
                       w_out_->bias.defined() ? w_out_->bias : torch::Tensor());
  }

  // ── Fused QKV projection (1 GEMM instead of 3) ──
  auto qkv = w_qkv_(x);  // [B, S, q_size + 2 * kv_size]

  // Split into Q, K, V without copying (narrow views)
  auto q = qkv.narrow(2, 0, q_size_);
  auto k = qkv.narrow(2, q_size_, kv_size_);
  auto v = qkv.narrow(2, q_size_ + kv_size_, kv_size_);

  // QK-norm before reshape (if not per-head)
  if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);

  // Reshape to [B, n_heads, S, head_dim]
  q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
  k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);

  // QK-norm per head after reshape
  if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);

  // ── RoPE ──
  if (rope_bufs) {
    auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, start_pos);
    q = q_rot;
    k = k_rot;
  }

  // ── KV Cache ──
  if (layer_cache) {
    auto [full_k, full_v] = layer_cache->update(k, v);
    k = full_k;
    v = full_v;
  }

  // ── Efficient GQA ──
  // Instead of repeat_interleave (allocates new tensor), use expand+reshape
  // which creates a view with stride tricks (zero-copy)
  if (n_heads_rep_ > 1) {
    // k: [B, n_kv_heads, S_full, head_dim]
    // expand to [B, n_kv_heads, n_heads_rep, S_full, head_dim]
    // then reshape to [B, n_heads, S_full, head_dim]
    auto full_S = k.size(2);
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, full_S, head_dim_})
         .reshape({B, n_heads_, full_S, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, full_S, head_dim_})
         .reshape({B, n_heads_, full_S, head_dim_});
  }

  // ── Attention ──
  bool is_causal = (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
  auto full_S = k.size(2);

  torch::Tensor attn_out;

  if (sliding_window_size_ > 0 && S > 1) {
    // Reuse cached mask when dimensions haven't changed. Same fix as in
    // attention.cpp: build the band mask via tril/triu (one kernel each)
    // instead of two arange tensors and a chain of boolean ops, and use
    // rank-0 zero/-inf scalars in torch::where to skip allocating two SxS
    // tensors only to throw one away.
    if (S != cached_mask_S_ || full_S != cached_mask_full_S_) {
      const auto offset = full_S - S;
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, full_S}, bool_opts);
      auto allowed   = torch::tril(ones, offset)
                         & torch::triu(ones, offset - sliding_window_size_);

      auto mask_opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
      cached_attn_mask_ = torch::where(
          allowed,
          torch::zeros({}, mask_opts),
          torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
      cached_mask_S_      = S;
      cached_mask_full_S_ = full_S;
    }

    attn_out = at::scaled_dot_product_attention(q, k, v, cached_attn_mask_, 0.0, false);
  } else {
    // On CUDA this dispatches to FlashAttention2 automatically
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  // ── Output projection ──
  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

}  // namespace olmo_cpp
