/**
 * src/model/attention.cpp
 *
 * Implementation of the un-fused multi-head / grouped-query self-attention
 * module (AttentionImpl). Constructs Q, K, V, output linear projections,
 * optional QK-RMSNorm, and RoPE, then computes attention via ATen's
 * scaled_dot_product_attention (which dispatches to FlashAttention on
 * CUDA where supported). Sliding-window masks are built lazily and cached
 * across forwards while their dimensions stay constant.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/attention.hpp: own header (declares AttentionImpl,
 *     RoPE, RMSNorm, KVCache types)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp: ReorderedNormTransformerBlockImpl::forward
 *     calls Attention(...) — i.e., this forward — once per layer
 *   - src/model/block_variants.cpp: PeriNorm/LayerNormScaled/Normalized
 *     /MoE block forwards likewise dispatch through Attention(...)
 *
 * --- Role in training pipeline ---
 *   On the un-fused training path, this is the per-layer attention
 *   compute. One call per layer per forward; gradients flow back through
 *   ATen's autograd over the SDPA op and the linear layers.
 */
#include "olmo_cpp/model/attention.hpp"
#include "olmo_cpp/backend/paged_attention.hpp"
#include "olmo_cpp/backend/cublas_direct.hpp"
#include "olmo_cpp/backend/fused_qkv_rope.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>
#include <limits>

namespace olmo_cpp {

/// Construct Q/K/V/output linears, optional QK-norms, and RoPE module.
/// The K/V projections are sized for n_kv_heads (GQA-aware), while the
/// Q and output projections use the full n_heads count.
// A4 — return the cached packed [w_q ; w_k ; w_v] weight, rebuilding only
// when any source weight has incremented its version counter (i.e. after
// the optimizer touched it). torch::cat itself stays in the autograd
// graph so grad_w_packed splits back into grad_w_q / grad_w_k / grad_w_v
// correctly. The savings come from skipping the bandwidth-bound concat
// (≈3.5 MB / layer / forward at 125M) and its dispatcher overhead on
// every forward within an unchanged-weights window.
torch::Tensor AttentionImpl::packed_qkv_weight() {
  const uint32_t v_q = w_q_->weight._version();
  const uint32_t v_k = w_k_->weight._version();
  const uint32_t v_v = w_v_->weight._version();
  if (cached_w_packed_valid_ &&
      v_q == cached_w_packed_v_q_ &&
      v_k == cached_w_packed_v_k_ &&
      v_v == cached_w_packed_v_v_) {
    return cached_w_packed_;
  }
  cached_w_packed_ = torch::cat({w_q_->weight, w_k_->weight, w_v_->weight}, /*dim=*/0);
  cached_w_packed_v_q_ = v_q;
  cached_w_packed_v_k_ = v_k;
  cached_w_packed_v_v_ = v_v;
  cached_w_packed_valid_ = true;
  return cached_w_packed_;
}

AttentionImpl::AttentionImpl(const TransformerConfig& cfg, int64_t /*layer_idx*/)
    : w_q_(register_module("w_q", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.n_heads * cfg.get_head_dim()).bias(false)))),
      w_k_(register_module("w_k", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.get_n_kv_heads() * cfg.get_head_dim()).bias(false)))),
      w_v_(register_module("w_v", torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model, cfg.get_n_kv_heads() * cfg.get_head_dim()).bias(false)))),
      w_out_(register_module("w_out", torch::nn::Linear(torch::nn::LinearOptions(cfg.n_heads * cfg.get_head_dim(), cfg.d_model).bias(false)))),
      n_heads_(cfg.n_heads),
      n_kv_heads_(cfg.get_n_kv_heads()),
      head_dim_(cfg.get_head_dim()),
      n_heads_rep_(cfg.n_heads / cfg.get_n_kv_heads()),
      use_head_qk_norm_(cfg.use_head_qk_norm),
      sliding_window_size_(cfg.sliding_window_size) {
  // Optional QK-norm: stabilizes attention scores at large depths/widths.
  if (cfg.use_qk_norm) {
    if (cfg.use_head_qk_norm) {
      // Per-head RMSNorm: parameter vector has length head_dim.
      q_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_head_dim(), cfg.layer_norm_eps);
    } else {
      // Per-tensor RMSNorm: applied before head reshape over the full
      // n_heads*head_dim (Q) or n_kv_heads*head_dim (K) feature width.
      q_norm_ = RMSNorm(cfg.n_heads * cfg.get_head_dim(), cfg.layer_norm_eps);
      k_norm_ = RMSNorm(cfg.get_n_kv_heads() * cfg.get_head_dim(), cfg.layer_norm_eps);
    }
    // Register so they get serialized with the module hierarchy.
    register_module("q_norm", q_norm_.value());
    register_module("k_norm", k_norm_.value());
  }
  // RoPE module is always created (gated at forward by rope_bufs != null).
  rope_ = RotaryEmbedding(cfg.get_head_dim(), cfg.rope_theta);
  register_module("rope", rope_.value());

  // FP8 emulation state (I-5 / T-6). One amax-history tracker per Linear's
  // input and per Linear's weight. Allocated only when FP8 is enabled in
  // the config so the disabled path pays zero extra memory.
  use_float8_ = cfg.use_float8;
  if (use_float8_) {
    fp8_qx_ = std::make_unique<Float8ScaleState>(16);
    fp8_kx_ = std::make_unique<Float8ScaleState>(16);
    fp8_vx_ = std::make_unique<Float8ScaleState>(16);
    fp8_ox_ = std::make_unique<Float8ScaleState>(16);
    fp8_qw_ = std::make_unique<Float8ScaleState>(16);
    fp8_kw_ = std::make_unique<Float8ScaleState>(16);
    fp8_vw_ = std::make_unique<Float8ScaleState>(16);
    fp8_ow_ = std::make_unique<Float8ScaleState>(16);
  }
}

torch::Tensor AttentionImpl::forward(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    std::optional<int64_t> start_pos,
    LayerKVCache* layer_cache) {
  auto B = x.size(0);
  auto S = x.size(1);

  // Hottest path: CUDA + no FP8 STE + no QK-norm + RoPE active.
  // Call the fully fused QKV+RoPE kernel (item G). One launch produces
  // q/k/v already in head-major layout with RoPE applied. Replaces
  // 3 Linears + 3 reshapes + 1 RoPE = 7 ATen calls.
  torch::Tensor q, k, v;
  const bool can_use_fused_qkv =
      x.is_cuda() && !use_float8_ && !q_norm_ && !k_norm_ && rope_bufs != nullptr;
  if (can_use_fused_qkv) {
    auto w_packed = packed_qkv_weight();
    // RoPEBuffers stores pos_cos/pos_sin as [seq_len, head_dim] full-dim
    // broadcast; the fused kernel takes [S, head_dim/2] (first half is
    // enough under half-rotation).
    const int64_t s_offset = start_pos.value_or(0);
    auto cos_full = rope_bufs->pos_cos.narrow(0, s_offset, S);
    auto sin_full = rope_bufs->pos_sin.narrow(0, s_offset, S);
    auto cos_half = cos_full.narrow(-1, 0, head_dim_ / 2);
    auto sin_half = sin_full.narrow(-1, 0, head_dim_ / 2);
    auto out = fused_qkv_rope_autograd(x, w_packed, cos_half, sin_half,
                                         n_heads_, n_kv_heads_, head_dim_);
    q = std::get<0>(out);  // [B, n_q,  S, head_dim] with RoPE
    k = std::get<1>(out);
    v = std::get<2>(out);
    // GQA expand + cache append + SDPA all happen below. Skip RoPE
    // (already applied) and the projection lines that follow.
    if (layer_cache) {
      auto [full_k, full_v] = layer_cache->update(k, v);
      k = full_k;
      v = full_v;
    }
    if (n_heads_rep_ > 1) {
      auto kS = k.size(2);
      k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
      v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
    }
    bool is_causal_fused = (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
    auto attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal_fused);
    attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
    return fast_linear(attn_out, w_out_->weight,
                       w_out_->bias.defined() ? w_out_->bias : torch::Tensor());
  }

  // Packed-QKV path (CPU or QK-norm active): still concat weights and do
  // one Linear, just don't use the fused-with-RoPE kernel. Saves 2 of
  // the 3 Linear launches without changing numerics.
  if (!use_float8_) {
    auto w_packed = packed_qkv_weight();
    auto qkv = fast_linear(x, w_packed, torch::Tensor());
    const int64_t q_dim  = n_heads_    * head_dim_;
    const int64_t kv_dim = n_kv_heads_ * head_dim_;
    q = qkv.narrow(-1, 0,                  q_dim);
    k = qkv.narrow(-1, q_dim,              kv_dim);
    v = qkv.narrow(-1, q_dim + kv_dim,     kv_dim);
  } else {
    // FP8 STE path: each Linear runs through float8_linear_emulated with
    // its own scale tracker; can't pack-and-share because each has its
    // own per-tensor scale.
    auto linear_fp8 = [&](torch::nn::Linear& lin,
                          Float8ScaleState* sx, Float8ScaleState* sw,
                          const torch::Tensor& in) -> torch::Tensor {
      return float8_linear_emulated(in, lin->weight,
                                    lin->bias.defined() ? lin->bias : torch::Tensor(),
                                    *sx, *sw);
    };
    q = linear_fp8(w_q_, fp8_qx_.get(), fp8_qw_.get(), x);
    k = linear_fp8(w_k_, fp8_kx_.get(), fp8_kw_.get(), x);
    v = linear_fp8(w_v_, fp8_vx_.get(), fp8_vw_.get(), x);
  }

  // QK-norm before reshape
  if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);

  q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
  k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);

  // QK-norm per head after reshape
  if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);

  if (rope_bufs) {
    auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, start_pos);
    q = q_rot;
    k = k_rot;
  }

  if (layer_cache) {
    auto [full_k, full_v] = layer_cache->update(k, v);
    k = full_k;
    v = full_v;
  }

  if (n_heads_rep_ > 1) {
    // Use expand (view-only, no allocation) instead of repeat_interleave (copies)
    auto kS = k.size(2);
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
  }

  bool is_causal = (S > 1) && (layer_cache == nullptr || layer_cache->seq_len() == S);
  auto full_S = k.size(2);

  torch::Tensor attn_out;

  if (sliding_window_size_ > 0 && S > 1) {
    // Reuse cached mask when dimensions haven't changed.
    if (S != cached_mask_S_ || full_S != cached_mask_full_S_) {
      // The mask we want is the band of diagonals [offset - window, offset]
      // (causal upper bound at offset, lower bound at offset - window). Build
      // it as a single bool kernel via tril/triu so we don't allocate two
      // arange tensors and pile up boolean ops. The previous implementation
      // also had a no-op refinement: the original `mask` already enforced
      // `cols <= (rows + offset)`, so the `if (is_causal)` clause re-ANDed
      // an identical condition for free.
      const auto offset = full_S - S;
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, full_S}, bool_opts);
      auto allowed   = torch::tril(ones, offset)
                         & torch::triu(ones, offset - sliding_window_size_);

      // Mask dtype must match the activation dtype so SDPA doesn't reject a
      // BF16 q/k/v with an FP32 attention bias (same class of bug as the one
      // that killed 7B startup). torch::where with rank-0 tensors avoids the
      // alloc-of-two-SxS-tensors-then-discard pattern the old code used.
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
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return use_float8_
      ? float8_linear_emulated(attn_out, w_out_->weight,
                               w_out_->bias.defined() ? w_out_->bias : torch::Tensor(),
                               *fp8_ox_, *fp8_ow_)
      : fast_linear(attn_out, w_out_->weight,
                    w_out_->bias.defined() ? w_out_->bias : torch::Tensor());
}

// Tree-attention forward (item 8.1). Standard projections + RoPE, then
// SDPA with the caller-supplied additive 2-D mask. No cache.
torch::Tensor AttentionImpl::forward_with_mask(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    torch::Tensor attn_mask) {
  auto B = x.size(0);
  auto S = x.size(1);
  auto q = w_q_(x);
  auto k = w_k_(x);
  auto v = w_v_(x);
  if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);
  q = q.view({B, S, n_heads_,   head_dim_}).transpose(1, 2);
  k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
  if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
  if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);
  if (rope_bufs) {
    auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, std::optional<int64_t>(0));
    q = q_rot; k = k_rot;
  }
  if (n_heads_rep_ > 1) {
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, S, head_dim_})
            .reshape({B, n_heads_, S, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, S, head_dim_})
            .reshape({B, n_heads_, S, head_dim_});
  }
  auto attn_out = at::scaled_dot_product_attention(q, k, v, attn_mask, 0.0, false);
  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

// ──────────────────────────────────────────────────────────────────────────
// Paged-KV variant. Differs from `forward` only in cache I/O — the rest of
// the path (projections, QK-norm, RoPE, GQA expand, sliding-window mask,
// SDPA) is identical. Kept as a parallel function so the legacy concat
// path stays untouched.
// ──────────────────────────────────────────────────────────────────────────
torch::Tensor AttentionImpl::forward_paged(
    torch::Tensor x,
    const RoPEBuffers* rope_bufs,
    int64_t start_pos,
    IPagedKVCache* paged,
    int64_t layer_idx) {
  TORCH_CHECK(paged != nullptr, "AttentionImpl::forward_paged: paged is null");

  auto B = x.size(0);
  auto S = x.size(1);

  torch::Tensor q, k, v;

  // Fused QKV + reshape + RoPE in one kernel — the same fast path the
  // training-side `forward` uses, dropped into the inference (paged) path.
  // Replaces 3 Linears + 3 reshapes + 1 RoPE with a single launch.
  const bool can_use_fused_qkv =
      x.is_cuda() && !use_float8_ && !q_norm_ && !k_norm_ && rope_bufs != nullptr;
  if (can_use_fused_qkv) {
    auto w_packed = packed_qkv_weight();
    auto cos_full = rope_bufs->pos_cos.narrow(0, start_pos, S);
    auto sin_full = rope_bufs->pos_sin.narrow(0, start_pos, S);
    auto cos_half = cos_full.narrow(-1, 0, head_dim_ / 2);
    auto sin_half = sin_full.narrow(-1, 0, head_dim_ / 2);
    auto out = fused_qkv_rope(x, w_packed, cos_half, sin_half,
                                n_heads_, n_kv_heads_, head_dim_);
    q = std::get<0>(out);
    k = std::get<1>(out);
    v = std::get<2>(out);
  } else {
    auto linear_maybe_fp8 = [&](torch::nn::Linear& lin,
                                Float8ScaleState* sx, Float8ScaleState* sw,
                                const torch::Tensor& in) -> torch::Tensor {
      if (!use_float8_) return fast_linear(in, lin->weight,
                                            lin->bias.defined() ? lin->bias : torch::Tensor());
      return float8_linear_emulated(in, lin->weight,
                                    lin->bias.defined() ? lin->bias : torch::Tensor(),
                                    *sx, *sw);
    };
    q = linear_maybe_fp8(w_q_, fp8_qx_.get(), fp8_qw_.get(), x);
    k = linear_maybe_fp8(w_k_, fp8_kx_.get(), fp8_kw_.get(), x);
    v = linear_maybe_fp8(w_v_, fp8_vx_.get(), fp8_vw_.get(), x);

    if (q_norm_ && !use_head_qk_norm_) q = (*q_norm_)(q);
    if (k_norm_ && !use_head_qk_norm_) k = (*k_norm_)(k);

    q = q.view({B, S, n_heads_, head_dim_}).transpose(1, 2);
    k = k.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);
    v = v.view({B, S, n_kv_heads_, head_dim_}).transpose(1, 2);

    if (q_norm_ && use_head_qk_norm_) q = (*q_norm_)(q);
    if (k_norm_ && use_head_qk_norm_) k = (*k_norm_)(k);

    if (rope_bufs) {
      auto [q_rot, k_rot] = (*rope_)->apply(q, k, *rope_bufs, std::optional<int64_t>(start_pos));
      q = q_rot;
      k = k_rot;
    }
  }

  // Append new K/V into the page pool. From here we either:
  //  (1) dispatch the single-query paged kernel (S==1, sliding window off,
  //      real BlockManager-backed pools), or
  //  (2) materialize the full cached K/V and run SDPA (prefill, sliding
  //      window, or shim-backed caches).
  paged->append(layer_idx, k, v);

  const bool kernel_eligible =
      (S == 1) && (sliding_window_size_ <= 0) && paged->has_page_table();
  if (kernel_eligible) {
    // q is [1, n_heads, 1, head_dim]; the kernel wants [n_q_heads, head_dim].
    // It handles GQA internally (maps q_head -> kv_head) and reads K/V
    // straight out of the page pool via the page table — no materialize.
    //
    // For an INT4-backed cache (4× memory drop) the kernel
    // dequantizes inline in the dot product; otherwise the bf16 _dyn
    // kernel reads the raw pool tensor. Both are graph-capture-safe via
    // the stable n_tokens scalar.
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto q2 = q.select(0, 0).select(1, 0).contiguous();           // [n_heads, head_dim]
    torch::Tensor attn_flat;
    if (paged->is_int4()) {
      attn_flat = paged_attention_decode_int4(
          q2,
          paged->k_pool(layer_idx),    paged->k_scales(layer_idx),
          paged->v_pool(layer_idx),    paged->v_scales(layer_idx),
          paged->page_table_tensor_stable(),
          paged->n_tokens_tensor(),
          sm_scale);
    } else {
      attn_flat = paged_attention_decode_dyn(
          q2,
          paged->k_pool(layer_idx),
          paged->v_pool(layer_idx),
          paged->page_table_tensor_stable(),
          paged->n_tokens_tensor(),
          sm_scale);
    }
    auto attn_out_one = attn_flat.view({B, S, n_heads_ * head_dim_});
    return use_float8_
        ? float8_linear_emulated(attn_out_one, w_out_->weight,
                                 w_out_->bias.defined() ? w_out_->bias : torch::Tensor(),
                                 *fp8_ox_, *fp8_ow_)
        : fast_linear(attn_out_one, w_out_->weight,
                       w_out_->bias.defined() ? w_out_->bias : torch::Tensor());
  }

  auto [full_k, full_v] = paged->materialize(layer_idx);
  k = full_k;
  v = full_v;

  if (n_heads_rep_ > 1) {
    auto kS = k.size(2);
    k = k.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
    v = v.unsqueeze(2).expand({B, n_kv_heads_, n_heads_rep_, kS, head_dim_}).reshape({B, n_heads_, kS, head_dim_});
  }

  const int64_t total_S = k.size(2);
  // Three cases:
  //  (a) S == 1: decode. q's single position attends to all of [0, total_S).
  //      SDPA with is_causal=false is correct (the new K/V is at the end).
  //  (b) S > 1, prefix == 0 (i.e. total_S == S): unchunked prefill. SDPA's
  //      is_causal handles the standard square causal mask.
  //  (c) S > 1, prefix > 0 (total_S > S): chunked prefill. q's position i
  //      corresponds to absolute position prefix + i and must mask out k[j]
  //      for j > prefix + i. We build the shifted-causal mask explicitly
  //      rather than rely on SDPA's L<S is_causal semantics, which only
  //      landed in PyTorch 2.1+ and is brittle across backends.
  const int64_t prefix_len = total_S - S;
  bool is_causal = (S > 1) && (prefix_len == 0);

  torch::Tensor attn_out;
  if (sliding_window_size_ > 0 && S > 1) {
    if (S != cached_mask_S_ || total_S != cached_mask_full_S_) {
      const auto offset = total_S - S;
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, total_S}, bool_opts);
      auto allowed   = torch::tril(ones, offset)
                         & torch::triu(ones, offset - sliding_window_size_);
      auto mask_opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
      cached_attn_mask_ = torch::where(
          allowed,
          torch::zeros({}, mask_opts),
          torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
      cached_mask_S_      = S;
      cached_mask_full_S_ = total_S;
    }
    attn_out = at::scaled_dot_product_attention(q, k, v, cached_attn_mask_, 0.0, false);
  } else if (S > 1 && prefix_len > 0) {
    // Chunked-prefill mask: rows [0, S), cols [0, total_S). Allow
    // k[j] iff j <= prefix_len + i. Build once per (S, total_S) pair —
    // cached_attn_mask_ is reused if dims unchanged across calls.
    if (S != cached_mask_S_ || total_S != cached_mask_full_S_) {
      auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(x.device());
      auto ones      = torch::ones({S, total_S}, bool_opts);
      auto allowed   = torch::tril(ones, prefix_len);   // diag offset = prefix_len
      auto mask_opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
      cached_attn_mask_ = torch::where(
          allowed,
          torch::zeros({}, mask_opts),
          torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));
      cached_mask_S_      = S;
      cached_mask_full_S_ = total_S;
    }
    attn_out = at::scaled_dot_product_attention(q, k, v, cached_attn_mask_, 0.0, false);
  } else {
    attn_out = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, is_causal);
  }

  attn_out = attn_out.transpose(1, 2).reshape({B, S, -1});
  return w_out_(attn_out);
}

}  // namespace olmo_cpp
