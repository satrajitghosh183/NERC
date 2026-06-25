/**
 * src/model/rope.cpp
 *
 * Implements Rotary Position Embedding (RoPE) — the relative-position scheme
 * used by LLaMA, OLMo, and most modern LLMs. RoPE encodes absolute position
 * by rotating the (q, k) feature pairs in 2-D subspaces by an angle
 *   theta_d * pos
 * with d-specific frequencies inv_freq[i] = theta^(-2i/d). The dot product
 * q^T k then depends only on the relative offset, giving translation-
 * equivariant attention. This file builds and caches the sin/cos buffers and
 * applies the rotation via a backend-dispatched fused kernel.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/rope.hpp: RotaryEmbeddingImpl declaration, RoPEBuffers
 *     struct (the cached sin/cos pair).
 *   - olmo_cpp/backend/backend.hpp: get_backend().apply_rope() — fused
 *     kernel for the q*cos + rotate_half(q)*sin combination.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp: rope_ = RotaryEmbedding(cfg.get_head_dim(),
 *     cfg.rope_theta); applied to q, k inside MultiHeadAttention.
 *   - src/model/fused_attention.cpp: same wiring for the FusedAttention path.
 *   - src/model/transformer.cpp / fused_transformer.cpp: construct a
 *     RotaryEmbedding once at model build time (line 60 / 58) and feed its
 *     buffers to every block.
 *
 * --- Role in training pipeline ---
 *   Called inside every attention forward (and gradient computation backward
 *   pass). The buffer cache means we recompute sin/cos only when seq_len /
 *   dtype / device changes — typically once per training run, then reused
 *   for every step. apply() is on the hot path of every step.
 */
#include "olmo_cpp/model/rope.hpp"
#include "olmo_cpp/backend/backend.hpp"
#include <cmath>

namespace olmo_cpp {

/// Construct a RoPE module.
/// head_size: per-head feature dimension (must be even — RoPE rotates pairs).
/// theta    : base angular frequency. OLMo and LLaMA-3 use 500000 to
///            comfortably support long contexts; original GPT-NeoX used 10000.
RotaryEmbeddingImpl::RotaryEmbeddingImpl(int64_t head_size, int64_t theta)
    : dim_(head_size), theta_(theta) {}

/// Compute inv_freq[i] = 1 / theta^(2i/dim) for i in [0, dim/2).
/// Returned tensor is FP32 on the requested device. These are the angular
/// frequencies along the dim/2 rotation planes.
torch::Tensor RotaryEmbeddingImpl::compute_inv_freqs(torch::Device device) {
  auto half_dim = dim_ / 2;
  // i = 0..half_dim-1
  auto indices = torch::arange(half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  // Scale to the standard exponent 2i/dim.
  indices = indices * 2.0 / static_cast<double>(dim_);
  // inv_freq = theta^(-2i/dim).
  return 1.0 / torch::pow(static_cast<double>(theta_), indices);
}

/// Half-rotation helper: split last dim into two halves [a; b], return [-b; a].
/// This is the operator that, combined with cos/sin scaling, realizes a 2-D
/// rotation by 90° between the paired components.
torch::Tensor RotaryEmbeddingImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

/// Apply rotation to a tensor `t` using precomputed sin/cos.
/// Output: t * cos + rotate_half(t) * sin. Backend-dispatched so CUDA gets
/// a fused kernel from kernels/rope.cu.
torch::Tensor RotaryEmbeddingImpl::apply_rotary(
    torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
  return get_backend().apply_rope(t, sin, cos);
}

/// Get sin/cos buffers for positions [0, seq_len), dtype-matched to the
/// caller's q/k tensors. Buffers are cached across calls; we only recompute
/// when seq_len exceeds cached_seq_len_, or when device/dtype change.
/// Returned tensors share storage with the cache (slice is a view).
RoPEBuffers RotaryEmbeddingImpl::get_buffers(int64_t seq_len, torch::Device device,
                                              torch::Dtype dtype) {
  // Cache hit: device + dtype match, cached length covers request.
  if (cached_seq_len_ >= seq_len && cached_device_ == device && cached_dtype_ == dtype) {
    if (cached_seq_len_ == seq_len) return cached_bufs_;
    // Cache holds more positions than needed — slice the leading [0, seq_len).
    RoPEBuffers bufs;
    bufs.pos_sin = cached_bufs_.pos_sin.slice(0, 0, seq_len);
    bufs.pos_cos = cached_bufs_.pos_cos.slice(0, 0, seq_len);
    return bufs;
  }

  // Cache miss: rebuild. Allocate at least 2048 positions to amortize cost
  // across short sequences; subsequent calls with seq_len <= alloc_len hit
  // the slice fast path above.
  int64_t alloc_len = std::max(seq_len, static_cast<int64_t>(2048));
  auto inv_freq = compute_inv_freqs(device);
  // seq[t] = t (the position index).
  auto seq = torch::arange(alloc_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  // outer product seq * inv_freq -> [alloc_len, half_dim] of angles.
  auto freqs = seq.unsqueeze(1) * inv_freq.unsqueeze(0);
  // Duplicate across the second half of head_dim so positions has shape
  // [alloc_len, dim] — matches the layout expected by apply_rotary.
  auto positions = torch::cat({freqs, freqs}, -1);
  // Compute sin/cos in FP32 for precision, then cast once to target dtype.
  cached_bufs_.pos_sin = positions.sin().to(dtype);
  cached_bufs_.pos_cos = positions.cos().to(dtype);
  cached_seq_len_ = alloc_len;
  cached_device_ = device;
  cached_dtype_ = dtype;

  // Return exactly seq_len rows so callers don't accidentally read padding.
  RoPEBuffers bufs;
  bufs.pos_sin = cached_bufs_.pos_sin.slice(0, 0, seq_len);
  bufs.pos_cos = cached_bufs_.pos_cos.slice(0, 0, seq_len);
  return bufs;
}

/// Apply RoPE to query and key tensors in attention.
/// q: (B, n_heads, q_len, head_dim), k: (B, n_kv_heads, k_len, head_dim).
/// start_pos: absolute starting position when q_len < k_len (KV-cache decoding).
/// Returns (q_rotated, k_rotated) with the same shapes as inputs.
std::pair<torch::Tensor, torch::Tensor> RotaryEmbeddingImpl::apply(
    torch::Tensor q,
    torch::Tensor k,
    const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  // q: (B, n_heads, q_len, head_dim), k: (B, n_kv_heads, k_len, head_dim)
  auto q_len = q.size(2);
  auto k_len = k.size(2);
  // During training q_len == k_len and we start at pos 0. During incremental
  // decoding q_len < k_len; q rotations need to start at the new position
  // (k_len - q_len), while k rotations start at 0 because k contains the full
  // history. Caller can override via start_pos for prefix prefill semantics.
  int64_t q_abs_start = start_pos ? *start_pos : (k_len - q_len);
  int64_t k_abs_start = start_pos ? *start_pos : 0;

  // unsqueeze(0).unsqueeze(0) broadcasts across (B, n_heads). Slice picks
  // the rows for our actual position window.
  auto sin_q = bufs.pos_sin.slice(0, q_abs_start, q_abs_start + q_len).unsqueeze(0).unsqueeze(0);
  auto cos_q = bufs.pos_cos.slice(0, q_abs_start, q_abs_start + q_len).unsqueeze(0).unsqueeze(0);
  auto sin_k = bufs.pos_sin.slice(0, k_abs_start, k_abs_start + k_len).unsqueeze(0).unsqueeze(0);
  auto cos_k = bufs.pos_cos.slice(0, k_abs_start, k_abs_start + k_len).unsqueeze(0).unsqueeze(0);

  // Buffers are already in target dtype and on target device (set in get_buffers).
  // Backend dispatch: apply_rope is fused (q*cos + rotate_half(q)*sin) on CUDA.
  auto q_rot = apply_rotary(q, sin_q, cos_q);
  auto k_rot = apply_rotary(k, sin_k, cos_k);
  return {q_rot, k_rot};
}

}  // namespace olmo_cpp
