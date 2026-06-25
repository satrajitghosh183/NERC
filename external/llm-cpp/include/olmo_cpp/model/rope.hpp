#pragma once
/**
 * include/olmo_cpp/model/rope.hpp
 *
 * Host-side RoPE. Owns:
 *   - the precomputed cos/sin tables (RoPEBuffers struct), and
 *   - the public apply(q, k) helper that calls into the fused
 *     CUDA / SIMD kernels via get_backend().apply_rope_qk().
 *
 * The tables are computed ONCE at model construction from
 * cfg.rope_theta + cfg.head_dim + max sequence length, possibly
 * post-processed by a scaler from rope_scaling.hpp.
 *
 * For the math behind rotary position embeddings, see
 * kernels/rope.cu's docblock — that's the most pedagogical
 * description in the codebase.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/rope.cpp : implementation.
 *   - src/model/attention.cpp / fused_attention.cpp : pull
 *     RoPEBuffers from the parent model and apply them to Q and K
 *     just before the SDPA call.
 *
 * --- Role in training pipeline ---
 *   Foundational — every attention block needs this on every forward.
 */

#include <torch/torch.h>
#include <memory>
#include <optional>
#include <unordered_map>

namespace olmo_cpp {

// Shared dtype-cast cache. RoPEBuffers holds a shared_ptr to one of these
// so when the trainer broadcasts the buffer struct to every layer
// (`assign(n_layers, bufs)` in transformer.cpp), every layer's view shares
// the *same* cast cache. Without this, the first layer's apply() fills its
// own cache slot but layers 1..n_layers-1 each pay the cast independently.
struct RoPECastCache {
  torch::Tensor pos_sin_cast;
  torch::Tensor pos_cos_cast;
  torch::Dtype  dtype = torch::kFloat32;
};

struct RoPEBuffers {
  torch::Tensor pos_sin;  // (seq_len, head_dim) — built in compute dtype
                          // (typically FP32 for stability).
  torch::Tensor pos_cos;
  // Lazy per-target-dtype cache. Scaled RoPE variants build the master
  // buffers in FP32 unconditionally for numerical stability; under bf16
  // training/inference, downstream apply() needs a bf16-cast version on
  // every call. Caching that cast here turns repeated `.to(q.dtype())`
  // ops into a single cast on the first miss.
  // shared_ptr so all layer-views point at one cache — first miss fills it,
  // every subsequent layer hits.
  std::shared_ptr<RoPECastCache> cast = std::make_shared<RoPECastCache>();
};

/// Rotary Position Embedding (RoPE)
/// inv_freq[i] = 1 / (theta^(2i/d))
/// Apply: output = x * cos + rotate_half(x) * sin
class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int64_t head_size, int64_t theta = 500000);

  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device,
                          torch::Dtype dtype = torch::kFloat32);

  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q,
      torch::Tensor k,
      const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_;
  int64_t theta_;
  // Cached buffers — avoid recomputing sin/cos every forward pass
  int64_t cached_seq_len_ = 0;
  torch::Device cached_device_ = torch::kCPU;
  torch::Dtype cached_dtype_ = torch::kFloat32;
  RoPEBuffers cached_bufs_;

  torch::Tensor compute_inv_freqs(torch::Device device);
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin, torch::Tensor cos);
};

TORCH_MODULE(RotaryEmbedding);

}  // namespace olmo_cpp
