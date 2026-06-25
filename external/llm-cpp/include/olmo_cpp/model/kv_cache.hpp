#pragma once

/**
 * include/olmo_cpp/model/kv_cache.hpp
 *
 * Dynamic per-layer KV cache used by autoregressive decoding. Each layer
 * owns a pre-allocated `(B, n_kv_heads, max_seq_len, head_dim)` buffer for
 * keys and values plus a cursor `len_` recording how many positions are
 * filled. Update appends new K/V into the cursor region and returns views
 * of the filled prefix; on overflow it doubles the buffer (rare path).
 *
 * --- Includes from this project ---
 *   (No project headers — uses only torch/torch.h.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp / fused_attention.cpp: append K/V each
 *     forward when a LayerKVCache pointer is provided
 *   - src/model/transformer.cpp / fused_transformer.cpp: forward_backbone
 *     accepts a KVCache* and routes the per-layer slice into each block
 *   - src/generate/speculative_decode.cpp: snapshot/rollback of cache
 *   - tools/chat.cpp: keeps a KVCache across user turns
 *
 * --- Role in training pipeline ---
 *   Inference / generation only. The training loss path passes nullptr
 *   for kv_cache so attention recomputes K/V from x each step.
 */

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Per-layer KV cache for incremental autoregressive decoding.
/// Uses pre-allocated buffers to avoid per-token allocations.
/// Tensors shaped (B, n_kv_heads, max_len, head_dim), with a cursor
/// tracking how many positions are filled.
struct LayerKVCache {
  torch::Tensor k;  // (B, n_kv_heads, max_len, head_dim) pre-allocated
  torch::Tensor v;  // same shape
  int64_t len_ = 0; // current filled length

  /// Current cached sequence length.
  int64_t seq_len() const { return len_; }

  /// Append new K/V and return views of the filled region.
  /// On first call, allocates a buffer for max_seq_len positions.
  /// Subsequent calls use copy_ into the pre-allocated buffer.
  std::pair<torch::Tensor, torch::Tensor> update(
      torch::Tensor new_k, torch::Tensor new_v, int64_t max_seq_len = 2048) {
    // new_k/new_v are [B, H, new_len, D]; record incoming length.
    int64_t new_len = new_k.size(2);

    if (!k.defined()) {
      // First call: pre-allocate buffer for max_seq_len
      auto opts = new_k.options();
      int64_t B = new_k.size(0);
      int64_t H = new_k.size(1);
      int64_t D = new_k.size(3);
      // Allocate a [B, H, max_seq_len, D] zero buffer for both K and V.
      k = torch::zeros({B, H, max_seq_len, D}, opts);
      v = torch::zeros({B, H, max_seq_len, D}, opts);
      // Copy initial data
      // Place the incoming K/V at the front of the buffer.
      k.narrow(2, 0, new_len).copy_(new_k);
      v.narrow(2, 0, new_len).copy_(new_v);
      len_ = new_len;
    } else if (len_ + new_len <= k.size(2)) {
      // Append into pre-allocated buffer (no allocation)
      // Common path during decode: copy into the [len_, len_+new_len) slot.
      k.narrow(2, len_, new_len).copy_(new_k);
      v.narrow(2, len_, new_len).copy_(new_v);
      len_ += new_len;
    } else {
      // Buffer overflow — grow by 2x (rare path)
      // Compute new capacity = max(2x, exact need). Rare past max_seq_len.
      int64_t new_max = std::max(k.size(2) * 2, len_ + new_len);
      auto new_k_buf = torch::zeros({k.size(0), k.size(1), new_max, k.size(3)}, k.options());
      auto new_v_buf = torch::zeros({v.size(0), v.size(1), new_max, v.size(3)}, v.options());
      // Copy existing valid prefix [0, len_).
      new_k_buf.narrow(2, 0, len_).copy_(k.narrow(2, 0, len_));
      new_v_buf.narrow(2, 0, len_).copy_(v.narrow(2, 0, len_));
      // Append the new K/V right after.
      new_k_buf.narrow(2, len_, new_len).copy_(new_k);
      new_v_buf.narrow(2, len_, new_len).copy_(new_v);
      // Swap buffers; old ones are released by tensor refcount.
      k = new_k_buf;
      v = new_v_buf;
      len_ += new_len;
    }

    // Return views of the filled region
    return {k.narrow(2, 0, len_), v.narrow(2, 0, len_)};
  }

  /// Truncate cache to keep only the first `len` positions.
  void truncate(int64_t len) {
    if (len <= 0) {
      k = torch::Tensor();
      v = torch::Tensor();
      len_ = 0;
    } else if (len < len_) {
      // Just move the cursor back — buffer stays allocated
      len_ = len;
    }
  }
};

/// Full model KV cache — one LayerKVCache per transformer layer.
struct KVCache {
  std::vector<LayerKVCache> layers;

  explicit KVCache(int64_t n_layers, torch::Device device = torch::kCPU)
      : layers(static_cast<size_t>(n_layers)), device_(device) {}

  torch::Device device() const { return device_; }

  /// Current cached sequence length (same across all layers).
  int64_t seq_len() const {
    return layers.empty() ? 0 : layers[0].seq_len();
  }

  /// Reset all cached state.
  void clear() {
    for (auto& l : layers) {
      l.k = torch::Tensor();
      l.v = torch::Tensor();
      l.len_ = 0;
    }
  }

  /// Save a snapshot of the current cache length for later rollback.
  int64_t snapshot() const { return seq_len(); }

  /// Rollback all layers to the given sequence length.
  void rollback(int64_t len) {
    for (auto& l : layers) {
      l.truncate(len);
    }
  }

 private:
  torch::Device device_;
};

}  // namespace olmo_cpp
