/**
 * include/olmo_cpp/model/block_manager.hpp
 *
 * Block manager for paged KV cache (fast-inference [9]).
 *
 * Maintains a pool of fixed-size pages of K/V memory. Each request gets
 * a "page table" mapping logical block index → physical page index.
 * Append-only: when a request's last page fills, allocate a new one;
 * never compact or rebalance.
 *
 * DRAFT — not yet wired into the attention forward path. The full
 * integration requires changing every attention block to use the paged
 * gather kernel instead of contiguous K/V tensors. That refactor is
 * deferred; this file lets the kernel + allocator exist and be tested
 * standalone.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace olmo_cpp {

/// Per-layer page pool plus a per-request page table.
/// One BlockManager per decode session. NOT thread-safe (single stream).
class BlockManager {
 public:
  BlockManager(int64_t n_layers,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t page_size,
               int64_t max_pages,
               torch::Device device,
               torch::Dtype dtype);

  /// Allocate `n` new pages for this request. Returns physical page indices.
  std::vector<int32_t> allocate(int64_t n);

  /// Free all pages held by this request (resets the request's page table).
  void free_all();

  /// Return the K page pool for the given layer: shape
  /// [max_pages, page_size, n_kv_heads, head_dim].
  torch::Tensor& k_pool(int64_t layer) { return k_pools_[static_cast<size_t>(layer)]; }
  torch::Tensor& v_pool(int64_t layer) { return v_pools_[static_cast<size_t>(layer)]; }

  /// Read-only accessors for the layout constants.
  int64_t page_size() const { return page_size_; }
  int64_t max_pages() const { return max_pages_; }
  int64_t n_kv_heads() const { return n_kv_heads_; }
  int64_t head_dim()  const { return head_dim_; }
  torch::Device device() const { return device_; }

  /// Current page table for the active request.
  const std::vector<int32_t>& page_table() const { return page_table_; }

 private:
  int64_t page_size_;
  int64_t max_pages_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  torch::Device device_;

  // Allocator state: a freelist of free page indices.
  std::vector<int32_t> free_list_;
  // Page table for the (single) active request: logical block → physical page.
  std::vector<int32_t> page_table_;

  // One [max_pages, page_size, n_kv_heads, head_dim] buffer per layer.
  std::vector<torch::Tensor> k_pools_;
  std::vector<torch::Tensor> v_pools_;
};

}  // namespace olmo_cpp
