#pragma once

/**
 * include/olmo_cpp/model/shared_block_pool.hpp
 *
 * Shared, ref-counted, content-hashed page pool for paged KV caching.
 * Single owner of the per-layer K/V tensors. Multiple PrefixCachingKVCache
 * instances point into it via their own logical page tables.
 *
 * Two primitives:
 *
 *   1. allocate(): hand out a fresh page index from the freelist (ref=1).
 *      Used for tokens whose K/V is about to be computed for the first
 *      time.
 *
 *   2. find_or_lease(prev_hash, token_id_block): hash chain lookup. If a
 *      page already exists for this token-id block continuation, return
 *      its index and bump its refcount. Otherwise allocate fresh and
 *      register it under that hash. This is the prefix-caching
 *      fast-path: the second request with the same prompt prefix gets
 *      back the first request's already-populated pages.
 *
 * Release: dec_ref() pages whose refcount drops to 0 return to the
 * freelist; their hash entry is invalidated. This means populated pages
 * with refcount > 0 outlive a single request, which is the whole point.
 *
 * Concurrency: not thread-safe. A serving scheduler must serialize calls.
 *
 * --- Role in fast-inference [I-7 / I-2] ---
 *   Foundation for prefix caching (I-7): two requests with a shared
 *   system prompt land on the same physical pages and skip recomputing
 *   their K/V. Also foundation for continuous batching (I-2): multiple
 *   in-flight requests each have their own page table referencing into
 *   the same pool.
 */

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

class SharedBlockPool {
 public:
  /// Construct a pool with `max_pages` physical pages per layer.
  /// Each layer's pool tensor is shape
  ///   [max_pages, page_size, n_kv_heads, head_dim]
  /// on `device` with `dtype`. Pages are reusable across logical
  /// positions; lookup is via content (token-id-block) hash.
  SharedBlockPool(int64_t n_layers,
                  int64_t n_kv_heads,
                  int64_t head_dim,
                  int64_t page_size,
                  int64_t max_pages,
                  torch::Device device,
                  torch::Dtype dtype);

  /// Allocate a fresh page. Returns its physical index with refcount=1.
  /// Throws if the freelist is empty.
  int32_t allocate();

  /// Hash chain lookup: try to find a page whose token-ids exactly match
  /// `token_id_block` AND whose preceding-block hash is `prev_hash`. On
  /// hit: bump refcount, return the existing physical page index and
  /// (was_hit=true). On miss: allocate fresh and register; return
  /// (was_hit=false). `out_hash` receives the combined hash for chaining
  /// to the next block.
  struct LeaseResult {
    int32_t page_idx;
    bool was_hit;
    uint64_t hash;
  };
  LeaseResult find_or_lease(uint64_t prev_hash,
                            const int32_t* token_ids,
                            int64_t n_tokens);

  /// Bump the refcount on an existing page (caller already knows the
  /// index because it was returned by find_or_lease in a prior request).
  void inc_ref(int32_t page_idx);

  /// Drop a reference; if refcount hits 0, the page returns to the
  /// freelist and its hash entry is dropped.
  void dec_ref(int32_t page_idx);

  /// Drop a vector of pages all at once (the typical "request done"
  /// path).
  void release_all(const std::vector<int32_t>& page_indices);

  /// Layer pool accessors — kernel-facing.
  torch::Tensor& k_pool(int64_t layer) { return k_pools_[static_cast<size_t>(layer)]; }
  torch::Tensor& v_pool(int64_t layer) { return v_pools_[static_cast<size_t>(layer)]; }

  int64_t page_size() const  { return page_size_; }
  int64_t max_pages() const  { return max_pages_; }
  int64_t n_kv_heads() const { return n_kv_heads_; }
  int64_t head_dim() const   { return head_dim_; }
  int64_t n_layers() const   { return static_cast<int64_t>(k_pools_.size()); }
  torch::Device device() const { return device_; }
  torch::Dtype  dtype()  const { return dtype_; }

  /// Number of currently-allocated pages (debug + tests).
  int64_t allocated_count() const {
    return max_pages_ - static_cast<int64_t>(free_list_.size());
  }

 private:
  // Single-page hash: combine token-ids + prev_hash via FNV-1a-style
  // mixing. Fast, no allocations, good enough collision resistance.
  static uint64_t hash_block(uint64_t prev, const int32_t* ids, int64_t n);

  int64_t page_size_;
  int64_t max_pages_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  torch::Device device_;
  torch::Dtype  dtype_;

  std::vector<torch::Tensor> k_pools_;  // one [max_pages, page_size, ...] tensor per layer
  std::vector<torch::Tensor> v_pools_;

  std::vector<int32_t>  free_list_;
  std::vector<int32_t>  ref_counts_;    // size = max_pages_, indexed by physical page
  std::vector<uint64_t> page_hashes_;   // hash registered for each allocated page
  std::unordered_map<uint64_t, int32_t> hash_to_page_;  // for find_or_lease
};

}  // namespace olmo_cpp
