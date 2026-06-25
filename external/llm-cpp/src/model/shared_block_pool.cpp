/**
 * src/model/shared_block_pool.cpp
 *
 * Implementation of the shared, ref-counted, content-hashed page pool.
 * Backs prefix caching (I-7) and is the substrate for the continuous-
 * batching scheduler (I-2): every in-flight request maps logical pages
 * into one pool via its own page table, and matching prefixes
 * automatically share physical pages.
 */

#include "olmo_cpp/model/shared_block_pool.hpp"

#include <stdexcept>

namespace olmo_cpp {

SharedBlockPool::SharedBlockPool(int64_t n_layers,
                                 int64_t n_kv_heads,
                                 int64_t head_dim,
                                 int64_t page_size,
                                 int64_t max_pages,
                                 torch::Device device,
                                 torch::Dtype dtype)
    : page_size_(page_size),
      max_pages_(max_pages),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      device_(device),
      dtype_(dtype) {
  TORCH_CHECK(page_size > 0 && max_pages > 0,
              "SharedBlockPool: page_size and max_pages must be > 0");

  free_list_.reserve(static_cast<size_t>(max_pages));
  // Push in reverse so allocate() returns ascending page indices first.
  for (int64_t i = max_pages - 1; i >= 0; --i) {
    free_list_.push_back(static_cast<int32_t>(i));
  }
  ref_counts_.assign(static_cast<size_t>(max_pages), 0);
  page_hashes_.assign(static_cast<size_t>(max_pages), 0);

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  k_pools_.reserve(static_cast<size_t>(n_layers));
  v_pools_.reserve(static_cast<size_t>(n_layers));
  for (int64_t l = 0; l < n_layers; ++l) {
    k_pools_.push_back(torch::empty({max_pages, page_size, n_kv_heads, head_dim}, opts));
    v_pools_.push_back(torch::empty({max_pages, page_size, n_kv_heads, head_dim}, opts));
  }
}

int32_t SharedBlockPool::allocate() {
  if (free_list_.empty()) {
    throw std::runtime_error("SharedBlockPool: out of pages");
  }
  int32_t pg = free_list_.back();
  free_list_.pop_back();
  ref_counts_[static_cast<size_t>(pg)] = 1;
  // Anonymous allocation — not registered in hash_to_page_; caller will
  // populate the page and may register a hash later via find_or_lease
  // for subsequent identical content.
  return pg;
}

SharedBlockPool::LeaseResult SharedBlockPool::find_or_lease(
    uint64_t prev_hash,
    const int32_t* token_ids,
    int64_t n_tokens) {
  const uint64_t h = hash_block(prev_hash, token_ids, n_tokens);
  auto it = hash_to_page_.find(h);
  if (it != hash_to_page_.end()) {
    const int32_t pg = it->second;
    ref_counts_[static_cast<size_t>(pg)]++;
    return LeaseResult{pg, /*was_hit=*/true, h};
  }
  // Miss — allocate fresh and register.
  const int32_t pg = allocate();
  page_hashes_[static_cast<size_t>(pg)] = h;
  hash_to_page_[h] = pg;
  return LeaseResult{pg, /*was_hit=*/false, h};
}

void SharedBlockPool::inc_ref(int32_t page_idx) {
  TORCH_CHECK(page_idx >= 0 && page_idx < max_pages_,
              "SharedBlockPool::inc_ref: page index out of range");
  TORCH_CHECK(ref_counts_[static_cast<size_t>(page_idx)] > 0,
              "SharedBlockPool::inc_ref: page is not allocated");
  ref_counts_[static_cast<size_t>(page_idx)]++;
}

void SharedBlockPool::dec_ref(int32_t page_idx) {
  TORCH_CHECK(page_idx >= 0 && page_idx < max_pages_,
              "SharedBlockPool::dec_ref: page index out of range");
  auto& rc = ref_counts_[static_cast<size_t>(page_idx)];
  TORCH_CHECK(rc > 0, "SharedBlockPool::dec_ref: page already freed");
  if (--rc == 0) {
    // Page is now eligible for reuse. Drop its hash entry so a future
    // find_or_lease for the same content doesn't return a stale page.
    const uint64_t h = page_hashes_[static_cast<size_t>(page_idx)];
    if (h != 0) {
      auto it = hash_to_page_.find(h);
      if (it != hash_to_page_.end() && it->second == page_idx) {
        hash_to_page_.erase(it);
      }
      page_hashes_[static_cast<size_t>(page_idx)] = 0;
    }
    free_list_.push_back(page_idx);
  }
}

void SharedBlockPool::release_all(const std::vector<int32_t>& page_indices) {
  for (auto pg : page_indices) dec_ref(pg);
}

uint64_t SharedBlockPool::hash_block(uint64_t prev,
                                     const int32_t* ids,
                                     int64_t n) {
  // 64-bit FNV-1a, seeded with prev so the hash forms a chain that
  // depends on every preceding block's content.
  uint64_t h = prev ? prev : 0xcbf29ce484222325ULL;  // FNV offset basis
  constexpr uint64_t prime = 0x100000001b3ULL;
  // Mix the chain seed itself so the first block of a request still
  // hashes its content (instead of returning prev when n == 0).
  h ^= 0x9E3779B97F4A7C15ULL;
  h *= prime;
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t v = static_cast<uint32_t>(ids[i]);
    h ^= static_cast<uint64_t>(v & 0xff);          h *= prime;
    h ^= static_cast<uint64_t>((v >> 8) & 0xff);   h *= prime;
    h ^= static_cast<uint64_t>((v >> 16) & 0xff);  h *= prime;
    h ^= static_cast<uint64_t>((v >> 24) & 0xff);  h *= prime;
  }
  // Avoid h == 0 (we use 0 as "unset" sentinel in page_hashes_).
  return h == 0 ? 1ULL : h;
}

}  // namespace olmo_cpp
