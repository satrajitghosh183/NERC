/**
 * src/model/block_manager.cpp
 *
 * Implementation of the BlockManager allocator (fast-inference [9]).
 *
 * Pages are pre-allocated as one big [max_pages, page_size, ...] buffer
 * per layer. The allocator just hands out indices from a freelist.
 * Append-only — a request keeps its pages until free_all().
 *
 * DRAFT — not wired into model attention yet.
 */

#include "olmo_cpp/model/block_manager.hpp"

#include <stdexcept>

namespace olmo_cpp {

BlockManager::BlockManager(int64_t n_layers,
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
      device_(device) {
  if (page_size <= 0 || max_pages <= 0) {
    throw std::invalid_argument("BlockManager: page_size and max_pages must be > 0");
  }

  // Freelist: pages 0..max_pages-1 are all initially free, in reverse
  // order so allocate() pops in ascending order.
  free_list_.reserve(static_cast<size_t>(max_pages));
  for (int64_t i = max_pages - 1; i >= 0; --i) {
    free_list_.push_back(static_cast<int32_t>(i));
  }
  page_table_.reserve(64);

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  k_pools_.reserve(static_cast<size_t>(n_layers));
  v_pools_.reserve(static_cast<size_t>(n_layers));
  for (int64_t l = 0; l < n_layers; ++l) {
    k_pools_.push_back(torch::empty({max_pages, page_size, n_kv_heads, head_dim}, opts));
    v_pools_.push_back(torch::empty({max_pages, page_size, n_kv_heads, head_dim}, opts));
  }
}

std::vector<int32_t> BlockManager::allocate(int64_t n) {
  if (static_cast<int64_t>(free_list_.size()) < n) {
    throw std::runtime_error("BlockManager: out of pages");
  }
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    int32_t pg = free_list_.back();
    free_list_.pop_back();
    out.push_back(pg);
    page_table_.push_back(pg);
  }
  return out;
}

void BlockManager::free_all() {
  // Return every page in the active page table to the freelist.
  for (auto pg : page_table_) {
    free_list_.push_back(pg);
  }
  page_table_.clear();
}

}  // namespace olmo_cpp
