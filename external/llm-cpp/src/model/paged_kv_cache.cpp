/**
 * src/model/paged_kv_cache.cpp
 *
 * Concat-backed implementation of IPagedKVCache — the shim that lets us
 * thread the new interface through the codebase before the real paged
 * allocator (fast-inference [9]) lands.
 *
 * Internally just wraps a KVCache and forwards every call. Append uses
 * KVCache's existing pre-allocated-buffer logic. Materialize returns
 * views over the filled region, identical to what the model code
 * currently sees from KVCache directly.
 *
 * Performance: same as KVCache, no improvement. The point is interface
 * stability — once kernels/paged_attention.cu lands, only this file is
 * replaced; call sites stay put.
 */

#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/block_manager.hpp"
#include "olmo_cpp/backend/paged_attention.hpp"

#include <stdexcept>
#include <algorithm>

namespace olmo_cpp {

namespace {

class ConcatKvCacheShim : public IPagedKVCache {
 public:
  ConcatKvCacheShim(int64_t n_layers, torch::Device device)
      : kv_(n_layers, device) {}

  int64_t seq_len() const override { return kv_.seq_len(); }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_.layers.size()) {
      throw std::out_of_range("paged_kv_cache_shim: layer index out of range");
    }
    auto& l = kv_.layers[static_cast<size_t>(layer)];
    auto [_, __] = l.update(k, v);
    (void)_; (void)__;
    return l.seq_len();
  }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_.layers.size()) {
      throw std::out_of_range("paged_kv_cache_shim: layer index out of range");
    }
    const auto& l = kv_.layers[static_cast<size_t>(layer)];
    if (!l.k.defined()) {
      // Empty cache — return empty tensors. Caller should typically check
      // seq_len() before calling materialize on a fresh layer.
      return {torch::Tensor(), torch::Tensor()};
    }
    int64_t n = l.seq_len();
    return {l.k.narrow(2, 0, n), l.v.narrow(2, 0, n)};
  }

  int64_t snapshot() const override { return kv_.snapshot(); }

  void rollback(int64_t len) override { kv_.rollback(len); }

  void clear() override { kv_.clear(); }

  /// Internal accessor for code paths that still take KVCache* directly.
  /// This is the bridge: existing callers continue to work, new callers
  /// use the IPagedKVCache interface.
  KVCache* underlying() { return &kv_; }

 private:
  KVCache kv_;
};

}  // namespace

std::unique_ptr<IPagedKVCache> make_concat_kv_cache_shim(
    int64_t n_layers, torch::Device device) {
  return std::make_unique<ConcatKvCacheShim>(n_layers, device);
}

// ──────────────────────────────────────────────────────────────────────────
// Real paged implementation, backed by BlockManager.
// ──────────────────────────────────────────────────────────────────────────

namespace {

class PagedKVCache : public IPagedKVCache {
 public:
  PagedKVCache(int64_t n_layers,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t page_size,
               int64_t max_pages,
               torch::Device device,
               torch::Dtype dtype,
               bool use_dyn_write)
      : mgr_(n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype),
        n_layers_(n_layers),
        page_size_(page_size),
        max_pages_(max_pages),
        device_(device),
        use_dyn_write_(use_dyn_write) {
    // Stable-address mirrors of the host-side cursor + page table. The
    // capture-friendly kernel (paged_attention_decode_dyn) reads from
    // these tensors at launch time; storage must outlive any captured
    // CUDA graph that holds the pointers, so we allocate once and update
    // in place.
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    page_table_t_ = torch::zeros({max_pages}, i32_opts);
    n_tokens_t_   = torch::zeros({}, i32_opts);
  }

  int64_t seq_len() const override { return logical_len_; }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    // k, v: [B=1, n_kv_heads, S, head_dim]. We support B==1 only for now.
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache: layer index out of range");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                "PagedKVCache: k/v must be 4D [B, n_kv_heads, S, head_dim]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1,
                "PagedKVCache: batch>1 not supported on this path");
    TORCH_CHECK(k.sizes() == v.sizes(), "PagedKVCache: k and v must match");
    const int64_t S = k.size(2);
    if (S == 0) return logical_len_;

    // Layer 0 is the "leader": it allocates any new pages and advances the
    // cursor. Layers 1..N-1 just write into the slots that layer 0 reserved.
    int64_t step_start, step_end;
    if (external_advance_) {
      // Cursor was bumped by the caller (typically right before a CUDA
      // graph replay). All layers in this step use the same slot range.
      step_end = logical_len_;
      step_start = step_end - S;
      TORCH_CHECK(step_start >= 0,
                  "PagedKVCache: external_advance=true but cursor not advanced for this step");
    } else if (layer == 0) {
      step_start = logical_len_;
      step_end = logical_len_ + S;
      const int64_t blocks_needed = (step_end + page_size_ - 1) / page_size_;
      const int64_t blocks_current = static_cast<int64_t>(mgr_.page_table().size());
      if (blocks_needed > blocks_current) {
        mgr_.allocate(blocks_needed - blocks_current);
        sync_page_table_tensor_(blocks_current, blocks_needed);
      }
      logical_len_ = step_end;
      // Mirror the new seq_len into the device-side stable scalar so the
      // capture-friendly kernel sees the right bound on its next launch.
      write_n_tokens_(static_cast<int32_t>(logical_len_));
    } else {
      step_end = logical_len_;
      step_start = step_end - S;
      TORCH_CHECK(step_start >= 0,
                  "PagedKVCache: append called for layer>0 before layer 0 advanced cursor");
    }

    write_layer_slots_(layer, k, v, step_start, step_end);
    return logical_len_;
  }

  // ── Graph-capture-friendly split: cursor advance separate from write ──
  int64_t advance_cursor(int64_t S) override {
    if (S <= 0) return logical_len_;
    const int64_t new_len = logical_len_ + S;
    const int64_t blocks_needed = (new_len + page_size_ - 1) / page_size_;
    const int64_t blocks_current = static_cast<int64_t>(mgr_.page_table().size());
    if (blocks_needed > blocks_current) {
      mgr_.allocate(blocks_needed - blocks_current);
      sync_page_table_tensor_(blocks_current, blocks_needed);
    }
    logical_len_ = new_len;
    write_n_tokens_(static_cast<int32_t>(logical_len_));
    return logical_len_;
  }

  void set_external_advance(bool on) override { external_advance_ = on; }
  bool external_advance() const override { return external_advance_; }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache: layer index out of range");
    if (logical_len_ == 0) {
      return {torch::Tensor(), torch::Tensor()};
    }
    return gather_layer_(layer, logical_len_);
  }

  int64_t snapshot() const override { return logical_len_; }

  void rollback(int64_t len) override {
    TORCH_CHECK(len >= 0 && len <= logical_len_,
                "PagedKVCache: rollback length out of range");
    // Pages stay allocated; subsequent appends will overwrite the rolled-back
    // slots. Cheap (just a cursor move) — matches LayerKVCache::truncate.
    logical_len_ = len;
    write_n_tokens_(static_cast<int32_t>(logical_len_));
  }

  void clear() override {
    mgr_.free_all();
    logical_len_ = 0;
    write_n_tokens_(0);
  }

  // ── Kernel-facing accessors (consumed by paged_attention_decode) ───────
  bool has_page_table() const override { return true; }
  int64_t page_size() const override { return page_size_; }

  torch::Tensor k_pool(int64_t layer) const override {
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache::k_pool: layer index out of range");
    return const_cast<BlockManager&>(mgr_).k_pool(layer);
  }
  torch::Tensor v_pool(int64_t layer) const override {
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "PagedKVCache::v_pool: layer index out of range");
    return const_cast<BlockManager&>(mgr_).v_pool(layer);
  }

  torch::Tensor page_table_tensor() const override {
    // Variable-length view of the stable buffer. Length = current block
    // count. Safe for use cases that do not need graph capture.
    const int64_t n = static_cast<int64_t>(mgr_.page_table().size());
    return page_table_t_.narrow(0, 0, n);
  }

  torch::Tensor page_table_tensor_stable() const override { return page_table_t_; }
  torch::Tensor n_tokens_tensor() const override        { return n_tokens_t_; }
  int64_t block_count() const override {
    return static_cast<int64_t>(mgr_.page_table().size());
  }

 private:
  void sync_page_table_tensor_(int64_t old_blocks, int64_t new_blocks) {
    // Copy any newly-allocated page indices from the host vector into the
    // stable device tensor. Pages already mirrored stay put.
    const auto& pt = mgr_.page_table();
    const int64_t n_new = new_blocks - old_blocks;
    if (n_new <= 0) return;
    std::vector<int32_t> tmp(static_cast<size_t>(n_new));
    for (int64_t i = 0; i < n_new; ++i) {
      tmp[static_cast<size_t>(i)] = pt[static_cast<size_t>(old_blocks + i)];
    }
    auto opts_cpu = torch::TensorOptions().dtype(torch::kInt32);
    auto src = torch::from_blob(tmp.data(), {n_new}, opts_cpu).clone();
    page_table_t_.narrow(0, old_blocks, n_new).copy_(src.to(device_));
  }

  void write_n_tokens_(int32_t v) {
    // CPU-only fast path: write_ into the underlying storage. On CUDA the
    // .fill_() is enough (issues an async memset/H2D under the hood); the
    // value is visible to the next kernel launch on the same stream.
    if (device_.is_cpu()) {
      n_tokens_t_.fill_(v);
    } else {
      // Stage on host to keep the fill on this stream's allocator; .fill_
      // with a Python int works on CUDA too without going through a host
      // intermediate, but we avoid scalar-via-PyObject overhead.
      auto staging = torch::tensor(v, torch::TensorOptions().dtype(torch::kInt32));
      n_tokens_t_.copy_(staging.to(device_, /*non_blocking=*/true));
    }
  }

  void write_layer_slots_(int64_t layer,
                          torch::Tensor k,
                          torch::Tensor v,
                          int64_t start,
                          int64_t end) {
    const int64_t S = end - start;
    if (S == 0) return;

    auto& k_pool = mgr_.k_pool(layer);
    auto& v_pool = mgr_.v_pool(layer);

    // Graph-safe path: source/dest both computed from the stable n_tokens
    // tensor inside the kernel. Caller must have already updated n_tokens_t_
    // for THIS step before calling write_layer_slots_ (we set logical_len_
    // = step_end on layer 0 above, which writes n_tokens_t_).
    if (use_dyn_write_) {
      // k, v come in as [1, n_kv_heads, S, head_dim]. The kernel wants
      // [S, n_kv_heads, head_dim].
      auto k_src = k.select(0, 0).permute({1, 0, 2}).contiguous();
      auto v_src = v.select(0, 0).permute({1, 0, 2}).contiguous();
      paged_kv_write_dyn(k_src, v_src, k_pool, v_pool,
                         page_table_t_, n_tokens_t_);
      return;
    }

    // Legacy index_put_ path. Not graph-capture-safe (destination
    // addresses are baked into the captured launch) but correct.
    const auto& pt = mgr_.page_table();

    // Build host-side index vectors. n_blocks is small (~ ceil(max_seq/16))
    // and S is typically 1 (decode) or prompt_len (prefill); doing this on
    // the host then transferring is cheaper than launching tiny GPU index ops.
    std::vector<int64_t> pg_idx(static_cast<size_t>(S));
    std::vector<int64_t> slot_idx(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
      const int64_t g = start + i;
      const int64_t blk = g / page_size_;
      const int64_t off = g % page_size_;
      pg_idx[static_cast<size_t>(i)]   = static_cast<int64_t>(pt[static_cast<size_t>(blk)]);
      slot_idx[static_cast<size_t>(i)] = off;
    }

    auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64);
    auto pg_t   = torch::from_blob(pg_idx.data(),   {S}, opts_cpu).clone().to(device_);
    auto slot_t = torch::from_blob(slot_idx.data(), {S}, opts_cpu).clone().to(device_);

    // Source: k has shape [1, n_kv_heads, S, head_dim]. We need
    // [S, n_kv_heads, head_dim] to match the pool slice ordering.
    auto k_src = k.select(0, 0).permute({1, 0, 2}).contiguous();  // [S, n_kv_heads, head_dim]
    auto v_src = v.select(0, 0).permute({1, 0, 2}).contiguous();

    // index_put_ with two index tensors performs gather-style scatter:
    // k_pool[pg_t[i], slot_t[i], :, :] = k_src[i]   for i in [0, S).
    k_pool.index_put_({pg_t, slot_t}, k_src.to(k_pool.dtype()));
    v_pool.index_put_({pg_t, slot_t}, v_src.to(v_pool.dtype()));
  }

  std::pair<torch::Tensor, torch::Tensor> gather_layer_(int64_t layer, int64_t total_len) const {
    // Pool layout: [max_pages, page_size, n_kv_heads, head_dim].
    // Gather the in-use pages, flatten to [n_blocks * page_size, ...], narrow
    // down to the logical length, then reshape to [1, n_kv_heads, total_len, head_dim].
    const auto& pt = mgr_.page_table();
    const int64_t n_blocks = (total_len + page_size_ - 1) / page_size_;
    TORCH_CHECK(n_blocks <= static_cast<int64_t>(pt.size()),
                "PagedKVCache: page table too short for requested length");

    auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> pt_int64(static_cast<size_t>(n_blocks));
    for (int64_t i = 0; i < n_blocks; ++i) {
      pt_int64[static_cast<size_t>(i)] = static_cast<int64_t>(pt[static_cast<size_t>(i)]);
    }
    auto pt_t = torch::from_blob(pt_int64.data(), {n_blocks}, opts_cpu).clone().to(device_);

    const auto& k_pool = const_cast<BlockManager&>(mgr_).k_pool(layer);
    const auto& v_pool = const_cast<BlockManager&>(mgr_).v_pool(layer);
    const int64_t n_kv_heads = mgr_.n_kv_heads();
    const int64_t head_dim   = mgr_.head_dim();

    auto k_blocks = k_pool.index_select(0, pt_t);  // [n_blocks, page_size, n_kv_heads, head_dim]
    auto v_blocks = v_pool.index_select(0, pt_t);
    auto k_flat = k_blocks.reshape({n_blocks * page_size_, n_kv_heads, head_dim})
                          .narrow(0, 0, total_len);                  // [total_len, H, D]
    auto v_flat = v_blocks.reshape({n_blocks * page_size_, n_kv_heads, head_dim})
                          .narrow(0, 0, total_len);
    auto k_out = k_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();  // [1, H, total_len, D]
    auto v_out = v_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();
    return {k_out, v_out};
  }

  BlockManager mgr_;
  int64_t n_layers_;
  int64_t page_size_;
  int64_t max_pages_;
  int64_t logical_len_ = 0;
  torch::Device device_;
  bool use_dyn_write_ = false;
  bool external_advance_ = false;
  // Stable-address mirrors for graph-capture-friendly launches. Updated
  // in place — never reallocated — so any captured kernel launch holds a
  // valid device pointer across replays.
  torch::Tensor page_table_t_;
  torch::Tensor n_tokens_t_;
};

}  // namespace

std::unique_ptr<IPagedKVCache> make_paged_kv_cache(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype) {
  return std::make_unique<PagedKVCache>(
      n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype,
      /*use_dyn_write=*/false);
}

std::unique_ptr<IPagedKVCache> make_paged_kv_cache_graph_safe(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype) {
  return std::make_unique<PagedKVCache>(
      n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype,
      /*use_dyn_write=*/true);
}

// ──────────────────────────────────────────────────────────────────────────
// Int4PagedKVCache — INT4-quantized variant. K/V stored as uint8 nibble
// pairs plus per-vector fp16 scales. 4× memory drop on the cache. Decode
// kernel: paged_attention_decode_int4 (dequantizes inline in the dot
// product). Materialize dequantizes the in-use slots back to compute_dtype.
// ──────────────────────────────────────────────────────────────────────────

namespace {

class Int4PagedKVCache : public IPagedKVCache {
 public:
  Int4PagedKVCache(int64_t n_layers,
                    int64_t n_kv_heads,
                    int64_t head_dim,
                    int64_t page_size,
                    int64_t max_pages,
                    torch::Device device,
                    torch::Dtype compute_dtype)
      : n_layers_(n_layers),
        n_kv_heads_(n_kv_heads),
        head_dim_(head_dim),
        head_dim_half_(head_dim / 2),
        page_size_(page_size),
        max_pages_(max_pages),
        device_(device),
        compute_dtype_(compute_dtype) {
    TORCH_CHECK(head_dim % 2 == 0, "Int4PagedKVCache: head_dim must be even");

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    auto fp16_opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    k_pools_.reserve(n_layers_);
    v_pools_.reserve(n_layers_);
    k_scales_.reserve(n_layers_);
    v_scales_.reserve(n_layers_);
    for (int64_t l = 0; l < n_layers_; ++l) {
      k_pools_.push_back(torch::zeros(
          {max_pages_, page_size_, n_kv_heads_, head_dim_half_}, u8_opts));
      v_pools_.push_back(torch::zeros(
          {max_pages_, page_size_, n_kv_heads_, head_dim_half_}, u8_opts));
      k_scales_.push_back(torch::zeros(
          {max_pages_, page_size_, n_kv_heads_}, fp16_opts));
      v_scales_.push_back(torch::zeros(
          {max_pages_, page_size_, n_kv_heads_}, fp16_opts));
    }

    page_table_t_ = torch::zeros({max_pages_}, i32_opts);
    n_tokens_t_   = torch::zeros({}, i32_opts);

    free_list_.reserve(static_cast<size_t>(max_pages_));
    for (int64_t i = max_pages_ - 1; i >= 0; --i) {
      free_list_.push_back(static_cast<int32_t>(i));
    }
  }

  int64_t seq_len() const override { return logical_len_; }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    TORCH_CHECK(layer >= 0 && layer < n_layers_,
                "Int4PagedKVCache: layer index out of range");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                "Int4PagedKVCache: k/v must be 4D [B, n_kv_heads, S, head_dim]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1,
                "Int4PagedKVCache: batch>1 not supported");
    const int64_t S = k.size(2);
    if (S == 0) return logical_len_;

    int64_t step_start, step_end;
    if (layer == 0) {
      step_start = logical_len_;
      step_end = logical_len_ + S;
      const int64_t blocks_needed = (step_end + page_size_ - 1) / page_size_;
      const int64_t blocks_current = static_cast<int64_t>(page_table_.size());
      if (blocks_needed > blocks_current) {
        allocate_(blocks_needed - blocks_current);
        sync_page_table_(blocks_current, blocks_needed);
      }
      logical_len_ = step_end;
      write_n_tokens_(static_cast<int32_t>(logical_len_));
    } else {
      step_end = logical_len_;
      step_start = step_end - S;
      TORCH_CHECK(step_start >= 0,
                  "Int4PagedKVCache: append called for layer>0 before layer 0");
    }
    quantize_and_write_(layer, k, v, step_start, step_end);
    return logical_len_;
  }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    if (logical_len_ == 0) return {torch::Tensor(), torch::Tensor()};
    return dequantize_layer_(layer, logical_len_);
  }

  int64_t snapshot() const override { return logical_len_; }
  void rollback(int64_t len) override {
    TORCH_CHECK(len >= 0 && len <= logical_len_,
                "Int4PagedKVCache: rollback length out of range");
    logical_len_ = len;
    write_n_tokens_(static_cast<int32_t>(logical_len_));
  }
  void clear() override {
    free_list_.clear();
    for (int64_t i = max_pages_ - 1; i >= 0; --i) {
      free_list_.push_back(static_cast<int32_t>(i));
    }
    page_table_.clear();
    logical_len_ = 0;
    write_n_tokens_(0);
  }

  // INT4-specific accessors.
  bool is_int4() const override { return true; }
  bool has_page_table() const override { return true; }
  int64_t page_size() const override { return page_size_; }
  torch::Tensor k_pool(int64_t layer) const override {
    return k_pools_[static_cast<size_t>(layer)];
  }
  torch::Tensor v_pool(int64_t layer) const override {
    return v_pools_[static_cast<size_t>(layer)];
  }
  torch::Tensor k_scales(int64_t layer) const override {
    return k_scales_[static_cast<size_t>(layer)];
  }
  torch::Tensor v_scales(int64_t layer) const override {
    return v_scales_[static_cast<size_t>(layer)];
  }
  torch::Tensor page_table_tensor() const override {
    const int64_t n = static_cast<int64_t>(page_table_.size());
    return page_table_t_.narrow(0, 0, n);
  }
  torch::Tensor page_table_tensor_stable() const override { return page_table_t_; }
  torch::Tensor n_tokens_tensor() const override { return n_tokens_t_; }
  int64_t block_count() const override {
    return static_cast<int64_t>(page_table_.size());
  }

 private:
  void allocate_(int64_t n) {
    TORCH_CHECK(static_cast<int64_t>(free_list_.size()) >= n,
                "Int4PagedKVCache: out of free pages");
    for (int64_t i = 0; i < n; ++i) {
      page_table_.push_back(free_list_.back());
      free_list_.pop_back();
    }
  }

  void sync_page_table_(int64_t old_blocks, int64_t new_blocks) {
    const int64_t n_new = new_blocks - old_blocks;
    if (n_new <= 0) return;
    std::vector<int32_t> tmp(static_cast<size_t>(n_new));
    for (int64_t i = 0; i < n_new; ++i) {
      tmp[static_cast<size_t>(i)] = page_table_[static_cast<size_t>(old_blocks + i)];
    }
    auto src = torch::from_blob(tmp.data(), {n_new},
                                 torch::TensorOptions().dtype(torch::kInt32)).clone();
    page_table_t_.narrow(0, old_blocks, n_new).copy_(src.to(device_));
  }

  void write_n_tokens_(int32_t v) const {
    if (device_.is_cpu()) {
      n_tokens_t_.fill_(v);
    } else {
      auto staging = torch::tensor(v, torch::TensorOptions().dtype(torch::kInt32));
      n_tokens_t_.copy_(staging.to(device_, /*non_blocking=*/true));
    }
  }

  // Quantize one (k, v) batch and write into the page pools at [start, end).
  // Per-vector dynamic max-abs scaling, 2 nibbles per byte along head_dim.
  void quantize_and_write_(int64_t layer,
                            torch::Tensor k,
                            torch::Tensor v,
                            int64_t start,
                            int64_t end) {
    const int64_t S = end - start;
    if (S == 0) return;

    // Reorder to [S, n_kv_heads, head_dim] in fp32 for stable quantization.
    auto k_src = k.select(0, 0).permute({1, 0, 2}).contiguous().to(torch::kFloat32);
    auto v_src = v.select(0, 0).permute({1, 0, 2}).contiguous().to(torch::kFloat32);

    auto [k_packed, k_scale] = quantize_(k_src);
    auto [v_packed, v_scale] = quantize_(v_src);

    // Scatter (page, slot) → pool. Build host index vectors of length S.
    std::vector<int64_t> pg_idx(static_cast<size_t>(S));
    std::vector<int64_t> slot_idx(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
      const int64_t g = start + i;
      pg_idx[static_cast<size_t>(i)] =
          static_cast<int64_t>(page_table_[static_cast<size_t>(g / page_size_)]);
      slot_idx[static_cast<size_t>(i)] = g % page_size_;
    }
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64);
    auto pg_t   = torch::from_blob(pg_idx.data(),   {S}, opts_i64).clone().to(device_);
    auto slot_t = torch::from_blob(slot_idx.data(), {S}, opts_i64).clone().to(device_);

    k_pools_[static_cast<size_t>(layer)].index_put_({pg_t, slot_t}, k_packed);
    v_pools_[static_cast<size_t>(layer)].index_put_({pg_t, slot_t}, v_packed);
    k_scales_[static_cast<size_t>(layer)].index_put_({pg_t, slot_t}, k_scale);
    v_scales_[static_cast<size_t>(layer)].index_put_({pg_t, slot_t}, v_scale);
  }

  // Returns {packed [S, n_kv_heads, head_dim/2] uint8,
  //          scales [S, n_kv_heads]              fp16}.
  std::pair<torch::Tensor, torch::Tensor> quantize_(torch::Tensor t_fp32) const {
    // t_fp32: [S, n_kv_heads, head_dim].
    auto max_abs = std::get<0>(t_fp32.abs().max(/*dim=*/-1, /*keepdim=*/true));   // [S, n_kv_heads, 1]
    auto scales = (max_abs / 7.0f).clamp_min(1e-8f);
    auto q = (t_fp32 / scales).round().clamp(-8.0f, 7.0f);
    auto q_shifted = (q + 8.0f).to(torch::kUInt8);                                 // [S, n_kv_heads, head_dim]
    auto q_pair = q_shifted.reshape({q_shifted.size(0), q_shifted.size(1),
                                       head_dim_half_, 2});
    auto lo = q_pair.select(-1, 0);
    auto hi = q_pair.select(-1, 1);
    auto packed = (lo | (hi.bitwise_left_shift(4))).contiguous();                  // [S, n_kv_heads, head_dim/2]
    auto scales_out = scales.squeeze(-1).to(torch::kFloat16).contiguous();          // [S, n_kv_heads]
    return {packed, scales_out};
  }

  // Dequantize the in-use slots of one layer back to compute_dtype.
  std::pair<torch::Tensor, torch::Tensor> dequantize_layer_(int64_t layer,
                                                             int64_t total_len) const {
    const int64_t n_blocks = (total_len + page_size_ - 1) / page_size_;
    std::vector<int64_t> pt_int64(static_cast<size_t>(n_blocks));
    for (int64_t i = 0; i < n_blocks; ++i) {
      pt_int64[static_cast<size_t>(i)] =
          static_cast<int64_t>(page_table_[static_cast<size_t>(i)]);
    }
    auto pt_t = torch::from_blob(pt_int64.data(), {n_blocks},
                                   torch::TensorOptions().dtype(torch::kInt64))
                     .clone().to(device_);

    const auto& k_pool = k_pools_[static_cast<size_t>(layer)];
    const auto& v_pool = v_pools_[static_cast<size_t>(layer)];
    const auto& k_scale = k_scales_[static_cast<size_t>(layer)];
    const auto& v_scale = v_scales_[static_cast<size_t>(layer)];

    auto k_blocks = k_pool.index_select(0, pt_t);   // [n_blocks, page, H, D/2] uint8
    auto v_blocks = v_pool.index_select(0, pt_t);
    auto k_sc_b   = k_scale.index_select(0, pt_t);  // [n_blocks, page, H] fp16
    auto v_sc_b   = v_scale.index_select(0, pt_t);

    auto k_flat = k_blocks.reshape({n_blocks * page_size_, n_kv_heads_, head_dim_half_})
                          .narrow(0, 0, total_len);
    auto v_flat = v_blocks.reshape({n_blocks * page_size_, n_kv_heads_, head_dim_half_})
                          .narrow(0, 0, total_len);
    auto k_sc_f = k_sc_b.reshape({n_blocks * page_size_, n_kv_heads_})
                        .narrow(0, 0, total_len);
    auto v_sc_f = v_sc_b.reshape({n_blocks * page_size_, n_kv_heads_})
                        .narrow(0, 0, total_len);

    auto k_dq = dequantize_(k_flat, k_sc_f);   // [total_len, H, D] compute_dtype
    auto v_dq = dequantize_(v_flat, v_sc_f);

    auto k_out = k_dq.permute({1, 0, 2}).unsqueeze(0).contiguous();   // [1, H, T, D]
    auto v_out = v_dq.permute({1, 0, 2}).unsqueeze(0).contiguous();
    return {k_out, v_out};
  }

  torch::Tensor dequantize_(torch::Tensor packed, torch::Tensor scales) const {
    // packed: [T, H, D/2] uint8. scales: [T, H] fp16.
    auto p32 = packed.to(torch::kInt32);
    auto lo  = (p32.bitwise_and(0xF) - 8).to(torch::kFloat32);              // [T, H, D/2]
    auto hi  = ((p32.bitwise_right_shift(4)).bitwise_and(0xF) - 8).to(torch::kFloat32);
    auto stacked = torch::stack({lo, hi}, /*dim=*/-1);                       // [T, H, D/2, 2]
    auto dq = stacked.reshape({packed.size(0), packed.size(1), head_dim_});  // [T, H, D]
    auto sc = scales.to(torch::kFloat32).unsqueeze(-1);                      // [T, H, 1]
    auto out = dq * sc;
    return out.to(compute_dtype_);
  }

  int64_t n_layers_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  int64_t head_dim_half_;
  int64_t page_size_;
  int64_t max_pages_;
  int64_t logical_len_ = 0;
  torch::Device device_;
  torch::Dtype compute_dtype_;

  std::vector<int32_t> free_list_;
  std::vector<int32_t> page_table_;
  std::vector<torch::Tensor> k_pools_;
  std::vector<torch::Tensor> v_pools_;
  std::vector<torch::Tensor> k_scales_;
  std::vector<torch::Tensor> v_scales_;

  mutable torch::Tensor page_table_t_;
  mutable torch::Tensor n_tokens_t_;
};

}  // namespace

std::unique_ptr<IPagedKVCache> make_paged_kv_cache_int4(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype compute_dtype) {
  return std::make_unique<Int4PagedKVCache>(
      n_layers, n_kv_heads, head_dim, page_size, max_pages, device, compute_dtype);
}

}  // namespace olmo_cpp
