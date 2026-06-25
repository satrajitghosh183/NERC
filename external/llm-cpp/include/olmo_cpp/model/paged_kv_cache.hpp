/**
 * include/olmo_cpp/model/paged_kv_cache.hpp
 *
 * Fast-inference roadmap item [1] — interface stub.
 *
 * The current KVCache (kv_cache.hpp) holds K/V as torch::Tensors that get
 * concat()-ed each step. Tensor shape changes every step, which:
 *   - blocks CUDA graph capture (item [2])
 *   - makes batched decoding with variable lengths painful
 *   - re-allocates and copies HBM repeatedly
 *
 * The plan: replace the concat-based store with a paged allocator
 * (vLLM-style PagedAttention). Fixed-size pages (e.g. 16 tokens),
 * a per-request page table, and an attention kernel that gathers from
 * pages via the table.
 *
 * This file is a stub. It defines the interface the rest of the code
 * will eventually call and provides a no-op fallback that delegates to
 * the existing concat KVCache. Real implementation lands in:
 *   - kernels/paged_attention.cu  (the gathered attention kernel)
 *   - src/model/paged_kv_cache.cpp (allocator + page table)
 *
 * Until then this header lets us thread `IPagedKVCache*` through the
 * model and bench code without committing to an implementation. Once
 * the paged allocator lands, only the implementations swap.
 *
 * NOT IMPLEMENTED YET. Do not call from production paths. The bench
 * harness uses the existing KVCache; that's deliberate.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <utility>

namespace olmo_cpp {

/// Minimal abstraction over a per-request KV cache. The current
/// concat-based KVCache will satisfy this shim trivially; the paged
/// implementation will satisfy it with O(1) append and zero reshape.
class IPagedKVCache {
 public:
  virtual ~IPagedKVCache() = default;

  /// Number of tokens currently cached for this request.
  virtual int64_t seq_len() const = 0;

  /// Append `k` and `v` (shape [B, T, n_heads, head_dim]) to the layer's cache.
  /// Returns the new total seq_len for that layer.
  virtual int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) = 0;

  /// Materialize layer-`layer` K and V as contiguous tensors of shape
  /// [B, total_T, n_heads, head_dim] for the existing attention kernel.
  /// Paged impl will return *views* into pages where possible. Concat impl
  /// returns the underlying tensor directly.
  virtual std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const = 0;

  /// Snapshot current length for rollback (used by speculative decoding
  /// at chat.cpp:429 / speculative_decode_step).
  virtual int64_t snapshot() const = 0;

  /// Truncate all layers back to `len`.
  virtual void rollback(int64_t len) = 0;

  /// Drop all cached state.
  virtual void clear() = 0;

  // ── Optional kernel-facing accessors ───────────────────────────────────
  // The paged_attention_decode kernel (kernels/paged_attention.cu) needs
  // direct access to the page pools and page table. Implementations that
  // don't physically page (e.g. the concat shim) should leave the default
  // throws in place; AttentionImpl::forward_paged checks has_page_table()
  // first and falls back to materialize() + SDPA when the kernel path is
  // unavailable.

  /// True if k_pool / v_pool / page_table_tensor return real device data.
  virtual bool has_page_table() const { return false; }

  /// Page size in tokens (page dimension stride in the per-layer pool).
  virtual int64_t page_size() const { return 0; }

  /// Per-layer K/V page pool: [max_pages, page_size, n_kv_heads, head_dim].
  virtual torch::Tensor k_pool(int64_t /*layer*/) const {
    throw std::runtime_error("k_pool() not supported on this IPagedKVCache impl");
  }
  virtual torch::Tensor v_pool(int64_t /*layer*/) const {
    throw std::runtime_error("v_pool() not supported on this IPagedKVCache impl");
  }

  /// Page table as a 1-D int32 tensor on the cache's device, length =
  /// number of currently allocated blocks. The kernel walks it as
  /// logical_block_idx -> physical_page_idx.
  ///
  /// For graph-capture use, prefer page_table_tensor_stable() — that
  /// variant returns a tensor with fixed storage so the captured launch
  /// holds a valid pointer across replays even as new pages are added.
  virtual torch::Tensor page_table_tensor() const {
    throw std::runtime_error("page_table_tensor() not supported on this IPagedKVCache impl");
  }

  /// Stable-address page table view sized for the worst case. The first
  /// `block_count()` entries are valid; the kernel relies on the n_tokens
  /// scalar to bound its iteration so the unused tail does not matter.
  /// Required if the caller wants to capture the decode launch in a CUDA
  /// graph (storage must stay put across replays).
  virtual torch::Tensor page_table_tensor_stable() const {
    return page_table_tensor();
  }

  /// 0-D int32 tensor on the cache's device whose value is the current
  /// seq_len(). Storage stays put across appends — only the value gets
  /// written in place. Pair with paged_attention_decode_dyn.
  virtual torch::Tensor n_tokens_tensor() const {
    throw std::runtime_error("n_tokens_tensor() not supported on this IPagedKVCache impl");
  }

  /// Number of pages currently in use (= length of the valid prefix of
  /// page_table_tensor_stable()).
  virtual int64_t block_count() const { return 0; }

  // ── INT4 KV variant (item U follow-on) ─────────────────────────────────
  // When is_int4() returns true, k_pool / v_pool hold uint8 nibble-packed
  // codes with shape [max_pages, page_size, n_kv_heads, head_dim/2] and the
  // matching per-vector fp16 scales are accessible via k_scales / v_scales.
  // The decode kernel that consumes them is paged_attention_decode_int4
  // (kernels/paged_attention.cu). Default impl: not int4; the scale
  // accessors throw.
  virtual bool is_int4() const { return false; }
  virtual torch::Tensor k_scales(int64_t /*layer*/) const {
    throw std::runtime_error("k_scales() only valid on int4-backed caches");
  }
  virtual torch::Tensor v_scales(int64_t /*layer*/) const {
    throw std::runtime_error("v_scales() only valid on int4-backed caches");
  }

  // ── Split append for CUDA-graph capture ────────────────────────────────
  // append() does two things: (a) advance the cursor (host bookkeeping +
  // allocate pages + bump n_tokens_t_) and (b) launch the K/V write
  // kernel. In a captured graph, (a) runs once at capture time and is
  // never re-run; (b) re-runs on every replay. For replays to write to
  // the correct slot, the kernel must compute its destination from
  // n_tokens_t_ at launch — and n_tokens_t_ must be advanced externally
  // between replays.
  //
  // To enable that, two pieces of API:
  //   1. advance_cursor(S): host-only update. Bumps logical_len_,
  //      allocates pages, writes new n_tokens_t_. No kernel launch.
  //   2. external_advance mode: when set, append() skips its internal
  //      cursor advance and just runs the K/V write. The caller is
  //      responsible for calling advance_cursor before each forward.
  //
  // When external_advance is OFF (default), append() is the all-in-one
  // call described above and behavior is unchanged.

  /// Bump the cursor by S without launching any K/V writes. Allocates
  /// new pages if needed and refreshes the device-side n_tokens scalar.
  /// Returns the new seq_len.
  virtual int64_t advance_cursor(int64_t /*S*/) {
    throw std::runtime_error("advance_cursor() not supported on this IPagedKVCache impl");
  }

  /// Toggle whether append() includes the cursor advance. Default false
  /// (cursor advance is internal to append). Set true before capturing a
  /// CUDA graph of forward_paged: the caller must invoke advance_cursor
  /// before each replay.
  virtual void set_external_advance(bool /*on*/) {
    throw std::runtime_error("set_external_advance() not supported on this IPagedKVCache impl");
  }
  virtual bool external_advance() const { return false; }
};

/// Construct a paged KV cache backed by the existing concat KVCache.
/// Used as the fallback for code paths that need IPagedKVCache but cannot
/// yet take advantage of the paged kernels (e.g. CPU-only build).
std::unique_ptr<IPagedKVCache> make_concat_kv_cache_shim(
    int64_t n_layers, torch::Device device);

/// Construct a real paged KV cache backed by a BlockManager.
///
/// Layout: per-layer K/V pools of shape
///   [max_pages, page_size, n_kv_heads, head_dim].
/// Block (page) allocation happens lazily as logical_len_ crosses page
/// boundaries. The caller is expected to invoke `append` for layers in
/// ascending order each step (layer 0 first); the implementation advances
/// the cursor on layer 0 and back-computes destination slots for the rest.
///
/// `max_pages * page_size` is the hard cap on cached sequence length.
///
/// Currently used as a drop-in storage replacement: `materialize(layer)`
/// returns contiguous [1, n_kv_heads, logical_len, head_dim] views, so the
/// existing SDPA-based attention path works unchanged. Decode-side attention
/// can later dispatch to `paged_attention_decode` (kernels/paged_attention.cu)
/// using the BlockManager pools + page table directly, without materializing.
std::unique_ptr<IPagedKVCache> make_paged_kv_cache(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype);

/// Same as make_paged_kv_cache, but K/V writes go through the graph-safe
/// paged_kv_write_dyn kernel instead of torch::Tensor::index_put_. Required
/// when the caller plans to capture forward_paged in a CUDA graph: the
/// index_put_ scatter hardcodes destination addresses at launch time, so
/// captured graphs would always write to the same slot. The dyn write
/// kernel reads `start = n_tokens - S` from the cache's stable n_tokens
/// scalar at kernel entry, so the captured launch tracks logical_len_ as
/// the cache grows.
std::unique_ptr<IPagedKVCache> make_paged_kv_cache_graph_safe(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype dtype);

/// INT4-quantized paged KV cache (item U). K/V are stored as uint8
/// nibble-packed codes with per-vector fp16 scales — 4× memory
/// reduction vs bf16 at the cost of ~1% perplexity on long context.
/// Quantization happens at append() time (per-vector dynamic max-abs).
/// Decode-side attention dispatches through paged_attention_decode_int4
/// when the cache reports is_int4() == true. Prefill / SDPA-fallback
/// paths get dequantized bf16 K/V from materialize().
///
/// `compute_dtype` is the dtype the dequantized K/V are produced in
/// (bf16 for OLMo). `page_size`, `max_pages` follow the bf16 cache.
std::unique_ptr<IPagedKVCache> make_paged_kv_cache_int4(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    int64_t max_pages,
    torch::Device device,
    torch::Dtype compute_dtype);

}  // namespace olmo_cpp
