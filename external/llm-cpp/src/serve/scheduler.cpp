/**
 * src/serve/scheduler.cpp
 *
 * Multi-request scheduler implementation (fast-inference [I-2]).
 *
 * Each Scheduler::step() picks one live request (FCFS over decoding +
 * prefilling requests) and advances it one quantum:
 *   - if Pending: kick off prefix-cache page leasing, transition to
 *     Prefilling, run the first prefill chunk.
 *   - if Prefilling: run one prefill_chunk_size-token chunk.
 *   - if Decoding: run one decode step.
 *
 * The model forward goes through a per-request IPagedKVCache view that
 * points into the shared pool with the request's logical page table.
 * Cache hits from find_or_lease() during admit mean those positions'
 * K/V is already populated; the actual prefill only computes K/V for
 * the suffix past the longest cache-hit prefix.
 */

#include "olmo_cpp/serve/scheduler.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/backend/paged_attention.hpp"
#include "olmo_cpp/backend/gpu_sample.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace olmo_cpp {

namespace {

// ── IPagedKVCache view backed by a SharedBlockPool ─────────────────────
//
// Per-request: holds a reference to the shared pool and the request's
// logical page_table (mirrored as a stable device tensor for the kernel
// path). cursor advances on layer 0, n_tokens_t_ tracks logical_len_,
// kernel-facing accessors return stable tensors.
class SharedPoolKVCache : public IPagedKVCache {
 public:
  SharedPoolKVCache(SharedBlockPool* pool,
                    std::vector<int32_t>* page_table,
                    int64_t* logical_len,
                    int64_t max_pages)
      : pool_(pool),
        page_table_ref_(page_table),
        logical_len_ref_(logical_len),
        max_pages_(max_pages) {
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(pool_->device());
    page_table_t_ = torch::zeros({max_pages}, i32_opts);
    n_tokens_t_   = torch::zeros({}, i32_opts);
    sync_page_table_(0, static_cast<int64_t>(page_table_ref_->size()));
    write_n_tokens_(static_cast<int32_t>(*logical_len_ref_));
  }

  int64_t seq_len() const override { return *logical_len_ref_; }

  int64_t append(int64_t layer, torch::Tensor k, torch::Tensor v) override {
    TORCH_CHECK(layer >= 0 && layer < pool_->n_layers(),
                "SharedPoolKVCache: layer index out of range");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4 && k.size(0) == 1 && v.size(0) == 1,
                "SharedPoolKVCache: k/v shape must be [1, n_kv_heads, S, head_dim]");
    const int64_t S = k.size(2);
    if (S == 0) return *logical_len_ref_;

    int64_t step_start, step_end;
    if (layer == 0) {
      step_start = *logical_len_ref_;
      step_end   = step_start + S;
      // Page table must already have enough blocks (admit() pre-allocates
      // for the prompt; decode() add_page_ extends as needed).
      const int64_t blocks_needed = (step_end + pool_->page_size() - 1) / pool_->page_size();
      const int64_t blocks_have   = static_cast<int64_t>(page_table_ref_->size());
      TORCH_CHECK(blocks_needed <= blocks_have,
                  "SharedPoolKVCache: append crosses unallocated page boundary; "
                  "caller must allocate first");
      *logical_len_ref_ = step_end;
      write_n_tokens_(static_cast<int32_t>(step_end));
    } else {
      step_end   = *logical_len_ref_;
      step_start = step_end - S;
    }

    write_layer_(layer, k, v, step_start, step_end);
    return *logical_len_ref_;
  }

  std::pair<torch::Tensor, torch::Tensor> materialize(int64_t layer) const override {
    if (*logical_len_ref_ == 0) return {torch::Tensor(), torch::Tensor()};
    return gather_(layer, *logical_len_ref_);
  }

  int64_t snapshot() const override { return *logical_len_ref_; }
  void rollback(int64_t len) override {
    TORCH_CHECK(len >= 0 && len <= *logical_len_ref_, "SharedPoolKVCache: bad rollback");
    *logical_len_ref_ = len;
    write_n_tokens_(static_cast<int32_t>(len));
  }
  void clear() override { /* lifecycle managed by Scheduler */ }

  bool has_page_table() const override { return true; }
  int64_t page_size() const override   { return pool_->page_size(); }
  torch::Tensor k_pool(int64_t layer) const override { return pool_->k_pool(layer); }
  torch::Tensor v_pool(int64_t layer) const override { return pool_->v_pool(layer); }
  torch::Tensor page_table_tensor() const override {
    return page_table_t_.narrow(0, 0, static_cast<int64_t>(page_table_ref_->size()));
  }
  torch::Tensor page_table_tensor_stable() const override { return page_table_t_; }
  torch::Tensor n_tokens_tensor() const override         { return n_tokens_t_; }
  int64_t block_count() const override { return static_cast<int64_t>(page_table_ref_->size()); }

  /// Caller signals new pages were appended to the host page_table vector;
  /// mirror them into the device tensor.
  void notify_pages_added(int64_t old_n, int64_t new_n) {
    sync_page_table_(old_n, new_n);
  }

 private:
  void sync_page_table_(int64_t old_n, int64_t new_n) {
    const int64_t n = new_n - old_n;
    if (n <= 0) return;
    std::vector<int32_t> tmp(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      tmp[static_cast<size_t>(i)] = (*page_table_ref_)[static_cast<size_t>(old_n + i)];
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto src = torch::from_blob(tmp.data(), {n}, opts).clone();
    page_table_t_.narrow(0, old_n, n).copy_(src.to(pool_->device()));
  }

  void write_n_tokens_(int32_t v) {
    if (pool_->device().is_cpu()) {
      n_tokens_t_.fill_(v);
    } else {
      auto stage = torch::tensor(v, torch::TensorOptions().dtype(torch::kInt32));
      n_tokens_t_.copy_(stage.to(pool_->device(), /*non_blocking=*/true));
    }
  }

  void write_layer_(int64_t layer, torch::Tensor k, torch::Tensor v,
                    int64_t start, int64_t end) {
    const int64_t S = end - start;
    auto& pt = *page_table_ref_;

    std::vector<int64_t> pg_idx(static_cast<size_t>(S));
    std::vector<int64_t> slot_idx(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
      const int64_t g = start + i;
      const int64_t blk = g / pool_->page_size();
      const int64_t off = g % pool_->page_size();
      pg_idx[static_cast<size_t>(i)]   = static_cast<int64_t>(pt[static_cast<size_t>(blk)]);
      slot_idx[static_cast<size_t>(i)] = off;
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto pg_t   = torch::from_blob(pg_idx.data(),   {S}, opts).clone().to(pool_->device());
    auto slot_t = torch::from_blob(slot_idx.data(), {S}, opts).clone().to(pool_->device());

    auto k_src = k.select(0, 0).permute({1, 0, 2}).contiguous();
    auto v_src = v.select(0, 0).permute({1, 0, 2}).contiguous();

    auto k_pool_ref = pool_->k_pool(layer);
    auto v_pool_ref = pool_->v_pool(layer);
    k_pool_ref.index_put_({pg_t, slot_t}, k_src.to(k_pool_ref.dtype()));
    v_pool_ref.index_put_({pg_t, slot_t}, v_src.to(v_pool_ref.dtype()));
  }

  std::pair<torch::Tensor, torch::Tensor> gather_(int64_t layer, int64_t total) const {
    const auto& pt = *page_table_ref_;
    const int64_t page_size = pool_->page_size();
    const int64_t n_blocks  = (total + page_size - 1) / page_size;
    std::vector<int64_t> idx(static_cast<size_t>(n_blocks));
    for (int64_t i = 0; i < n_blocks; ++i) {
      idx[static_cast<size_t>(i)] = static_cast<int64_t>(pt[static_cast<size_t>(i)]);
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto idx_t = torch::from_blob(idx.data(), {n_blocks}, opts).clone().to(pool_->device());

    auto k_blocks = pool_->k_pool(layer).index_select(0, idx_t);
    auto v_blocks = pool_->v_pool(layer).index_select(0, idx_t);
    auto k_flat = k_blocks.reshape({n_blocks * page_size, pool_->n_kv_heads(), pool_->head_dim()})
                          .narrow(0, 0, total);
    auto v_flat = v_blocks.reshape({n_blocks * page_size, pool_->n_kv_heads(), pool_->head_dim()})
                          .narrow(0, 0, total);
    auto k_out = k_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();
    auto v_out = v_flat.permute({1, 0, 2}).unsqueeze(0).contiguous();
    return {k_out, v_out};
  }

  SharedBlockPool* pool_;
  std::vector<int32_t>* page_table_ref_;
  int64_t* logical_len_ref_;
  int64_t max_pages_;
  torch::Tensor page_table_t_;
  torch::Tensor n_tokens_t_;
};

}  // namespace

// ── Scheduler ────────────────────────────────────────────────────────────

Scheduler::Scheduler(Transformer model,
                     SharedBlockPool* pool,
                     int64_t prefill_chunk_size)
    : model_(std::move(model)),
      pool_(pool),
      prefill_chunk_size_(prefill_chunk_size),
      device_(pool->device()),
      model_dtype_(pool->dtype()) {}

int64_t Scheduler::admit(const std::vector<int32_t>& prompt_tokens,
                         int64_t max_new_tokens,
                         double temperature,
                         int64_t top_k,
                         double top_p,
                         double repetition_penalty,
                         uint32_t seed,
                         int64_t eos_id) {
  SchedulerRequest req;
  req.id = next_id_++;
  req.prompt_tokens = prompt_tokens;
  req.max_new_tokens = max_new_tokens;
  req.temperature = temperature;
  req.top_k = top_k;
  req.top_p = top_p;
  req.repetition_penalty = repetition_penalty;
  req.rng.seed(seed);
  req.eos_id = eos_id;

  // Walk the prompt in page-size blocks and find_or_lease each block.
  // Track the longest cache-hit prefix: those positions' K/V is already
  // populated by a prior request, so prefill can skip them.
  const int64_t page_size = pool_->page_size();
  const int64_t n_full_blocks = static_cast<int64_t>(prompt_tokens.size()) / page_size;
  int64_t lcp_blocks = 0;
  uint64_t prev_hash = 0;
  bool still_hitting = true;

  for (int64_t b = 0; b < n_full_blocks; ++b) {
    const int32_t* ids = prompt_tokens.data() + b * page_size;
    auto lease = pool_->find_or_lease(prev_hash, ids, page_size);
    req.page_table.push_back(lease.page_idx);
    prev_hash = lease.hash;
    if (still_hitting && lease.was_hit) {
      lcp_blocks++;
    } else {
      still_hitting = false;
    }
  }

  // Final partial block (if prompt length not page-aligned): always a
  // fresh allocation — we never hash partial blocks.
  const int64_t leftover = static_cast<int64_t>(prompt_tokens.size()) - n_full_blocks * page_size;
  if (leftover > 0) {
    req.page_table.push_back(pool_->allocate());
  }

  // K/V for [0, lcp_blocks * page_size) is already valid in the pool.
  req.logical_len  = lcp_blocks * page_size;
  req.prefill_done = lcp_blocks * page_size;
  req.status = SchedulerRequest::Status::Prefilling;

  // If everything was a hit AND prompt_len is page-aligned, prefill is
  // effectively done — sample directly from the last cached position's
  // logits. We still need one prefill step to compute logits for the
  // last position (the K/V is there but logits aren't cached). Handled
  // uniformly by the prefill chunk loop with chunk_done == prompt_len.

  requests_.push_back(std::move(req));
  return requests_.back().id;
}

SchedulerRequest* Scheduler::pick_() {
  for (auto& r : requests_) {
    if (r.status != SchedulerRequest::Status::Done) return &r;
  }
  return nullptr;
}

bool Scheduler::step() {
  auto* req = pick_();
  if (!req) return false;

  if (req->status == SchedulerRequest::Status::Prefilling) {
    run_prefill_chunk_(*req);
  } else if (req->status == SchedulerRequest::Status::Decoding) {
    run_decode_step_(*req);
  }
  return true;
}

void Scheduler::run_until_done() {
  while (step()) {}
}

int64_t Scheduler::active_count() const {
  int64_t n = 0;
  for (const auto& r : requests_) {
    if (r.status != SchedulerRequest::Status::Done) n++;
  }
  return n;
}

std::unique_ptr<IPagedKVCache> Scheduler::build_view_(SchedulerRequest& req) {
  return std::make_unique<SharedPoolKVCache>(
      pool_, &req.page_table, &req.logical_len, pool_->max_pages());
}

int64_t Scheduler::sample_logits_(torch::Tensor logits_1d,
                                   SchedulerRequest& req) {
  // Defer to the shared device-resident sampler: repetition penalty,
  // temperature, top-k, softmax, top-p (O(V) bucket-radix kernel) and the draw
  // all run on the device the logits live on; the only D->H is the token id —
  // not the full [V] vocab this used to .cpu() every step. (Sampling uses
  // torch's generator rather than req.rng; greedy is deterministic.)
  std::vector<int64_t> rep_tokens;
  if (req.repetition_penalty != 1.0) {
    rep_tokens.reserve(req.prompt_tokens.size() + req.generated_tokens.size());
    for (auto t : req.prompt_tokens)    rep_tokens.push_back(static_cast<int64_t>(t));
    for (auto t : req.generated_tokens) rep_tokens.push_back(static_cast<int64_t>(t));
  }
  return gpu_sample(logits_1d, req.temperature, req.top_k, req.top_p,
                    rep_tokens, req.repetition_penalty);
}

void Scheduler::run_prefill_chunk_(SchedulerRequest& req) {
  const int64_t prompt_len = static_cast<int64_t>(req.prompt_tokens.size());
  if (req.prefill_done >= prompt_len) {
    // No more prefill work — last sample should have transitioned us
    // to Decoding; if not, do that now.
    req.status = SchedulerRequest::Status::Decoding;
    return;
  }
  const int64_t this_chunk = std::min(prefill_chunk_size_, prompt_len - req.prefill_done);

  auto view = build_view_(req);

  // Build input tensor: tokens [prefill_done, prefill_done + this_chunk).
  std::vector<int64_t> ids(static_cast<size_t>(this_chunk));
  for (int64_t i = 0; i < this_chunk; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int64_t>(
        req.prompt_tokens[static_cast<size_t>(req.prefill_done + i)]);
  }
  auto input = torch::from_blob(ids.data(), {1, this_chunk},
                                torch::TensorOptions().dtype(torch::kInt64))
                   .clone()
                   .to(device_);

  torch::NoGradGuard no_grad;
  auto logits = model_->forward_paged(input, view.get());

  req.prefill_done += this_chunk;
  if (req.prefill_done >= prompt_len) {
    // Sample first generated token from the last position's logits.
    auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
    int64_t next_id = sample_logits_(next_logits, req);
    req.generated_tokens.push_back(static_cast<int32_t>(next_id));
    req.last_token = next_id;
    if (req.eos_id >= 0 && next_id == req.eos_id) {
      req.status = SchedulerRequest::Status::Done;
    } else if (static_cast<int64_t>(req.generated_tokens.size()) >= req.max_new_tokens) {
      req.status = SchedulerRequest::Status::Done;
    } else {
      req.status = SchedulerRequest::Status::Decoding;
    }
  }
}

void Scheduler::run_decode_step_(SchedulerRequest& req) {
  // Need a page for the slot we're about to write (if we just crossed
  // a page boundary).
  const int64_t page_size = pool_->page_size();
  const int64_t blocks_needed = (req.logical_len + 1 + page_size - 1) / page_size;
  const int64_t blocks_have   = static_cast<int64_t>(req.page_table.size());
  std::unique_ptr<SharedPoolKVCache> view_owned;
  SharedPoolKVCache* view_raw = nullptr;
  if (blocks_needed > blocks_have) {
    req.page_table.push_back(pool_->allocate());
  }
  auto view = build_view_(req);
  view_raw = static_cast<SharedPoolKVCache*>(view.get());
  // The view's stable page_table tensor was synced at construction; if we
  // added a new page above it picked it up. Good.

  // H5 (partial): reuse one [1,1] device buffer for the decode input instead of
  // a per-token vector + from_blob + clone + H->D. (The bigger server-side win —
  // caching the SharedPoolKVCache view per request instead of rebuilding it
  // every token in build_view_ — is tracked separately; it needs the page-table
  // device tensor to grow in place when a page is added.)
  static thread_local torch::Tensor in_buf;
  if (!in_buf.defined() || in_buf.device() != device_) {
    in_buf = torch::empty({1, 1},
                          torch::TensorOptions().dtype(torch::kInt64).device(device_));
  }
  in_buf.fill_(req.last_token);

  torch::NoGradGuard no_grad;
  auto logits = model_->forward_paged(in_buf, view.get());
  auto next_logits = logits.select(1, 0).squeeze(0);
  int64_t next_id = sample_logits_(next_logits, req);
  req.generated_tokens.push_back(static_cast<int32_t>(next_id));
  req.last_token = next_id;
  (void)view_raw;

  if (req.eos_id >= 0 && next_id == req.eos_id) {
    req.status = SchedulerRequest::Status::Done;
  } else if (static_cast<int64_t>(req.generated_tokens.size()) >= req.max_new_tokens) {
    req.status = SchedulerRequest::Status::Done;
  }
}

}  // namespace olmo_cpp
