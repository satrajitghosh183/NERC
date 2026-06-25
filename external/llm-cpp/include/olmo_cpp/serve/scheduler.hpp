#pragma once

/**
 * include/olmo_cpp/serve/scheduler.hpp
 *
 * Continuous-batching style multi-request scheduler — fast-inference [I-2].
 *
 * Runs many in-flight requests against a single Transformer + a single
 * SharedBlockPool. Each request keeps:
 *   - its own token stream (prompt + generated)
 *   - its own logical page table (vector<int32_t> indexing into the pool)
 *   - its own sampling params
 *   - its own status (Pending / Prefilling / Decoding / Done)
 *
 * Each call to step() picks ONE request (FCFS over decoding requests, with
 * a configurable per-step prefill quantum so a long prompt does not
 * starve running decodes) and advances it by one chunk-of-prefill or one
 * decode step.
 *
 * Prefix caching: when a new request is admitted, the scheduler walks its
 * prompt in page-size chunks and find_or_lease()s each block from the
 * pool. Already-populated pages (matching prefix from a prior request)
 * are reused — only the suffix's K/V is recomputed during prefill. The
 * net effect is "system prompt" or "shared instruction" requests share
 * their first N pages physically.
 *
 * This is NOT a batched-forward implementation: each step still runs the
 * model on exactly one request's tokens. The win is in scheduling
 * (a long prefill no longer blocks the world; new requests admit into
 * the middle of running decodes) and in prefix sharing (memory + skip-
 * compute). True batched forward across requests is a follow-on that
 * needs a kernel extension to accept per-request page tables.
 *
 * Single-threaded; no internal synchronization. The caller serializes.
 */

#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/shared_block_pool.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <deque>

namespace olmo_cpp {

class SchedulerRequest {
 public:
  enum class Status { Pending, Prefilling, Decoding, Done };

  int64_t id;
  std::vector<int32_t> prompt_tokens;      // input prompt (set once)
  std::vector<int32_t> generated_tokens;   // tokens emitted by the model
  std::vector<int32_t> page_table;         // logical_block -> physical page
  int64_t logical_len = 0;                 // current cached seq length
  int64_t prefill_done = 0;                // how many prompt tokens have been
                                           // physically processed so far
  int64_t max_new_tokens;
  // Sampling params (same shape as chat.cpp's sample_logits).
  double temperature = 0.0;
  int64_t top_k = 0;
  double top_p = 1.0;
  double repetition_penalty = 1.0;
  // RNG: per-request so two identical prompts can still diverge under
  // sampling, and so a request is reproducible by seed.
  std::mt19937 rng;
  Status status = Status::Pending;

  /// Last sampled logit -> next generated token. -1 = no token yet.
  int64_t last_token = -1;
  /// EOS id; populated by the scheduler at admit time.
  int64_t eos_id = -1;
};

class Scheduler {
 public:
  /// `prefill_chunk_size` is the per-step prefill quantum. With chunk=64
  /// and 4 concurrent requests in mixed prefill/decode states, the
  /// scheduler interleaves: every 4 steps each request advances either
  /// by one decode token or by `chunk` prefill tokens.
  Scheduler(Transformer model,
            SharedBlockPool* pool,
            int64_t prefill_chunk_size = 64);

  /// Admit a new request. Returns its request id. Walks the prompt in
  /// page-size blocks, leases pages from the pool (hit -> share with an
  /// existing request, miss -> fresh page). Sets logical_len to the
  /// length of the longest cache-hit prefix.
  int64_t admit(const std::vector<int32_t>& prompt_tokens,
                int64_t max_new_tokens,
                double temperature = 0.0,
                int64_t top_k = 0,
                double top_p = 1.0,
                double repetition_penalty = 1.0,
                uint32_t seed = 0xC0FFEE,
                int64_t eos_id = -1);

  /// Advance one request by one step (prefill chunk or one decode token).
  /// Returns true if work was done; false when no live requests remain.
  bool step();

  /// Drive step() in a loop until every request is Done.
  void run_until_done();

  /// Snapshot of all requests, including completed ones. Tests / drivers
  /// inspect this to read generated_tokens out.
  const std::deque<SchedulerRequest>& requests() const { return requests_; }
  std::deque<SchedulerRequest>& requests() { return requests_; }

  /// Active (non-Done) request count.
  int64_t active_count() const;

 private:
  // Find the next request to advance. FCFS: lowest-id active request.
  SchedulerRequest* pick_();

  // Run one prefill chunk on `req`. Writes K/V into the pool for any new
  // logical positions [logical_len, logical_len + chunk). Transitions to
  // Decoding when prefill_done == prompt.size().
  void run_prefill_chunk_(SchedulerRequest& req);

  // Run one decode step on `req`. Generates one token, updates state,
  // marks Done on EOS or max_new_tokens.
  void run_decode_step_(SchedulerRequest& req);

  // Build a temporary IPagedKVCache view over the pool + this request's
  // page table for the model forward. Lifetime: per call.
  std::unique_ptr<IPagedKVCache> build_view_(SchedulerRequest& req);

  // Sampling — identical semantics to tools/chat.cpp's sample_logits but
  // pulled in here so the scheduler is self-contained.
  int64_t sample_logits_(torch::Tensor logits_1d,
                         SchedulerRequest& req);

  Transformer model_;
  SharedBlockPool* pool_;
  int64_t prefill_chunk_size_;
  int64_t next_id_ = 0;
  std::deque<SchedulerRequest> requests_;
  torch::Device device_;
  torch::Dtype  model_dtype_;
};

}  // namespace olmo_cpp
