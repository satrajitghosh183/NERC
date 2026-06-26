#pragma once

#include "zwt/core/tensor.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace zwt::data {

// Zero-stall token loader.
//
// Design intent: the forward pass must never block waiting for a batch.
//
// Architecture:
//   * mmap the token file (single large i64 array on disk, as produced by
//     the existing prepare_data tool).
//   * A producer thread builds the next batch on CPU (gather into a pinned
//     host buffer) using a ring of N slots.
//   * A second thread kicks off the H2D copy on the dataloader's copy stream
//     the moment a slot is ready.
//   * get_batch() returns a device tensor whose data is already on the GPU
//     by the time the caller sees it. If compute is slow (typical), the
//     queue stays saturated; if compute is fast, get_batch() blocks — but
//     this is the same bottleneck every trainer hits. We add no overhead.
class TokenLoader {
 public:
  struct Options {
    std::string path;               // path to .npy or raw i64 token file
    int64_t     seq_len = 2048;
    int64_t     batch_size = 4;
    bool        shuffle = true;
    uint64_t    seed = 0xC0FFEE;
    int         ring_size = 3;
    Device      device = Device::cpu();
    int64_t     start_cursor = 0;   // resume: first chunk to serve
  };

  struct Batch {
    Tensor input;   // [B, S] i64
    Tensor target;  // [B, S] i64 — shifted by one
  };

  explicit TokenLoader(Options opts);
  ~TokenLoader();

  TokenLoader(const TokenLoader&) = delete;
  TokenLoader& operator=(const TokenLoader&) = delete;

  // Blocks until a batch is ready, then advances. Safe to call from the
  // training thread.
  Batch next();

  // Start the producer thread. Must be called after construction; get_batch
  // blocks on an empty ring otherwise.
  void start();

  int64_t steps_per_epoch() const;

  // Current position in the chunk order (for checkpointing).
  int64_t cursor() const { return chunk_cursor_.load(); }

 private:
  struct Slot {
    Tensor host_input;   // pinned or plain host memory
    Tensor host_target;
    Tensor dev_input;    // device-resident copy (if device is CUDA)
    Tensor dev_target;
    std::atomic<bool> ready{false};
    std::atomic<bool> consumed{true};
  };

  void producer_loop_();
  void fill_slot_(int slot_idx);

  Options                     opts_;
  std::vector<int64_t>        tokens_;   // mmapped / loaded tokens
  int64_t                     num_chunks_ = 0;

  std::vector<std::unique_ptr<Slot>> ring_;
  std::atomic<int64_t>        produce_cursor_{0};
  std::atomic<int64_t>        consume_cursor_{0};
  std::mutex                  ring_mu_;
  std::condition_variable     ring_cv_;

  std::atomic<bool>           stop_{false};
  std::thread                 producer_thread_;

  std::vector<int64_t>        chunk_order_;
  std::atomic<int64_t>        chunk_cursor_{0};
};

}  // namespace zwt::data
