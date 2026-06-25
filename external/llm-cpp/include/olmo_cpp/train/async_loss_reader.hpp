#pragma once

/**
 * include/olmo_cpp/train/async_loss_reader.hpp
 *
 * Non-blocking loss-scalar readout for training logs (item AA).
 *
 * Problem: at log boundaries, `accum_loss_tensor.item<float>()` issues a
 * full device→host sync that drains the compute stream. At ~60ms drain
 * per log step (per the speed-xp-sg profile) and a log_interval of 10
 * steps, that's a 0.6% straight-line slowdown for nothing — the logged
 * loss value can lag by a step or two with zero training impact.
 *
 * Mechanism: a small ring buffer of pinned host scalars + CUDA events.
 * On queue(), we cudaMemcpyAsync the device loss into the next pinned
 * slot on a side stream and record an event. On poll(), we check the
 * read-head event; if it's complete, hand back the slot's value and
 * advance the read head. Effective latency: log_interval-many steps
 * behind real-time, which is invisible in the log output.
 *
 * CPU fallback: poll() does the read inline (no async path needed).
 */

#include <torch/torch.h>
#include <cstdint>
#include <vector>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  define OLMO_HAS_CUDA_STREAM 1
#  include <c10/cuda/CUDAStream.h>
#  include <c10/cuda/CUDAGuard.h>
#endif

namespace olmo_cpp {

class AsyncLossReader {
 public:
  /// `depth` is the ring-buffer length. depth=4 means up to 4 outstanding
  /// log-step copies in flight before a poll(); larger = more tolerance
  /// for slow log handlers, smaller = lower latency. Default is fine.
  explicit AsyncLossReader(size_t depth = 4);
  ~AsyncLossReader();

  /// Non-blocking. Schedules `loss` for transfer to a pinned-host slot.
  /// On CPU `loss`, snapshots the value into the next slot synchronously
  /// (still cheap). The caller passes a 0-D float or float-castable tensor.
  void queue(const torch::Tensor& loss);

  /// Returns the oldest ready value, or `last_value_` if none ready yet.
  /// Always non-blocking. Use this in the log handler in place of
  /// `.item<float>()`.
  float poll();

  /// Drain everything queued, blocking. Use at end of training to flush
  /// the final loss value into the log.
  float flush();

  /// Last value returned by poll() / flush(). Useful for end-of-epoch
  /// summaries when the queue is empty.
  float last_value() const { return last_value_; }

 private:
  struct Slot {
    torch::Tensor pinned_host;   // [1] float on pinned host (CUDA only)
    float cpu_value = 0.0f;       // CPU fallback storage
    bool valid = false;           // has data been written?
    bool consumed = true;         // has poll() read it?
#ifdef OLMO_HAS_CUDA_STREAM
    cudaEvent_t event = nullptr;
    bool event_recorded = false;
#endif
  };

  std::vector<Slot> slots_;
  size_t write_head_ = 0;
  size_t read_head_  = 0;
  float last_value_  = 0.0f;
#ifdef OLMO_HAS_CUDA_STREAM
  c10::cuda::CUDAStream copy_stream_;
  bool stream_inited_ = false;
  cudaEvent_t compute_barrier_ = nullptr;  // cross-stream fence: compute→copy
#endif
};

}  // namespace olmo_cpp
