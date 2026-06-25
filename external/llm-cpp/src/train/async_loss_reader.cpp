/**
 * src/train/async_loss_reader.cpp
 *
 * Implementation of the non-blocking loss reader (item AA).
 */

#include "olmo_cpp/train/async_loss_reader.hpp"

#include <stdexcept>

#ifdef OLMO_HAS_CUDA_STREAM
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()
#endif

namespace olmo_cpp {

AsyncLossReader::AsyncLossReader(size_t depth)
#ifdef OLMO_HAS_CUDA_STREAM
    : copy_stream_(c10::cuda::getStreamFromPool())
#endif
{
  slots_.resize(depth);
  for (auto& s : slots_) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(false);
    s.pinned_host = torch::zeros({1}, opts);
  }
}

AsyncLossReader::~AsyncLossReader() {
#ifdef OLMO_HAS_CUDA_STREAM
  for (auto& s : slots_) {
    if (s.event) cudaEventDestroy(s.event);
  }
  if (compute_barrier_) cudaEventDestroy(compute_barrier_);
#endif
}

void AsyncLossReader::queue(const torch::Tensor& loss) {
  Slot& s = slots_[write_head_];
  s.valid = true;
  s.consumed = false;

#ifdef OLMO_HAS_CUDA_STREAM
  if (loss.is_cuda()) {
    if (!stream_inited_) {
      // First CUDA use: re-allocate pinned host buffers, lazy-create events.
      for (auto& slot : slots_) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        slot.pinned_host = torch::zeros({1}, opts);
        if (!slot.event) {
          cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming);
        }
      }
      // Reusable cross-stream fence event (no timing overhead).
      if (!compute_barrier_) {
        cudaEventCreateWithFlags(&compute_barrier_, cudaEventDisableTiming);
      }
      stream_inited_ = true;
    }

    // Cross-stream ordering fix: accum_loss_tensor.add_() ran on the compute
    // stream. Without this fence, copy_stream_ can race ahead and read the
    // zero-initialised value (from .zero_() earlier in the step) instead of
    // the accumulated loss. Record on the compute stream, wait on copy_stream_
    // so the cast+copy are guaranteed to see the updated device memory.
    cudaStream_t compute_stream = at::cuda::getCurrentCUDAStream().stream();
    cudaEventRecord(compute_barrier_, compute_stream);
    cudaStreamWaitEvent(copy_stream_.stream(), compute_barrier_, 0);

    c10::cuda::CUDAStreamGuard guard(copy_stream_);
    auto loss_f = loss.detach().to(torch::kFloat32).reshape({1});
    s.pinned_host.copy_(loss_f, /*non_blocking=*/true);
    cudaEventRecord(s.event, copy_stream_.stream());
    s.event_recorded = true;
  } else
#endif
  {
    s.cpu_value = loss.detach().to(torch::kFloat32).reshape({1}).item<float>();
#ifdef OLMO_HAS_CUDA_STREAM
    s.event_recorded = false;
#endif
  }

  write_head_ = (write_head_ + 1) % slots_.size();
}

float AsyncLossReader::poll() {
  for (size_t tries = 0; tries < slots_.size(); ++tries) {
    Slot& s = slots_[read_head_];
    if (!s.valid || s.consumed) {
      // Nothing here — try advancing the read head once to see if a
      // later slot has something (rare). Don't block.
      break;
    }
#ifdef OLMO_HAS_CUDA_STREAM
    if (s.event_recorded) {
      auto status = cudaEventQuery(s.event);
      if (status == cudaErrorNotReady) {
        return last_value_;  // not ready yet, keep last value
      }
      last_value_ = s.pinned_host.data_ptr<float>()[0];
    } else
#endif
    {
      last_value_ = s.cpu_value;
    }
    s.consumed = true;
    read_head_ = (read_head_ + 1) % slots_.size();
    return last_value_;
  }
  return last_value_;
}

float AsyncLossReader::flush() {
#ifdef OLMO_HAS_CUDA_STREAM
  if (stream_inited_) cudaStreamSynchronize(copy_stream_.stream());
#endif
  // Now every recorded event is complete. Drain the queue.
  while (true) {
    Slot& s = slots_[read_head_];
    if (!s.valid || s.consumed) break;
#ifdef OLMO_HAS_CUDA_STREAM
    if (s.event_recorded) {
      last_value_ = s.pinned_host.data_ptr<float>()[0];
    } else
#endif
    {
      last_value_ = s.cpu_value;
    }
    s.consumed = true;
    read_head_ = (read_head_ + 1) % slots_.size();
  }
  return last_value_;
}

}  // namespace olmo_cpp
