/**
 * src/backend/persistent_decode.cpp
 *
 * Stub implementation of the persistent decode kernel handle
 * (fast-inference [13]).
 *
 * DRAFT — currently delegates each enqueue() to the existing
 * fused_lm_head_sample synchronous call. The real persistent kernel
 * (one long-running CUDA kernel polling a host-mapped queue) is the
 * actual deliverable; this stub satisfies the API so calling code can
 * be written against the right interface.
 *
 * Why a stub: writing a production persistent CUDA kernel that handles
 * a queue, signaled events, and graceful shutdown is its own multi-week
 * project. The CUDA primitives needed (cuStreamWaitValue, host-mapped
 * memory, polling) all exist but the orchestration is non-trivial.
 *
 * Once the real kernel is in place, swap out the body of Impl::enqueue
 * to write the request into the device-side queue and Impl::poll to
 * wait on the device-side completion flag.
 */

#include "olmo_cpp/backend/persistent_decode.hpp"
#include "olmo_cpp/backend/fused_lm_head_sample.hpp"

#include <vector>
#include <stdexcept>

namespace olmo_cpp {

struct PersistentDecode::Impl {
  uint64_t seed;
  int max_in_flight;
  torch::Device device;

  // Per-slot pending result. Synchronous stub: enqueue computes the
  // result immediately, poll just reads it back.
  std::vector<int64_t> results;
  std::vector<bool>    occupied;

  Impl(int64_t /*V*/, int /*H*/, torch::Device dev, uint64_t s, int n)
      : seed(s), max_in_flight(n), device(dev),
        results(static_cast<size_t>(n), -1),
        occupied(static_cast<size_t>(n), false) {}

  int allocate_slot() {
    for (int i = 0; i < max_in_flight; ++i) {
      if (!occupied[static_cast<size_t>(i)]) {
        occupied[static_cast<size_t>(i)] = true;
        return i;
      }
    }
    throw std::runtime_error("PersistentDecode: no free slots");
  }
};

PersistentDecode::PersistentDecode(int64_t vocab_size, int hidden_dim,
                                   torch::Device device,
                                   uint64_t seed, int max_in_flight)
    : impl_(std::make_unique<Impl>(vocab_size, hidden_dim, device, seed, max_in_flight)) {}

PersistentDecode::~PersistentDecode() = default;

int PersistentDecode::enqueue(torch::Tensor hidden, torch::Tensor W_U,
                              float temperature, uint32_t position) {
  int slot = impl_->allocate_slot();
  // Synchronous stub: compute the result immediately. Real implementation
  // would push (hidden, W_U, T, position, slot) into the device queue and
  // return without blocking.
  impl_->results[static_cast<size_t>(slot)] =
      fused_lm_head_sample(hidden, W_U, temperature, impl_->seed, position);
  return slot;
}

int64_t PersistentDecode::poll(int slot_id) {
  if (slot_id < 0 || slot_id >= impl_->max_in_flight) {
    throw std::out_of_range("PersistentDecode::poll: bad slot id");
  }
  int64_t r = impl_->results[static_cast<size_t>(slot_id)];
  impl_->occupied[static_cast<size_t>(slot_id)] = false;
  return r;
}

void PersistentDecode::stop() {
  // No-op in the stub.
}

}  // namespace olmo_cpp
