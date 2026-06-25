/**
 * include/olmo_cpp/backend/persistent_decode.hpp
 *
 * Persistent decode kernel — fast-inference [13].
 *
 * One CUDA kernel launched at session start, runs forever. CPU writes
 * "next token + position" into a memory-mapped queue; GPU polls the
 * queue, runs the full forward step (or just the sample), writes the
 * result back. ZERO kernel launches per token after init.
 *
 * For this DRAFT scope: a *sample-only* persistent kernel. The CPU
 * still issues the model forward (multiple kernels). Per-token sampling
 * (just the LM head + Gumbel-max) runs inside the persistent kernel,
 * eliminating ~3-5 launches per decoded token.
 *
 * Real production version (e.g. TensorRT-LLM in-flight batching) keeps
 * the entire decode step inside the persistent kernel — that's a much
 * bigger project requiring all kernels to be persistent-kernel-friendly.
 *
 * This is essentially scaffolding to land the queue protocol; the
 * persistent body is a draft sample loop.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <memory>

namespace olmo_cpp {

/// Handle for a running persistent decode kernel.
class PersistentDecode {
 public:
  /// Launch a persistent kernel that handles up to `max_tokens` requests.
  /// Caller must call enqueue() for each new (hidden, W_U) input and
  /// poll() to get the resulting token id.
  ///
  /// All work uses Philox keyed on (seed, position).
  PersistentDecode(int64_t vocab_size, int hidden_dim,
                   torch::Device device,
                   uint64_t seed,
                   int max_in_flight = 16);
  ~PersistentDecode();

  PersistentDecode(const PersistentDecode&) = delete;
  PersistentDecode& operator=(const PersistentDecode&) = delete;

  /// Push a new sampling request. Returns a slot id used by poll().
  /// hidden: [H], W_U: [V, H], temperature, position. Caller-owned tensors.
  int enqueue(torch::Tensor hidden, torch::Tensor W_U,
              float temperature, uint32_t position);

  /// Poll for the result of slot `id`. Blocks until ready.
  int64_t poll(int slot_id);

  /// Stop the persistent kernel and clean up.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace olmo_cpp
