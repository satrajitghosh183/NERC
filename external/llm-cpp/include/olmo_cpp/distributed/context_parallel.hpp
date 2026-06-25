#pragma once

/**
 * include/olmo_cpp/distributed/context_parallel.hpp
 *
 * Public API for Context Parallelism (CP). CP shards the sequence dimension S
 * across `cp_size` ranks: each rank owns a contiguous chunk S/cp of every
 * (B,S,D) activation tensor. The model parameters are *replicated* across CP
 * ranks (unlike TP/FSDP which shard parameters). The non-trivial part is
 * self-attention, which is globally coupled across the sequence; we resolve
 * that with *ring attention*: each rank starts with its local (Q,K,V) and
 * rotates K/V around a logical ring via point-to-point send/recv (one
 * neighbour pair per step, cp_size steps total), running an online softmax
 * accumulator so the final per-rank output equals what a non-distributed
 * attention would have produced for that rank's query slice. Collective
 * primitives used: p2p send/recv (ring KV exchange) and allgather
 * (gather_sequence reassembly). Causal masking uses absolute positions
 * derived from `rank_ * S_local` so the ring step that wraps "into the
 * future" is masked out.
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/context_parallel.cpp: out-of-line method definitions
 *   Direct callers in model code not located via quick grep (CP is wired in
 *   at the attention call site by the trainer when enabled).
 *
 * --- Role in training pipeline ---
 *   Used during forward pass of attention when sequences are too long to fit
 *   on a single device. Lets us train with effective sequence length S
 *   while each GPU only ever materialises Q,K,V of length S/cp_size.
 */

#include <torch/torch.h>
#include <optional>

#ifdef OLMO_HAS_DDP
// c10d::Backend is the abstract collective-comm interface (Gloo/NCCL/MPI).
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Context parallelism: splits sequence dimension across ranks.
/// Uses ring attention to compute full causal attention over distributed sequence chunks.
/// Parameters are replicated; only activations along S are sharded.
class ContextParallelContext {
 public:
#ifdef OLMO_HAS_DDP
  /// Factory: returns nullopt if `backend` is null or `cp_size < 2` (no CP).
  /// `cp_size` is the size of the CP sub-group; rank within the group is
  /// derived as `backend->getRank() % cp_size`.
  static std::optional<ContextParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend, int cp_size);
#endif

  /// Scatter: split sequence into per-rank chunks [B,S,D] -> [B,S/cp,D].
  /// Local-only operation (no collective): each rank narrows along dim=1.
  torch::Tensor scatter_sequence(torch::Tensor x);

  /// Ring attention: compute full causal attention across distributed chunks.
  /// q,k,v: [B,H,S/cp,D] -> output: [B,H,S/cp,D]. Performs cp_size-1 p2p
  /// send/recv pairs to rotate K/V around the ring while accumulating an
  /// online softmax (numerically stable log-sum-exp combine).
  torch::Tensor ring_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

  /// Gather: reassemble full sequence [B,S/cp,D] -> [B,S,D] via allgather.
  torch::Tensor gather_sequence(torch::Tensor x);

  /// Rank within the CP sub-group (0..cp_size-1).
  int cp_rank() const { return rank_; }
  /// Total ranks participating in CP.
  int cp_size() const { return cp_size_; }

 private:
#ifdef OLMO_HAS_DDP
  // Real ctor used only when distributed support is compiled in.
  ContextParallelContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int cp_size);
  c10::intrusive_ptr<c10d::Backend> backend_;  // c10d collective backend (Gloo/NCCL).
#else
  // Stub ctor compiled when OLMO_HAS_DDP is undefined; CP becomes a no-op pass-through.
  ContextParallelContext(int rank, int cp_size);
#endif
  int rank_;     // Rank within CP sub-group.
  int cp_size_;  // Number of ranks in the sub-group.
};

}  // namespace olmo_cpp
