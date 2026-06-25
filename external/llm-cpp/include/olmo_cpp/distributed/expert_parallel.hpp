#pragma once

/**
 * include/olmo_cpp/distributed/expert_parallel.hpp
 *
 * Public API for Expert Parallelism (EP) used in Mixture-of-Experts (MoE)
 * layers. EP shards the *experts* themselves across ranks: each of `ep_size`
 * ranks owns `num_experts / ep_size` experts (must divide evenly). Token
 * routing assigns each token to top_k experts. Because tokens and the
 * experts they are routed to may live on different ranks, we must shuffle
 * tokens across the network in two phases:
 *   - dispatch: token -> rank-owning-its-expert (forward all-to-all)
 *   - combine:  expert output -> originating rank (reverse all-to-all)
 *
 * Collective ops used:
 *   - allgather (small) to exchange per-rank send-counts so every rank
 *     learns its recv-counts.
 *   - point-to-point send/recv per (sender,receiver) pair to realise the
 *     all-to-all (this implementation does not call a fused alltoall).
 * Non-expert MLP/attention parameters are replicated (or sharded by some
 * other axis like FSDP/TP); only the expert FFN weights are EP-sharded.
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/expert_parallel.cpp: implements dispatch/combine
 *   Direct call sites in MoE layer code not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   Invoked inside the MoE forward pass: dispatch before per-expert FFN,
 *   combine after, then the routing weights blend the per-expert outputs
 *   back into per-token activations.
 */

#include <torch/torch.h>
#include <optional>
#include <tuple>

#ifdef OLMO_HAS_DDP
// c10d::Backend gives us the send/recv/allgather primitives we need.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Expert parallelism: distributes MoE experts across ranks.
/// Uses all-to-all (built from per-pair send/recv) to dispatch tokens to the
/// rank owning their assigned expert and to combine results back.
class ExpertParallelContext {
 public:
#ifdef OLMO_HAS_DDP
  /// Factory: returns nullopt unless a backend exists and ep_size >= 2.
  /// Throws if num_experts is not a multiple of ep_size.
  static std::optional<ExpertParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend, int num_experts, int ep_size);
#endif

  /// Dispatch tokens to expert-owning ranks via all-to-all.
  /// tokens: [num_tokens, D], expert_ids: [num_tokens, top_k]
  /// Returns: local_tokens [local_count, D], metadata for combine.
  /// Since each token may go to top_k experts, rows are duplicated top_k
  /// times before routing, so local_count = sum_r (tokens going to rank r).
  struct DispatchResult {
    torch::Tensor local_tokens;  // tokens assigned to this rank's experts (post-shuffle)
    torch::Tensor local_ids;     // local expert indices (0..experts_per_rank-1)
    torch::Tensor weights;       // routing weights for local tokens
    torch::Tensor send_counts;   // how many tokens this rank sent to each peer
    torch::Tensor recv_counts;   // how many tokens this rank received from each peer
  };

  /// Forward shuffle: send tokens to the rank hosting their assigned expert.
  /// Performs a small allgather of send-counts followed by per-pair p2p
  /// send/recv to realise the all-to-all.
  DispatchResult dispatch(torch::Tensor tokens, torch::Tensor expert_ids,
                          torch::Tensor weights);

  /// Combine: gather expert outputs back to original ranks via reverse
  /// all-to-all (send/recv counts swapped vs. dispatch).
  torch::Tensor combine(torch::Tensor expert_outputs,
                         const DispatchResult& dispatch_info);

  /// First global expert id owned by this rank (inclusive).
  int local_expert_start() const { return rank_ * experts_per_rank_; }
  /// One past the last global expert id owned by this rank.
  int local_expert_end() const { return (rank_ + 1) * experts_per_rank_; }
  /// Number of experts hosted on this rank.
  int num_local_experts() const { return experts_per_rank_; }
  int rank() const { return rank_; }
  int ep_size() const { return ep_size_; }

 private:
#ifdef OLMO_HAS_DDP
  ExpertParallelContext(c10::intrusive_ptr<c10d::Backend> backend,
                        int rank, int ep_size, int num_experts);
  c10::intrusive_ptr<c10d::Backend> backend_;  // collective backend for all-to-all.
#else
  // Fallback ctor: dispatch/combine become identity-like in single-process mode.
  ExpertParallelContext(int rank, int ep_size, int num_experts);
#endif
  int rank_, ep_size_, num_experts_, experts_per_rank_;
  // Invariant: experts_per_rank_ * ep_size_ == num_experts_.
};

}  // namespace olmo_cpp
