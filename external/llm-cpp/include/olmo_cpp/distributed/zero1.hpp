#pragma once

/**
 * include/olmo_cpp/distributed/zero1.hpp
 *
 * ZeRO-1 (sharded optimizer state) — fast-inference [T-4].
 *
 * Idea: optimizer state (Adam first/second moments, Muon momentum, etc.)
 * is the largest memory-consumer after the parameters themselves. Sharding
 * the state across DP ranks cuts that per-rank memory by world_size× at
 * the cost of one extra allgather per step.
 *
 * Protocol (per training step):
 *   1. Forward / backward as usual. DDP allreduce makes the gradient on
 *      every rank identical (the average across ranks).
 *   2. Each rank applies the optimizer update to ONLY the parameters it
 *      owns. (Other params still receive the same gradient but it's not
 *      consumed locally — those updates happen on other ranks.)
 *   3. allgather_params() broadcasts each updated parameter from its
 *      owning rank to all other ranks, so the next forward sees identical
 *      weights on every replica.
 *
 * Partitioning strategy: parameters are assigned to ranks by their index
 * mod world_size. Simple and load-balances reasonably across deep
 * Transformers where most params are similar-sized matmul weights. For
 * heavily imbalanced cases (very large vocab embedding + smaller MLPs)
 * a size-aware bin-packed partition would do better; not worth it here.
 *
 * Integration with existing optimizers: this class is intentionally an
 * orchestrator, not a wrapper. The caller calls `partition()` to get
 * just-this-rank's parameter subset, constructs its optimizer of choice
 * over that subset (AdamW, Muon, Lion, Dion — whatever), and calls
 * `allgather_params()` after every optimizer.step() to sync replicas.
 * Works for any optimizer because we never touch its internals.
 *
 * Without a real c10d backend (single-process build, no Gloo), the
 * partition is a no-op (returns the full list) and allgather is a no-op,
 * so wrapper code is single-rank-clean.
 */

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <vector>
#include <memory>

namespace olmo_cpp {

class OptimizerStateSharder {
 public:
  /// `backend` may be null (single-process build). `rank` / `world_size`
  /// match DDPContext. With backend==null, every call is a no-op pass-
  /// through.
  OptimizerStateSharder(c10::intrusive_ptr<c10d::Backend> backend,
                        int rank, int world_size)
      : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {}

  /// Returns the subset of `all_params` this rank owns. The caller
  /// constructs its optimizer over this subset only. Order within the
  /// subset matches creation order so the optimizer's internal state
  /// vector lines up the way the underlying optimizer expects.
  std::vector<torch::Tensor> partition(
      const std::vector<torch::Tensor>& all_params) const;

  /// Broadcast each parameter from its owning rank to all other ranks.
  /// Call once per training step, after the local optimizer's step().
  /// Modifies `all_params` in place (each tensor's data is overwritten
  /// with the owning rank's copy).
  void allgather_params(std::vector<torch::Tensor>& all_params);

  /// Owner rank for a given parameter index. Trivially `idx % world_size`.
  int owner_of(size_t idx) const {
    return world_size_ == 0 ? 0 : static_cast<int>(idx % static_cast<size_t>(world_size_));
  }

  bool is_active() const { return backend_ && world_size_ > 1; }
  int rank() const { return rank_; }
  int world_size() const { return world_size_; }

 private:
  c10::intrusive_ptr<c10d::Backend> backend_;
  int rank_;
  int world_size_;
};

}  // namespace olmo_cpp
