#pragma once

/**
 * include/olmo_cpp/distributed/fsdp.hpp
 *
 * Public API for Fully Sharded Data Parallel (FSDP), the C++ analogue of
 * PyTorch's torch.distributed.fsdp. FSDP shards *parameters*, *gradients*,
 * and *optimizer state* across world_size ranks, materialising the full
 * parameter tensors only briefly when needed for computation.
 *
 * Lifecycle of a forward+backward step (FULL_SHARD = ZeRO-3):
 *   1. unshard_params:     allgather   shards -> full param (compute)
 *   2. forward+backward run with full params; produces full grads.
 *   3. reshard_params:     drop full param, keep local shard (memory)
 *   4. reduce_scatter_grads: reduce-scatter SUM grads -> per-rank grad shard,
 *      divided by world_size (i.e. mean reduction across replicas).
 *   5. optimizer step runs on the local shard only.
 *
 * SHARD_GRAD_OP (ZeRO-2) keeps params replicated and only shards
 * grads/optim state; here we approximate this by short-circuiting
 * shard/unshard/reshard (a real impl would still reduce-scatter grads).
 * NO_SHARD degenerates to DDP semantics (allreduce grads, divide by W).
 *
 * HSDP variant: two process groups -- intra-node (shard) and inter-node
 * (replicate). Intra-group does the FSDP allgather/reduce-scatter; the
 * inter-group adds a final allreduce-mean across the replicate dimension
 * after reduce-scatter.
 *
 * Collective ops used:
 *   - allgather       (unshard params)
 *   - reduce_scatter  (gradients, with mean via /world_size)
 *   - allreduce       (NO_SHARD path; HSDP inter-group reduction)
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/fsdp.cpp: implementation
 *   Direct call sites in trainer not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   Memory-saving alternative to DDP: lets us train models too big to
 *   replicate per-GPU. Each rank's GPU memory holds only 1/W of params,
 *   grads, and optim state in steady state.
 */

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

#if defined(OLMO_HAS_DDP) || defined(OLMO_HAS_NCCL)
// c10d::Backend provides allgather / reduce_scatter / allreduce primitives.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#endif

namespace olmo_cpp {

/// Sharding strategy for FSDP. Mirrors the ZeRO-{1,2,3} hierarchy.
enum class ShardingStrategy {
  FULL_SHARD,    ///< ZeRO-3: shard params + grads + optimizer states
  SHARD_GRAD_OP, ///< ZeRO-2: shard grads + optimizer states only
  NO_SHARD       ///< DDP-like: only allreduce gradients
};

/// FSDP (Fully Sharded Data Parallel) context.
/// Shards model parameters across ranks for memory-efficient distributed training.
class FSDPContext {
 public:
#if defined(OLMO_HAS_DDP) || defined(OLMO_HAS_NCCL)
  /// Create FSDP context. Returns nullopt if backend is null or world_size < 2.
  /// `strategy` selects between FULL_SHARD (ZeRO-3), SHARD_GRAD_OP (ZeRO-2)
  /// and NO_SHARD (DDP-equivalent).
  static std::optional<FSDPContext> create(
      c10::intrusive_ptr<c10d::Backend> backend,
      ShardingStrategy strategy = ShardingStrategy::FULL_SHARD);

  /// HSDP variant: separate intra-node (shard via `intra_backend`) and
  /// inter-node (replicate via `inter_backend`) process groups. The intra
  /// group is the FSDP shard axis; the inter group does the final mean
  /// allreduce in reduce_scatter_grads.
  static std::optional<FSDPContext> create_hsdp(
      c10::intrusive_ptr<c10d::Backend> intra_backend,
      c10::intrusive_ptr<c10d::Backend> inter_backend,
      ShardingStrategy strategy = ShardingStrategy::FULL_SHARD);
#endif

  /// Initially shard full parameters: each rank keeps the (rank-th) 1/N
  /// flat slice of each tensor. Pads each param up to a multiple of N.
  /// In-place: reassigns `.data()` of each parameter to its local shard.
  void shard_params(std::vector<torch::Tensor>& params);

  /// Unshard (allgather) params for forward pass: each rank reconstructs
  /// the full tensor by allgathering all peer shards and concatenating.
  void unshard_params(std::vector<torch::Tensor>& params);

  /// Alias for unshard_params (kept for naming clarity at call sites).
  void allgather_params(std::vector<torch::Tensor>& params) { unshard_params(params); }

  /// Reduce-scatter gradients after backward: SUM-reduce across ranks and
  /// scatter so each rank holds the gradient slice corresponding to its
  /// param shard. Divided by world_size for mean semantics. NO_SHARD path
  /// falls back to allreduce-mean.
  void reduce_scatter_grads(std::vector<torch::Tensor>& grads);

  /// Re-shard params after forward (free full param memory by pointing the
  /// parameter tensor's data back at the small local shard).
  void reshard_params(std::vector<torch::Tensor>& params);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  ShardingStrategy strategy() const { return strategy_; }

 private:
#if defined(OLMO_HAS_DDP) || defined(OLMO_HAS_NCCL)
  FSDPContext(c10::intrusive_ptr<c10d::Backend> backend,
              c10::intrusive_ptr<c10d::Backend> inter_backend,
              int rank, int world_size, ShardingStrategy strategy);

  c10::intrusive_ptr<c10d::Backend> backend_;        // intra-node for HSDP, or single shard group.
  c10::intrusive_ptr<c10d::Backend> inter_backend_;  // inter-node for HSDP (nullable for plain FSDP).
#else
  // Stub ctor when distributed support is not compiled in.
  FSDPContext(int rank, int world_size, ShardingStrategy strategy);
#endif
  int rank_;
  int world_size_;
  ShardingStrategy strategy_;

  // Stored local shards (one per original param). Indexed parallel to the
  // user's `params` vector passed to shard_params.
  std::vector<torch::Tensor> sharded_params_;
  // Original (unflattened) shapes for unsharding -- needed because we
  // flatten + pad before sharding.
  std::vector<std::vector<int64_t>> original_shapes_;
  // Pre-pad numel so we can trim the padding zeros after allgather.
  std::vector<int64_t> original_numels_;
  // Tracks whether shard_params() has run (idempotency guard).
  bool is_sharded_ = false;
};

}  // namespace olmo_cpp
