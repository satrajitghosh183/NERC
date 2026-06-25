#pragma once

/**
 * include/olmo_cpp/distributed/ddp.hpp
 *
 * Public API for Distributed Data Parallel (DDP). DDP replicates the *full*
 * model on every rank and shards the *data*: each rank consumes a different
 * mini-batch slice. After backward, gradients are made bit-identical across
 * ranks via an allreduce-sum followed by division by world_size (i.e.
 * arithmetic mean of per-rank gradients). At init time, parameters are
 * broadcast from rank 0 so all replicas start from identical weights.
 *
 * Collective ops used:
 *   - broadcast(rootRank=0)  -> sync initial parameters
 *   - allreduce(SUM)         -> sum gradients across all ranks
 * Note: divide-by-world_size is done locally after the allreduce; combined
 * this is mathematically equivalent to a MEAN reduction.
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/ddp.cpp: real implementation when Gloo is available
 *   - src/distributed/ddp_stub.cpp: no-op fallback when Gloo is missing
 *   Direct call sites in the trainer not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   The simplest data-parallel strategy. Trainer calls broadcast_parameters
 *   once after model construction, and allreduce_gradients after every
 *   backward, before the optimizer step.
 */

#include <torch/torch.h>
// c10d::Backend is the abstract collective interface; concrete backend is
// ProcessGroupGloo here (CPU/TCP) but could be NCCL on GPU clusters.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

/// Distributed Data Parallel (DDP) helper.
/// When RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set, initializes
/// ProcessGroupGloo and provides gradient allreduce. Otherwise no-op.
/// Model is replicated on every rank; data is sharded.
class DDPContext {
 public:
  /// Initialize from environment (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
  /// Returns nullopt if any of those vars are missing (single-process mode).
  /// Spins up a TCPStore (rank 0 = server) and a ProcessGroupGloo on top.
  static std::optional<DDPContext> init_from_env();

  /// Broadcast parameters from rank 0 to all ranks (modifies in place).
  /// One broadcast per parameter; rank 0 is rootRank. Ensures every replica
  /// starts the run with bitwise-identical weights.
  void broadcast_parameters(std::vector<torch::Tensor>& parameters);

  /// Allreduce gradients across all parameters, then divide by world_size.
  /// Net effect: each rank's `.grad` becomes the arithmetic mean of all
  /// per-rank gradients (sum-allreduce + local divide).
  ///
  /// Behavior depends on whether register_grad_hooks() was called:
  ///   - With hooks: this is a "finalize" — wait on any outstanding
  ///     bucket Works that the hooks kicked off during backward, flush
  ///     any final partial bucket, then divide. Real overlap with
  ///     backward happens through the hooks.
  ///   - Without hooks: original bucketed end-of-step pass (no overlap
  ///     with backward; still bucketed for collective pipelining).
  void allreduce_gradients(const std::vector<torch::Tensor>& parameters);

  /// Register post-accumulate-grad hooks on every trainable parameter so
  /// bucket allreduce fires DURING backward. Buckets are filled in
  /// reverse parameter order (deeper layers first, matching backward
  /// order). When all parameters in a bucket have accumulated their
  /// gradient, the hook dispatches a bucket allreduce; the resulting
  /// c10d::Work is added to pending_works_ for later wait+divide.
  ///
  /// Call once after model construction. With gradient accumulation, the
  /// driver should set sync_required(false) on all but the last accum
  /// step so the hooks skip the collective.
  void register_grad_hooks(std::vector<torch::Tensor>& parameters,
                           int64_t bucket_bytes = 25 * 1024 * 1024);

  /// Hook-mode sync gate. When false, the registered hooks do nothing
  /// (so per-accum-step gradient accumulation isn't reduced). Set true
  /// before the last backward in an accumulation cycle.
  void set_sync_required(bool on) { sync_required_ = on; }
  bool sync_required() const { return sync_required_; }

  /// True iff register_grad_hooks() has been called and a hook structure
  /// is live. Drives the allreduce_gradients() branch.
  bool has_hooks() const { return !hook_state_.buckets.empty(); }

  /// This process's rank in [0, world_size).
  int rank() const { return rank_; }
  /// Total number of replicas participating in DDP.
  int world_size() const { return world_size_; }
  /// True iff a real backend was initialised (distributed env was present).
  bool is_distributed() const { return backend_ != nullptr; }
  /// Access the c10d backend for higher-level orchestration (ZeRO-1,
  /// expert-parallel, etc.). Returns null when DDP is inactive.
  c10::intrusive_ptr<c10d::Backend> backend() const { return backend_; }

 private:
  // Private ctor; instances are produced by init_from_env().
  DDPContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size);

  c10::intrusive_ptr<c10d::Backend> backend_;  // c10d collective backend (Gloo).
  int rank_;          // This process's rank.
  int world_size_;    // Total number of ranks.

  // ── Backward-hook bucket state (T-1) ────────────────────────────────
  // A bucket is a fixed-size group of consecutive parameters (in reverse
  // creation order — deeper layers first). Each parameter's hook sets a
  // bit in its bucket's `ready` mask; when the mask hits all-ones, the
  // hook dispatches an allreduce over the bucket's gradient list and
  // records the c10::Work in pending_works_ for finalize-time wait.
  struct Bucket {
    std::vector<at::Tensor> grads;  // populated as hooks fire
    int64_t total_count = 0;        // expected fill count
    int64_t ready_count = 0;        // current fill count
  };
  struct HookState {
    std::vector<Bucket> buckets;
    std::vector<c10::intrusive_ptr<c10d::Work>> pending_works;
    // Heap-held so HookState (and thus DDPContext) stays MOVABLE — a bare
    // std::mutex member is non-movable, which broke
    // `std::optional<DDPContext> init_from_env() { return DDPContext(...); }`.
    std::unique_ptr<std::mutex> mu = std::make_unique<std::mutex>();
  };
  HookState hook_state_;
  bool sync_required_ = true;
};

}  // namespace olmo_cpp
