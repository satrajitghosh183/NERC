#pragma once

/**
 * include/olmo_cpp/distributed/tensor_parallel.hpp
 *
 * Public API for Tensor Parallelism (TP), Megatron-style. TP shards the
 * *weight matrices* of linear layers across TP ranks along one axis:
 *   - Column-parallel: split W along the OUTPUT dim. Each rank computes a
 *     local slice of the output features. Forward needs no collective
 *     (output is "naturally" sharded along feature dim); the gradient on
 *     the input has an implicit allreduce in the autograd graph.
 *   - Row-parallel:    split W along the INPUT dim. Each rank multiplies
 *     its slice and the partial outputs are summed across ranks via
 *     allreduce. Inputs are assumed to already be sharded along the
 *     contracted dim (typical pairing: column-parallel feeds row-parallel).
 *
 * In transformer FFNs we use column-parallel for up_proj/gate_proj and
 * row-parallel for down_proj; in attention we use column-parallel for the
 * Q/K/V projections (split along heads) and row-parallel for o_proj.
 *
 * Sequence parallel (SP) extension: when the tensor-parallel ranks all
 * hold replicated activations, we can additionally shard along the
 * sequence dimension. allgather_sequence reassembles the full sequence
 * before a TP region; the inverse (reduce-scatter) lives elsewhere in
 * the model code.
 *
 * Collective ops used:
 *   - allreduce  (row_parallel_linear: SUM partial outputs)
 *   - allgather  (allgather_sequence:  reassemble full S along dim 1)
 *
 * --- Includes from this project ---
 *   - (none from this project; pure declaration)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/distributed/tensor_parallel.cpp: implementation
 *   Direct call sites in attention/FFN modules not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   The "intra-layer" parallelism: complements DDP/FSDP (inter-data) and
 *   PP (inter-layer). Critical for very large hidden dims that don't fit
 *   on one device.
 */

#include <torch/torch.h>
// c10d::Backend supplies the allreduce / allgather we use.
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <memory>
#include <optional>

namespace olmo_cpp {

/// Tensor parallelism (Megatron-style) context.
/// Splits linear layers across TP ranks: column-parallel (output split),
/// row-parallel (input split, allreduce after matmul).
class TensorParallelContext {
 public:
  /// Create from ProcessGroup backend. Returns nullopt if world_size < 2.
  static std::optional<TensorParallelContext> create(
      c10::intrusive_ptr<c10d::Backend> backend);

  /// Column-parallel linear: output dim split across ranks.
  /// y = x @ W_local^T  (W_local is the rank's slice of full W along the
  /// OUTPUT axis). No collective in forward; result is sharded along the
  /// last (feature) dim. Caller is responsible for the autograd-side
  /// allreduce on x.grad if x is replicated.
  torch::Tensor column_parallel_linear(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  /// Row-parallel linear: input dim split, allreduce output.
  /// Assumes x is already sharded along the contracted dim (typically the
  /// output of a column-parallel layer). After the local matmul, performs
  /// a SUM allreduce so every rank holds the full unsharded output.
  torch::Tensor row_parallel_linear(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  /// Allgather tensors along the sequence axis (dim 1) -- used to
  /// reassemble a full sequence after sequence-parallel regions before
  /// entering a tensor-parallel region.
  torch::Tensor allgather_sequence(const torch::Tensor& x);

  /// Reduce-scatter along the sequence axis (dim 1). Input is [B, S, D]
  /// with each rank holding partial sums of the same D values across the
  /// full S sequence. Output is [B, S/world_size, D], summed across
  /// ranks and scattered. This is the inverse of allgather_sequence and
  /// the boundary OUT of a TP region into an SP region.
  torch::Tensor reduce_scatter_sequence(const torch::Tensor& x);

  /// Column-parallel linear at an SP boundary. Input is
  /// [B, S/world_size, in_features] (sequence-sharded, feature-full);
  /// allgathers along seq, performs the local matmul, returns
  /// [B, S, out_features_local].
  torch::Tensor column_parallel_linear_sp(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  /// Row-parallel linear at an SP boundary. Input is
  /// [B, S, in_features_local] (feature-sharded); matmul yields partial
  /// [B, S, out_features], reduce-scatter on seq gives the SP-region
  /// shape [B, S/world_size, out_features].
  torch::Tensor row_parallel_linear_sp(
      const torch::Tensor& x,
      const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias = c10::nullopt);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  /// Convenience: skip TP machinery entirely if running on a single rank.
  bool is_tensor_parallel() const { return world_size_ > 1; }

 private:
  TensorParallelContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size);

  c10::intrusive_ptr<c10d::Backend> backend_;  // c10d backend for allreduce/allgather.
  int rank_;
  int world_size_;
};

}  // namespace olmo_cpp
