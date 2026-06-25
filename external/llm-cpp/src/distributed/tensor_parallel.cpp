/**
 * src/distributed/tensor_parallel.cpp
 *
 * ─── What "Tensor Parallelism" is ───────────────────────────────────
 *
 * TP (Megatron-LM, 2019) splits the *weight matrices themselves*
 * across GPUs. For a Linear layer y = x · W with W of shape [in, out]:
 *
 *   "column-parallel": each rank holds W[:, slice], computes a partial
 *     output [in, out/world_size], and the activations are concatenated
 *     across ranks (all_gather, but typically deferred).
 *
 *   "row-parallel":    each rank holds W[slice, :], computes a partial
 *     sum over a slice of the input dim, and an **all_reduce** sums
 *     the partials to get the true output.
 *
 * In a transformer block the standard recipe is:
 *   QKV projection — column-parallel
 *   attention output projection — row-parallel  (allreduce here)
 *   FFN up/gate — column-parallel
 *   FFN down — row-parallel  (allreduce here)
 *
 * So TP costs two allreduces per block per fwd, two per bwd. That's
 * heavy bandwidth — TP is usually only used inside a single node where
 * NVLink can keep up, and combined with DP/PP across nodes.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/tensor_parallel.hpp : TP context + helpers.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp / feed_forward.cpp: when a TP context is
 *     present, the linear ops dispatch to TP-aware variants that
 *     allreduce.
 *
 * --- Role in training pipeline ---
 *   Used when a single layer's weights don't fit on one device.
 *   Disabled by default; activated when world_size_tp > 1.
 */
#include "olmo_cpp/distributed/tensor_parallel.hpp"

namespace olmo_cpp {

std::optional<TensorParallelContext> TensorParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend) {
  if (!backend) return std::nullopt;
  int world_size = backend->getSize();
  if (world_size < 2) return std::nullopt;
  return TensorParallelContext(backend, backend->getRank(), world_size);
}

TensorParallelContext::TensorParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size)
    : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {}

torch::Tensor TensorParallelContext::column_parallel_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // x: [B, S, in_features], weight: [out_features_local, in_features]
  // output: [B, S, out_features_local] - sharded along output dim
  return torch::nn::functional::linear(x, weight, bias);
}

torch::Tensor TensorParallelContext::row_parallel_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // x: [B, S, in_features_local] (sharded), weight: [out_features, in_features_local]
  auto out = torch::nn::functional::linear(x, weight, bias);
  std::vector<at::Tensor> tensors = {out};
  backend_->allreduce(tensors)->wait();
  return out;
}

torch::Tensor TensorParallelContext::allgather_sequence(const torch::Tensor& x) {
  // Allgather: each rank has [B, S_local, D], gather to [B, S_local*W, D]
  // outputTensors[rank][0] = buffer for rank to receive full gathered tensor
  std::vector<int64_t> out_sizes = x.sizes().vec();
  out_sizes[1] *= world_size_;  // sequence dim
  std::vector<std::vector<at::Tensor>> outputs(world_size_);
  for (int r = 0; r < world_size_; ++r) {
    outputs[r] = {torch::empty(out_sizes, x.options())};
  }
  std::vector<at::Tensor> inputs = {x};
  backend_->allgather(outputs, inputs, c10d::AllgatherOptions())->wait();
  return outputs[rank_][0];
}

torch::Tensor TensorParallelContext::reduce_scatter_sequence(const torch::Tensor& x) {
  // Inverse of allgather_sequence at a sequence-parallel boundary.
  // x shape: [B, S, D] — each rank holds the SAME shape with PARTIAL
  // sums of the per-output-element values. After reduce_scatter:
  //   - SUM is applied across ranks
  //   - Result is scattered along the seq dim
  // Output: [B, S/world_size, D].
  const int64_t S = x.size(1);
  TORCH_CHECK(S % world_size_ == 0,
              "reduce_scatter_sequence: S must be divisible by world_size");
  const int64_t s_local = S / world_size_;

  // Split x along dim 1 into world_size chunks; build inputTensors[rank][0]
  // for c10d::reduce_scatter. Each rank's input shard ends up summed and
  // scattered to the corresponding rank.
  std::vector<at::Tensor> input_chunks;
  input_chunks.reserve(world_size_);
  for (int r = 0; r < world_size_; ++r) {
    input_chunks.push_back(x.narrow(1, r * s_local, s_local).contiguous());
  }
  std::vector<int64_t> out_sizes = x.sizes().vec();
  out_sizes[1] = s_local;
  auto out = torch::empty(out_sizes, x.options());
  std::vector<at::Tensor> output_tensors = {out};
  std::vector<std::vector<at::Tensor>> input_tensors = {input_chunks};
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = c10d::ReduceOp::SUM;
  backend_->reduce_scatter(output_tensors, input_tensors, opts)->wait();
  return out;
}

torch::Tensor TensorParallelContext::column_parallel_linear_sp(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // SP boundary IN: x is [B, S/W, D_full]. Allgather along seq dim, then
  // do the column-parallel matmul (output is feature-sharded).
  auto x_full = allgather_sequence(x);                       // [B, S, D_full]
  return torch::nn::functional::linear(x_full, weight, bias); // [B, S, D_local]
}

torch::Tensor TensorParallelContext::row_parallel_linear_sp(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  // SP boundary OUT: x is [B, S, D_local] (feature-sharded). Local matmul
  // yields partial [B, S, D_full]; reduce_scatter on seq sums + scatters
  // to [B, S/W, D_full].
  auto partial = torch::nn::functional::linear(x, weight, bias);  // [B, S, D_full]
  return reduce_scatter_sequence(partial);                         // [B, S/W, D_full]
}

}  // namespace olmo_cpp
