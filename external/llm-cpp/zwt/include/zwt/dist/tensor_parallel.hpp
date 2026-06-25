#pragma once

#include "zwt/core/stream.hpp"
#include "zwt/layers/linear.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace zwt::dist {

// Megatron-style tensor parallelism primitives.
//
// ColumnParallelLinear shards a Linear along out_features:
//   global W  : [O, I]                 global b : [O]
//   local  W  : [O/W, I] on rank r     local  b : [O/W]
// Forward: y_local = x @ W_local^T + b_local               shape [.., O/W]
//   — x is replicated across ranks (no collective in);
//   — if the downstream consumer is a RowParallelLinear, the [.., O/W]
//     split is exactly what it wants, so we skip the output allgather.
//   — if the downstream consumer expects the full [.., O], an allgather
//     over the O dimension fuses it back.
//
// RowParallelLinear shards a Linear along in_features:
//   global W  : [O, I]                 global b : [O]
//   local  W  : [O, I/W] on rank r     local  b : [O]  (replicated)
// Forward: y_partial = x_local @ W_local^T                shape [.., O]
//   y = allreduce(y_partial) + b
//   — x_local is the sharded output of a preceding ColumnParallelLinear.
//
// Pairing: attention's (Q,K,V) projections are column-parallel, O is
// row-parallel; SwiGLU's gate_up is column-parallel, down is row-parallel.
// That pairing keeps the intermediate activations sharded and only inserts
// one all-reduce per sub-layer.
//
// The actual NCCL collective enters via a callback — same discipline as
// BucketManager. CPU / world_size=1 builds become pass-throughs.

struct TpContext {
  int rank;
  int world_size;
  Stream stream;      // stream to issue allreduce/allgather on
  // Callback signatures. n = element count of the local buffer.
  std::function<void(float* buf, size_t n, StreamHandle s)> allreduce;
  std::function<void(const float* in_local, float* out_full,
                     size_t n_local, StreamHandle s)> allgather;
};

// Trivial loopback context — world_size=1, collectives are no-ops.
TpContext make_loopback_tp(Stream s);

class ColumnParallelLinear final : public Module {
 public:
  // out_features must be divisible by ctx.world_size.
  // `gather_output` controls whether forward() produces the full [..,O]
  // tensor (true → allgather at the end) or the sharded [..,O/W] tensor
  // (false → skip allgather; downstream must handle the shard).
  ColumnParallelLinear(int64_t in_features, int64_t out_features,
                       bool use_bias, DType dtype, Device device,
                       TpContext ctx, bool gather_output,
                       uint64_t init_seed = 0xABCD'EF01ULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  int64_t local_out_features() const { return local_out_; }

 private:
  TpContext ctx_;
  Linear    inner_;     // local [O/W, I] Linear does the compute
  bool      gather_;
  int64_t   local_out_;
  int64_t   global_out_;
};

class RowParallelLinear final : public Module {
 public:
  // in_features must be divisible by ctx.world_size. The input to forward()
  // is the sharded [..,I/W] tensor produced by a preceding
  // ColumnParallelLinear(gather_output=false).
  RowParallelLinear(int64_t in_features, int64_t out_features,
                    bool use_bias, DType dtype, Device device,
                    TpContext ctx,
                    uint64_t init_seed = 0xDEAD'BEEFULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  int64_t local_in_features() const { return local_in_; }

 private:
  TpContext ctx_;
  Linear    inner_;     // local [O, I/W] Linear does the compute
  int64_t   local_in_;
};

}  // namespace zwt::dist
