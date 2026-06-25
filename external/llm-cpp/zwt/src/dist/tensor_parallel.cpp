#include "zwt/dist/tensor_parallel.hpp"

#include <cstring>
#include <stdexcept>

namespace zwt::dist {

TpContext make_loopback_tp(Stream s) {
  TpContext c;
  c.rank       = 0;
  c.world_size = 1;
  c.stream     = s;
  c.allreduce = [](float*, size_t, StreamHandle) {};
  c.allgather = [](const float* in, float* out, size_t n, StreamHandle) {
    if (in != out) std::memcpy(out, in, n * sizeof(float));
  };
  return c;
}

// ---------------------------------------------------------------------------
// ColumnParallelLinear
// ---------------------------------------------------------------------------
ColumnParallelLinear::ColumnParallelLinear(
    int64_t in_features, int64_t out_features,
    bool use_bias, DType dtype, Device device,
    TpContext ctx, bool gather_output, uint64_t init_seed)
    : ctx_(std::move(ctx)),
      inner_(in_features,
             out_features / std::max(ctx_.world_size, 1),
             use_bias, dtype, device, init_seed),
      gather_(gather_output),
      local_out_(out_features / std::max(ctx_.world_size, 1)),
      global_out_(out_features) {
  if (ctx_.world_size < 1) {
    throw std::runtime_error("ColumnParallelLinear: world_size < 1");
  }
  if (out_features % ctx_.world_size != 0) {
    throw std::runtime_error(
        "ColumnParallelLinear: out_features not divisible by world_size");
  }
}

Tensor ColumnParallelLinear::forward(const Tensor& x) {
  Tensor local = inner_.forward(x);
  if (!gather_ || ctx_.world_size == 1) return local;
  // Allgather local [..O/W] → full [..O] along the last dim. The callback
  // is responsible for the cross-rank data movement; with world_size=1 the
  // loopback allgather is a memcpy, which we skip by returning local above.
  Shape out_shape = local.shape();
  out_shape.dims[out_shape.rank - 1] = global_out_;
  Tensor full = empty_scratch(out_shape, local.dtype(), local.device());
  ctx_.allgather(reinterpret_cast<const float*>(local.data()),
                 reinterpret_cast<float*>(full.data()),
                 static_cast<size_t>(local.numel()),
                 ctx_.stream.handle);
  return full;
}

Tensor ColumnParallelLinear::backward(const Tensor& grad_y) {
  // grad_y has shape [..,O] if gather_output else [..,O/W].
  // Inner Linear expects [..,O/W] grad_y. For gather_output=true we'd
  // allreduce-scatter grad_y along the last dim; world_size=1 loopback
  // does nothing. Here we simply forward grad_y to inner when not gathered,
  // and for gathered we take the rank-0 slice (correct under world_size=1).
  if (gather_ && ctx_.world_size > 1) {
    throw std::runtime_error(
        "ColumnParallelLinear: gathered-output backward needs "
        "reduce-scatter (unimplemented in loopback build)");
  }
  return inner_.backward(grad_y);
}

void ColumnParallelLinear::collect_params(std::vector<Parameter*>& out) {
  inner_.collect_params(out);
}

// ---------------------------------------------------------------------------
// RowParallelLinear
// ---------------------------------------------------------------------------
RowParallelLinear::RowParallelLinear(
    int64_t in_features, int64_t out_features,
    bool use_bias, DType dtype, Device device,
    TpContext ctx, uint64_t init_seed)
    : ctx_(std::move(ctx)),
      inner_(in_features / std::max(ctx_.world_size, 1),
             out_features,
             use_bias, dtype, device, init_seed),
      local_in_(in_features / std::max(ctx_.world_size, 1)) {
  if (ctx_.world_size < 1) {
    throw std::runtime_error("RowParallelLinear: world_size < 1");
  }
  if (in_features % ctx_.world_size != 0) {
    throw std::runtime_error(
        "RowParallelLinear: in_features not divisible by world_size");
  }
}

Tensor RowParallelLinear::forward(const Tensor& x) {
  // x is expected to be sharded along the last dim [..,I/W]. inner_
  // computes the partial [..,O] and we all-reduce sum it across ranks.
  Tensor partial = inner_.forward(x);
  if (ctx_.world_size > 1) {
    ctx_.allreduce(reinterpret_cast<float*>(partial.data()),
                   static_cast<size_t>(partial.numel()),
                   ctx_.stream.handle);
  }
  return partial;
}

Tensor RowParallelLinear::backward(const Tensor& grad_y) {
  // grad_y is replicated [..,O] across ranks (it came from an all-reduce
  // output). Per-rank we backprop through the local [O,I/W] Linear and
  // return the sharded [..,I/W] grad_x directly — the producer on the
  // other side (ColumnParallelLinear with gather_output=false) expects
  // exactly that layout for its grad.
  return inner_.backward(grad_y);
}

void RowParallelLinear::collect_params(std::vector<Parameter*>& out) {
  inner_.collect_params(out);
}

}  // namespace zwt::dist
