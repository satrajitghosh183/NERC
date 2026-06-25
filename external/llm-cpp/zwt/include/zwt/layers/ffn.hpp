#pragma once

#include "zwt/layers/linear.hpp"

namespace zwt {

// SwiGLU FFN with a fused gate+up projection:
//   y = down( silu(gate(x)) * up(x) )
// The gate and up projections are combined into a single Linear of out
// size 2*hidden so the forward path executes ONE GEMM that reads x once,
// instead of two GEMMs that each re-read x from HBM. The SiLU-gated
// elementwise op then consumes the combined [N, 2H] output per row.
//
// Backward mirrors this: a single combined grad_gate_up tensor of shape
// [N, 2H] is produced by silu_mul_gated_backward, and one GEMM against
// gate_up_'s weight recovers dL/dx (and accumulates the weight gradient).
class FFN final : public Module {
 public:
  FFN(int64_t d_model, int64_t hidden, DType dtype, Device device,
      uint64_t init_seed = 0xFF77000ULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

 private:
  Linear gate_up_;   // [2*hidden, d_model] — the fused projection
  Linear down_;

  Tensor saved_combined_;   // gate_up_ output [..., 2*hidden]
  Tensor saved_silu_mul_;   // silu(gate) * up — input to down_
};

}  // namespace zwt
