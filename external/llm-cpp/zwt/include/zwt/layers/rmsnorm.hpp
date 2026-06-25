#pragma once

#include "zwt/layers/module.hpp"

namespace zwt {

class RMSNorm final : public Module {
 public:
  RMSNorm(int64_t dim, float eps, DType dtype, Device device);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  // Fused residual path: compute (x+res) -> rmsnorm -> out, and also return
  // the pre-norm sum (useful for the next block's residual connection).
  // Caller passes a pre-allocated `sum_out` tensor.
  Tensor forward_residual(const Tensor& x, const Tensor& residual, Tensor& sum_out);

 private:
  Parameter weight_;
  float     eps_;
  int64_t   dim_;

  Tensor saved_input_;
  Tensor saved_rstd_;
};

}  // namespace zwt
