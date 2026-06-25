#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
//   x:      [..., D]
//   weight: [D]
//   y:      [..., D]    (same shape as x)
//   rstd:   [..., 1]    (scratch used by backward; must be supplied)
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& y, Tensor& rstd,
             float eps);

// Fused residual form: y = rmsnorm(x + residual) * weight, and also writes
// the pre-norm sum (x + residual) to `sum_out` for use in the next layer's
// residual shortcut. Saves a full activation read/write per block.
void rmsnorm_residual(const Tensor& x, const Tensor& residual,
                      const Tensor& weight, Tensor& y, Tensor& sum_out,
                      Tensor& rstd, float eps);

// Backward: given grad_y, x, weight, rstd compute grad_x and grad_weight.
// Accumulates into grad_weight (beta=1), overwrites grad_x.
void rmsnorm_backward(const Tensor& grad_y, const Tensor& x, const Tensor& weight,
                      const Tensor& rstd,
                      Tensor& grad_x, Tensor& grad_weight, float eps);

}  // namespace zwt::ops
