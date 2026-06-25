#pragma once

#include "zwt/core/tensor.hpp"
#include <string>

namespace zwt {

// A trainable parameter. Owns its value and its gradient (same shape/dtype/device).
// Gradient is allocated lazily on the first backward pass.
struct Parameter {
  std::string name;
  Tensor      value;
  Tensor      grad;   // same shape as value; zero until first backward; fp32 if value is bf16

  Parameter() = default;
  Parameter(std::string n, Tensor v) : name(std::move(n)), value(std::move(v)) {}

  // Materialize grad buffer if not done yet. FP32 grad for BF16 params so
  // optimizer state math is stable; f32 params get f32 grad.
  void ensure_grad();

  void zero_grad();

  int64_t numel() const { return value.numel(); }
};

}  // namespace zwt
