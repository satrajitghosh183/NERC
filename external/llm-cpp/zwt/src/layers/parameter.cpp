#include "zwt/layers/parameter.hpp"

namespace zwt {

void Parameter::ensure_grad() {
  if (grad.data() != nullptr) return;
  // Grad buffer is always fp32 for stability — optimizer reads it in fp32
  // regardless of the param dtype. This matches how every serious LLM trainer
  // runs today (the "master gradient" pattern) and removes the main source of
  // bf16 training instability without costing any forward perf.
  grad = zeros(value.shape(), DType::F32, value.device());
}

void Parameter::zero_grad() {
  if (grad.data() != nullptr) grad.zero_();
}

}  // namespace zwt
