#pragma once

#include "zwt/core/tensor.hpp"
#include "zwt/layers/parameter.hpp"
#include <vector>

namespace zwt {

// Minimal module interface. Implementations keep activations in member state
// so backward() can reuse them without a graph-tape lookup.
//
// Lifecycle:
//   * construct with config  — allocates parameters in the device pool
//   * forward(x)             — returns an arena-allocated output; stashes state
//   * backward(grad_y)       — returns an arena-allocated grad_x; accumulates
//                              into each Parameter::grad
//   * collect_params(out)    — append pointers to all owned parameters
class Module {
 public:
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor& x) = 0;
  virtual Tensor backward(const Tensor& grad_y) = 0;
  virtual void   collect_params(std::vector<Parameter*>& out) = 0;
};

// Reset the activation arena. Call once per training step, after optimizer
// and before the next forward.
void step_begin();

}  // namespace zwt
