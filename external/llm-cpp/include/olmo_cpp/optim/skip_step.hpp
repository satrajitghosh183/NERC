#pragma once
/**
 * include/olmo_cpp/optim/skip_step.hpp
 *
 * Header for SkipStepOptimizer. This is a defensive wrapper that
 * intercepts step() — if the gradient is NaN/Inf or has a norm well
 * above the recent moving average, it just doesn't apply the update
 * (the model is left untouched). See src/optim/skip_step.cpp for
 * the longer explanation.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/optim/skip_step.cpp : implementation.
 *   - src/train.cpp           : optional wrap of the inner optimizer.
 *
 * --- Role in training pipeline ---
 *   Optional defensive layer. Especially useful in BF16/FP8 training.
 */
#include <torch/torch.h>
#include <deque>
#include <memory>

namespace olmo_cpp {

/// Skip-step optimizer wrapper: detects loss spikes and skips optimizer steps
class SkipStepOptimizer {
 public:
  SkipStepOptimizer(std::unique_ptr<torch::optim::Optimizer> inner,
                    double spike_threshold = 5.0,
                    int64_t window_size = 100);
  /// Returns true if step was taken, false if skipped due to spike
  bool step(float current_loss);
  void zero_grad();
  torch::optim::Optimizer& inner() { return *inner_; }
  int64_t skipped_steps() const { return skipped_steps_; }
 private:
  std::unique_ptr<torch::optim::Optimizer> inner_;
  double spike_threshold_;
  int64_t window_size_;
  std::deque<float> loss_history_;
  double running_mean_ = 0.0;
  double running_var_ = 1.0;
  int64_t skipped_steps_ = 0;
};

}  // namespace olmo_cpp
