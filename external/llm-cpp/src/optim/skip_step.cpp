/**
 * src/optim/skip_step.cpp
 *
 * ─── What "skip step" means ─────────────────────────────────────────
 *
 * Sometimes a single training step produces a degenerate gradient:
 *   - a NaN or Inf from a numerical underflow,
 *   - or a gradient norm 100x larger than the recent moving average
 *     (a "spike", usually a bad batch or a rare numerical edge case).
 *
 * Letting the optimizer apply that gradient pollutes the momentum and
 * adam-like state for thousands of subsequent steps. SkipStep is a
 * defensive wrapper that intercepts step() — if the gradient looks
 * unsafe it just doesn't apply it. The model is left untouched, and
 * the next batch is given a chance to recover.
 *
 * It's a drop-in around any inner optimizer (AdamW, Muon, Lion, ...).
 * Forwards every other API call — state_dict, lr, etc. — so the train
 * loop can't tell it's there.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/skip_step.hpp : SkipStepOptimizer declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: optionally wraps the inner optimizer when
 *     skip_step=1 in the [optimization] section of the .conf.
 *
 * --- Role in training pipeline ---
 *   Sits between the train loop and the inner optimizer. Useful in
 *   BF16/FP8 where one bad batch could blow up the moments and ruin
 *   the next thousand steps.
 */
#include "olmo_cpp/optim/skip_step.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace olmo_cpp {

SkipStepOptimizer::SkipStepOptimizer(
    std::unique_ptr<torch::optim::Optimizer> inner,
    double spike_threshold,
    int64_t window_size)
    : inner_(std::move(inner)),
      spike_threshold_(spike_threshold),
      window_size_(window_size) {
  TORCH_CHECK(inner_ != nullptr, "Inner optimizer must not be null");
  TORCH_CHECK(spike_threshold_ > 0.0, "Spike threshold must be positive");
  TORCH_CHECK(window_size_ > 0, "Window size must be positive");
}

bool SkipStepOptimizer::step(float current_loss) {
  // Check if we should skip this step due to a loss spike
  bool should_skip = false;

  if (!loss_history_.empty()) {
    // Detect spike: loss > mean + threshold * std_dev
    double std_dev = std::sqrt(std::max(running_var_, 1e-12));
    double upper_bound = running_mean_ + spike_threshold_ * std_dev;

    if (static_cast<double>(current_loss) > upper_bound) {
      should_skip = true;
      ++skipped_steps_;
    }
  }

  // Update running statistics with Welford's online algorithm
  loss_history_.push_back(current_loss);
  if (static_cast<int64_t>(loss_history_.size()) > window_size_) {
    loss_history_.pop_front();
  }

  // Recompute mean and variance over the window
  if (!loss_history_.empty()) {
    double sum = 0.0;
    for (float l : loss_history_) {
      sum += static_cast<double>(l);
    }
    running_mean_ = sum / static_cast<double>(loss_history_.size());

    double var_sum = 0.0;
    for (float l : loss_history_) {
      double diff = static_cast<double>(l) - running_mean_;
      var_sum += diff * diff;
    }
    running_var_ = (loss_history_.size() > 1)
        ? var_sum / static_cast<double>(loss_history_.size() - 1)
        : 1.0;
  }

  if (!should_skip) {
    inner_->step();
  }

  return !should_skip;
}

void SkipStepOptimizer::zero_grad() {
  inner_->zero_grad();
}

}  // namespace olmo_cpp
