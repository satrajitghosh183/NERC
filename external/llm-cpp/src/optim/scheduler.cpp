/**
 * src/optim/scheduler.cpp
 *
 * ─── What an "LR scheduler" is ──────────────────────────────────────
 *
 * The optimizer step is `w ← w − lr · update`. If lr is fixed at a
 * sensible value, training tends to plateau early. Empirically, two
 * tricks help a lot:
 *
 *   1. **Warmup**: start at a tiny lr and linearly ramp up over the
 *      first ~1% of steps. Without this, the first few gradients are
 *      so noisy that they kick the model off the manifold.
 *
 *   2. **Cosine decay**: after warmup, smoothly decay lr along
 *      lr_max · 0.5 · (1 + cos(π · progress)) to ≈ 0 by the end of
 *      training. The model gets fine-grained refinement near the end
 *      where the loss landscape is shallow.
 *
 * This file implements the canonical four shapes:
 *   - constant
 *   - linear            : lr → 0 over total_steps (no warmup)
 *   - cosine            : warmup then cosine decay to 0
 *   - cosine_with_floor : same but floors at (1−decay_ratio) · lr_max
 *
 * Every scheduler is **stateless** beyond its constructor arguments —
 * lr is a pure function of the global step. Easy to checkpoint.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/scheduler.hpp : LRScheduler interface + factories.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: built once, then queried each microbatch to set
 *     param_groups()[g].options().lr() before optimizer step.
 *
 * --- Role in training pipeline ---
 *   Read-only function of the global step. The .conf's "scheduler"
 *   key selects which one to instantiate.
 */
#include "olmo_cpp/optim/scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace olmo_cpp {

// ============================================================================
// LRScheduler base
// ============================================================================

void LRScheduler::apply(torch::optim::Optimizer& optimizer, int64_t step, int64_t total_steps) const {
  double lr = get_lr(step, total_steps);
  for (auto& group : optimizer.param_groups()) {
    // LibTorch OptimizerOptions base class provides set_lr()/get_lr() since v1.8+
    group.options().set_lr(lr);
  }
}

// ============================================================================
// Helper: linear warmup factor
// ============================================================================

namespace {

/// Returns a warmup multiplier in [0, 1] for the given step.
/// At step=0 returns 0, at step=warmup_steps returns 1.
inline double warmup_factor(int64_t step, int64_t warmup_steps) {
  if (warmup_steps <= 0 || step >= warmup_steps) {
    return 1.0;
  }
  return static_cast<double>(step) / static_cast<double>(warmup_steps);
}

}  // namespace

// ============================================================================
// LinearDecayLR
// ============================================================================

LinearDecayLR::LinearDecayLR(double base_lr, int64_t warmup_steps, double min_lr)
    : base_lr_(base_lr), min_lr_(min_lr), warmup_steps_(warmup_steps) {}

double LinearDecayLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Decay phase: linearly decay from base_lr to min_lr
  int64_t decay_steps = total_steps - warmup_steps_;
  if (decay_steps <= 0) {
    return base_lr_;
  }

  double progress = static_cast<double>(step - warmup_steps_) / static_cast<double>(decay_steps);
  progress = std::min(progress, 1.0);
  return base_lr_ + (min_lr_ - base_lr_) * progress;
}

// ============================================================================
// CosineAnnealingLR
// ============================================================================

CosineAnnealingLR::CosineAnnealingLR(double base_lr, int64_t warmup_steps, double min_lr)
    : base_lr_(base_lr), min_lr_(min_lr), warmup_steps_(warmup_steps) {}

double CosineAnnealingLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Cosine decay phase
  int64_t decay_steps = total_steps - warmup_steps_;
  if (decay_steps <= 0) {
    return base_lr_;
  }

  double progress = static_cast<double>(step - warmup_steps_) / static_cast<double>(decay_steps);
  progress = std::min(progress, 1.0);

  // Cosine annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
  return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + std::cos(M_PI * progress));
}

// ============================================================================
// ExponentialDecayLR
// ============================================================================

ExponentialDecayLR::ExponentialDecayLR(double base_lr, double gamma, int64_t warmup_steps)
    : base_lr_(base_lr), gamma_(gamma), warmup_steps_(warmup_steps) {}

double ExponentialDecayLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Exponential decay: base_lr * gamma^(step - warmup_steps)
  int64_t decay_step = step - warmup_steps_;
  return base_lr_ * std::pow(gamma_, static_cast<double>(decay_step));
}

// ============================================================================
// InverseSqrtLR
// ============================================================================

InverseSqrtLR::InverseSqrtLR(double base_lr, int64_t warmup_steps)
    : base_lr_(base_lr), warmup_steps_(warmup_steps) {
  TORCH_CHECK(warmup_steps > 0, "InverseSqrtLR requires warmup_steps > 0");
}

double InverseSqrtLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Inverse sqrt decay: base_lr * sqrt(warmup_steps) / sqrt(step)
  // At step = warmup_steps, this equals base_lr
  return base_lr_ * std::sqrt(static_cast<double>(warmup_steps_)) /
         std::sqrt(static_cast<double>(step));
}

// ============================================================================
// PolynomialDecayLR
// ============================================================================

PolynomialDecayLR::PolynomialDecayLR(double base_lr, double power, int64_t warmup_steps, double min_lr)
    : base_lr_(base_lr), power_(power), min_lr_(min_lr), warmup_steps_(warmup_steps) {}

double PolynomialDecayLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Polynomial decay
  int64_t decay_steps = total_steps - warmup_steps_;
  if (decay_steps <= 0) {
    return base_lr_;
  }

  double progress = static_cast<double>(step - warmup_steps_) / static_cast<double>(decay_steps);
  progress = std::min(progress, 1.0);

  // lr = (base_lr - min_lr) * (1 - progress)^power + min_lr
  return (base_lr_ - min_lr_) * std::pow(1.0 - progress, power_) + min_lr_;
}

// ============================================================================
// StepDecayLR
// ============================================================================

StepDecayLR::StepDecayLR(double base_lr, double gamma, int64_t step_size, int64_t warmup_steps)
    : base_lr_(base_lr), gamma_(gamma), step_size_(step_size), warmup_steps_(warmup_steps) {}

double StepDecayLR::get_lr(int64_t step, int64_t total_steps) const {
  // Warmup phase
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Step decay: base_lr * gamma^floor((step - warmup) / step_size)
  int64_t decay_step = step - warmup_steps_;
  int64_t num_decays = decay_step / step_size_;
  return base_lr_ * std::pow(gamma_, static_cast<double>(num_decays));
}

// ============================================================================
// WSDLR (Warmup-Stable-Decay)
// ============================================================================

WSDLR::WSDLR(double base_lr, int64_t warmup_steps, int64_t stable_steps, double min_lr)
    : base_lr_(base_lr), min_lr_(min_lr), warmup_steps_(warmup_steps), stable_steps_(stable_steps) {}

double WSDLR::get_lr(int64_t step, int64_t total_steps) const {
  // Phase 1: Warmup - linear from 0 to base_lr
  if (step < warmup_steps_) {
    return base_lr_ * warmup_factor(step, warmup_steps_);
  }

  // Phase 2: Stable - constant at base_lr
  int64_t stable_end = warmup_steps_ + stable_steps_;
  if (step < stable_end) {
    return base_lr_;
  }

  // Phase 3: Decay - cosine decay from base_lr to min_lr over remaining steps
  int64_t decay_steps = total_steps - stable_end;
  if (decay_steps <= 0) {
    return base_lr_;
  }

  double progress = static_cast<double>(step - stable_end) / static_cast<double>(decay_steps);
  progress = std::min(progress, 1.0);

  // Cosine decay
  return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + std::cos(M_PI * progress));
}

// ============================================================================
// Factory function
// ============================================================================

std::unique_ptr<LRScheduler> create_scheduler(
    const std::string& name, double base_lr, int64_t warmup_steps,
    double min_lr, double gamma, int64_t step_size,
    double power, int64_t stable_steps) {

  if (name == "constant") {
    return std::make_unique<ConstantLR>(base_lr);
  } else if (name == "linear" || name == "linear_decay") {
    return std::make_unique<LinearDecayLR>(base_lr, warmup_steps, min_lr);
  } else if (name == "cosine" || name == "cosine_annealing") {
    return std::make_unique<CosineAnnealingLR>(base_lr, warmup_steps, min_lr);
  } else if (name == "exponential" || name == "exponential_decay") {
    return std::make_unique<ExponentialDecayLR>(base_lr, gamma, warmup_steps);
  } else if (name == "inverse_sqrt") {
    return std::make_unique<InverseSqrtLR>(base_lr, warmup_steps);
  } else if (name == "polynomial" || name == "polynomial_decay") {
    return std::make_unique<PolynomialDecayLR>(base_lr, power, warmup_steps, min_lr);
  } else if (name == "step" || name == "step_decay") {
    return std::make_unique<StepDecayLR>(base_lr, gamma, step_size, warmup_steps);
  } else if (name == "wsd" || name == "warmup_stable_decay") {
    return std::make_unique<WSDLR>(base_lr, warmup_steps, stable_steps, min_lr);
  } else {
    throw std::invalid_argument("Unknown scheduler name: " + name);
  }
}

}  // namespace olmo_cpp
