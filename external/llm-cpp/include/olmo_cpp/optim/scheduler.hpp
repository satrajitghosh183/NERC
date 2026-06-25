#pragma once
/**
 * include/olmo_cpp/optim/scheduler.hpp
 *
 * Learning-rate schedulers. Defines the LRScheduler base class plus
 * factories for the four canonical shapes: constant, linear,
 * cosine-with-warmup, cosine-with-floor.
 *
 * Schedulers are stateless — `get_lr(step)` is a pure function of
 * the global step. This makes them trivial to checkpoint (just
 * record the step) and predictable across resumes.
 *
 * See src/optim/scheduler.cpp for the implementation and the
 * formulae for each shape.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/optim/scheduler.cpp : implementation.
 *   - src/train.cpp           : queries scheduler.get_lr(step) every
 *                                microbatch and updates the optimizer
 *                                param-group lr field.
 *
 * --- Role in training pipeline ---
 *   Drives lr over the run. The .conf's "scheduler" key picks the shape.
 */
#include <torch/torch.h>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace olmo_cpp {

/// Base LR scheduler interface
class LRScheduler {
 public:
  virtual ~LRScheduler() = default;
  virtual double get_lr(int64_t step, int64_t total_steps) const = 0;
  void apply(torch::optim::Optimizer& optimizer, int64_t step, int64_t total_steps) const;
};

/// Constant LR
class ConstantLR : public LRScheduler {
 public:
  explicit ConstantLR(double lr) : lr_(lr) {}
  double get_lr(int64_t step, int64_t total_steps) const override { return lr_; }
 private:
  double lr_;
};

/// Linear decay from base_lr to 0
class LinearDecayLR : public LRScheduler {
 public:
  LinearDecayLR(double base_lr, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_;
};

/// Cosine annealing with optional warmup
class CosineAnnealingLR : public LRScheduler {
 public:
  CosineAnnealingLR(double base_lr, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_;
};

/// Exponential decay
class ExponentialDecayLR : public LRScheduler {
 public:
  ExponentialDecayLR(double base_lr, double gamma, int64_t warmup_steps = 0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, gamma_;
  int64_t warmup_steps_;
};

/// Inverse square root schedule (used in original Transformer paper)
class InverseSqrtLR : public LRScheduler {
 public:
  InverseSqrtLR(double base_lr, int64_t warmup_steps);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_;
  int64_t warmup_steps_;
};

/// Polynomial decay
class PolynomialDecayLR : public LRScheduler {
 public:
  PolynomialDecayLR(double base_lr, double power = 1.0, int64_t warmup_steps = 0, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, power_, min_lr_;
  int64_t warmup_steps_;
};

/// Step decay: multiply lr by gamma every step_size steps
class StepDecayLR : public LRScheduler {
 public:
  StepDecayLR(double base_lr, double gamma, int64_t step_size, int64_t warmup_steps = 0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, gamma_;
  int64_t step_size_, warmup_steps_;
};

/// Warmup-Stable-Decay (WSD) schedule used by OLMo
class WSDLR : public LRScheduler {
 public:
  WSDLR(double base_lr, int64_t warmup_steps, int64_t stable_steps, double min_lr = 0.0);
  double get_lr(int64_t step, int64_t total_steps) const override;
 private:
  double base_lr_, min_lr_;
  int64_t warmup_steps_, stable_steps_;
};

/// Custom schedule from a user-provided function
class CustomLR : public LRScheduler {
 public:
  using ScheduleFn = std::function<double(int64_t step, int64_t total_steps)>;
  explicit CustomLR(ScheduleFn fn) : fn_(std::move(fn)) {}
  double get_lr(int64_t step, int64_t total_steps) const override { return fn_(step, total_steps); }
 private:
  ScheduleFn fn_;
};

/// Factory to create schedulers by name
std::unique_ptr<LRScheduler> create_scheduler(
    const std::string& name, double base_lr, int64_t warmup_steps = 0,
    double min_lr = 0.0, double gamma = 0.1, int64_t step_size = 1000,
    double power = 1.0, int64_t stable_steps = 0);

}  // namespace olmo_cpp
