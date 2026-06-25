#pragma once
/**
 * include/olmo_cpp/train/grad_scaler.hpp
 *
 * GradScaler declaration. In mixed-precision (FP16) training,
 * gradients can be too small to represent in 16-bit float. The
 * scaler multiplies the loss by a large factor S before backward,
 * lifting the gradients into FP16's representable range, then
 * divides them by S before the optimizer step.
 *
 * S is dynamic: auto-grows when training is calm, halves when an
 * Inf/NaN appears. See src/train/grad_scaler.cpp for the longer
 * pedagogical explanation.
 *
 * BF16 doesn't underflow as easily, so this is mostly a no-op for
 * BF16 runs — use_grad_scaler=0 in the BF16 path.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train/grad_scaler.cpp : implementation.
 *   - src/train.cpp             : wraps loss.backward / optim.step
 *                                  when train_cfg.use_grad_scaler=1.
 *
 * --- Role in training pipeline ---
 *   FP16 plumbing. Off by default in the quickstart flow.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Gradient scaler for mixed precision training.
/// Scales loss to prevent float16 gradient underflow.
class GradScaler {
 public:
  GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f,
             float backoff_factor = 0.5f, int growth_interval = 2000);

  /// Scale loss before backward()
  torch::Tensor scale(torch::Tensor loss);

  /// Unscale gradients. Returns true if all grads are finite.
  bool unscale_and_check(torch::optim::Optimizer& optimizer);

  /// Step optimizer (only if grads are finite)
  void step(torch::optim::Optimizer& optimizer);

  /// Update scale factor after step
  void update();

  float current_scale() const { return scale_; }
  bool found_inf() const { return found_inf_; }

 private:
  float scale_;
  float growth_factor_;
  float backoff_factor_;
  int growth_interval_;
  int steps_since_growth_ = 0;
  bool found_inf_ = false;
};

}  // namespace olmo_cpp
