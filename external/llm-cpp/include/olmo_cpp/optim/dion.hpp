#pragma once
/**
 * include/olmo_cpp/optim/dion.hpp
 *
 * DION: a diagonal-adaptive learning-rate optimizer in the AdamW family.
 * Update rule (per parameter, per step):
 *     m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
 *     v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2          (diagonal Hessian proxy)
 *     mhat  = m_t / (1 - beta1^t)                             (bias correction)
 *     vhat  = v_t / (1 - beta2^t)
 *     theta = theta - lr * mhat / (sqrt(vhat) + eps)
 *     theta = theta - lr * weight_decay * theta               (decoupled WD, AdamW-style)
 * It is essentially AdamW with the WD applied AFTER the adaptive update so the
 * effective regularization is decoupled from the per-coordinate scaling.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: LibTorch optimizer base classes (Optimizer,
 *     OptimizerCloneableOptions) and tensor types.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp:126,313,648: constructed via std::make_unique<DION> when
 *     the [training] config selects optimizer="dion".
 *
 * --- Role in training pipeline ---
 *   Constructed once after model init by train_loop based on cfg.optimizer;
 *   .step() called each microbatch after backward + optional grad clip.
 *   DION competes with ForeachAdamW as a default adaptive optimizer for
 *   hyper-parameter sweeps; the implementation here is per-parameter (not
 *   foreach-batched) so it is slower but easier to read and modify.
 */
#include <torch/torch.h>

namespace olmo_cpp {

/// Hyper-parameters for the DION optimizer. Inherits from LibTorch's
/// CloneableOptions so the optimizer state machine can deep-copy options
/// across param groups. Defaults match Adam(W) folklore.
struct DIONOptions : public torch::optim::OptimizerCloneableOptions<DIONOptions> {
  /// Convenience ctor: only the learning rate must be supplied at construction.
  DIONOptions(double lr = 1e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-3;             ///< Step size η in the update rule.
  TORCH_ARG(double, beta1) = 0.9;           ///< First-moment EMA decay (momentum on g).
  TORCH_ARG(double, beta2) = 0.999;         ///< Second-moment EMA decay (momentum on g^2).
  TORCH_ARG(double, eps) = 1e-8;            ///< Numerical floor inside sqrt(vhat)+eps.
  TORCH_ARG(double, weight_decay) = 0.0;    ///< Decoupled weight decay coefficient λ.
  /// Required override so an LR scheduler can mutate η between steps.
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

/// DION optimizer. Per-parameter adaptive update with decoupled weight decay.
/// State per tensor: exp_avg (m), exp_avg_sq (v), step counter.
class DION : public torch::optim::Optimizer {
 public:
  /// Multi-group constructor — each group can carry its own DIONOptions.
  explicit DION(std::vector<torch::optim::OptimizerParamGroup> param_groups, DIONOptions defaults = {});
  /// Flat-list constructor — wraps params into a single param group.
  explicit DION(std::vector<torch::Tensor> params, DIONOptions defaults = {});
  using torch::optim::Optimizer::step;
  /// Run one optimizer step. `closure` is an optional loss-recomputation
  /// callable supported by the LibTorch base class; rarely used in this
  /// codebase. Returns the recomputed loss (empty tensor if no closure).
  torch::Tensor step(LossClosure closure = nullptr) override;
};

}  // namespace olmo_cpp
