#pragma once
/**
 * include/olmo_cpp/optim/lion.hpp
 *
 * Lion (EvoLved Sign Momentum) optimizer — Chen et al., NeurIPS 2023
 * (arXiv:2302.06675). Update rule, per parameter:
 *     update = sign( beta1 * m_{t-1} + (1 - beta1) * g_t )    (NOT in-place)
 *     theta -= lr * update
 *     theta -= lr * weight_decay * theta                       (decoupled WD)
 *     m_t   = beta2 * m_{t-1} + (1 - beta2) * g_t              (momentum updated AFTER step)
 * Notable properties:
 *   - Only ONE state tensor per param (momentum), so half the optimizer
 *     memory of AdamW — useful for training larger models on the same VRAM.
 *   - The update direction is the sign of a separate (faster) EMA of g, while
 *     the persistent momentum tracks a slower EMA. This decoupling is what
 *     makes Lion qualitatively different from sign-SGD.
 *   - Uses sign() so the step magnitude is exactly lr — typically Lion needs
 *     lr ~ 1/3 to 1/10 of an AdamW lr at the same WD.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: optimizer base classes and tensor types.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp:124,305,642: instantiated when cfg.optimizer == "lion".
 *
 * --- Role in training pipeline ---
 *   Constructed once after model init by train_loop based on cfg.optimizer;
 *   .step() is called each microbatch after backward + optional grad clip.
 */
#include <torch/torch.h>

namespace olmo_cpp {

/// Lion optimizer: uses sign of momentum for updates
/// Simpler than Adam, less memory (no second moment)
struct LionOptions : public torch::optim::OptimizerCloneableOptions<LionOptions> {
  LionOptions(double lr = 1e-4) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-4;            ///< Step size η. Lion typically uses 3-10x smaller LR than AdamW.
  TORCH_ARG(double, beta1) = 0.9;          ///< Fast EMA coefficient used inside sign().
  TORCH_ARG(double, beta2) = 0.99;         ///< Slow EMA coefficient used to update persistent momentum.
  TORCH_ARG(double, weight_decay) = 0.0;   ///< Decoupled weight decay coefficient.
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

/// Lion optimizer. State per param: momentum_buffer (single tensor) +
/// step counter. Half the memory footprint of AdamW.
class Lion : public torch::optim::Optimizer {
 public:
  explicit Lion(std::vector<torch::optim::OptimizerParamGroup> param_groups, LionOptions defaults = {});
  explicit Lion(std::vector<torch::Tensor> params, LionOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;
};

}  // namespace olmo_cpp
