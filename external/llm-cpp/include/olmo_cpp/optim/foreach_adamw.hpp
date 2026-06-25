#pragma once
/**
 * include/olmo_cpp/optim/foreach_adamw.hpp
 *
 * ForeachAdamW: AdamW (Loshchilov & Hutter, 2017) implemented with PyTorch's
 * batched _foreach_* primitives. The math is identical to standard AdamW:
 *     m_t  = beta1 * m_{t-1} + (1 - beta1) * g_t
 *     v_t  = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *     p_t  = (1 - lr*wd) * p_{t-1}                          (decoupled WD)
 *     p_t -= lr * (m_t/(1-beta1^t)) / (sqrt(v_t/(1-beta2^t)) + eps)
 * but instead of running ~5-6 elementwise CUDA kernels per parameter (≈53k
 * launches for a 350M-param model with ~10k tensors), the foreach variant
 * issues 6-7 launches TOTAL by feeding vectors of tensors into a single fused
 * kernel. On H100 this often turns optimizer-step time from ~30 ms into <2 ms.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: torch::optim::Optimizer base + OptimizerParamGroup.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp:128,317,651: instantiated when cfg.optimizer == "adamw"
 *     and the foreach path is enabled (default).
 *
 * --- Role in training pipeline ---
 *   The default optimizer for OLMo C++ training. Constructed once after model
 *   init by train_loop based on cfg.optimizer; .step() called each microbatch
 *   after backward + optional grad clip. The scratch vectors below are reused
 *   across steps to avoid heap churn on the hot path.
 */
#include <torch/torch.h>

namespace olmo_cpp {

/// Hyper-parameters for ForeachAdamW. Field-for-field equivalent to
/// torch::optim::AdamWOptions; we keep our own type so an LR scheduler can
/// downcast options() and so we are not bound to LibTorch's internal layout.
struct ForeachAdamWOptions : public torch::optim::OptimizerCloneableOptions<ForeachAdamWOptions> {
  ForeachAdamWOptions(double lr = 1e-3) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1e-3;             ///< Step size η.
  TORCH_ARG(double, beta1) = 0.9;           ///< First-moment EMA decay.
  TORCH_ARG(double, beta2) = 0.999;         ///< Second-moment EMA decay.
  TORCH_ARG(double, eps) = 1e-8;            ///< Numerical floor in the denominator.
  TORCH_ARG(double, weight_decay) = 0.01;   ///< Decoupled weight decay (AdamW default 0.01).
  /// When true, the optimizer keeps an fp32 master copy of each (bf16) param,
  /// runs the whole Adam update in fp32 (master + fp32 moments + upcast grad),
  /// then writes the result back down into the bf16 param. This is the standard
  /// mixed-precision master-weights pattern: it removes the bf16 update-rounding
  /// (lr*grad ~ 1e-7 vanishing against a bf16 weight ULP ~ 1e-4) that makes pure
  /// bf16 training stall/diverge. Enable for bf16 runs; harmless (no-op) on fp32.
  TORCH_ARG(bool, master_weights) = false;
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

/// Fused AdamW using _foreach_* batched operations.
/// ~7 kernel launches per step regardless of parameter count,
/// vs ~53,000 with LibTorch's default per-parameter AdamW.
class ForeachAdamW : public torch::optim::Optimizer {
 public:
  /// Flat-list ctor — one param group, default options.
  explicit ForeachAdamW(std::vector<torch::Tensor> params, ForeachAdamWOptions defaults = {});
  /// Multi-group ctor — pre-built groups (e.g. separate WD per group).
  explicit ForeachAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups, ForeachAdamWOptions defaults = {});
  using torch::optim::Optimizer::step;
  /// Run one batched optimizer update over all param groups.
  torch::Tensor step(LossClosure closure = nullptr) override;

 private:
  /// Global step counter — used as the bias-correction exponent t in
  /// (1 - beta^t). Shared across all param groups; grows monotonically.
  int64_t step_count_ = 0;
  /// Running products beta1^t / beta2^t, updated incrementally per step
  /// so we don't call std::pow(beta, step) every step. Initialized to 1.0
  /// (matches t=0). The first ::step() multiplies by beta -> beta^1.
  double  beta1_pow_ = 1.0;
  double  beta2_pow_ = 1.0;
  /// Last (beta1, beta2) seen — if the user changes betas mid-run we
  /// rebuild the running powers from scratch via std::pow(beta, step_count_).
  double  last_beta1_ = -1.0;
  double  last_beta2_ = -1.0;
  // Scratch vectors reused across steps — we .clear() at the start of each
  // step so the capacity (sized to n_params on the first call) is retained
  // and subsequent push_backs allocate nothing. These hold tensor handles
  // (cheap shallow refs), not data, so the memory cost is negligible.
  std::vector<torch::Tensor> params_scratch_;
  std::vector<torch::Tensor> grads_scratch_;
  std::vector<torch::Tensor> exp_avg_scratch_;
  std::vector<torch::Tensor> exp_avg_sq_scratch_;
  // Master-weights mode only: parallel to params_scratch_. bf16_param_scratch_
  // holds the bf16 model params (write-back targets); params_scratch_/grads_scratch_
  // then hold the fp32 master + persistent fp32 grad buffers fed to the fused
  // update. bf16_grad_scratch_ holds the raw bf16 grads, batch-cast into the
  // fp32 grad buffers with a single _foreach_copy_ (instead of N per-param .to()).
  std::vector<torch::Tensor> bf16_param_scratch_;
  std::vector<torch::Tensor> bf16_grad_scratch_;
};

}  // namespace olmo_cpp
