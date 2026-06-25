/**
 * src/optim/dion.cpp
 *
 * Implementation of the DION optimizer declared in dion.hpp. DION is AdamW
 * with the decoupled weight decay step kept verbatim:
 *     m_t = beta1*m_{t-1} + (1-beta1)*g_t
 *     v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
 *     theta -= lr * (m_t/(1-beta1^t)) / (sqrt(v_t/(1-beta2^t)) + eps)
 *     theta -= lr * wd * theta
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/dion.hpp: declarations of DION and DIONOptions; this
 *     file provides their definitions plus the file-local DIONParamState.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: instantiates DION when cfg.optimizer == "dion".
 *
 * --- Role in training pipeline ---
 *   The .step() method is invoked by the trainer after backward + grad clip
 *   each microbatch. State (m, v, step) is allocated lazily on first contact
 *   with each parameter and keyed by raw TensorImpl pointer.
 */
#include "olmo_cpp/optim/dion.hpp"

namespace olmo_cpp {

namespace {

/// Per-parameter state stored by the optimizer keyed on TensorImpl*.
/// Inherits OptimizerCloneableParamState so LibTorch can deep-copy state
/// across param-group reassignments and serialize/deserialize it.
struct DIONParamState : public torch::optim::OptimizerCloneableParamState<DIONParamState> {
  TORCH_ARG(torch::Tensor, exp_avg);      // First moment estimate m_t (same shape as param).
  TORCH_ARG(torch::Tensor, exp_avg_sq);   // Second moment estimate v_t (diagonal Hessian proxy).
  TORCH_ARG(int64_t, step) = 0;           // Number of completed updates for this param.

  /// Persist state to a checkpoint archive. step is stored as a 0-dim int64
  /// tensor because the archive format only supports tensor values.
  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (exp_avg().defined()) {
      archive.write("exp_avg", exp_avg());
    }
    if (exp_avg_sq().defined()) {
      archive.write("exp_avg_sq", exp_avg_sq());
    }
  }

  /// Restore state from a checkpoint archive. try_read is tolerant of missing
  /// keys so that older checkpoints (without one of the moments) still load.
  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("exp_avg", buf)) {
      exp_avg(buf);
    }
    if (archive.try_read("exp_avg_sq", buf)) {
      exp_avg_sq(buf);
    }
  }
};

}  // namespace

/// Flat-list ctor: wraps params into a single OptimizerParamGroup with the
/// given default options.
DION::DION(std::vector<torch::Tensor> params, DIONOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<DIONOptions>(defaults)) {}

/// Multi-group ctor: callers can pre-build param groups (e.g. with separate
/// LR/WD per group) before passing them in.
DION::DION(std::vector<torch::optim::OptimizerParamGroup> param_groups, DIONOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<DIONOptions>(defaults)) {}

/// One DION update across all parameter groups.
torch::Tensor DION::step(LossClosure closure) {
  // Forbid grad tracking inside the optimizer body — all the math we do here
  // is parameter-side bookkeeping and should never end up in the autograd graph.
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    // The optional closure may want gradients (re-running fwd+loss).
    // Re-enable autograd just for that block.
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    // Pull hyper-parameters out of the group's options object once per group.
    auto& options = static_cast<DIONOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double eps = options.eps();
    const double weight_decay = options.weight_decay();

    for (auto& p : group.params()) {
      // Skip params that did not receive a gradient this step (e.g. frozen
      // tied embeddings or a head that was bypassed for this microbatch).
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      // unsafeGetTensorImpl() is a stable per-tensor identity — works even
      // when the python-side autograd hands us a fresh Tensor wrapper.
      auto key = p.unsafeGetTensorImpl();

      // Lazily allocate per-param state on first contact. Both moments are
      // zero-initialized to match the AdamW convention (m_0 = v_0 = 0).
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<DIONParamState>();
        s->exp_avg(torch::zeros_like(p.data()));
        s->exp_avg_sq(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<DIONParamState&>(*state_[key]);
      auto& m = state.exp_avg();
      auto& v = state.exp_avg_sq();
      // Pre-increment t — this is the step we are about to perform, used
      // both as the bias-correction exponent and the persisted counter.
      state.step(state.step() + 1);
      int64_t t = state.step();

      // Step 1: Update biased first moment estimate
      //   m = beta1 * m + (1 - beta1) * g     (in-place fused via mul_ + add_)
      m.mul_(beta1).add_(grad, 1.0 - beta1);

      // Step 2: Update biased second moment estimate (diagonal Hessian proxy)
      //   v = beta2 * v + (1 - beta2) * g^2   (addcmul_ does v += a * g * g)
      v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

      // Step 3: Bias correction. Adam's m/v are biased toward 0 early in
      // training because they start at 0; dividing by (1 - beta^t) removes
      // the bias.
      double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(t));
      double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(t));
      auto m_hat = m / bias_correction1;
      auto v_hat = v / bias_correction2;

      // Step 4: Parameter update
      //   p -= lr * m_hat / (sqrt(v_hat) + eps)
      // addcdiv_ computes p += scalar * (m_hat / denom). NB: v_hat.sqrt()
      // returns a fresh tensor; the .add_(eps) here mutates that fresh
      // tensor, NOT the persistent v.
      p.data().addcdiv_(m_hat, v_hat.sqrt().add_(eps), -lr);

      // Step 5: Decoupled weight decay (AdamW-style).
      //   p -= lr * weight_decay * p
      // Applied AFTER the adaptive update so the decay rate doesn't get
      // re-scaled by the per-coordinate vhat denominator.
      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }
    }
  }

  return loss;
}

}  // namespace olmo_cpp
