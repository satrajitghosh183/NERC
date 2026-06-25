/**
 * src/optim/lion.cpp
 *
 * Lion optimizer implementation (Chen et al. 2023, arXiv:2302.06675). Update:
 *     update = sign(beta1 * m + (1 - beta1) * g)   (NOT stored — transient)
 *     theta -= lr * update
 *     theta -= lr * wd * theta
 *     m      = beta2 * m + (1 - beta2) * g          (persisted)
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/lion.hpp: declarations being implemented here.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: instantiated when cfg.optimizer == "lion".
 *
 * --- Role in training pipeline ---
 *   Constructed once after model init; .step() called each microbatch after
 *   backward + grad clip. Half the optimizer-state memory of AdamW since
 *   we keep only one EMA tensor per parameter.
 */
#include "olmo_cpp/optim/lion.hpp"

namespace olmo_cpp {

namespace {

/// Per-parameter state for Lion: a single momentum buffer and a step counter.
/// Notably absent: the second-moment v we would have in Adam/DION.
struct LionParamState : public torch::optim::OptimizerCloneableParamState<LionParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);   ///< Slow-EMA momentum m, same shape as param.
  TORCH_ARG(int64_t, step) = 0;                ///< Persisted step count for this param.

  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (momentum_buffer().defined()) {
      archive.write("momentum_buffer", momentum_buffer());
    }
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("momentum_buffer", buf)) {
      momentum_buffer(buf);
    }
  }
};

}  // namespace

Lion::Lion(std::vector<torch::Tensor> params, LionOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<LionOptions>(defaults)) {}

Lion::Lion(std::vector<torch::optim::OptimizerParamGroup> param_groups, LionOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<LionOptions>(defaults)) {}

/// One Lion update across all param groups.
torch::Tensor Lion::step(LossClosure closure) {
  // Disable autograd inside the optimizer body.
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    // Pull hyper-parameters once per group.
    auto& options = static_cast<LionOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double weight_decay = options.weight_decay();

    for (auto& p : group.params()) {
      // Skip parameters with no gradient this step.
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      // Lazily allocate the momentum buffer on first contact.
      if (state_.find(key) == state_.end()) {
        auto s = std::make_unique<LionParamState>();
        s->momentum_buffer(torch::zeros_like(p.data()));
        s->step(0);
        state_[key] = std::move(s);
      }

      auto& state = static_cast<LionParamState&>(*state_[key]);
      auto& m = state.momentum_buffer();
      state.step(state.step() + 1);

      // Step 1: Decoupled weight decay applied BEFORE the sign update so
      // the WD direction is unaffected by the sign() call.
      //   theta -= lr * wd * theta
      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }

      // Step 2: Update direction = sign( beta1 * m + (1 - beta1) * g ).
      // We must compute this from the OLD m (not yet updated for this step),
      // hence the temporary tensor — no in-place ops on m here.
      // sign_() is in-place on the temporary, not on m.
      auto update = (m * beta1 + grad * (1.0 - beta1)).sign_();

      // Step 3: theta -= lr * update.
      p.data().add_(update, -lr);

      // Step 4: NOW update the persistent momentum with the slower EMA:
      //   m = beta2 * m + (1 - beta2) * g
      // Order matters — Lion uses the OLD m for the update direction and
      // the NEW m only on subsequent steps.
      m.mul_(beta2).add_(grad, 1.0 - beta2);
    }
  }

  return loss;
}

}  // namespace olmo_cpp
