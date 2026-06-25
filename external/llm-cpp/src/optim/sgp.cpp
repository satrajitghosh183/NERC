/**
 * src/optim/sgp.cpp
 *
 * Speculative Gradient Prediction v1. The original of this idea —
 * see sgp_v2.cpp's docblock for a longer description of the
 * "predict-then-verify" approach.
 *
 * v1 vs v2 in one line: v1 uses a single shared predictor across all
 * parameter groups and a looser acceptance criterion; v2 reuses that
 * predictor across more parameters but tightens the threshold.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/sgp.hpp : SGPOptimizer declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when sgp=1 + sgp_version=1, the train loop wraps
 *     the inner optimizer in SGPOptimizer.
 *
 * --- Role in training pipeline ---
 *   Wall-clock optimisation, opt-in. Off by default in the quickstart
 *   flow.
 */
#include "olmo_cpp/optim/sgp.hpp"
#include <ATen/ATen.h>

namespace olmo_cpp {

namespace {

/// Lazy-initialize a 0-dim FP32 scalar on the same device as `ref`.
/// Used for SGP coefficients (alpha, beta) so that all per-param updates stay
/// on-device — no GPU→CPU syncs in the hot path.
inline void ensure_scalar(torch::Tensor& t, float init_value, const torch::Tensor& ref) {
  if (!t.defined()) {
    t = torch::full({}, init_value,
                    torch::TensorOptions().dtype(torch::kFloat).device(ref.device()));
  }
}

}  // namespace

SGPPredictor::SGPPredictor(const std::vector<torch::Tensor>& params, SGPConfig config)
    : config_(config),
      params_(params),
      states_(params.size()),
      k_(config.initial_k),
      steps_since_anchor_(0) {}

bool SGPPredictor::should_skip_backward(int64_t global_step) const noexcept {
  if (global_step < config_.warmup_steps) return false;
  if (steps_since_anchor_ == 0) return false;
  return steps_since_anchor_ < k_;
}

void SGPPredictor::observe_real_gradients() {
  total_++;

  torch::NoGradGuard no_grad;

  // Device-resident error accumulator: one scalar, reduced at the end of the
  // loop with a single .item() sync. The CPU counter tracks how many params
  // contributed so we can normalize without a second sync.
  torch::Tensor error_sum;
  int64_t error_count = 0;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!p.grad().defined()) continue;
    if (p.numel() < config_.min_param_numel) continue;

    const auto true_grad = p.grad().detach();
    const auto dtype = true_grad.scalar_type();

    // Prediction-quality measurement — skipped on warmup and the first real
    // step (nothing to compare against). All tensor ops, zero host syncs.
    if (ps.has_history && steps_since_anchor_ > 0 && ps.alpha.defined()) {
      const auto alpha_t = ps.alpha.to(dtype);  // 0-dim, negligible
      torch::Tensor diff = true_grad - ps.prev_grad * alpha_t;
      if (ps.has_two_history) {
        const auto beta_t = ps.beta.to(dtype);
        diff = diff - ps.prev_prev_grad * beta_t;
      }
      const auto res_norm = diff.norm().to(torch::kFloat);
      const auto true_norm = true_grad.norm().to(torch::kFloat);

      if (!error_sum.defined()) {
        error_sum = torch::zeros({},
            torch::TensorOptions().dtype(torch::kFloat).device(true_grad.device()));
      }
      // epsilon in denominator guards against zero-norm grads without a
      // CPU-side comparison that would force a sync.
      error_sum.add_(res_norm / (true_norm + 1e-12f));
      ++error_count;

      update_predictor_coefficients(ps, true_grad);
    }

    // Shift history: prev_prev inherits prev's storage (no copy), prev gets
    // a fresh clone of the new real gradient.
    if (ps.has_history) {
      ps.prev_prev_grad = ps.prev_grad;
      ps.has_two_history = true;
    }
    ps.prev_grad = true_grad.clone();
    ps.has_history = true;
  }

  // Single device→host sync for K adaptation. Cost: ~5-10μs vs. ~2-4ms for
  // the per-param .item() pattern this replaces on a 3B model.
  if (error_count > 0 && error_sum.defined()) {
    const auto avg = (error_sum / static_cast<float>(error_count)).item<double>();
    last_error_ = avg;

    if (avg < config_.grow_threshold && k_ < config_.max_k) {
      ++k_;
    } else if (avg > config_.shrink_threshold && k_ > config_.min_k) {
      --k_;
    }
  }

  steps_since_anchor_ = 1;
}

void SGPPredictor::apply_predicted_gradients() {
  ++skipped_;
  ++total_;

  torch::NoGradGuard no_grad;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!ps.has_history) continue;
    if (p.numel() < config_.min_param_numel) continue;
    if (!ps.alpha.defined()) continue;

    const auto dtype = ps.prev_grad.scalar_type();
    const auto alpha_t = ps.alpha.to(dtype);

    torch::Tensor predicted = ps.prev_grad * alpha_t;
    if (ps.has_two_history) {
      const auto beta_t = ps.beta.to(dtype);
      predicted = predicted + ps.prev_prev_grad * beta_t;
    }

    if (p.grad().defined()) {
      p.grad().copy_(predicted);
    } else {
      p.mutable_grad() = std::move(predicted);
    }
  }

  ++steps_since_anchor_;
}

void SGPPredictor::update_predictor_coefficients(ParamState& ps, const torch::Tensor& true_grad) {
  // Online least-squares for G_pred = alpha*G_{t-1} + beta*G_{t-2}, computed
  // entirely on-device. Per-param cost: a handful of elementwise multiplies
  // and three reductions, no syncs.
  if (!ps.has_history) return;

  ensure_scalar(ps.alpha, 1.0f, true_grad);
  ensure_scalar(ps.beta, 0.0f, true_grad);

  const auto g = true_grad.view(-1);
  const auto g1 = ps.prev_grad.view(-1);

  // alpha ← 0.7*alpha + 0.3*<g,g1>/<g1,g1>
  // Reductions kept in native dtype (fits tensor cores); only the 0-dim
  // scalar result is promoted to FP32 for a stable division.
  const auto dot_gg1 = (g * g1).sum().to(torch::kFloat);
  const auto ns_g1 = (g1 * g1).sum().to(torch::kFloat);
  const auto new_alpha = (dot_gg1 / (ns_g1 + 1e-12f)).clamp(-10.0f, 10.0f);
  ps.alpha.lerp_(new_alpha, 0.3f);

  if (ps.has_two_history) {
    // beta ← 0.7*beta + 0.3*<residual, g2>/<g2, g2> where residual = g - alpha*g1.
    const auto alpha_t = ps.alpha.to(true_grad.scalar_type());
    const auto res = g - g1 * alpha_t;
    const auto g2 = ps.prev_prev_grad.view(-1);
    const auto dot_r_g2 = (res * g2).sum().to(torch::kFloat);
    const auto ns_g2 = (g2 * g2).sum().to(torch::kFloat);
    const auto new_beta = (dot_r_g2 / (ns_g2 + 1e-12f)).clamp(-10.0f, 10.0f);
    ps.beta.lerp_(new_beta, 0.3f);
    ps.beta.clamp_(-0.5f, 0.5f);
  }
}

}  // namespace olmo_cpp
