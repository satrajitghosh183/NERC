/**
 * src/optim/sgp_v2.cpp
 *
 * ─── What "Speculative Gradient Prediction" is ──────────────────────
 *
 * Backward passes are expensive — typically as costly as the forward
 * pass. SGP is an experimental technique in this repo that asks: can
 * we sometimes *predict* the next k gradients from history, take an
 * optimistic step, and only run a real backward to verify every k
 * steps?
 *
 * The predictor is a tiny low-rank linear model fit to the recent
 * sequence of (parameter, gradient) pairs. If its prediction lands
 * close enough to the verified gradient, k can grow; if it's wrong,
 * we roll back, fall back to k=1, and retrain the predictor.
 *
 * v2 differs from v1 (sgp.cpp) by:
 *   - reusing the predictor across more parameter groups,
 *   - using a tighter rejection criterion, so it accepts predictions
 *     less often but mis-applies them less often.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/sgp_v2.hpp : SGPv2 wrapper + state.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when sgp=1 + sgp_version=2 in the .conf, the
 *     train loop wraps the inner optimizer in SGPv2.
 *
 * --- Role in training pipeline ---
 *   Optional wall-clock optimisation. Off by default in the quickstart
 *   flow.
 */
#include "olmo_cpp/optim/sgp_v2.hpp"
#include <ATen/ATen.h>
#include <algorithm>
#include <iostream>

namespace olmo_cpp {

namespace {

inline void ensure_scalar(torch::Tensor& t, float init_value, const torch::Tensor& ref) {
  if (!t.defined()) {
    t = torch::full({}, init_value,
                    torch::TensorOptions().dtype(torch::kFloat).device(ref.device()));
  }
}

}  // namespace

SGPv2Predictor::SGPv2Predictor(const std::vector<torch::Tensor>& params,
                               SGPConfig config, int64_t rank)
    : config_(config),
      rank_(rank),
      params_(params),
      states_(params.size()),
      k_(config.initial_k),
      steps_since_anchor_(0) {
  int64_t rank2d_count = 0;
  int64_t linear_count = 0;
  for (size_t i = 0; i < params_.size(); ++i) {
    const auto& p = params_[i];
    auto& ps = states_[i];
    if (p.dim() == 2
        && p.size(0) >= 2 * rank_
        && p.size(1) >= 2 * rank_
        && p.numel() >= config_.min_param_numel) {
      ps.mode = Mode::Rank2D;
      ps.m = p.size(0);
      ps.n = p.size(1);
      ++rank2d_count;
    } else {
      ps.mode = Mode::Linear;
      if (p.numel() >= config_.min_param_numel) ++linear_count;
    }
  }
  std::cout << "SGP v2: " << rank2d_count << " rank-" << rank_ << " params, "
            << linear_count << " linear-predictor params" << std::endl;
}

bool SGPv2Predictor::should_skip_backward(int64_t global_step) const noexcept {
  if (global_step < config_.warmup_steps) return false;
  if (steps_since_anchor_ == 0) return false;
  return steps_since_anchor_ < k_;
}

void SGPv2Predictor::update_basis(ParamState& ps, const torch::Tensor& G_2d) {
  // Halko-style randomized SVD: sketch → QR → SVD of small projection.
  // Bulk matmuls stay in the gradient's native dtype (BF16 on H100 tensor
  // cores). Only small m×sketch and sketch×n projections are promoted to
  // FP32 for QR/SVD, which avoids materializing a full FP32 copy of G.
  const int64_t oversample = 5;
  const int64_t min_dim = std::min(ps.m, ps.n);
  const int64_t r = std::min<int64_t>(rank_, min_dim - 1);
  if (r < 1) return;
  const int64_t sketch = std::min<int64_t>(r + oversample, min_dim);

  const auto dtype = G_2d.scalar_type();
  const auto device = G_2d.device();

  // Sketch in FP32 first then cast down so we don't depend on BF16 randn.
  auto omega_f = torch::randn({ps.n, sketch},
      torch::TensorOptions().dtype(torch::kFloat).device(device));
  const auto omega = (dtype == torch::kFloat) ? omega_f : omega_f.to(dtype);

  // Y = G · Ω — native dtype, tensor-core eligible.
  const auto Y = torch::matmul(G_2d, omega);                  // m × sketch

  // QR requires FP32 on CUDA; promote only the small Y.
  const auto Y_f = (dtype == torch::kFloat) ? Y : Y.to(torch::kFloat);
  auto qr_result = at::linalg_qr(Y_f, "reduced");
  const auto Q_f = std::get<0>(qr_result);                    // m × sketch FP32

  // B = Q^T · G — cast Q back to native dtype so the big matmul stays native.
  const auto Q_native = (dtype == torch::kFloat) ? Q_f : Q_f.to(dtype);
  const auto B = torch::matmul(Q_native.transpose(0, 1), G_2d);  // sketch × n native

  // SVD requires FP32; promote only the small B.
  const auto B_f = (dtype == torch::kFloat) ? B : B.to(torch::kFloat);
  auto svd_result = at::_linalg_svd(B_f, /*full_matrices=*/false, /*compute_uv=*/true);
  const auto U_b = std::get<0>(svd_result);
  const auto Vh = std::get<2>(svd_result);

  const int64_t r_eff = std::min<int64_t>(r, std::min(U_b.size(1), Vh.size(0)));
  if (r_eff < 1) return;

  // Keep the basis in FP32 — it's tiny (m×r, n×r) and accuracy matters for
  // projections done many times between anchor steps.
  ps.U_left = torch::matmul(Q_f, U_b.slice(1, 0, r_eff)).contiguous();
  ps.U_right = Vh.slice(0, 0, r_eff).transpose(0, 1).contiguous();
  ps.has_basis = true;
}

void SGPv2Predictor::observe_real_gradients() {
  ++total_;

  torch::NoGradGuard no_grad;

  torch::Tensor error_sum;
  int64_t error_count = 0;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    auto& ps = states_[i];
    if (!p.grad().defined()) continue;
    if (p.numel() < config_.min_param_numel) continue;

    const auto true_grad = p.grad().detach();
    const auto dtype = true_grad.scalar_type();

    // Lazy device-tensor accumulator, initialized on the first param.
    const auto accum_device_opts =
        torch::TensorOptions().dtype(torch::kFloat).device(true_grad.device());

    if (ps.mode == Mode::Linear) {
      // v1 linear predictor fallback — identical math to SGPPredictor but
      // with tensor-based alpha/beta to keep this pass sync-free.
      if (ps.has_history && steps_since_anchor_ > 0 && ps.alpha.defined()) {
        const auto alpha_t = ps.alpha.to(dtype);
        torch::Tensor diff = true_grad - ps.prev_grad * alpha_t;
        if (ps.has_two_history) {
          const auto beta_t = ps.beta.to(dtype);
          diff = diff - ps.prev_prev_grad * beta_t;
        }
        const auto res_norm = diff.norm().to(torch::kFloat);
        const auto true_norm = true_grad.norm().to(torch::kFloat);
        if (!error_sum.defined()) error_sum = torch::zeros({}, accum_device_opts);
        error_sum.add_(res_norm / (true_norm + 1e-12f));
        ++error_count;

        update_linear_coefficients(ps, true_grad);
      }
      if (ps.has_history) {
        ps.prev_prev_grad = ps.prev_grad;
        ps.has_two_history = true;
      }
      ps.prev_grad = true_grad.clone();
      ps.has_history = true;
      continue;
    }

    // Rank-r subspace path.
    const auto G_2d = true_grad.view({ps.m, ps.n});

    // Prediction-quality measurement against the *previous* basis, still
    // current at this point because update_basis hasn't run yet.
    if (ps.has_basis && ps.has_history && steps_since_anchor_ > 0 && ps.alpha.defined()) {
      const auto alpha_t = ps.alpha.to(dtype);
      const auto U_native = ps.U_left.to(dtype);
      const auto V_native = ps.U_right.to(dtype);
      const auto U_native_t = U_native.transpose(0, 1);
      const auto V_native_t = V_native.transpose(0, 1);

      const auto prev_coord = torch::matmul(
          torch::matmul(U_native_t, ps.prev_grad), V_native);  // r × r
      auto pred_coord = prev_coord * alpha_t;
      if (ps.has_two_history) {
        const auto beta_t = ps.beta.to(dtype);
        const auto pp_coord = torch::matmul(
            torch::matmul(U_native_t, ps.prev_prev_grad), V_native);
        pred_coord = pred_coord + pp_coord * beta_t;
      }
      const auto rank_r_part = torch::matmul(torch::matmul(U_native, pred_coord), V_native_t);
      const auto off_subspace = ps.prev_grad
          - torch::matmul(torch::matmul(U_native, prev_coord), V_native_t);
      const auto predicted = rank_r_part + off_subspace;

      const auto res_norm = (G_2d - predicted).norm().to(torch::kFloat);
      const auto true_norm = G_2d.norm().to(torch::kFloat);
      if (!error_sum.defined()) error_sum = torch::zeros({}, accum_device_opts);
      error_sum.add_(res_norm / (true_norm + 1e-12f));
      ++error_count;
    }

    // Refresh basis from the current real gradient.
    update_basis(ps, G_2d);

    // Update alpha, beta in coordinate space using the freshly-computed
    // basis. All on-device, no syncs.
    if (ps.has_basis && ps.has_history) {
      ensure_scalar(ps.alpha, 1.0f, true_grad);
      ensure_scalar(ps.beta, 0.0f, true_grad);

      const auto U_native = ps.U_left.to(dtype);
      const auto V_native = ps.U_right.to(dtype);
      const auto U_native_t = U_native.transpose(0, 1);

      const auto cur_coord = torch::matmul(torch::matmul(U_native_t, G_2d), V_native);
      const auto prev_coord = torch::matmul(torch::matmul(U_native_t, ps.prev_grad), V_native);

      const auto dot = (cur_coord * prev_coord).sum().to(torch::kFloat);
      const auto ns = (prev_coord * prev_coord).sum().to(torch::kFloat);
      const auto new_alpha = (dot / (ns + 1e-12f)).clamp(-10.0f, 10.0f);
      ps.alpha.lerp_(new_alpha, 0.3f);

      if (ps.has_two_history) {
        const auto alpha_t = ps.alpha.to(dtype);
        const auto pp_coord = torch::matmul(
            torch::matmul(U_native_t, ps.prev_prev_grad), V_native);
        const auto res_coord = cur_coord - prev_coord * alpha_t;
        const auto dot2 = (res_coord * pp_coord).sum().to(torch::kFloat);
        const auto ns2 = (pp_coord * pp_coord).sum().to(torch::kFloat);
        const auto new_beta = (dot2 / (ns2 + 1e-12f)).clamp(-10.0f, 10.0f);
        ps.beta.lerp_(new_beta, 0.3f);
        ps.beta.clamp_(-0.5f, 0.5f);
      }
    }

    if (ps.has_history) {
      ps.prev_prev_grad = ps.prev_grad;
      ps.has_two_history = true;
    }
    ps.prev_grad = G_2d.clone();
    ps.has_history = true;
  }

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

void SGPv2Predictor::apply_predicted_gradients() {
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

    torch::Tensor predicted;
    if (ps.mode == Mode::Linear) {
      predicted = ps.prev_grad * alpha_t;
      if (ps.has_two_history) {
        const auto beta_t = ps.beta.to(dtype);
        predicted = predicted + ps.prev_prev_grad * beta_t;
      }
    } else {
      if (!ps.has_basis) continue;
      const auto U_native = ps.U_left.to(dtype);
      const auto V_native = ps.U_right.to(dtype);
      const auto U_native_t = U_native.transpose(0, 1);
      const auto V_native_t = V_native.transpose(0, 1);

      const auto prev_coord = torch::matmul(
          torch::matmul(U_native_t, ps.prev_grad), V_native);
      auto pred_coord = prev_coord * alpha_t;
      if (ps.has_two_history) {
        const auto beta_t = ps.beta.to(dtype);
        const auto pp_coord = torch::matmul(
            torch::matmul(U_native_t, ps.prev_prev_grad), V_native);
        pred_coord = pred_coord + pp_coord * beta_t;
      }
      const auto rank_r_part = torch::matmul(torch::matmul(U_native, pred_coord), V_native_t);
      const auto off_subspace = ps.prev_grad
          - torch::matmul(torch::matmul(U_native, prev_coord), V_native_t);
      predicted = (rank_r_part + off_subspace).view(p.sizes());
    }

    if (p.grad().defined()) {
      p.grad().copy_(predicted);
    } else {
      p.mutable_grad() = std::move(predicted);
    }
  }

  ++steps_since_anchor_;
}

void SGPv2Predictor::update_linear_coefficients(ParamState& ps, const torch::Tensor& true_grad) {
  // Same math as SGPPredictor::update_predictor_coefficients, kept inlined
  // rather than shared across translation units because both versions need
  // different ParamState types and the body is small.
  if (!ps.has_history) return;

  ensure_scalar(ps.alpha, 1.0f, true_grad);
  ensure_scalar(ps.beta, 0.0f, true_grad);

  const auto g = true_grad.view(-1);
  const auto g1 = ps.prev_grad.view(-1);

  const auto dot_gg1 = (g * g1).sum().to(torch::kFloat);
  const auto ns_g1 = (g1 * g1).sum().to(torch::kFloat);
  const auto new_alpha = (dot_gg1 / (ns_g1 + 1e-12f)).clamp(-10.0f, 10.0f);
  ps.alpha.lerp_(new_alpha, 0.3f);

  if (ps.has_two_history) {
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
