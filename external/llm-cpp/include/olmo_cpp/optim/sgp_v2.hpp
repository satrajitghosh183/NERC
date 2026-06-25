#pragma once
/**
 * include/olmo_cpp/optim/sgp_v2.hpp
 *
 * Header for the v2 variant of Speculative Gradient Prediction.
 * Differences vs v1 (sgp.hpp): tighter rejection criterion, predictor
 * reused across more parameter groups. See src/optim/sgp_v2.cpp.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/sgp.hpp : shared interfaces / state types.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/optim/sgp_v2.cpp : implementation.
 *   - src/train.cpp        : opt-in wrap when sgp=1 + sgp_version=2.
 *
 * --- Role in training pipeline ---
 *   Optional wall-clock optimisation. Off in the quickstart flow.
 */
#include "olmo_cpp/optim/sgp.hpp"
#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// SGP v2 — rank-r subspace gradient predictor.
///
/// For 2D parameters above a size threshold, tracks a rank-r orthonormal basis
/// (U_left, U_right) updated every real backward via randomized SVD of the
/// current gradient. Predicts future gradients as:
///
///   G_pred = U_left * extrap(coord_{t-1}, coord_{t-2}) * U_right^T   (rank-r part)
///          + (G_{t-1} - U_left * coord_{t-1} * U_right^T)            (off-subspace, assumed stable)
///
/// where coord_{t-k} = U_left^T * G_{t-k} * U_right are r×r coordinates in the
/// current basis, extrap(.) is the same online least-squares linear extrapolator
/// used by v1 but evaluated in coordinate space. This denoises the prediction
/// by splitting the gradient into a low-rank "principal" part that extrapolates
/// well, and a stable off-subspace part copied from the last real step.
///
/// For 1D parameters or small 2D parameters, falls back to the v1 linear
/// predictor exactly so that biases, norms and embeddings are still handled.
class SGPv2Predictor final : public ISGPPredictor {
 public:
  explicit SGPv2Predictor(const std::vector<torch::Tensor>& params,
                          SGPConfig config = {},
                          int64_t rank = 4);

  [[nodiscard]] bool should_skip_backward(int64_t global_step) const noexcept override;
  void observe_real_gradients() override;
  void apply_predicted_gradients() override;

  [[nodiscard]] int64_t current_k() const noexcept override { return k_; }
  [[nodiscard]] double last_prediction_error() const noexcept override { return last_error_; }
  [[nodiscard]] int64_t skipped_steps() const noexcept override { return skipped_; }
  [[nodiscard]] int64_t total_steps() const noexcept override { return total_; }
  [[nodiscard]] double skip_rate() const noexcept override {
    return total_ > 0 ? static_cast<double>(skipped_) / static_cast<double>(total_) : 0.0;
  }

 private:
  enum class Mode { Rank2D, Linear };

  struct ParamState {
    Mode mode = Mode::Linear;
    // Gradient history (used by both Linear and Rank2D; for Rank2D it holds
    // the already-reshaped 2D view so we never re-view on the hot path).
    torch::Tensor prev_grad;
    torch::Tensor prev_prev_grad;
    // 0-dim FP32 scalars on device. Lazy-initialized on first real backward.
    // For Rank2D mode these live in coordinate space; for Linear they live in
    // flat-grad space, but the math is identical.
    torch::Tensor alpha;
    torch::Tensor beta;
    bool has_history = false;
    bool has_two_history = false;
    // Rank-r basis (only populated when mode == Rank2D). Stored in FP32 since
    // they are small (m × r, n × r for r = 4) and numerical stability matters
    // for the SVD that produced them.
    torch::Tensor U_left;
    torch::Tensor U_right;
    bool has_basis = false;
    int64_t m = 0;
    int64_t n = 0;
  };

  void update_basis(ParamState& ps, const torch::Tensor& G_2d);
  void update_linear_coefficients(ParamState& ps, const torch::Tensor& true_grad);

  SGPConfig config_;
  int64_t rank_;
  std::vector<torch::Tensor> params_;
  std::vector<ParamState> states_;
  int64_t k_;
  int64_t steps_since_anchor_;
  double last_error_ = 0.0;
  int64_t skipped_ = 0;
  int64_t total_ = 0;
};

}  // namespace olmo_cpp
