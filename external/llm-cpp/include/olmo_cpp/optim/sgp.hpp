#pragma once
/**
 * include/olmo_cpp/optim/sgp.hpp
 *
 * Header for the Speculative Gradient Prediction v1 optimizer wrapper.
 * SGP attempts to predict the next gradient from history with a tiny
 * linear model so the optimizer can step k>1 times between actual
 * backward passes; see src/optim/sgp.cpp for the longer explanation.
 *
 * --- Includes from this project ---
 *   (none — declares a class and uses LibTorch types.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/optim/sgp.cpp        : implements every method here.
 *   - src/train.cpp            : wraps the inner optimizer in SGP
 *                                 when sgp=1 + sgp_version=1.
 *
 * --- Role in training pipeline ---
 *   Optional wall-clock optimisation. Off in the quickstart flow.
 */
#include <torch/torch.h>
#include <vector>
#include <string>

namespace olmo_cpp {

/// Speculative Gradient Prediction (SGP) — skip backward passes by predicting
/// gradients from their recent history. Anchor-correct every K steps with a
/// real backward. Quality bounded by anchor frequency; speed from skip rate.
///
/// Usage: wrap the inner training loop. On each step, call should_skip_backward()
/// to decide whether to use predicted gradients or compute real ones. After a
/// real backward, call observe_real_gradients(). After a predicted step, call
/// apply_predicted_gradients().

struct SGPConfig {
  int64_t initial_k = 2;          // initial skip interval (predict K-1, anchor 1)
  int64_t max_k = 8;              // max skip interval
  int64_t min_k = 1;              // min skip interval (1 = no skipping)
  double grow_threshold = 0.1;    // grow K if prediction error < this
  double shrink_threshold = 0.3;  // shrink K if prediction error > this
  int64_t warmup_steps = 100;     // no skipping during warmup (gather statistics)
  int64_t min_param_numel = 1024; // only predict for params above this size
};

/// Abstract interface so v1 (linear) and v2 (rank-r subspace) can share a call site.
class ISGPPredictor {
 public:
  virtual ~ISGPPredictor() = default;
  [[nodiscard]] virtual bool should_skip_backward(int64_t global_step) const noexcept = 0;
  virtual void observe_real_gradients() = 0;
  virtual void apply_predicted_gradients() = 0;
  [[nodiscard]] virtual int64_t current_k() const noexcept = 0;
  [[nodiscard]] virtual double last_prediction_error() const noexcept = 0;
  [[nodiscard]] virtual int64_t skipped_steps() const noexcept = 0;
  [[nodiscard]] virtual int64_t total_steps() const noexcept = 0;
  [[nodiscard]] virtual double skip_rate() const noexcept = 0;
};

class SGPPredictor final : public ISGPPredictor {
 public:
  explicit SGPPredictor(const std::vector<torch::Tensor>& params, SGPConfig config = {});

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
  struct ParamState {
    torch::Tensor prev_grad;       // G_{t-1}
    torch::Tensor prev_prev_grad;  // G_{t-2}
    // 0-dim FP32 scalars on the same device as the gradient. Lazy-initialized
    // on first real backward so we can adopt the grad's device automatically.
    torch::Tensor alpha;
    torch::Tensor beta;
    bool has_history = false;
    bool has_two_history = false;
  };

  void update_predictor_coefficients(ParamState& ps, const torch::Tensor& true_grad);

  SGPConfig config_;
  std::vector<torch::Tensor> params_;
  std::vector<ParamState> states_;
  int64_t k_;                      // current skip interval
  int64_t steps_since_anchor_;     // steps since last real backward
  double last_error_ = 0.0;
  int64_t skipped_ = 0;
  int64_t total_ = 0;
};

}  // namespace olmo_cpp
