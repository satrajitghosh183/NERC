#pragma once
/**
 * include/olmo_cpp/optim/muon.hpp
 *
 * Muon: orthogonalized momentum optimizer (Jordan, Bernstein et al., 2024).
 * Update rule, per parameter (2D matrices only — others fall back to plain
 * heavy-ball momentum):
 *     buf   = momentum * buf + g_t                       (no (1-mu) factor — Polyak-style)
 *     U     = NewtonSchulzOrthogonalize(buf, ns_steps)   (nearest semi-orthogonal matrix)
 *     theta -= lr * U
 *     theta -= lr * wd * theta                            (decoupled WD)
 *
 * The 5-step Newton-Schulz iteration converges to U = (buf buf^T)^{-1/2} buf,
 * i.e. the orthogonal factor of buf's polar decomposition. Replacing the raw
 * momentum with its orthogonal projection prevents the spectral norm of the
 * update from blowing up along a few dominant directions, which is what hurts
 * SGD/Adam on transformer weights at high LR.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: optimizer base classes; tensor + matmul.
 *   - <c10/cuda/CUDAStream.h>, <ATen/cuda/CUDAEvent.h>: side-stream support
 *     for the optional async-NS path that overlaps the (expensive) NS
 *     iteration with the next forward+backward.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp:122,309,645: instantiated when cfg.optimizer == "muon".
 *
 * --- Role in training pipeline ---
 *   Constructed once after model init by train_loop based on cfg.optimizer;
 *   .step() called each microbatch after backward + optional grad clip.
 *   With async_ns=true (cfg.async_muon), the NS iteration for large 2D
 *   weights runs on a separate CUDA stream and the resulting orthogonalized
 *   update is applied one step LATE — sacrificing one step of staleness to
 *   hide ~70% of NS compute behind the next training step.
 */
#include <torch/torch.h>
#include <vector>
#include <unordered_map>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

namespace olmo_cpp {

/// Hyper-parameters for Muon.
struct MuonOptions : public torch::optim::OptimizerCloneableOptions<MuonOptions> {
  MuonOptions(double lr = 0.02) : lr_(lr) {}
  TORCH_ARG(double, lr) = 0.02;             ///< Step size η. Muon tolerates much larger LR than AdamW.
  TORCH_ARG(double, momentum) = 0.95;       ///< Polyak heavy-ball momentum (no (1-mu) on the gradient).
  TORCH_ARG(double, weight_decay) = 0.0;    ///< Decoupled WD coefficient.
  TORCH_ARG(int64_t, ns_steps) = 5;         ///< Number of Newton-Schulz iterations (5 is empirically saturated).
  TORCH_ARG(bool, async_ns) = false;        ///< If true, run NS on a side stream and apply 1 step late.
  TORCH_ARG(int64_t, async_min_numel) = 65536;  ///< Only async-ize tensors above this size — small NS is faster sync.
  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { return lr_; }
};

/// Muon optimizer. State per param: momentum_buffer + step counter (+ async
/// staging buffers if async_ns is enabled).
class Muon : public torch::optim::Optimizer {
 public:
  explicit Muon(std::vector<torch::optim::OptimizerParamGroup> param_groups, MuonOptions defaults = {});
  explicit Muon(std::vector<torch::Tensor> params, MuonOptions defaults = {});
  using torch::optim::Optimizer::step;
  torch::Tensor step(LossClosure closure = nullptr) override;

 private:
  /// Run the Newton-Schulz iteration to compute the nearest semi-orthogonal
  /// matrix to G. Tall matrices are handled by transposition. Static so we
  /// can call it without an instance from the side stream path.
  static torch::Tensor newton_schulz_orthogonalize(torch::Tensor G, int64_t steps);

#ifdef USE_CUDA
  /// Per-parameter state for the async-NS path. The NS iteration for the
  /// CURRENT step lands here; the NEXT call to step() consumes it.
  struct AsyncState {
    torch::Tensor pending_update;     ///< Orthogonalized momentum from previous step, lr=1.
    at::cuda::CUDAEvent ready_event;  ///< Signals when pending_update is ready on the side stream.
    bool has_pending = false;         ///< False on the first step or after consumption.
  };
  /// Keyed on TensorImpl* to match the LibTorch state_ map.
  std::unordered_map<void*, AsyncState> async_states_;
  /// Side stream lazily acquired from the CUDA stream pool on first 2D param.
  c10::optional<at::cuda::CUDAStream> ns_stream_;
#endif
};

}  // namespace olmo_cpp
