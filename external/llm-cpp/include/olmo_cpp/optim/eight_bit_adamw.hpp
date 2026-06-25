#pragma once

/**
 * include/olmo_cpp/optim/eight_bit_adamw.hpp
 *
 * Block-quantized 8-bit AdamW (item O).
 *
 * Storage: exp_avg and exp_avg_sq are kept as INT8 in HBM with per-2048-
 * element block scales. ~4× drop on optimizer-state memory (8 bytes/param
 * → ~2.1 bytes/param). Numerics: dequantize-on-read, requantize-on-write
 * around the standard Adam math. Block size 2048 is bitsandbytes
 * convention.
 *
 * Trajectory parity vs FP32 Adam holds within Adam-style noise (the
 * bitsandbytes paper's main empirical result); we use the same dynamic
 * (max-abs) per-block scale.
 *
 * This wrapper plugs into the train loop where ForeachAdamW does today:
 * same step() / parameters() / state_dict() / load_state_dict() surface.
 */

#include <torch/torch.h>
#include <torch/optim.h>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

struct EightBitAdamWOptions
    : public torch::optim::OptimizerCloneableOptions<EightBitAdamWOptions> {
  TORCH_ARG(double, lr)             = 1e-3;
  TORCH_ARG(double, beta1)          = 0.9;
  TORCH_ARG(double, beta2)          = 0.999;
  TORCH_ARG(double, eps)            = 1e-8;
  TORCH_ARG(double, weight_decay)   = 0.0;
  TORCH_ARG(int64_t, block_size)    = 2048;
};

class EightBitAdamW : public torch::optim::Optimizer {
 public:
  EightBitAdamW(std::vector<torch::Tensor> params,
                 EightBitAdamWOptions defaults = {});
  explicit EightBitAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                          EightBitAdamWOptions defaults = {});

  torch::Tensor step(LossClosure closure = nullptr) override;

  /// Reset state (drops INT8 moment buffers); next step() will re-init
  /// them at zero.
  void reset_state();

  /// Total bytes used by optimizer state. Useful for memory reporting.
  int64_t state_bytes() const;

  void save(torch::serialize::OutputArchive& archive) const override;
  void load(torch::serialize::InputArchive& archive) override;

 private:
  int64_t step_count_ = 0;
};

}  // namespace olmo_cpp
