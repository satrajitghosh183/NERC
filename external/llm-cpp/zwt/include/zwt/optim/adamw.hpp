#pragma once

#include "zwt/layers/parameter.hpp"
#include <vector>

namespace zwt::optim {

struct AdamWConfig {
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps   = 1e-8f;
  float weight_decay = 0.1f;
  float grad_clip = 0.0f;   // 0 = disabled
};

// Fused multi-param AdamW. One kernel launch updates all params at once by
// packing pointers + sizes into a device "param plan" that the kernel walks.
//
// This is the cost-center in every Python trainer: tiny per-param launches.
// We go from N launches to one, regardless of model size.
class AdamW {
 public:
  AdamW(std::vector<Parameter*> params, AdamWConfig cfg);
  ~AdamW();

  void step();
  void zero_grad();
  int64_t step_count() const { return step_count_; }
  void    set_step_count(int64_t s) { step_count_ = s; }

  AdamWConfig& config() { return cfg_; }

  // Access to moment buffers (fp32, same shape as the corresponding param).
  // The checkpoint path uses these to serialize/restore optimizer state.
  size_t  n_params()    const { return params_.size(); }
  Tensor& moment_m(size_t i);
  Tensor& moment_v(size_t i);

 private:
  struct State;
  State* state_ = nullptr;  // pimpl to keep CUDA types out of the header

  AdamWConfig cfg_;
  int64_t step_count_ = 0;
  std::vector<Parameter*> params_;
};

}  // namespace zwt::optim
