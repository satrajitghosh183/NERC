#pragma once

#include "zwt/layers/module.hpp"

namespace zwt {

// y = x @ W^T + b
// Weight stored as [out_features, in_features] (row-major). This is the
// layout cuBLAS GEMM wants for a "transposed A, not-transposed B" call.
class Linear final : public Module {
 public:
  Linear(int64_t in_features, int64_t out_features, bool use_bias,
         DType dtype, Device device, uint64_t init_seed = 0xC00FFEEULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  Parameter& weight() { return weight_; }
  Parameter* bias()   { return has_bias_ ? &bias_ : nullptr; }

 private:
  Parameter weight_;
  Parameter bias_;
  bool      has_bias_;
  int64_t   in_features_;
  int64_t   out_features_;

  Tensor    saved_input_;  // view of forward input (not owned)
  Shape     saved_input_shape_;
};

}  // namespace zwt
