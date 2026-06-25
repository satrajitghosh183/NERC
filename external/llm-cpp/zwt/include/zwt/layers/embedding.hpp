#pragma once

#include "zwt/layers/module.hpp"

namespace zwt {

// Token embedding: value = weight[token_ids].
// Weight shape: [vocab, d_model].
// Forward: input i64 [B, S] -> output dtype [B, S, d_model].
// Backward: scatter-add grad_y rows into grad_weight by saved ids.
class Embedding final : public Module {
 public:
  Embedding(int64_t vocab_size, int64_t d_model, DType dtype, Device device);

  Tensor forward(const Tensor& token_ids) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

 private:
  Parameter weight_;
  int64_t   vocab_size_;
  int64_t   d_model_;
  Tensor    saved_ids_;
  Shape     saved_out_shape_;
};

}  // namespace zwt
