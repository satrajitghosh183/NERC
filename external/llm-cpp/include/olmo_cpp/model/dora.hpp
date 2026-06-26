#pragma once
/**
 * include/olmo_cpp/model/dora.hpp
 *
 * DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).
 *
 * A Linear's weight W [out, in] is decomposed into magnitude m and direction:
 *     W = m · (W / ||W||_row)
 * DoRA freezes W and trains a low-rank update plus the magnitude:
 *     W' = m · (W + s·B·A) / ||W + s·B·A||_row ,   s = alpha / rank
 * where A [rank, in], B [out, rank] (B init 0 so W'=W at step 0), m [out]
 * (init = per-output-row L2 norm of W). Only A, B, m are trained; W stays frozen.
 *
 * Drop-in for torch::nn::Linear (bias optional, frozen). To enable DoRA finetune,
 * the attention/FFN modules construct DoRALinear instead of nn::Linear when
 * TrainConfig.use_dora is set, passing the loaded base weight. Optimizer then
 * trains only requires_grad=true params (the adapters + magnitude + norms + MTP
 * heads); the frozen base contributes no optimizer state -> fits a 7B on one GPU.
 */
#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

class DoRALinearImpl : public torch::nn::Module {
 public:
  /// \param base_weight  [out, in] pretrained weight (copied, frozen)
  /// \param base_bias    optional bias (copied, frozen); pass undefined for none
  /// \param rank         low-rank dimension r
  /// \param alpha        LoRA scaling numerator (effective scale = alpha/rank)
  DoRALinearImpl(const torch::Tensor& base_weight,
                 const torch::Tensor& base_bias,
                 int rank, double alpha) {
    weight_ = register_parameter("weight", base_weight.detach().clone(),
                                 /*requires_grad=*/false);
    if (base_bias.defined())
      bias_ = register_parameter("bias", base_bias.detach().clone(),
                                 /*requires_grad=*/false);
    const int64_t out = weight_.size(0);
    const int64_t in = weight_.size(1);
    scaling_ = alpha / static_cast<double>(rank);
    auto fopts = torch::TensorOptions().dtype(weight_.dtype()).device(weight_.device());
    // A ~ N(0, 1/in), B = 0  ->  initial delta is zero (W' == W at step 0).
    lora_A_ = register_parameter(
        "lora_A", torch::randn({rank, in}, fopts) * (1.0 / std::sqrt(static_cast<double>(in))));
    lora_B_ = register_parameter("lora_B", torch::zeros({out, rank}, fopts));
    // Magnitude = per-output-row norm of the base weight (fp32 for stability).
    magnitude_ = register_parameter(
        "magnitude", weight_.detach().to(torch::kFloat32).norm(2, /*dim=*/1).to(weight_.dtype()));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto delta = torch::matmul(lora_B_, lora_A_) * scaling_;          // [out, in]
    auto adapted = weight_ + delta;                                  // [out, in]
    auto row_norm = adapted.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-6);  // [out,1]
    auto W = (magnitude_.unsqueeze(1) / row_norm) * adapted;         // [out, in]
    auto y = torch::matmul(x, W.transpose(0, 1));                    // [..., out]
    if (bias_.defined()) y = y + bias_;
    return y;
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_, bias_, lora_A_, lora_B_, magnitude_;
  double scaling_ = 1.0;
};
TORCH_MODULE(DoRALinear);

/// DoRAAdapter — the practical wiring form. Holds ONLY the trainable adapter
/// params (lora_A, lora_B, magnitude); the frozen base weight stays in the host
/// module's existing nn::Linear and is passed into forward(). This avoids the
/// construct-before-load ordering problem (the base .pt loads into the nn::Linear
/// normally; the adapter never owns a base copy). magnitude lazy-inits from the
/// base weight on the first forward (which runs after the base is loaded), so at
/// step 0 W' == base exactly. Params init fp32; main.cpp's bf16 pass casts them.
class DoRAAdapterImpl : public torch::nn::Module {
 public:
  DoRAAdapterImpl(int64_t out, int64_t in, int rank, double alpha) {
    scaling_ = alpha / static_cast<double>(rank);
    lora_A_ = register_parameter(
        "lora_A", torch::randn({rank, in}) * (1.0 / std::sqrt(static_cast<double>(in))));
    lora_B_ = register_parameter("lora_B", torch::zeros({out, rank}));
    magnitude_ = register_parameter("magnitude", torch::ones({out}));  // lazy-init in forward
  }

  /// x [..., in], base_weight [out, in] (frozen). Returns [..., out].
  torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& base_weight) {
    if (!mag_init_) {
      torch::NoGradGuard ng;
      magnitude_.set_data(
          base_weight.detach().to(torch::kFloat32).norm(2, 1).to(base_weight.dtype()));
      mag_init_ = true;
    }
    auto delta = torch::matmul(lora_B_, lora_A_) * scaling_;            // [out, in]
    auto adapted = base_weight + delta;                                // [out, in]
    auto rn = adapted.to(torch::kFloat32).norm(2, 1, /*keepdim=*/true)
                  .clamp_min(1e-6).to(base_weight.dtype());            // [out,1] (fp32 reduce)
    auto W = (magnitude_.unsqueeze(1) / rn) * adapted;                 // [out, in]
    return torch::matmul(x, W.transpose(0, 1));
  }

 private:
  torch::Tensor lora_A_, lora_B_, magnitude_;
  double scaling_ = 1.0;
  bool mag_init_ = false;
};
TORCH_MODULE(DoRAAdapter);

}  // namespace olmo_cpp
