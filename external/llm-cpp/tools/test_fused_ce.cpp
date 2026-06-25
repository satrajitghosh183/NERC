// tools/test_fused_ce.cpp
//
// Parity check: fused_lm_head_ce_cpu / fused_lm_head_ce_autograd must
// agree with the unfused reference (linear + cross_entropy) on both
// loss values and gradients.

#include <torch/torch.h>
#include "olmo_cpp/backend/fused_lm_head_ce.hpp"

#include <cmath>
#include <iostream>

int main() {
  torch::manual_seed(0x5EED);

  const int64_t N = 64;
  const int64_t d = 128;
  const int64_t V = 257;
  const int64_t ignore_index = -100;

  auto h_opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto h = torch::randn({N, d}, h_opts).set_requires_grad(true);
  auto w = torch::randn({V, d}, h_opts).set_requires_grad(true);

  // Mix in some ignore_index labels.
  auto labels = torch::randint(0, V, {N}, torch::kInt64);
  labels[3] = ignore_index;
  labels[7] = ignore_index;
  labels[40] = ignore_index;

  // ── Reference path: linear + cross_entropy ───────────────────────
  auto h_ref = h.detach().clone().set_requires_grad(true);
  auto w_ref = w.detach().clone().set_requires_grad(true);
  auto logits_ref = torch::nn::functional::linear(h_ref, w_ref);
  auto loss_ref = torch::nn::functional::cross_entropy(
      logits_ref, labels,
      torch::nn::functional::CrossEntropyFuncOptions()
          .ignore_index(ignore_index)
          .reduction(torch::kMean));
  loss_ref.backward();

  // ── Fused path: fused_lm_head_ce_autograd ────────────────────────
  auto loss_fused = olmo_cpp::fused_lm_head_ce_autograd(h, w, labels, ignore_index);
  loss_fused.backward();

  const float loss_diff = std::abs(loss_ref.item<float>() - loss_fused.item<float>());
  std::cout << "loss ref   = " << loss_ref.item<float>() << "\n"
            << "loss fused = " << loss_fused.item<float>() << "\n"
            << "loss |diff| = " << loss_diff << "\n";
  if (loss_diff > 1e-4) {
    std::cerr << "FAIL: loss mismatch beyond tolerance\n";
    return 1;
  }

  auto grad_h_diff = (h.grad() - h_ref.grad()).abs().max().item<float>();
  auto grad_w_diff = (w.grad() - w_ref.grad()).abs().max().item<float>();
  std::cout << "grad_h max |diff| = " << grad_h_diff << "\n"
            << "grad_w max |diff| = " << grad_w_diff << "\n";
  if (grad_h_diff > 1e-3 || grad_w_diff > 1e-3) {
    std::cerr << "FAIL: gradient mismatch beyond tolerance\n";
    return 1;
  }

  std::cout << "FusedLMHeadCE parity OK\n";
  return 0;
}
