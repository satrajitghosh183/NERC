// tools/test_fused_qkv_rope.cpp
//
// Parity check: fused_qkv_rope_autograd (CPU dispatch) must match a
// reference path of separate Q/K/V linears + half-rotation RoPE on
// both outputs and gradients.

#include <torch/torch.h>
#include "olmo_cpp/backend/fused_qkv_rope.hpp"

#include <cmath>
#include <iostream>

namespace {

// Reference half-rotation RoPE applied to a head-major tensor
// [B, H, S, head_dim]. cos/sin shape [S, head_dim/2].
torch::Tensor apply_rope_ref(torch::Tensor t,
                              torch::Tensor cos,
                              torch::Tensor sin) {
  const int64_t head_dim = t.size(3);
  const int64_t half = head_dim / 2;
  auto first  = t.narrow(-1, 0,    half);
  auto second = t.narrow(-1, half, half);
  auto cb = cos.view({1, 1, cos.size(0), cos.size(1)});
  auto sb = sin.view({1, 1, sin.size(0), sin.size(1)});
  auto y_first  = first * cb - second * sb;
  auto y_second = first * sb + second * cb;
  return torch::cat({y_first, y_second}, /*dim=*/-1);
}

}  // namespace

int main() {
  torch::manual_seed(0xBEEF);

  const int64_t B  = 2;
  const int64_t S  = 8;
  const int64_t d  = 32;
  const int64_t n_q  = 4;
  const int64_t n_kv = 2;
  const int64_t hd   = d / n_q;     // head_dim = 8
  const int64_t half = hd / 2;       // 4

  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto x     = torch::randn({B, S, d},                   opts).set_requires_grad(true);
  auto w_qkv = torch::randn({(n_q + 2 * n_kv) * hd, d},  opts).set_requires_grad(true);
  auto cos   = torch::randn({S, half}, opts);
  auto sin   = torch::randn({S, half}, opts);

  // ── Reference: separate ops ──────────────────────────────────────
  auto x_ref     = x.detach().clone().set_requires_grad(true);
  auto w_qkv_ref = w_qkv.detach().clone().set_requires_grad(true);
  auto qkv = torch::nn::functional::linear(x_ref, w_qkv_ref);  // [B, S, (n_q+2*n_kv)*hd]
  auto q_flat = qkv.narrow(-1, 0,                  n_q  * hd);
  auto k_flat = qkv.narrow(-1, n_q * hd,           n_kv * hd);
  auto v_flat = qkv.narrow(-1, (n_q + n_kv) * hd,  n_kv * hd);
  auto q_ref = q_flat.view({B, S, n_q,  hd}).transpose(1, 2).contiguous();
  auto k_ref = k_flat.view({B, S, n_kv, hd}).transpose(1, 2).contiguous();
  auto v_ref = v_flat.view({B, S, n_kv, hd}).transpose(1, 2).contiguous();
  q_ref = apply_rope_ref(q_ref, cos, sin);
  k_ref = apply_rope_ref(k_ref, cos, sin);

  // ── Fused autograd ───────────────────────────────────────────────
  auto out = olmo_cpp::fused_qkv_rope_autograd(x, w_qkv, cos, sin, n_q, n_kv, hd);
  auto q_fused = std::get<0>(out);
  auto k_fused = std::get<1>(out);
  auto v_fused = std::get<2>(out);

  auto q_diff = (q_fused - q_ref).abs().max().item<float>();
  auto k_diff = (k_fused - k_ref).abs().max().item<float>();
  auto v_diff = (v_fused - v_ref).abs().max().item<float>();
  std::cout << "forward q max |diff| = " << q_diff << "\n"
            << "forward k max |diff| = " << k_diff << "\n"
            << "forward v max |diff| = " << v_diff << "\n";
  if (q_diff > 1e-4 || k_diff > 1e-4 || v_diff > 1e-4) {
    std::cerr << "FAIL: forward output mismatch\n";
    return 1;
  }

  // Backward through a random scalar reduction of all three outputs.
  auto upstream_q = torch::randn_like(q_ref);
  auto upstream_k = torch::randn_like(k_ref);
  auto upstream_v = torch::randn_like(v_ref);
  auto loss_ref   = (q_ref * upstream_q + 0.0).sum()
                  + (k_ref * upstream_k + 0.0).sum()
                  + (v_ref * upstream_v + 0.0).sum();
  auto loss_fused = (q_fused * upstream_q).sum()
                  + (k_fused * upstream_k).sum()
                  + (v_fused * upstream_v).sum();
  loss_ref.backward();
  loss_fused.backward();

  auto gx_diff = (x.grad() - x_ref.grad()).abs().max().item<float>();
  auto gw_diff = (w_qkv.grad() - w_qkv_ref.grad()).abs().max().item<float>();
  std::cout << "backward grad_x  max |diff| = " << gx_diff << "\n"
            << "backward grad_W  max |diff| = " << gw_diff << "\n";
  if (gx_diff > 1e-3 || gw_diff > 1e-3) {
    std::cerr << "FAIL: gradient mismatch\n";
    return 1;
  }

  std::cout << "FusedQKVRopeAutograd parity OK\n";
  return 0;
}
