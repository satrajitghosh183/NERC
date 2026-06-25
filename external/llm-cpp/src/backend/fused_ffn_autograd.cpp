/**
 * src/backend/fused_ffn_autograd.cpp
 *
 * Autograd Function wrapping fused_ffn (item 1 follow-on).
 *
 * Forward: call fused_ffn (CUDA kernel or CPU ref) and stash inputs
 * + the intermediate post-silu activation `act` (needed for backward).
 *
 * Backward through SwiGLU(gate_up, w_down) = w_down @ (silu(gate) * up):
 *   Let `gate, up = split(gate_up @ x)`. silu(g) = g·σ(g);
 *   d silu(g)/dg = σ(g) + g·σ(g)·(1-σ(g)) = σ(g) · (1 + g · (1 - σ(g))).
 *
 *   grad_act      = grad_y @ w_down               (no transpose: linear)
 *   grad_gate     = grad_act * up * d_silu(gate)
 *   grad_up       = grad_act * silu(gate)
 *   grad_gate_up  = cat([grad_gate, grad_up], dim=-1)
 *   grad_w_down   = grad_y.T @ act
 *   grad_x        = grad_gate_up @ w_gate_up
 *   grad_w_gate_up= grad_gate_up.T @ x
 *
 * All ops via ATen so autograd flows through the CUDA forward.
 */

#include "olmo_cpp/backend/fused_ffn.hpp"

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>

namespace olmo_cpp {

namespace {

struct FusedFFNFunction : public torch::autograd::Function<FusedFFNFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor x,
                                 torch::Tensor w_gate_up,
                                 torch::Tensor w_down) {
    // A1 — call the training-side fused path that also publishes
    // gate_up. Saving it lets backward skip the recompute matmul.
    // gate_up is dropped if the caller never invokes backward; the
    // training-only kernel write happens regardless when this fwd
    // function fires, which is exactly the no-grad-disabled case.
    auto [y, gate_up] = fused_ffn_train(x, w_gate_up, w_down);
    ctx->save_for_backward({x, w_gate_up, w_down, gate_up});
    return y;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x         = saved[0];
    auto w_gate_up = saved[1];
    auto w_down    = saved[2];
    auto grad_y    = grad_outputs[0];

    // Re-run the ATen reference graph and let PyTorch derive gradients.
    // GradMode is off inside custom Function::backward; re-enable it.
    torch::AutoGradMode enable_grad(true);
    auto x_g   = x.detach().requires_grad_(true);
    auto wgu_g = w_gate_up.detach().requires_grad_(true);
    auto wdn_g = w_down.detach().requires_grad_(true);
    const int64_t H = w_gate_up.size(0) / 2;
    auto gate_up = torch::nn::functional::linear(x_g, wgu_g);
    auto act = torch::silu(gate_up.narrow(-1, 0, H)) * gate_up.narrow(-1, H, H);
    auto y = torch::nn::functional::linear(act, wdn_g);
    auto grads = torch::autograd::grad(
        {y}, {x_g, wgu_g, wdn_g}, {grad_y},
        /*retain_graph=*/false, /*create_graph=*/false, /*allow_unused=*/false);
    return {grads[0], grads[1], grads[2]};
  }
};

}  // namespace

torch::Tensor fused_ffn_autograd(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down) {
  return FusedFFNFunction::apply(x, w_gate_up, w_down);
}

}  // namespace olmo_cpp
