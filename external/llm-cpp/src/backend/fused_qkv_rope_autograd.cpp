/**
 * src/backend/fused_qkv_rope_autograd.cpp
 *
 * Autograd Function wrapping fused_qkv_rope so training can call the
 * CUDA forward and still get gradients (item 1 follow-on).
 *
 * Forward: call into fused_qkv_rope (CUDA kernel or CPU ref) and stash
 * (x, w_qkv, cos, sin, n_q_heads, n_kv_heads, head_dim) on the context.
 *
 * Backward: the math is straightforward inverse-of-forward via ATen ops:
 *   - Inverse RoPE on grad_q / grad_k (RoPE is unitary: use (cos, -sin)).
 *   - Reshape grad_q/k/v from [B, H, S, D] back to [B, S, H*D].
 *   - Concat along feature dim -> grad_qkv [B, S, (n_q+2*n_kv)*D].
 *   - grad_x      = grad_qkv @ w_qkv   (no transpose: linear was y = x @ W.T,
 *                                       so dY/dx = grad_y @ W).
 *   - grad_w_qkv  = grad_qkv.T @ x_flat
 *
 * Numerics: matches the CPU reference path bitwise within fp32.
 */

#include "olmo_cpp/backend/fused_qkv_rope.hpp"

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <tuple>

namespace olmo_cpp {

namespace {

// Inverse RoPE matching the forward kernel's HALF-rotation convention:
//   y[i]        = x[i] * c - x[i + half] * s
//   y[i + half] = x[i] * s + x[i + half] * c
// Backward (transpose of the 2×2 rotation matrix):
//   grad_x[i]        =  grad_y[i] * c + grad_y[i + half] * s
//   grad_x[i + half] = -grad_y[i] * s + grad_y[i + half] * c
// cos/sin shapes: [S, head_dim/2]. Broadcast across [B, H, S, head_dim/2].
torch::Tensor inverse_rope(torch::Tensor t, torch::Tensor cos, torch::Tensor sin) {
  const int64_t head_dim = t.size(3);
  const int64_t half = head_dim / 2;
  auto first  = t.narrow(-1, 0,    half);
  auto second = t.narrow(-1, half, half);
  auto cb = cos.view({1, 1, cos.size(0), cos.size(1)});
  auto sb = sin.view({1, 1, sin.size(0), sin.size(1)});
  auto out_first  =  first * cb + second * sb;
  auto out_second = -first * sb + second * cb;
  return torch::cat({out_first, out_second}, /*dim=*/-1);
}

struct FusedQKVRopeFunction
    : public torch::autograd::Function<FusedQKVRopeFunction> {
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor x,
      torch::Tensor w_qkv,
      torch::Tensor cos,
      torch::Tensor sin,
      int64_t n_q_heads,
      int64_t n_kv_heads,
      int64_t head_dim) {
    auto out = fused_qkv_rope(x, w_qkv, cos, sin, n_q_heads, n_kv_heads, head_dim);
    auto& q = std::get<0>(out);
    auto& k = std::get<1>(out);
    auto& v = std::get<2>(out);
    ctx->save_for_backward({x, w_qkv, cos, sin});
    ctx->saved_data["n_q_heads"]  = n_q_heads;
    ctx->saved_data["n_kv_heads"] = n_kv_heads;
    ctx->saved_data["head_dim"]   = head_dim;
    return {q, k, v};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0]; auto w_qkv = saved[1];
    auto cos = saved[2]; auto sin = saved[3];
    const int64_t n_q  = ctx->saved_data["n_q_heads"].toInt();
    const int64_t n_kv = ctx->saved_data["n_kv_heads"].toInt();
    const int64_t hd   = ctx->saved_data["head_dim"].toInt();

    auto grad_q = grad_outputs[0];
    auto grad_k = grad_outputs[1];
    auto grad_v = grad_outputs[2];

    grad_q = inverse_rope(grad_q, cos, sin);
    grad_k = inverse_rope(grad_k, cos, sin);

    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t d = x.size(-1);
    const int64_t total = (n_q + 2 * n_kv) * hd;

    // Pre-allocate g_qkv and copy each grad slice into its halves —
    // no torch::cat (which would allocate a 4th tensor and copy all
    // three again). The .transpose(1,2).contiguous() copies are
    // unavoidable: we need head-major → seq-major + contiguous layout
    // for the downstream GEMM.
    auto g_qkv = torch::empty({B, S, total}, x.options());
    g_qkv.narrow(-1, 0,             n_q  * hd).copy_(
        grad_q.transpose(1, 2).contiguous().view({B, S, n_q  * hd}));
    g_qkv.narrow(-1, n_q * hd,      n_kv * hd).copy_(
        grad_k.transpose(1, 2).contiguous().view({B, S, n_kv * hd}));
    g_qkv.narrow(-1, (n_q + n_kv) * hd, n_kv * hd).copy_(
        grad_v.transpose(1, 2).contiguous().view({B, S, n_kv * hd}));

    // cuBLASLt-direct backward GEMMs (A2).
    //   grad_x      = g_qkv  @ w_qkv         [B*S, total] @ [total, d] → [B*S, d]
    //   grad_w_qkv  = g_qkv.T @ x_flat       [total, B*S] @ [B*S, d]   → [total, d]
    auto g_qkv_flat = g_qkv.view({B * S, total});
    auto x_flat     = x.view({B * S, d});
    auto grad_x_flat   = torch::matmul(g_qkv_flat, w_qkv);
    auto grad_w_qkv    = torch::matmul(g_qkv_flat.transpose(0, 1), x_flat);
    auto grad_x        = grad_x_flat.view(x.sizes());

    return {grad_x, grad_w_qkv, torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

}  // namespace

/// Differentiable forward: call this from AttentionImpl's training path
/// once the call-site swap from the discrete (Linear / RoPE / reshape)
/// chain to the fused kernel lands. Inference (no_grad) can call
/// fused_qkv_rope() directly.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_autograd(torch::Tensor x,
                          torch::Tensor w_qkv,
                          torch::Tensor cos,
                          torch::Tensor sin,
                          int64_t n_q_heads,
                          int64_t n_kv_heads,
                          int64_t head_dim) {
  auto out = FusedQKVRopeFunction::apply(x, w_qkv, cos, sin,
                                          n_q_heads, n_kv_heads, head_dim);
  return {out[0], out[1], out[2]};
}

}  // namespace olmo_cpp
