/**
 * kernels/olmo_ops_autograd.cpp
 *
 * Autograd wrappers for the olmo_ops:: custom kernels.
 *
 * IMPORTANT: This file MUST live in olmo_kernels (shared library), NOT in
 * olmo_cpp (static library). The TORCH_LIBRARY_IMPL static constructor has no
 * exported symbols, so the linker strips it silently from static archives —
 * the Autograd dispatch key registration never fires. Shared libraries load all
 * object files unconditionally, so placement here guarantees execution.
 *
 * Init ordering within olmo_kernels.so:
 *   PyTorch's TORCH_LIBRARY_IMPL defers impl registration when the schema has
 *   not been registered yet. TORCH_LIBRARY(olmo_ops, m) lives in rms_norm.cu
 *   (same shared lib). Whichever static constructor runs first is fine —
 *   PyTorch applies deferred impls once the schema appears.
 *
 * Background: kernels/rms_norm.cu, silu_mul.cu, and rope.cu register their
 * fused implementations only at the `CUDA` dispatch key. When training on
 * CUDA with autograd-tracking inputs, the dispatcher looks for `Autograd`,
 * doesn't find it, falls back to the backend key, and the output tensor is
 * produced without a grad_fn. Gradients then cannot flow back through any of
 * these ops — training silently produces wrong models.
 *
 * Fix: provide an Autograd impl per op via torch::autograd::Function.
 * Forward calls the underlying kernel via the dispatcher with
 * AutoDispatchBelowAutograd active (skips this autograd key, hits the
 * CUDA impl). Backward computes the standard gradient via ATen ops and the
 * fused B3 CUDA backward kernel.
 */

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "olmo_cpp/backend/rms_norm_backward.hpp"

namespace olmo_cpp {

namespace {

using namespace torch::autograd;

c10::optional<c10::OperatorHandle> find_op(const char* name) {
  return c10::Dispatcher::singleton().findOp({name, ""});
}

// ── RMSNorm ─────────────────────────────────────────────────────────────
//
//     y = x * rsqrt(mean(x^2, -1) + eps) * w
//
// Backward (standard):
//   rms          = rsqrt(mean(x^2, -1, keepdim=True) + eps)        [...,1]
//   x_hat        = x * rms
//   grad_x_hat   = grad_y * w                                       [...,D]
//   mean_term    = (grad_x_hat * x_hat).mean(-1, keepdim=True)      [...,1]
//   grad_x       = rms * (grad_x_hat - x_hat * mean_term)           [...,D]
//   grad_w       = (grad_y * x_hat).sum over batch dims              [D]
struct RmsNormFn : public Function<RmsNormFn> {
  static torch::Tensor forward(AutogradContext* ctx,
                                 torch::Tensor x,
                                 c10::optional<torch::Tensor> weight,
                                 double eps) {
    at::AutoDispatchBelowAutograd guard;
    static const auto op = find_op("olmo_ops::rms_norm");
    TORCH_CHECK(op.has_value(), "olmo_ops::rms_norm not registered");
    using FnType =
        torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double);
    auto y = op->typed<FnType>().call(x, weight, eps);
    ctx->save_for_backward({x, weight.value_or(torch::Tensor())});
    ctx->saved_data["eps"] = eps;
    ctx->saved_data["has_weight"] = weight.has_value();
    return y;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w_saved = saved[1];
    const double eps = ctx->saved_data["eps"].toDouble();
    const bool has_weight = ctx->saved_data["has_weight"].toBool();
    auto grad_y = grads[0];

    // B3 — fused CUDA backward when possible, ATen fallback otherwise.
    if (x.is_cuda() &&
        (x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32)) {
      c10::optional<torch::Tensor> w_opt;
      if (has_weight && w_saved.defined()) w_opt = w_saved;
      auto [grad_x, grad_w] =
          rms_norm_backward_cuda(grad_y, x, w_opt, eps);
      return {grad_x, grad_w, torch::Tensor()};
    }

    // CPU / unsupported-dtype path: ATen recompute.
    auto x_fp32 = x.to(torch::kFloat32);
    auto rms = (x_fp32 * x_fp32).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    auto x_hat = x_fp32 * rms;
    torch::Tensor grad_x_hat;
    if (has_weight && w_saved.defined()) {
      grad_x_hat = grad_y.to(torch::kFloat32) * w_saved.to(torch::kFloat32);
    } else {
      grad_x_hat = grad_y.to(torch::kFloat32);
    }
    auto mean_term = (grad_x_hat * x_hat).mean(-1, /*keepdim=*/true);
    auto grad_x = (rms * (grad_x_hat - x_hat * mean_term)).to(x.dtype());

    torch::Tensor grad_w;
    if (has_weight && w_saved.defined()) {
      auto gw32 = (grad_y.to(torch::kFloat32) * x_hat);
      const int64_t lead = gw32.dim() - 1;
      std::vector<int64_t> reduce_dims;
      reduce_dims.reserve(lead);
      for (int64_t i = 0; i < lead; ++i) reduce_dims.push_back(i);
      grad_w = gw32.sum(reduce_dims).to(w_saved.dtype());
    }
    return {grad_x, grad_w, torch::Tensor()};
  }
};

torch::Tensor rms_norm_autograd_impl(const torch::Tensor& x,
                                        const c10::optional<torch::Tensor>& weight,
                                        double eps) {
  return RmsNormFn::apply(x, weight, eps);
}

// ── RMSNormAdd (fused: y = residual + rms_norm(x) * w) ──────────────────
struct RmsNormAddFn : public Function<RmsNormAddFn> {
  static torch::Tensor forward(AutogradContext* ctx,
                                 torch::Tensor x,
                                 torch::Tensor residual,
                                 c10::optional<torch::Tensor> weight,
                                 double eps) {
    at::AutoDispatchBelowAutograd guard;
    static const auto op = find_op("olmo_ops::rms_norm_add");
    TORCH_CHECK(op.has_value(), "olmo_ops::rms_norm_add not registered");
    using FnType = torch::Tensor(const torch::Tensor&, const torch::Tensor&,
                                  const c10::optional<torch::Tensor>&, double);
    auto y = op->typed<FnType>().call(x, residual, weight, eps);
    ctx->save_for_backward({x, weight.value_or(torch::Tensor())});
    ctx->saved_data["eps"] = eps;
    ctx->saved_data["has_weight"] = weight.has_value();
    return y;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w_saved = saved[1];
    const double eps = ctx->saved_data["eps"].toDouble();
    const bool has_weight = ctx->saved_data["has_weight"].toBool();
    auto grad_y = grads[0];

    // residual contributes identity: grad_residual = grad_y.
    auto grad_residual = grad_y;

    // Same as RmsNorm backward for the norm portion.
    auto x_fp32 = x.to(torch::kFloat32);
    auto rms = (x_fp32 * x_fp32).mean(-1, true).add(eps).rsqrt();
    auto x_hat = x_fp32 * rms;
    torch::Tensor grad_x_hat = grad_y.to(torch::kFloat32);
    if (has_weight && w_saved.defined()) {
      grad_x_hat = grad_x_hat * w_saved.to(torch::kFloat32);
    }
    auto mean_term = (grad_x_hat * x_hat).mean(-1, true);
    auto grad_x = (rms * (grad_x_hat - x_hat * mean_term)).to(x.dtype());

    torch::Tensor grad_w;
    if (has_weight && w_saved.defined()) {
      auto gw32 = (grad_y.to(torch::kFloat32) * x_hat);
      const int64_t lead = gw32.dim() - 1;
      std::vector<int64_t> reduce_dims;
      reduce_dims.reserve(lead);
      for (int64_t i = 0; i < lead; ++i) reduce_dims.push_back(i);
      grad_w = gw32.sum(reduce_dims).to(w_saved.dtype());
    }
    return {grad_x, grad_residual, grad_w, torch::Tensor()};
  }
};

torch::Tensor rms_norm_add_autograd_impl(const torch::Tensor& x,
                                            const torch::Tensor& residual,
                                            const c10::optional<torch::Tensor>& weight,
                                            double eps) {
  return RmsNormAddFn::apply(x, residual, weight, eps);
}

// ── silu_mul ────────────────────────────────────────────────────────────
//   y = silu(gate) * up
//   d silu(g)/dg = sig + g * sig * (1 - sig) = sig * (1 + g * (1 - sig))
//   grad_gate = grad_y * up * d_silu(gate)
//   grad_up   = grad_y * silu(gate)
struct SiluMulFn : public Function<SiluMulFn> {
  static torch::Tensor forward(AutogradContext* ctx,
                                 torch::Tensor gate,
                                 torch::Tensor up) {
    at::AutoDispatchBelowAutograd guard;
    static const auto op = find_op("olmo_ops::silu_mul");
    TORCH_CHECK(op.has_value(), "olmo_ops::silu_mul not registered");
    using FnType = torch::Tensor(const torch::Tensor&, const torch::Tensor&);
    auto y = op->typed<FnType>().call(gate, up);
    ctx->save_for_backward({gate, up});
    return y;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
    auto saved = ctx->get_saved_variables();
    auto gate = saved[0];
    auto up   = saved[1];
    auto grad_y = grads[0];
    auto sig = torch::sigmoid(gate);
    auto silu = gate * sig;
    auto d_silu = sig + gate * sig * (1 - sig);
    auto grad_gate = grad_y * up * d_silu;
    auto grad_up   = grad_y * silu;
    return {grad_gate, grad_up};
  }
};

torch::Tensor silu_mul_autograd_impl(const torch::Tensor& gate,
                                        const torch::Tensor& up) {
  return SiluMulFn::apply(gate, up);
}

// ── apply_rope (half-rotation) ──────────────────────────────────────────
//
// Forward: y[i]        = x[i]   * cos - x[i+half] * sin
//          y[i + half] = x[i]   * sin + x[i+half] * cos
// Backward (transposed rotation):
//          gx[i]        =  gy[i]   * cos + gy[i+half] * sin
//          gx[i + half] = -gy[i]   * sin + gy[i+half] * cos
//
// Note: the kernel uses the parameter order (x, sin, cos) per its
// schema; we keep that order here. cos/sin require no grad.
//
// CAVEAT: rope.cu actually implements the (apply_rope x cos sin) order
// in its CPU reference but the dispatcher schema is whatever rope.cu
// declared. We trust the schema-declared parameter order.
struct ApplyRopeFn : public Function<ApplyRopeFn> {
  static torch::Tensor forward(AutogradContext* ctx,
                                 torch::Tensor x,
                                 torch::Tensor cos,
                                 torch::Tensor sin) {
    at::AutoDispatchBelowAutograd guard;
    static const auto op = find_op("olmo_ops::apply_rope");
    TORCH_CHECK(op.has_value(), "olmo_ops::apply_rope not registered");
    using FnType = torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
    auto y = op->typed<FnType>().call(x, cos, sin);
    ctx->save_for_backward({cos, sin});
    ctx->saved_data["head_dim"] = x.size(-1);
    return y;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
    auto saved = ctx->get_saved_variables();
    auto cos = saved[0];
    auto sin = saved[1];
    auto grad_y = grads[0];
    const int64_t head_dim = ctx->saved_data["head_dim"].toInt();
    const int64_t half = head_dim / 2;
    // Apply inverse half-rotation: same op with negated sin.
    auto first  = grad_y.narrow(-1, 0,    half);
    auto second = grad_y.narrow(-1, half, half);
    auto cb = cos.unsqueeze(0).unsqueeze(0);   // broadcast leading dims
    auto sb = sin.unsqueeze(0).unsqueeze(0);
    // Shapes: cos/sin expected as [S, head_dim] in apply_rope schema.
    // Use them directly via broadcasting compatible with grad_y's rank.
    while (cb.dim() < grad_y.dim()) {
      cb = cb.unsqueeze(0);
      sb = sb.unsqueeze(0);
    }
    auto cb_first  = cb.narrow(-1, 0,    half);
    auto cb_second = cb.narrow(-1, half, half);
    auto sb_first  = sb.narrow(-1, 0,    half);
    auto sb_second = sb.narrow(-1, half, half);
    // For half-rotation, cos[i] in full-dim is identical at first and
    // second halves (and same for sin). The cb_first / cb_second views
    // therefore carry the same per-position values; both yield the
    // expected c factor. Same for sb_*. We choose cb_first / sb_first
    // for the math.
    auto gx_first  =  first * cb_first + second * sb_first;
    auto gx_second = -first * sb_first + second * cb_first;
    auto grad_x = torch::cat({gx_first, gx_second}, /*dim=*/-1);
    return {grad_x, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor apply_rope_autograd_impl(const torch::Tensor& x,
                                          const torch::Tensor& cos,
                                          const torch::Tensor& sin) {
  return ApplyRopeFn::apply(x, cos, sin);
}

}  // namespace

}  // namespace olmo_cpp

// Register autograd-aware impls at the Autograd dispatch key.
//
// PyTorch ≥ 2.3 changed custom-op dispatch: the AutogradCUDA key no longer
// fires for TORCH_LIBRARY_IMPL registrations. The generic Autograd key is
// the correct target — it fires for all backends (CPU and CUDA) when a
// tensor has requires_grad=True. Inside each Function::forward we use
// AutoDispatchBelowAutograd so the call re-dispatches to the backend-specific
// impl (CUDA or CPU) rather than looping back here.
//
// This file MUST be in olmo_kernels (shared library). If placed in olmo_cpp
// (static library), the linker strips the object file since all symbols here
// are in anonymous namespace and nothing in the link graph references them by
// name. Shared libraries load all objects unconditionally, so the static
// constructor always fires.
//
// PyTorch handles cross-TU init ordering within this .so via deferred
// registration: if TORCH_LIBRARY_IMPL fires before TORCH_LIBRARY has
// registered the schema, the impl is queued and applied when the schema
// appears. Both live in this .so, so the queue always drains before the
// binary starts executing user code.
TORCH_LIBRARY_IMPL(olmo_ops, Autograd, m) {
  m.impl("rms_norm",     TORCH_FN(olmo_cpp::rms_norm_autograd_impl));
  m.impl("rms_norm_add", TORCH_FN(olmo_cpp::rms_norm_add_autograd_impl));
  m.impl("silu_mul",     TORCH_FN(olmo_cpp::silu_mul_autograd_impl));
  m.impl("apply_rope",   TORCH_FN(olmo_cpp::apply_rope_autograd_impl));
}
