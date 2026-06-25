/**
 * src/backend/simd_backend.cpp
 *
 * ─── What "SIMD" is ─────────────────────────────────────────────────
 *
 * SIMD = **S**ingle **I**nstruction **M**ultiple **D**ata. Modern
 * CPUs have wide vector registers (NEON 128-bit on Apple Silicon /
 * ARM, AVX2 256-bit on most x86, AVX-512 512-bit on server x86) that
 * can execute the same operation on 4, 8, or 16 floats at once. To
 * get good CPU performance you have to write loops that the compiler
 * (or you, with intrinsics) can map onto these wide ops.
 *
 * This file is the **CPU equivalent** of the .cu kernels:
 *
 *   - SIMDBackend implements the IBackend interface (rms_norm,
 *     silu_mul, apply_rope, residual_rms_norm) using vectorised
 *     CPU code from src/backend/kernels/simd_elementwise.cpp.
 *
 *   - The fast paths only kick in when the input is contiguous, on
 *     CPU, and in a supported dtype. Anything else falls back to the
 *     default ATen recipe (the IBackend base class's implementation).
 *     The can_use_simd() predicate at the top of every op gates that.
 *
 * On the 3060 quickstart the SIMD backend is NOT installed because
 * device=cuda; CUDABackend takes over. SIMDBackend is what runs on a
 * Mac laptop or any GPU-less Linux box.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/simd_backend.hpp           : SIMDBackend class.
 *   - olmo_cpp/backend/kernels/simd_elementwise.hpp : the actual
 *                                                     vectorised kernels.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: use_simd_backend() is called when device=cpu,
 *     installing SIMDBackend as the global backend singleton.
 *
 * --- Role in training pipeline ---
 *   The fast-path for CPU-only training. Mirrors what the .cu kernels
 *   do but with NEON/AVX intrinsics instead of CUDA threads.
 */
#include "olmo_cpp/backend/simd_backend.hpp"
#include "olmo_cpp/backend/kernels/simd_elementwise.hpp"

namespace olmo_cpp {

bool SIMDBackend::can_use_simd(const torch::Tensor& t) {
  return t.is_contiguous() &&
         t.device().is_cpu() &&
         t.scalar_type() == torch::kFloat32;
}

torch::Tensor SIMDBackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  if (!can_use_simd(x)) {
    return IBackend::rms_norm(x, weight, eps);
  }

  // Flatten all dims except last to get (n_rows, d)
  auto orig_sizes = x.sizes().vec();
  int64_t d = orig_sizes.back();
  int64_t n_rows = x.numel() / d;
  auto x_flat = x.reshape({n_rows, d});

  auto out = torch::empty_like(x_flat);

  const float* w_ptr = (weight.defined() && weight.numel() > 0)
                            ? weight.contiguous().data_ptr<float>()
                            : nullptr;

  kernels::fused_rms_norm_f32(
      x_flat.data_ptr<float>(), w_ptr, out.data_ptr<float>(),
      n_rows, d, static_cast<float>(eps));

  return out.view(orig_sizes);
}

torch::Tensor SIMDBackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
  if (!can_use_simd(gate) || !can_use_simd(up)) {
    return IBackend::silu_mul(gate, up);
  }

  auto gate_c = gate.contiguous();
  auto up_c = up.contiguous();
  auto out = torch::empty_like(gate_c);

  kernels::fused_silu_mul_f32(
      gate_c.data_ptr<float>(), up_c.data_ptr<float>(),
      out.data_ptr<float>(), gate_c.numel());

  return out;
}

torch::Tensor SIMDBackend::apply_rope(torch::Tensor t, torch::Tensor sin,
                                       torch::Tensor cos) {
  if (!can_use_simd(t)) {
    return IBackend::apply_rope(t, sin, cos);
  }

  // t is (..., head_dim). We need contiguous sin/cos broadcast-expanded to t's shape.
  auto t_c = t.contiguous();
  auto sin_c = sin.expand_as(t_c).contiguous();
  auto cos_c = cos.expand_as(t_c).contiguous();

  auto out = torch::empty_like(t_c);

  int64_t head_dim = t_c.size(-1);
  int64_t n = t_c.numel() / head_dim;

  kernels::fused_rope_f32(
      t_c.data_ptr<float>(), sin_c.data_ptr<float>(), cos_c.data_ptr<float>(),
      out.data_ptr<float>(), n, head_dim);

  return out;
}

torch::Tensor SIMDBackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                              torch::Tensor weight, double eps) {
  // Fuse: add + rms_norm in place.
  // For SIMD path: compute x+residual directly then run fused rms_norm.
  if (!can_use_simd(x) || !can_use_simd(residual)) {
    return IBackend::residual_rms_norm(x, residual, weight, eps);
  }

  auto h = x + residual;  // ATen add (could also be SIMD'd but marginal)
  return rms_norm(h, weight, eps);
}

// ---------------------------------------------------------------------------
// Arena scope management
// ---------------------------------------------------------------------------

void SIMDBackend::begin_scope() {
  scope_stack_.push_back(thread_arena().mark());
}

void SIMDBackend::end_scope() {
  if (!scope_stack_.empty()) {
    thread_arena().reset(scope_stack_.back());
    scope_stack_.pop_back();
  }
}

torch::Tensor SIMDBackend::alloc_scratch(at::IntArrayRef sizes, torch::ScalarType dtype,
                                          torch::Device device) {
  return thread_arena().allocate_tensor(sizes, dtype, device);
}

void use_simd_backend() {
  set_backend(std::make_unique<SIMDBackend>());
}

}  // namespace olmo_cpp
