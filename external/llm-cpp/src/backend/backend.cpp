/**
 * src/backend/backend.cpp
 *
 * Defines the default ATen implementations of every IBackend op and
 * the global singleton plumbing (get_backend / set_backend). Each
 * concrete backend (SIMD, CUDA) inherits these as fallbacks: an op
 * that is not specialized for the active device just resolves to the
 * ATen recipe defined here.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/backend.hpp: IBackend declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: get_backend() at startup; use_simd_backend()/
 *     use_cuda_backend() (in those backends' .cpp files) call
 *     set_backend(...).
 *   - src/backend/cuda_backend.cpp, src/backend/simd_backend.cpp:
 *     fallback paths invoke IBackend::rms_norm etc. on the base class.
 *   - src/model/{block,attention,...}.cpp: every transformer block goes through
 *     get_backend().<op>(...) for fused elementwise pieces.
 *
 * --- Role in training pipeline ---
 *   This file *is* the LibTorch reference path: when no specialized
 *   backend is installed, every fused op runs the textbook ATen
 *   sequence here. It also doubles as the correctness oracle for the
 *   custom kernels — the SIMD and CUDA versions must produce results
 *   numerically equivalent to these implementations.
 */

#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// Default implementations (pure ATen, matches existing code exactly)
// ---------------------------------------------------------------------------

/// Reference RMSNorm. Note this is the LLaMA-style RMSNorm: it
/// normalizes by sqrt(mean(x^2) + eps) — there is no mean subtraction
/// (unlike LayerNorm). The optional weight is per-channel affine scale.
torch::Tensor IBackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  // Compute the variance/normalization in fp32. In bf16 the sum-of-squares
  // mean has ~8-bit mantissa, and the eps floor (1e-6) is swallowed whenever
  // variance >~ 4e-4 (bf16 ULP > eps), removing the divide-by-zero guard and
  // making the norm unstable. The custom CUDA kernel already accumulates in
  // fp32; this reference fallback (CPU, and Blackwell sm_120 where the fp32
  // kernel is gated off) must match.
  auto xf = x.to(torch::kFloat32);
  auto variance = xf.pow(2).mean(-1, true).add(eps);
  auto x_norm = (xf * torch::rsqrt(variance)).to(x.dtype());
  if (weight.defined()) {
    x_norm = x_norm * weight.to(x.dtype());
  }
  return x_norm;
}

/// SwiGLU's elementwise piece: silu(gate) * up. The default path
/// allocates one tensor for silu(gate) and another for the product;
/// fused backends do this in a single pass.
torch::Tensor IBackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
  return torch::silu(gate) * up;
}

/// Reference RoPE: rotate_half(t) is computed by chunking the last dim
/// in two and stacking [-x_high, x_low] back together. The result is
/// t*cos + rotate_half(t)*sin, then cast back to t's dtype (cos/sin
/// are typically fp32 even when t is bf16/fp16). The chunk+cat path
/// allocates 4 temporaries; the fused kernels avoid all of them.
torch::Tensor IBackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
  auto chunks = t.chunk(2, -1);
  auto rotated = torch::cat({-chunks[1], chunks[0]}, -1);
  return (t * cos + rotated * sin).to(t.dtype());
}

/// Pre-norm pattern: the residual has just been added in, normalize it.
/// The CUDA fused kernel does add+norm in one pass (residual_rms_norm.cu)
/// saving one full read/write of d_model.
torch::Tensor IBackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                           torch::Tensor weight, double eps) {
  auto h = x + residual;
  return rms_norm(h, weight, eps);
}

/// Mirror of residual_rms_norm but returning residual + norm(x): used
/// in some block layouts where the norm is applied to the sublayer
/// output before adding into the residual stream.
torch::Tensor IBackend::rms_norm_add(torch::Tensor x, torch::Tensor residual,
                                      torch::Tensor weight, double eps) {
  return residual + rms_norm(x, weight, eps);
}

// ---------------------------------------------------------------------------
// Default LibTorch backend (just inherits default impls)
// ---------------------------------------------------------------------------

/// Concrete IBackend that does nothing besides naming itself; all op
/// methods are inherited verbatim from the base class. Acts as the
/// "ATen reference" backend installed by default.
class LibTorchBackend : public IBackend {
 public:
  const char* name() const override { return "libtorch"; }
};

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// Process-wide backend pointer. Holds whichever IBackend was last
/// installed via set_backend(). Lazy-initialized to LibTorchBackend on
/// first read.
static std::unique_ptr<IBackend> g_backend;

/// Get the active backend, constructing the default LibTorchBackend
/// the first time anything asks. Not thread-safe with respect to
/// concurrent set_backend() calls — backend selection should happen
/// once at startup before worker threads start.
IBackend& get_backend() {
  if (!g_backend) {
    g_backend = std::make_unique<LibTorchBackend>();
  }
  return *g_backend;
}

/// Replace the global backend. Ownership of `backend` transfers in.
/// Passing a default-constructed unique_ptr (nullptr) clears the
/// pointer, in which case the next get_backend() call will reinstall
/// LibTorchBackend.
void set_backend(std::unique_ptr<IBackend> backend) {
  g_backend = std::move(backend);
}

}  // namespace olmo_cpp
