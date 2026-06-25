/**
 * src/backend/cuda_backend.cpp
 *
 * CUDABackend implementation. Each method (rms_norm, silu_mul,
 * apply_rope, residual_rms_norm) resolves a TORCH_LIBRARY-registered
 * "olmo_ops::*" op handle exactly once (cached in a static local) and
 * dispatches the call into our hand-written kernel. The handle lookup
 * involves a hash-map probe on a string; doing it on every layer per
 * step would be hundreds of redundant lookups per training step, so
 * caching matters. Static-local initialization is thread-safe since
 * C++11 (per-block magic-static guard).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/cuda_backend.hpp: declares CUDABackend.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: use_cuda_backend() at startup when CUDA device chosen.
 *   - kernels/rms_norm.cu / silu_mul.cu / rope.cu: each registers its
 *     impl into the olmo_ops:: schema via TORCH_LIBRARY_IMPL(... CUDA ...).
 *
 * --- Role in training pipeline ---
 *   This is the bridge between the device-agnostic transformer code
 *   and the .cu files. The forward pass of any TransformerBlock on
 *   CUDA hits these methods, which in turn launch CUDA kernels. If a
 *   tensor is in an unsupported dtype we fall through to IBackend's
 *   ATen recipe so correctness is preserved at the cost of speed.
 */

#include "olmo_cpp/backend/cuda_backend.hpp"

#ifdef USE_CUDA
#include <torch/library.h>
#include <cuda_runtime.h>
#endif

namespace olmo_cpp {

// Dispatch to custom CUDA kernels which now handle both FP32 and BF16
// natively (BF16 uses FP32 accumulation internally). No dtype promotion
// needed — zero allocation overhead, compatible with CUDA graphs.
//
// Each op resolves its OperatorHandle exactly once on first call (static
// local init is thread-safe since C++11). The old code called
// findSchemaOrThrow() on every invocation, which was a hash-map+string
// lookup per norm/silu/rope per layer per forward — ~100 string lookups
// per step on a 24-layer model. Now those are compile-time-constant
// loads after warmup.

namespace {

/// Our CUDA kernels currently support FP32 and BF16. FP16 falls back to
/// ATen. (BF16 is the default for H100 / A100 mixed-precision training.)
inline bool supported_dtype(torch::ScalarType t) {
  return t == torch::kFloat32 || t == torch::kBFloat16;
}

inline bool use_custom_cuda_kernels(torch::Device dev) {
#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
  if (!dev.is_cuda()) return false;
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, dev.index());
  return props.major < 12;
#else
  (void)dev;
  return false;
#endif
}

#ifdef OLMO_HAS_CUDA_KERNELS
/// Look up an op handle in the global LibTorch dispatcher. Returns
/// nullopt if the schema name is not registered (e.g. olmo_kernels.so
/// failed to load). The empty overload string "" picks the default.
inline c10::optional<c10::OperatorHandle> resolve_op(const char* name) {
  return c10::Dispatcher::singleton().findOp({name, ""});
}
#endif

}  // namespace

/// Dispatch RMSNorm to kernels/rms_norm.cu. The signature must match
/// the one in TORCH_LIBRARY(olmo_ops, ...) exactly, otherwise typed<>
/// will throw at runtime when the kernels load.
torch::Tensor CUDABackend::rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  // Function-pointer signature for the registered schema:
  //   rms_norm(Tensor x, Tensor? weight, float eps) -> Tensor
  using FnType =
      torch::Tensor(const torch::Tensor&, const c10::optional<torch::Tensor>&, double);
  // Cached on first call; thread-safe magic static.
  static const auto op = resolve_op("olmo_ops::rms_norm");
  if (op.has_value() && x.is_cuda() && supported_dtype(x.scalar_type())
      && use_custom_cuda_kernels(x.device())) {
    // Schema takes Tensor? — convert undefined Tensor() into nullopt.
    c10::optional<torch::Tensor> w = weight.defined()
        ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
    return op->typed<FnType>().call(x, w, eps);
  }
#endif
  // Either kernels not compiled in, op not registered, wrong device, or
  // unsupported dtype — fall back to the ATen reference implementation.
  return IBackend::rms_norm(x, weight, eps);
}

/// Dispatch silu(gate)*up to kernels/silu_mul.cu.
torch::Tensor CUDABackend::silu_mul(torch::Tensor gate, torch::Tensor up) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType = torch::Tensor(const torch::Tensor&, const torch::Tensor&);
  static const auto op = resolve_op("olmo_ops::silu_mul");
  if (op.has_value() && gate.is_cuda() && supported_dtype(gate.scalar_type())) {
    return op->typed<FnType>().call(gate, up);
  }
#endif
  return IBackend::silu_mul(gate, up);
}

/// Dispatch RoPE to kernels/rope.cu. IBackend's interface uses
/// (t, sin, cos), but the kernel registers itself as
/// apply_rope(t, cos, sin) — so we swap the last two before the call.
torch::Tensor CUDABackend::apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType =
      torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
  static const auto op = resolve_op("olmo_ops::apply_rope");
  if (op.has_value() && t.is_cuda() && supported_dtype(t.scalar_type())) {
    // Kernel takes (t, cos, sin) — note the argument order swap.
    return op->typed<FnType>().call(t, cos, sin);
  }
#endif
  return IBackend::apply_rope(t, sin, cos);
}

/// Dispatch fused (x+residual then RMSNorm) to kernels/rms_norm.cu's
/// residual_rms_norm_*_kernel. The kernel returns two tensors —
/// {normed_out, residual_out_for_next_block} — but IBackend's contract
/// only returns the normed result, so we drop residual_out here.
/// Callers that need both outputs would call the schema directly.
torch::Tensor CUDABackend::residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                               torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType = std::vector<torch::Tensor>(
      const torch::Tensor&, const torch::Tensor&,
      const c10::optional<torch::Tensor>&, double);
  static const auto op = resolve_op("olmo_ops::residual_rms_norm");
  if (op.has_value() && x.is_cuda() && supported_dtype(x.scalar_type())
      && use_custom_cuda_kernels(x.device())) {
    c10::optional<torch::Tensor> w = weight.defined()
        ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
    auto results = op->typed<FnType>().call(x, residual, w, eps);
    return results[0];
  }
#endif
  return IBackend::residual_rms_norm(x, residual, weight, eps);
}

/// Dispatch fused `out = residual + rms_norm(x) * weight` (item H).
/// Different op from residual_rms_norm above: norm-then-add vs add-then-norm.
/// Used by RMSNormImpl::forward_add, which is the per-block residual merge
/// in reordered-norm transformer blocks.
torch::Tensor CUDABackend::rms_norm_add(torch::Tensor x, torch::Tensor residual,
                                          torch::Tensor weight, double eps) {
#ifdef OLMO_HAS_CUDA_KERNELS
  using FnType = torch::Tensor(
      const torch::Tensor&, const torch::Tensor&,
      const c10::optional<torch::Tensor>&, double);
  static const auto op = resolve_op("olmo_ops::rms_norm_add");
  if (op.has_value() && x.is_cuda() && supported_dtype(x.scalar_type())
      && use_custom_cuda_kernels(x.device())) {
    c10::optional<torch::Tensor> w = weight.defined()
        ? c10::optional<torch::Tensor>(weight) : c10::nullopt;
    return op->typed<FnType>().call(x, residual, w, eps);
  }
#endif
  return IBackend::rms_norm_add(x, residual, weight, eps);
}

/// Public entry point: install CUDABackend as the global IBackend.
/// Called once from main.cpp when the configured device is CUDA.
void use_cuda_backend() {
  set_backend(std::make_unique<CUDABackend>());
}

}  // namespace olmo_cpp
