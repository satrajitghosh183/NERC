/**
 * src/backend/fused_ffn.cpp
 *
 * CPU reference + host dispatcher for fused SwiGLU FFN (item I).
 */

#include "olmo_cpp/backend/fused_ffn.hpp"
#include "olmo_cpp/backend/backend.hpp"
#include "olmo_cpp/backend/cublas_direct.hpp"

#include <torch/torch.h>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  include <cuda_runtime.h>
#endif

namespace olmo_cpp {

#ifdef OLMO_HAS_CUDA_KERNELS
namespace {
// The WMMA/TMA FFN kernels keep the whole [16, 2H] gate_up tile, the
// [16, H] act tile, (TMA) the [16, d] x tile, and a per-warp fp32
// scratch in shared memory. For large H this exceeds even Blackwell's
// ~227 KB opt-in cap — at d=1024/H=2816 it's ~300 KB. When it won't
// fit we must NOT launch the kernel (cudaFuncSetAttribute would reject
// it and the launch would fail); fall back to the cuBLAS chain.
bool ffn_fused_shmem_fits(int64_t d, int64_t H, c10::Device dev) {
  const size_t bf16 = 2;
  const size_t need =
        (size_t)16 * d * bf16            // sh_x (TMA variant; upper bound)
      + (size_t)16 * (2 * H) * bf16      // sh_gate_up
      + (size_t)16 * H * bf16            // sh_act
      + (size_t)4 * 16 * 16 * sizeof(float)  // per-warp WMMA scratch
      + 64;                              // mbarrier + alignment slack
  int max_optin = 0;
  cudaDeviceGetAttribute(&max_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev.index());
  return max_optin > 0 && need <= (size_t)max_optin;
}

// cuBLAS-direct FFN: y = w_down · (silu(gate) ⊙ up). Returns (y, gate_up)
// when want_gate_up so the training path can save it for backward.
std::pair<torch::Tensor, torch::Tensor>
ffn_cublas_chain(torch::Tensor x, torch::Tensor w_gate_up, torch::Tensor w_down,
                 bool want_gate_up) {
  auto gate_up = fast_linear(x, w_gate_up, torch::Tensor());   // [..., 2H]
  const int64_t H = gate_up.size(-1) / 2;
  auto gate = gate_up.narrow(-1, 0, H);
  auto up   = gate_up.narrow(-1, H, H);
  auto act  = get_backend().silu_mul(gate, up);                // [..., H]
  auto y    = fast_linear(act, w_down, torch::Tensor());       // [..., d]
  return {y, want_gate_up ? gate_up : torch::Tensor()};
}
}  // namespace
#endif  // OLMO_HAS_CUDA_KERNELS

torch::Tensor fused_ffn_cpu(torch::Tensor x,
                              torch::Tensor w_gate_up,
                              torch::Tensor w_down) {
  TORCH_CHECK(x.dim() == 3 && w_gate_up.dim() == 2 && w_down.dim() == 2,
              "fused_ffn_cpu: shapes wrong");
  // Reference path mirrors FeedForwardImpl::forward (fused-gate-up branch):
  //   gate_up = linear(x, w_gate_up)
  //   gate, up = split(gate_up, H, dim=-1)
  //   act = silu(gate) * up
  //   y = linear(act, w_down)
  auto gate_up = torch::nn::functional::linear(x, w_gate_up);   // [B,S,2H]
  const int64_t H = gate_up.size(-1) / 2;
  auto gate = gate_up.narrow(-1, 0, H);
  auto up   = gate_up.narrow(-1, H, H);
  auto act  = get_backend().silu_mul(gate, up);                 // [B,S,H]
  return torch::nn::functional::linear(act, w_down);             // [B,S,d]
}

// A1 — training-side dispatcher. Returns (y, gate_up). For CPU and
// non-aligned shapes, gate_up is computed via fast_linear so the
// numerics match the production forward path; CUDA-aligned shapes
// route through the WMMA/TMA train variants that produce gate_up
// as a kernel side-output (one extra HBM write per call).
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_train(torch::Tensor x,
                 torch::Tensor w_gate_up,
                 torch::Tensor w_down) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda() && x.scalar_type() == torch::kBFloat16) {
    const int64_t d = x.size(-1);
    const int64_t H = w_gate_up.size(0) / 2;
    // WMMA/TMA fused kernels are not yet parity-validated on Blackwell
    // (sm_120). Route through cuBLAS until the tensor-core path is fixed.
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, x.device().index());
    if (props.major >= 12) {
      return ffn_cublas_chain(x, w_gate_up, w_down, /*want_gate_up=*/true);
    }
    if (d % 16 == 0 && H % 16 == 0 && ffn_fused_shmem_fits(d, H, x.device())) {
      return fused_ffn_tma_train_cuda(x, w_gate_up, w_down);
    }
    // Aligned but the fused kernel's shmem won't fit (large H): use the
    // cuBLAS chain, which also materializes gate_up for the backward.
    if (d % 16 == 0 && H % 16 == 0) {
      return ffn_cublas_chain(x, w_gate_up, w_down, /*want_gate_up=*/true);
    }
  }
#endif
  // CPU / non-aligned fallback: produce gate_up explicitly. Slightly
  // slower than the kernel-side write but exercised only on edge cases.
  auto gate_up = torch::nn::functional::linear(x, w_gate_up);
  auto y = fused_ffn(x, w_gate_up, w_down);
  return {y, gate_up};
}

torch::Tensor fused_ffn(torch::Tensor x,
                         torch::Tensor w_gate_up,
                         torch::Tensor w_down) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda() && x.scalar_type() == torch::kBFloat16) {
    const int64_t d = x.size(-1);
    const int64_t H = w_gate_up.size(0) / 2;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, x.device().index());
    if (props.major >= 12) {
      return ffn_cublas_chain(x, w_gate_up, w_down, /*want_gate_up=*/false).first;
    }
    if (d % 16 == 0 && H % 16 == 0 && ffn_fused_shmem_fits(d, H, x.device())) {
      // Tensor-core path — TMA variant on sm_90+, plain WMMA otherwise.
      // fused_ffn_tma_cuda itself runtime-checks and falls back to the
      // WMMA path on older arches, so we always dispatch here.
      return fused_ffn_tma_cuda(x, w_gate_up, w_down);
    }
    if (d % 16 == 0 && H % 16 == 0) {
      // Aligned but shmem won't fit (large H): cuBLAS chain. Still uses
      // fast_linear GEMMs + the vectorized silu_mul kernel.
      return ffn_cublas_chain(x, w_gate_up, w_down, /*want_gate_up=*/false).first;
    }
    // Fallback to FMA-loop kernel for non-aligned shapes.
    return fused_ffn_cuda(x, w_gate_up, w_down);
  }
  if (x.is_cuda()) {
    return fused_ffn_cuda(x, w_gate_up, w_down);
  }
#endif
  return fused_ffn_cpu(x, w_gate_up, w_down);
}

}  // namespace olmo_cpp
