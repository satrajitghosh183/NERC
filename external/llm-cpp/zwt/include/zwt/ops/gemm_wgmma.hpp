#pragma once

#include "zwt/core/tensor.hpp"

// Hopper WGMMA + TMA GEMM via CUTLASS 3.x.
//
// Built opt-in behind -DZWT_USE_WGMMA=ON. The implementation is a thin
// wrapper around cutlass::gemm::device::GemmUniversalAdapter driven by
// the sm_90 CollectiveBuilder (KernelScheduleAuto, StageCountAuto,
// TmaWarpSpecializedCooperative epilogue). BF16 inputs, F32 accumulate,
// BF16 output — matches the rest of the training stack.
//
// Why a separate TU: WGMMA instructions require the "sm_90a"
// architectural suffix. A unit compiled for plain sm_90 cannot emit
// them. The CMake target zwt_wgmma is built with
// -gencode=arch=compute_90a,code=sm_90a in isolation so the rest of
// the library stays portable across sm_80..sm_90.
//
// Supported layouts (map to the three cases Linear actually issues):
//   * transa=false, transb=false : NN (grad_X = grad_Y @ W)
//   * transa=false, transb=true  : NT (Y = X @ W^T — forward projection)
//   * transa=true,  transb=false : TN (grad_W = grad_Y^T @ X)
// The TT combination is not wired (no call site needs it); calling it
// throws.
//
// Shape constraints: M, N, K must be divisible by 8 (128-bit alignment
// of BF16 pointers). Every Transformer shape we train satisfies this.
//
// When ZWT_USE_WGMMA is off, wgmma_available() returns false and
// gemm_wgmma() throws.  Callers that use ops::gemm() do not need to
// #ifdef — the dispatch is runtime.

namespace zwt::ops {

// True when (a) the TU was built with ZWT_USE_WGMMA, and
//          (b) the current default CUDA device is sm_90.
// Cached after the first call.
bool wgmma_available();

// Row-major GEMM with the same semantics as ops::gemm(): C = alpha * A[.T?]
// @ B[.T?] + beta * C.  Tensors must live on a CUDA device with BF16 dtype.
void gemm_wgmma(const Tensor& a, bool transa,
                const Tensor& b, bool transb,
                Tensor& c,
                float alpha = 1.0f,
                float beta  = 0.0f);

}  // namespace zwt::ops
