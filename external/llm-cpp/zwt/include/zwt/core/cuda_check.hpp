#pragma once

// Lightweight error-check macros for CUDA + cuBLAS.
//
// The runtime habit in zwt has been to discard return codes from CUDA / cuBLAS
// calls because failures are vanishingly rare and the launch overhead of a
// branch is non-trivial. That argument holds for hot kernel launches but not
// for handle setup, GEMM dispatch, or anything that runs O(once per step) —
// silent failure there means we ship corrupted weights.
//
// Use these wrappers at the boundary calls (cublasGemmEx, cudaMallocAsync,
// cudaMemcpyAsync at startup, etc). For inner-loop kernel launches keep the
// existing fire-and-forget pattern; cudaPeekAtLastError + a one-time check at
// step boundary is enough.

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace zwt::detail {

[[noreturn]] inline void throw_cuda(cudaError_t e, const char* expr,
                                    const char* file, int line) {
  std::string msg = "CUDA error: ";
  msg += cudaGetErrorString(e);
  msg += " (";
  msg += expr;
  msg += " at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  msg += ")";
  throw std::runtime_error(msg);
}

inline const char* cublas_status_string(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "LICENSE_ERROR";
    default:                             return "UNKNOWN";
  }
}

[[noreturn]] inline void throw_cublas(cublasStatus_t s, const char* expr,
                                      const char* file, int line) {
  std::string msg = "cuBLAS error: ";
  msg += cublas_status_string(s);
  msg += " (";
  msg += expr;
  msg += " at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  msg += ")";
  throw std::runtime_error(msg);
}

}  // namespace zwt::detail

#define ZWT_CUDA(call)                                                        \
  do {                                                                        \
    cudaError_t _zwt_e = (call);                                              \
    if (_zwt_e != cudaSuccess) {                                              \
      ::zwt::detail::throw_cuda(_zwt_e, #call, __FILE__, __LINE__);           \
    }                                                                         \
  } while (0)

#define ZWT_CUBLAS(call)                                                      \
  do {                                                                        \
    cublasStatus_t _zwt_s = (call);                                           \
    if (_zwt_s != CUBLAS_STATUS_SUCCESS) {                                    \
      ::zwt::detail::throw_cublas(_zwt_s, #call, __FILE__, __LINE__);         \
    }                                                                         \
  } while (0)

// Kernel-launch error check. Cheap (no sync). Catches bad grid/block
// configurations, unaligned shared-memory requests, etc. Use immediately
// after a <<<>>> launch. Does NOT catch in-kernel asserts — those need a
// device sync, which we deliberately avoid in the hot loop.
#define ZWT_KERNEL_OK()                                                       \
  do {                                                                        \
    cudaError_t _zwt_e = cudaPeekAtLastError();                               \
    if (_zwt_e != cudaSuccess) {                                              \
      ::zwt::detail::throw_cuda(_zwt_e, "kernel launch", __FILE__, __LINE__); \
    }                                                                         \
  } while (0)

#endif  // USE_CUDA
