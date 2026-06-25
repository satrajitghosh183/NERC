#include "zwt/ops/gemm.hpp"
#include "zwt/ops/gemm_wgmma.hpp"
#include "zwt/core/determinism.hpp"
#include "zwt/core/stream.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#ifdef USE_CUDA
#include "zwt/core/cuda_check.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace zwt::ops {

namespace {

#ifdef USE_CUDA
cublasHandle_t& cublas_handle() {
  static cublasHandle_t h = nullptr;
  static std::once_flag flag;
  std::call_once(flag, []() {
    ZWT_CUBLAS(cublasCreate(&h));
    // No math-mode override: BF16 gemms already run on tensor cores regardless,
    // and forcing TF32 would only affect F32 gemms we don't issue.
  });
  return h;
}

cudaDataType cuda_dt(DType t) {
  switch (t) {
    case DType::F32:  return CUDA_R_32F;
    case DType::F16:  return CUDA_R_16F;
    case DType::BF16: return CUDA_R_16BF;
    default:
      throw std::runtime_error("zwt::gemm: unsupported dtype");
  }
}

// ZWT_DISABLE_WGMMA is consulted on every gemm call. getenv() takes a process-
// scope lock and walks the env block — measurable on small projection gemms
// at high TPS. Cache it. The bench tool flips the var between phases and
// calls reset_wgmma_disable_cache() to invalidate. We use a relaxed atomic
// because the value only ever moves uninit → cached and a torn read can't
// produce an invalid bool.
std::atomic<int> g_wgmma_disabled_cache{-1};  // -1 unknown, 0 enabled, 1 disabled

bool wgmma_disabled_cached() {
  int v = g_wgmma_disabled_cache.load(std::memory_order_relaxed);
  if (v < 0) {
    v = (std::getenv("ZWT_DISABLE_WGMMA") != nullptr) ? 1 : 0;
    g_wgmma_disabled_cache.store(v, std::memory_order_relaxed);
  }
  return v != 0;
}

// cublasSetStream is a no-op when the stream hasn't changed, but the call
// itself takes a handle-mutex. We issue thousands of gemms per step on the
// same compute stream — skip the call when the stream matches the last one
// we set on this thread.
inline void set_stream_if_changed(cublasHandle_t h, cudaStream_t s) {
  thread_local cudaStream_t last = nullptr;
  if (s != last) {
    ZWT_CUBLAS(cublasSetStream(h, s));
    last = s;
  }
}
#endif

// CPU reference impl — f32 only. For correctness tests and tiny CPU work.
void gemm_cpu_f32(const float* A, bool ta, const float* B, bool tb,
                  float* C, int M, int N, int K, float alpha, float beta) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        float av = ta ? A[k * M + i] : A[i * K + k];
        float bv = tb ? B[j * K + k] : B[k * N + j];
        acc += av * bv;
      }
      C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
  }
}

}  // namespace

void gemm_init() {
#ifdef USE_CUDA
  (void)cublas_handle();
#endif
}

void reset_wgmma_disable_cache() {
#ifdef USE_CUDA
  g_wgmma_disabled_cache.store(-1, std::memory_order_relaxed);
#endif
}

void gemm(const Tensor& a, bool transa,
          const Tensor& b, bool transb,
          Tensor& c,
          float alpha, float beta) {
  // Derive M, N, K from shapes. A is [Ar, Ac]; B is [Br, Bc]; C is [M, N].
  const int Ar = static_cast<int>(a.dim(0));
  const int Ac = static_cast<int>(a.dim(1));
  const int Br = static_cast<int>(b.dim(0));
  const int Bc = static_cast<int>(b.dim(1));
  const int M = transa ? Ac : Ar;
  const int K = transa ? Ar : Ac;
  const int Kb = transb ? Bc : Br;
  const int N  = transb ? Br : Bc;
  if (K != Kb) throw std::runtime_error("zwt::gemm: inner dim mismatch");
  if (c.dim(0) != M || c.dim(1) != N) {
    throw std::runtime_error("zwt::gemm: C shape mismatch");
  }

  if (a.device().is_cuda()) {
#ifdef USE_CUDA
    // Hopper WGMMA path: BF16 tensor-core kernel via CUTLASS 3.x. Opt-in
    // at build time (ZWT_USE_WGMMA); runtime-checked for sm_90. Shape
    // constraint is M,N,K % 8 == 0, which every transformer projection
    // satisfies. Env var ZWT_DISABLE_WGMMA=1 forces the cuBLAS fallback —
    // cached at first call (see wgmma_disabled_cached).
    if (!wgmma_disabled_cached() &&
        wgmma_available() &&
        a.dtype() == DType::BF16 && b.dtype() == DType::BF16 &&
        c.dtype() == DType::BF16 &&
        (M & 7) == 0 && (N & 7) == 0 && (K & 7) == 0 &&
        !(transa && transb)) {
      gemm_wgmma(a, transa, b, transb, c, alpha, beta);
      return;
    }

    cublasHandle_t h = cublas_handle();
    set_stream_if_changed(h, reinterpret_cast<cudaStream_t>(
                                 compute_stream(a.device()).handle));

    // cuBLAS is column-major; we have row-major matrices. Use the identity
    //   C_row = A_row @ B_row  ==  C_col^T = (A_row @ B_row)^T = B_col @ A_col
    // i.e. swap operand order and ship row-major matrices to cuBLAS as-is
    // with dimensions transposed accordingly.
    int lda = transa ? M : K;
    int ldb = transb ? K : N;
    int ldc = N;

    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    cudaDataType dtA = cuda_dt(a.dtype());
    cudaDataType dtB = cuda_dt(b.dtype());
    cudaDataType dtC = cuda_dt(c.dtype());

    // Swap A,B to convert row-major to column-major.
    // In determinism mode, pick a fixed, workspace-backed algorithm rather
    // than cuBLAS's heuristic default. CUBLAS_GEMM_DEFAULT still picks from
    // the deterministic-algo set when CUBLAS_WORKSPACE_CONFIG=:4096:8 is set.
    cublasGemmAlgo_t algo = is_deterministic()
        ? CUBLAS_GEMM_DEFAULT
        : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    ZWT_CUBLAS(cublasGemmEx(
        h,
        opB, opA,
        N, M, K,
        &alpha,
        b.data(), dtB, ldb,
        a.data(), dtA, lda,
        &beta,
        c.data(), dtC, ldc,
        CUBLAS_COMPUTE_32F,
        algo));
    return;
#else
    throw std::runtime_error("zwt::gemm: cuda path requested on CPU-only build");
#endif
  }

  // CPU path — f32 only.
  if (a.dtype() != DType::F32 || b.dtype() != DType::F32 || c.dtype() != DType::F32) {
    throw std::runtime_error("zwt::gemm: CPU path supports only F32");
  }
  gemm_cpu_f32(a.as<float>(), transa, b.as<float>(), transb,
               c.as<float>(), M, N, K, alpha, beta);
}

void gemm_batched(const Tensor& a, bool transa,
                  const Tensor& b, bool transb,
                  Tensor& c,
                  float alpha, float beta) {
  if (a.rank() != 3 || b.rank() != 3 || c.rank() != 3) {
    throw std::runtime_error("zwt::gemm_batched: expected rank-3 tensors");
  }
  if (!a.device().is_cuda()) {
    throw std::runtime_error("zwt::gemm_batched: CPU path unimplemented");
  }
#ifdef USE_CUDA
  const int64_t B_ = a.dim(0);
  const int Ar = static_cast<int>(a.dim(1));
  const int Ac = static_cast<int>(a.dim(2));
  const int Br = static_cast<int>(b.dim(1));
  const int Bc = static_cast<int>(b.dim(2));
  const int M = transa ? Ac : Ar;
  const int K = transa ? Ar : Ac;
  const int Kb = transb ? Bc : Br;
  const int N  = transb ? Br : Bc;
  if (K != Kb) throw std::runtime_error("zwt::gemm_batched: inner dim mismatch");

  cublasHandle_t h = cublas_handle();
  set_stream_if_changed(h, reinterpret_cast<cudaStream_t>(
                               compute_stream(a.device()).handle));
  int lda = transa ? M : K;
  int ldb = transb ? K : N;
  int ldc = N;
  cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cudaDataType dtA = cuda_dt(a.dtype());
  cudaDataType dtB = cuda_dt(b.dtype());
  cudaDataType dtC = cuda_dt(c.dtype());

  long long strideA = (long long)Ar * Ac;
  long long strideB = (long long)Br * Bc;
  long long strideC = (long long)M * N;

  cublasGemmAlgo_t algo = is_deterministic()
      ? CUBLAS_GEMM_DEFAULT
      : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  ZWT_CUBLAS(cublasGemmStridedBatchedEx(
      h, opB, opA,
      N, M, K,
      &alpha,
      b.data(), dtB, ldb, strideB,
      a.data(), dtA, lda, strideA,
      &beta,
      c.data(), dtC, ldc, strideC,
      static_cast<int>(B_),
      CUBLAS_COMPUTE_32F,
      algo));
#else
  (void)transa; (void)transb; (void)c; (void)alpha; (void)beta;
#endif
}

}  // namespace zwt::ops
