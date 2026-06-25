// WGMMA + TMA GEMM via CUTLASS 3.x — sm_90a only.
//
// This translation unit is compiled only when -DZWT_USE_WGMMA=ON is set
// at cmake configure time (see CMakeLists.txt: target `zwt_wgmma` with
// -gencode=arch=compute_90a,code=sm_90a).  Without the flag the stub in
// gemm_wgmma_stub.cpp provides the API symbols and wgmma_available()
// reports false.

#ifdef ZWT_USE_WGMMA

#include "zwt/ops/gemm_wgmma.hpp"
#include "zwt/core/stream.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUTLASS 3.x collective-builder stack.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

#include <cstdio>
#include <stdexcept>

namespace zwt::ops {

namespace {

// ---- element + tile config ------------------------------------------------
using ElementA           = cutlass::bfloat16_t;
using ElementB           = cutlass::bfloat16_t;
using ElementC           = cutlass::bfloat16_t;
using ElementAccumulator = float;

// 128-bit global-load alignment: 16 bytes / sizeof(bf16) = 8 elements.
constexpr int kAlignA = 8;
constexpr int kAlignB = 8;
constexpr int kAlignC = 8;

// Tile & cluster shapes. 128x128x64 is the go-to cooperative tile for
// BF16 WGMMA on H100; cluster (1,1,1) works everywhere without
// thread-block-cluster constraints. We can tune later.
using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

// ---- one CUTLASS GemmUniversalAdapter per (LayoutA, LayoutB) pair --------
template <class LayoutA, class LayoutB>
struct Builder {
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          ElementA, LayoutA, kAlignA,
          ElementB, LayoutB, kAlignB,
          ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAuto,
          cutlass::gemm::collective::KernelScheduleAuto
      >::CollectiveOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          ElementC, cutlass::layout::RowMajor, kAlignC,
          ElementC, cutlass::layout::RowMajor, kAlignC,
          cutlass::epilogue::TmaWarpSpecializedCooperative
      >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// CUTLASS 3.x GEMM convention: C[m,n] = sum_k A[m,k] * B[n,k]
// i.e. the kernel iterates B as N×K (B^T in textbook notation), with the
// stride built from shape (N, K, 1). The LayoutB selection therefore
// describes the *N×K* indexing, not the user's logical (K,N) view:
//
//   LayoutB = RowMajor   → kernel reads base + n*K + k → matches a
//                          buffer stored as (N, K) row-major
//   LayoutB = ColumnMajor → kernel reads base + n + k*N → matches a
//                          buffer stored as (K, N) row-major
//
// Our `transb=false` (NN) means the user passed B as (K, N) row-major
// → ColumnMajor. Our `transb=true` (NT) means the user passed B as
// (N, K) row-major → RowMajor.
//
// LayoutA follows the textbook convention: RowMajor for transa=false
// (user buffer is M×K), ColumnMajor for transa=true (user buffer is
// (K, M) row-major == M×K col-major).

// NN: C = A @ B.  A=(M,K) row-major, B=(K,N) row-major.
using GemmNN = Builder<cutlass::layout::RowMajor,    cutlass::layout::ColumnMajor>::Gemm;
// NT: C = A @ B^T. A=(M,K) row-major, B=(N,K) row-major.
using GemmNT = Builder<cutlass::layout::RowMajor,    cutlass::layout::RowMajor   >::Gemm;
// TN: C = A^T @ B. A=(K,M) row-major (col-major M×K), B=(K,N) row-major.
using GemmTN = Builder<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>::Gemm;

bool check_sm90_once() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major == 9 && prop.minor == 0;
}

template <class Gemm>
void run(int M, int N, int K,
         const void* A, const void* B, void* C,
         float alpha, float beta, cudaStream_t stream) {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<const ElementA*>(A), stride_A,
          reinterpret_cast<const ElementB*>(B), stride_B
      },
      {
          { ElementAccumulator(alpha), ElementAccumulator(beta) },
          reinterpret_cast<const ElementC*>(C), stride_C,
          reinterpret_cast<      ElementC*>(C), stride_D
      }
  };

  Gemm op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("gemm_wgmma: can_implement failed: ")
                             + cutlass::cutlassGetStatusString(status));
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
      throw std::runtime_error("gemm_wgmma: cudaMalloc for workspace failed");
    }
  }

  status = op.initialize(args, workspace, stream);
  if (status == cutlass::Status::kSuccess) {
    status = op.run(stream);
  }

  if (workspace) cudaFree(workspace);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("gemm_wgmma: run failed: ")
                             + cutlass::cutlassGetStatusString(status));
  }
}

}  // namespace

bool wgmma_available() {
  static int cached = -1;
  if (cached < 0) cached = check_sm90_once() ? 1 : 0;
  return cached == 1;
}

void gemm_wgmma(const Tensor& a, bool transa,
                const Tensor& b, bool transb,
                Tensor& c, float alpha, float beta) {
  if (a.dtype() != DType::BF16 || b.dtype() != DType::BF16 ||
      c.dtype() != DType::BF16) {
    throw std::runtime_error("gemm_wgmma: bf16 only");
  }
  if (!a.device().is_cuda() || !b.device().is_cuda() || !c.device().is_cuda()) {
    throw std::runtime_error("gemm_wgmma: tensors must be on CUDA");
  }
  if (!wgmma_available()) {
    throw std::runtime_error("gemm_wgmma: default device is not sm_90");
  }

  const int Ar = static_cast<int>(a.dim(0));
  const int Ac = static_cast<int>(a.dim(1));
  const int Br = static_cast<int>(b.dim(0));
  const int Bc = static_cast<int>(b.dim(1));
  const int M  = transa ? Ac : Ar;
  const int K  = transa ? Ar : Ac;
  const int Kb = transb ? Bc : Br;
  const int N  = transb ? Br : Bc;
  if (K != Kb) throw std::runtime_error("gemm_wgmma: inner dim mismatch");
  if (c.dim(0) != M || c.dim(1) != N) {
    throw std::runtime_error("gemm_wgmma: C shape mismatch");
  }
  if ((M & 7) || (N & 7) || (K & 7)) {
    throw std::runtime_error("gemm_wgmma: M,N,K must be multiples of 8");
  }

  auto stream = reinterpret_cast<cudaStream_t>(
      compute_stream(a.device()).handle);

  if (!transa && !transb) {
    run<GemmNN>(M, N, K, a.data(), b.data(), c.data(), alpha, beta, stream);
  } else if (!transa && transb) {
    run<GemmNT>(M, N, K, a.data(), b.data(), c.data(), alpha, beta, stream);
  } else if (transa && !transb) {
    run<GemmTN>(M, N, K, a.data(), b.data(), c.data(), alpha, beta, stream);
  } else {
    throw std::runtime_error(
        "gemm_wgmma: TT layout not wired (no Linear call site needs it)");
  }
}

}  // namespace zwt::ops

#endif  // ZWT_USE_WGMMA
