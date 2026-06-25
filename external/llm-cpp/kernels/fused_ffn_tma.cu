/**
 * kernels/fused_ffn_tma.cu
 *
 * TMA (Tensor Memory Accelerator) variant of fused_ffn_wmma — item W
 * wiring. On sm_90+ (Hopper / Blackwell sm_120), the x input tile is
 * async-loaded into shared memory via cp.async.bulk.tensor with an
 * mbarrier-based completion signal; WMMA computes on the loaded tile.
 * The async load overlaps compute and hides HBM latency.
 *
 * Older arches: the host dispatcher detects compute capability < 9 and
 * falls back to fused_ffn_wmma_cuda; the kernel body's TMA path is
 * preprocessed out via __CUDA_ARCH__ guards. The kernel symbol is
 * always defined so the launch site links cleanly across all targets.
 *
 * Reference: NVIDIA PTX ISA 8.0 §9.7.8.24 (cp.async.bulk.tensor) and
 * §9.7.13.16 (mbarrier).
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#include "olmo_cpp/backend/fused_ffn.hpp"
#include "olmo_cpp/backend/tma_loads.hpp"

namespace olmo_cpp {

namespace {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using namespace nvcuda;
#endif

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float silu_f(float v) {
  return v / (1.0f + __expf(-v));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// PTX wrappers around TMA + mbarrier primitives. Inline so we don't
// pay a function-call cost on the hot path. Only compiled for sm_90+;
// older arches preprocessor-skip and use the WMMA-only path below.
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_smem, unsigned count) {
  unsigned mbar_ptr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_smem));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
               :: "r"(mbar_ptr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar_smem,
                                                            unsigned bytes) {
  unsigned mbar_ptr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_smem));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
               :: "r"(mbar_ptr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_smem, unsigned phase) {
  unsigned mbar_ptr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_smem));
  asm volatile(
      "{\n"
      ".reg .pred  p;\n"
      "L1: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra L1;\n"
      "}\n"
      :: "r"(mbar_ptr), "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    __nv_bfloat16* smem_dst,
    const void* tensor_map,
    int coord0, int coord1,
    uint64_t* mbar_smem) {
  unsigned dst_ptr  = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  unsigned mbar_ptr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_smem));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(dst_ptr),
         "l"(reinterpret_cast<uint64_t>(tensor_map)),
         "r"(coord0), "r"(coord1),
         "r"(mbar_ptr));
}

#endif  // sm_90+ PTX wrappers

// Kernel body. One block processes a 16-row tile of the (N, d) input.
// On sm_90+, x is loaded via TMA. On older arches the body is a stub —
// host dispatch will never reach it because the runtime check routes
// older devices to fused_ffn_wmma_cuda.
// The TMA descriptor is passed by value as a grid constant. The parameter
// type must be IDENTICAL in the host (launch) and device passes — switching
// it on __CUDA_ARCH__ makes the host see `void*` while the launch passes a
// CUtensorMap, which doesn't convert. Keep it CUtensorMap on every pass;
// the TMA *instructions* are still guarded by __CUDA_ARCH__ >= 900 below.
__global__ void fused_ffn_tma_kernel(
    const __grid_constant__ CUtensorMap x_desc,
    const __nv_bfloat16* __restrict__ w_gate_up,
    const __nv_bfloat16* __restrict__ w_down,
    __nv_bfloat16* __restrict__ y_out,
    __nv_bfloat16* __restrict__ gate_up_out,    // [N, 2H] or nullptr (A1)
    int N, int d, int H) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const int row_tile = blockIdx.x;
  const int row_base = row_tile * kWmmaM;
  if (row_base >= N) return;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane    = threadIdx.x & 31;

  // Shared-memory carve-up. mbarrier is 8-byte aligned at the tail.
  extern __shared__ __nv_bfloat16 smem_pool[];
  __nv_bfloat16* sh_x       = smem_pool;
  __nv_bfloat16* sh_gate_up = sh_x + 16 * d;
  __nv_bfloat16* sh_act     = sh_gate_up + 16 * (2 * H);
  // Per-warp fp32 scratch for wmma::store_matrix_sync. Placed before
  // the mbarrier so we keep mbar aligned at the end.
  float* sh_wmma = reinterpret_cast<float*>(sh_act + 16 * H);
  float* my_wmma = sh_wmma + warp_id * (kWmmaM * kWmmaN);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(
      (reinterpret_cast<uintptr_t>(sh_wmma + kWarpsPerBlock * kWmmaM * kWmmaN) + 7)
      & ~static_cast<uintptr_t>(7));

  // Phase 0 — TMA load of x[row_base:row_base+16, 0:d].
  if (threadIdx.x == 0) {
    mbarrier_init(mbar, 1);
    const unsigned bytes = 16u * static_cast<unsigned>(d) * sizeof(__nv_bfloat16);
    mbarrier_arrive_expect_tx(mbar, bytes);
    cp_async_bulk_tensor_2d(sh_x, &x_desc, /*col0=*/0, /*row1=*/row_base, mbar);
  }
  __syncthreads();
  mbarrier_wait(mbar, /*phase=*/0);
  // TMA writes sh_x via the async proxy; WMMA load_matrix_sync reads via
  // the generic proxy.  On Hopper (sm_90+) these are separate memory
  // proxies and the mbarrier completion alone does NOT make async writes
  // visible to generic reads.  A fence.proxy.async is required to bridge
  // the two proxies before any subsequent shared-memory access.
  // Without this, load_matrix_sync reads stale/uninitialised sh_x data,
  // producing large errors (~128) in the gate_up matmul.
  asm volatile("fence.proxy.async;" ::: "memory");

  // Phase 1 — WMMA matmul gate_up = sh_x @ w_gate_up.T.
  const int col_tiles_2H = (2 * H) / kWmmaN;
  for (int ct = warp_id; ct < col_tiles_2H; ct += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kt = 0; kt < d / kWmmaK; ++kt) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, sh_x + kt * kWmmaK, d);
      wmma::load_matrix_sync(b, w_gate_up + (int64_t)(ct * kWmmaN) * d + kt * kWmmaK, d);
      wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(my_wmma, c, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    // B2 — paired conversion + 4-byte store.
    constexpr int kPairs = kWmmaM * kWmmaN / 2;
    for (int p = lane; p < kPairs; p += 32) {
      const int i  = p / (kWmmaN / 2);
      const int c2 = (p % (kWmmaN / 2)) * 2;
      const float2 f = make_float2(my_wmma[i * kWmmaN + c2],
                                     my_wmma[i * kWmmaN + c2 + 1]);
      const __nv_bfloat162 b = __float22bfloat162_rn(f);
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
          &sh_gate_up[i * (2 * H) + ct * kWmmaN + c2]);
      *dst = b;
    }
  }
  __syncthreads();

  // A1 — optionally publish gate_up to HBM for the backward path. The
  // training entry point allocates this buffer; the inference entry
  // points pass nullptr.
  if (gate_up_out != nullptr) {
    const int total_gu = 16 * 2 * H;
    for (int idx = threadIdx.x; idx < total_gu; idx += blockDim.x) {
      const int i = idx / (2 * H);
      const int j = idx % (2 * H);
      const int gi = row_base + i;
      if (gi < N) {
        gate_up_out[(int64_t)gi * (2 * H) + j] = sh_gate_up[i * (2 * H) + j];
      }
    }
    __syncthreads();
  }

  // Phase 2 — silu_mul to act.
  const int total_act = 16 * H;
  for (int idx = threadIdx.x; idx < total_act; idx += blockDim.x) {
    int i = idx / H;
    int h = idx % H;
    float g = __bfloat162float(sh_gate_up[i * (2 * H) + h]);
    float u = __bfloat162float(sh_gate_up[i * (2 * H) + h + H]);
    sh_act[i * H + h] = __float2bfloat16(silu_f(g) * u);
  }
  __syncthreads();

  // Phase 3 — down matmul. y_out[row_base:row_base+16, 0:d] = sh_act @ w_down.T.
  const int col_tiles_d = d / kWmmaN;
  for (int ct = warp_id; ct < col_tiles_d; ct += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kt = 0; kt < H / kWmmaK; ++kt) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, sh_act + kt * kWmmaK, H);
      wmma::load_matrix_sync(b, w_down + (int64_t)(ct * kWmmaN) * H + kt * kWmmaK, H);
      wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(my_wmma, c, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    // B2 — paired conversion + 4-byte store.
    constexpr int kPairs = kWmmaM * kWmmaN / 2;
    for (int p = lane; p < kPairs; p += 32) {
      const int i  = p / (kWmmaN / 2);
      const int c2 = (p % (kWmmaN / 2)) * 2;
      const int gi = row_base + i;
      if (gi >= N) continue;
      const float2 f = make_float2(my_wmma[i * kWmmaN + c2],
                                     my_wmma[i * kWmmaN + c2 + 1]);
      const __nv_bfloat162 b = __float22bfloat162_rn(f);
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
          &y_out[(int64_t)gi * d + ct * kWmmaN + c2]);
      *dst = b;
    }
  }
#else
  // Pre-sm_90 builds: kernel is unreachable at runtime (host routes to
  // fused_ffn_wmma_cuda), but we keep the symbol so the launch site
  // links. No-op body keeps device-side preprocessing legal — the TMA
  // PTX is preprocessed out above.
  (void)x_desc;
  (void)w_gate_up; (void)w_down; (void)y_out; (void)gate_up_out;
  (void)N; (void)d; (void)H;
#endif
}

}  // namespace

torch::Tensor fused_ffn_tma_cuda(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fused_ffn_tma_cuda: bf16 CUDA inputs required");

  // Runtime gate: TMA needs sm_90+. Older devices route through the
  // WMMA-only fused kernel.
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, x.device().index());
  if (props.major < 9) {
    return fused_ffn_wmma_cuda(x, w_gate_up, w_down);
  }
  if (props.major >= 12) {
    return fused_ffn_wmma_cuda(x, w_gate_up, w_down);
  }

  c10::cuda::CUDAGuard guard(x.device());
  auto x_c  = x.contiguous();
  auto wg   = w_gate_up.contiguous();
  auto wd   = w_down.contiguous();
  const int64_t B = x_c.size(0);
  const int64_t S = x_c.size(1);
  const int64_t d = x_c.size(2);
  const int64_t H = wg.size(0) / 2;
  TORCH_CHECK(d % 16 == 0 && H % 16 == 0,
              "fused_ffn_tma_cuda: shapes must be multiples of 16");
  const int N = static_cast<int>(B * S);
  auto y = torch::empty_like(x_c);

  // Build TMA descriptor for x viewed as [N, d].
  auto x_flat = x_c.view({N, d});
  alignas(64) unsigned char x_desc_storage[128];
  make_tma_descriptor(x_flat, /*tile_rows=*/16, /*tile_cols=*/d, x_desc_storage);

  const size_t shmem =
        16 * d * sizeof(__nv_bfloat16)              // sh_x
      + 16 * (2 * H) * sizeof(__nv_bfloat16)         // sh_gate_up
      + 16 * H * sizeof(__nv_bfloat16)               // sh_act
      + kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float)  // per-warp WMMA scratch
      + 16;                                          // mbarrier slack

  // Opt into >48 KB dynamic shared memory. Host dispatcher guards that
  // shmem ≤ device opt-in max before routing here.
  if (shmem > 48 * 1024) {
    cudaFuncSetAttribute(fused_ffn_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
  }
  const int grid = (N + 16 - 1) / 16;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_ffn_tma_kernel<<<grid, kThreadsPerBlock, shmem, stream>>>(
      *reinterpret_cast<const CUtensorMap*>(x_desc_storage),
      reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      /*gate_up_out=*/nullptr,
      N, static_cast<int>(d), static_cast<int>(H));
  return y;
}

// A1 — training entry point. Mirrors fused_ffn_wmma_train_cuda but on
// the TMA-async-load path. Sm_90+ keeps TMA; older archs fall through
// to the WMMA train variant via the same runtime check.
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_tma_train_cuda(torch::Tensor x,
                           torch::Tensor w_gate_up,
                           torch::Tensor w_down) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fused_ffn_tma_train_cuda: bf16 CUDA inputs required");
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, x.device().index());
  if (props.major < 9) {
    return fused_ffn_wmma_train_cuda(x, w_gate_up, w_down);
  }
  // TMA path is validated on Hopper (sm_90). Blackwell (sm_120) uses WMMA
  // until the TMA descriptor/coords are re-verified on sm_120 hardware.
  if (props.major >= 12) {
    return fused_ffn_wmma_train_cuda(x, w_gate_up, w_down);
  }

  c10::cuda::CUDAGuard guard(x.device());
  auto x_c  = x.contiguous();
  auto wg   = w_gate_up.contiguous();
  auto wd   = w_down.contiguous();
  const int64_t B = x_c.size(0);
  const int64_t S = x_c.size(1);
  const int64_t d = x_c.size(2);
  const int64_t H = wg.size(0) / 2;
  TORCH_CHECK(d % 16 == 0 && H % 16 == 0,
              "fused_ffn_tma_train_cuda: shapes must be multiples of 16");
  const int N = static_cast<int>(B * S);
  auto y = torch::empty_like(x_c);
  auto gate_up = torch::empty({B, S, 2 * H}, x_c.options());

  auto x_flat = x_c.view({N, d});
  alignas(64) unsigned char x_desc_storage[128];
  make_tma_descriptor(x_flat, /*tile_rows=*/16, /*tile_cols=*/d, x_desc_storage);

  const size_t shmem =
        16 * d * sizeof(__nv_bfloat16)
      + 16 * (2 * H) * sizeof(__nv_bfloat16)
      + 16 * H * sizeof(__nv_bfloat16)
      + kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float)  // per-warp WMMA scratch
      + 16;                                          // mbarrier slack

  if (shmem > 48 * 1024) {
    cudaFuncSetAttribute(fused_ffn_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
  }

  const int grid = (N + 16 - 1) / 16;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_ffn_tma_kernel<<<grid, kThreadsPerBlock, shmem, stream>>>(
      *reinterpret_cast<const CUtensorMap*>(x_desc_storage),
      reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()),
      N, static_cast<int>(d), static_cast<int>(H));
  return {y, gate_up};
}

}  // namespace olmo_cpp
