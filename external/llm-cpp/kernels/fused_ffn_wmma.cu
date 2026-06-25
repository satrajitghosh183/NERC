/**
 * kernels/fused_ffn_wmma.cu
 *
 * Tensor-core (WMMA) FFN macro kernel — item 2 follow-on.
 *
 * Per-warp tile geometry: each warp produces a 16×16 output tile via
 * mma_sync over BF16 inputs with FP32 accumulator. The kernel is two
 * tiled matmuls fused with silu_mul between them, all intermediates
 * kept in shared memory.
 *
 * Sized for d_model ≤ 1024 and H ≤ 4096 (which covers OLMo's 125M /
 * 350M / 1B configs). Larger shapes fall back to the per-row FMA
 * kernel via the host-side dispatch.
 *
 * Speedup target on Blackwell sm_120: ~2-3× over the FMA-loop fused_ffn
 * kernel at training-shape B*S, because the inner matmul now uses
 * the 5th-gen tensor cores instead of FP32 FMA loops.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#include "olmo_cpp/backend/fused_ffn.hpp"

namespace olmo_cpp {

namespace {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
// WMMA headers omit nvcuda::wmma below sm_80. Host dispatch never routes
// here on such devices (falls back to the FMA-loop kernel).
__global__ void fused_ffn_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate_up,
    const __nv_bfloat16* __restrict__ w_down,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ gate_up_out,
    int N, int d, int H) {
  (void)x; (void)w_gate_up; (void)w_down; (void)y; (void)gate_up_out;
  (void)N; (void)d; (void)H;
}

#else

using namespace nvcuda;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float silu_f(float v) {
  return v / (1.0f + __expf(-v));
}

// One block per row-tile of 16 output rows. Within the block, 4 warps
// cooperate to compute 4 column-tiles of 16 (each producing a 16x16
// output tile of gate_up). After silu_mul collapses to H, the same
// warp layout produces the down output.
//
// Limitations: assumes d is a multiple of 16, H is a multiple of 16.
// Falls back to per-row kernel otherwise (host-side dispatch checks).
__global__ void fused_ffn_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,         // [N, d]
    const __nv_bfloat16* __restrict__ w_gate_up, // [2H, d]
    const __nv_bfloat16* __restrict__ w_down,    // [d, H]
    __nv_bfloat16* __restrict__ y,                // [N, d]
    __nv_bfloat16* __restrict__ gate_up_out,      // [N, 2H] or nullptr (A1)
    int N, int d, int H) {
  const int row_tile = blockIdx.x;
  const int row_base = row_tile * kWmmaM;
  if (row_base >= N) return;

  const int warp_id = threadIdx.x / kWarpSize;

  // Per-block shared memory: gate_up output tile [16, 2H] + post-silu
  // act tile [16, H]. For H=2048 these are 64KB and 32KB respectively
  // — fits in Blackwell's 228KB SM shared memory with room for the
  // matmul tiles.
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sh_gate_up = smem;                       // [16, 2H]
  __nv_bfloat16* sh_act     = smem + 16 * (2 * H);        // [16, H]
  // Per-warp fp32 scratch for wmma::store_matrix_sync. WMMA requires
  // every thread in the warp to pass the SAME pointer (shared or
  // global memory); per-thread stack arrays are UB. Each of the
  // kWarpsPerBlock warps gets its own [16, 16] scratch slot here.
  float* sh_wmma = reinterpret_cast<float*>(sh_act + 16 * H);
  float* my_wmma = sh_wmma + warp_id * (kWmmaM * kWmmaN);
  const int lane = threadIdx.x & 31;

  // Compute gate_up = x_tile @ w_gate_up.T.
  // gate_up tile [16, 2H]. Each warp handles 2H / (warps * 16) column tiles.
  const int col_tiles_2H = (2 * H) / kWmmaN;
  for (int ct = warp_id; ct < col_tiles_2H; ct += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kt = 0; kt < d / kWmmaK; ++kt) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, x + (int64_t)row_base * d + kt * kWmmaK, d);
      wmma::load_matrix_sync(b, w_gate_up + (int64_t)(ct * kWmmaN) * d + kt * kWmmaK, d);
      wmma::mma_sync(c, a, b, c);
    }
    // Collective store of the [16, 16] fp32 tile into the warp's
    // shared-memory scratch slot. All 32 threads pass `my_wmma`.
    wmma::store_matrix_sync(my_wmma, c, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    // B2 — paired fp32→bf16 conversion via __float22bfloat162_rn and a
    // single 4-byte store per pair. 128 pairs / warp / tile, 4 pairs / thread.
    constexpr int kPairs = kWmmaM * kWmmaN / 2;
    for (int p = lane; p < kPairs; p += 32) {
      const int i = p / (kWmmaN / 2);
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

  // A1 — optionally publish gate_up to HBM so the training-side backward
  // skips the recompute matmul. nullptr means inference (no_grad) and we
  // skip the write entirely.
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
  }

  // silu_mul: act[i, h] = silu(gate_up[i, h]) * gate_up[i, h + H]
  // for i in [0, 16), h in [0, H).
  const int total_act = 16 * H;
  for (int idx = threadIdx.x; idx < total_act; idx += blockDim.x) {
    int i = idx / H;
    int h = idx % H;
    float g = __bfloat162float(sh_gate_up[i * (2 * H) + h]);
    float u = __bfloat162float(sh_gate_up[i * (2 * H) + h + H]);
    sh_act[i * H + h] = __float2bfloat16(silu_f(g) * u);
  }
  __syncthreads();

  // down matmul: y[16, d] = act[16, H] @ w_down.T  where w_down is [d, H].
  const int col_tiles_d = d / kWmmaN;
  for (int ct = warp_id; ct < col_tiles_d; ct += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kt = 0; kt < H / kWmmaK; ++kt) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, sh_act + 0 * H + kt * kWmmaK, H);
      wmma::load_matrix_sync(b, w_down + (int64_t)(ct * kWmmaN) * H + kt * kWmmaK, H);
      wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(my_wmma, c, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    // B2 — paired conversion + 4-byte store. The (gi >= N) guard
    // applies per-row so we evaluate it per-pair (both elements share
    // the same row).
    constexpr int kPairs = kWmmaM * kWmmaN / 2;
    for (int p = lane; p < kPairs; p += 32) {
      const int i = p / (kWmmaN / 2);
      const int c2 = (p % (kWmmaN / 2)) * 2;
      const int gi = row_base + i;
      if (gi >= N) continue;
      const float2 f = make_float2(my_wmma[i * kWmmaN + c2],
                                     my_wmma[i * kWmmaN + c2 + 1]);
      const __nv_bfloat162 b = __float22bfloat162_rn(f);
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
          &y[(int64_t)gi * d + ct * kWmmaN + c2]);
      *dst = b;
    }
  }
}

#endif  // __CUDA_ARCH__ < 800

}  // namespace

// Host dispatch: prefer WMMA when shapes align (multiples of 16) and
// per-block shmem fits; otherwise fall back to the FMA-loop kernel.
torch::Tensor fused_ffn_wmma_cuda(torch::Tensor x,
                                    torch::Tensor w_gate_up,
                                    torch::Tensor w_down) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fused_ffn_wmma_cuda: bf16 CUDA inputs required");
  c10::cuda::CUDAGuard guard(x.device());
  auto x_c = x.contiguous();
  auto wg = w_gate_up.contiguous();
  auto wd = w_down.contiguous();
  const int64_t B = x_c.size(0);
  const int64_t S = x_c.size(1);
  const int64_t d = x_c.size(2);
  const int64_t H = wg.size(0) / 2;
  TORCH_CHECK(d % 16 == 0 && H % 16 == 0, "shapes must be multiples of 16");
  const int N = static_cast<int>(B * S);
  auto y = torch::empty_like(x_c);

  const size_t shmem = (16 * (2 * H) + 16 * H) * sizeof(__nv_bfloat16)
                     + (size_t)kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float);
  // ^ trailing term: per-warp scratch for wmma::store_matrix_sync.
  // Opt into >48 KB dynamic shared memory. The host dispatcher
  // (fused_ffn.cpp) already guarantees shmem ≤ device opt-in max before
  // routing here, so this set always succeeds.
  if (shmem > 48 * 1024) {
    cudaFuncSetAttribute(fused_ffn_wmma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
  }
  const int grid = (N + 16 - 1) / 16;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_ffn_wmma_kernel<<<grid, kThreadsPerBlock, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      /*gate_up_out=*/nullptr,
      N, static_cast<int>(d), static_cast<int>(H));
  return y;
}

// A1 — training entry point. Identical to fused_ffn_wmma_cuda but also
// materializes gate_up to HBM so the autograd backward can skip
// recomputing fast_linear(x, w_gate_up). The HBM write cost (≈143 µs
// for B*S=16384, H=2048 on 5060 Ti) is amortized by the ≈860 µs
// recompute it eliminates.
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_wmma_train_cuda(torch::Tensor x,
                            torch::Tensor w_gate_up,
                            torch::Tensor w_down) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fused_ffn_wmma_train_cuda: bf16 CUDA inputs required");
  c10::cuda::CUDAGuard guard(x.device());
  auto x_c = x.contiguous();
  auto wg = w_gate_up.contiguous();
  auto wd = w_down.contiguous();
  const int64_t B = x_c.size(0);
  const int64_t S = x_c.size(1);
  const int64_t d = x_c.size(2);
  const int64_t H = wg.size(0) / 2;
  TORCH_CHECK(d % 16 == 0 && H % 16 == 0, "shapes must be multiples of 16");
  const int N = static_cast<int>(B * S);
  auto y = torch::empty_like(x_c);
  auto gate_up = torch::empty({B, S, 2 * H}, x_c.options());

  const size_t shmem = (16 * (2 * H) + 16 * H) * sizeof(__nv_bfloat16)
                     + (size_t)kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float);
  // ^ trailing term: per-warp scratch for wmma::store_matrix_sync.
  if (shmem > 48 * 1024) {
    cudaFuncSetAttribute(fused_ffn_wmma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
  }
  const int grid = (N + 16 - 1) / 16;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_ffn_wmma_kernel<<<grid, kThreadsPerBlock, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()),
      N, static_cast<int>(d), static_cast<int>(H));
  return {y, gate_up};
}

}  // namespace olmo_cpp
