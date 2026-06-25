/**
 * kernels/fused_qkv_rope_wmma.cu
 *
 * Tensor-core (WMMA) variant of fused QKV projection + reshape + RoPE.
 *
 * Mirrors fused_ffn_wmma's geometry: one block per 16-row tile of the
 * (B*S, F) output where F = (n_q + 2*n_kv) * head_dim. Four warps
 * cooperate on column tiles; each warp owns a column tile and walks
 * the K axis (d) with WMMA 16x16x16 BF16 (fp32 accum, bf16 store).
 *
 * After the matmul lands the full [16, F] tile in shared memory, the
 * block scatters into q_out / k_out / v_out with half-rotation RoPE
 * applied to Q and K heads.
 *
 * Gated on:
 *   - bf16 inputs
 *   - N = B*S divisible by 16 (otherwise WMMA loads x rows OOB)
 *   - F divisible by 16, d divisible by 16
 *   - 16 * F * sizeof(bf16) fits in sm shared memory (228 KB on
 *     Blackwell sm_120, 96 KB on sm_80).
 *
 * Host dispatch (fused_qkv_rope.cpp) falls back to the FMA-loop kernel
 * when any precondition fails.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#include "olmo_cpp/backend/fused_qkv_rope.hpp"

namespace olmo_cpp {

namespace {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
__global__ void fused_qkv_rope_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_qkv,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int N, int d, int S, int F,
    int n_q, int n_kv, int hd) {
  (void)x; (void)w_qkv; (void)cos; (void)sin;
  (void)q_out; (void)k_out; (void)v_out;
  (void)N; (void)d; (void)S; (void)F; (void)n_q; (void)n_kv; (void)hd;
}

#else

using namespace nvcuda;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__global__ void fused_qkv_rope_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,         // [N, d]    (N = B*S, row-major)
    const __nv_bfloat16* __restrict__ w_qkv,     // [F, d]    (row-major)
    const __nv_bfloat16* __restrict__ cos,       // [S, hd/2]
    const __nv_bfloat16* __restrict__ sin,       // [S, hd/2]
    __nv_bfloat16* __restrict__ q_out,            // [B, n_q,  S, hd]
    __nv_bfloat16* __restrict__ k_out,            // [B, n_kv, S, hd]
    __nv_bfloat16* __restrict__ v_out,            // [B, n_kv, S, hd]
    int N, int d, int S, int F,
    int n_q, int n_kv, int hd) {
  const int row_tile = blockIdx.x;
  const int row_base = row_tile * kWmmaM;
  if (row_base >= N) return;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane    = threadIdx.x & 31;

  // Per-block shared memory layout:
  //   [0, 16 * F * sizeof(bf16))                         output tile (bf16)
  //   [trailing, + warps * 16 * 16 * sizeof(float))      per-warp wmma scratch
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sh_out  = smem;
  float* sh_wmma = reinterpret_cast<float*>(sh_out + 16 * F);
  float* my_wmma = sh_wmma + warp_id * (kWmmaM * kWmmaN);

  // Phase 1 — WMMA matmul. Compute 16 rows × F cols of x @ w_qkv.T.
  // x is row-major [N, d]; w_qkv is row-major [F, d]. To form x @ w_qkv.T
  // via WMMA we read x as row-major and w_qkv with the col-major layout
  // (which reinterprets the same buffer as the transpose with ld = d).
  const int col_tiles_total = F / kWmmaN;
  for (int ct = warp_id; ct < col_tiles_total; ct += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kt = 0; kt < d / kWmmaK; ++kt) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                      __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, x + (int64_t)row_base * d + kt * kWmmaK, d);
      wmma::load_matrix_sync(b, w_qkv + (int64_t)(ct * kWmmaN) * d + kt * kWmmaK, d);
      wmma::mma_sync(c, a, b, c);
    }
    // Collective store of the 16×16 tile into the warp's scratch slot.
    // All threads in the warp pass the same pointer (per WMMA contract).
    wmma::store_matrix_sync(my_wmma, c, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    // B2 — paired fp32→bf16 conversion + 4-byte store.
    constexpr int kPairs = kWmmaM * kWmmaN / 2;
    for (int p = lane; p < kPairs; p += 32) {
      const int i  = p / (kWmmaN / 2);
      const int c2 = (p % (kWmmaN / 2)) * 2;
      const float2 f = make_float2(my_wmma[i * kWmmaN + c2],
                                     my_wmma[i * kWmmaN + c2 + 1]);
      const __nv_bfloat162 b = __float22bfloat162_rn(f);
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
          &sh_out[i * F + ct * kWmmaN + c2]);
      *dst = b;
    }
  }
  __syncthreads();

  // Phase 2 — RoPE + scatter for Q and K heads.
  // Each thread handles one (row, qk_head, dim_pair) — a pair of elements
  // at columns (head_start + di, head_start + di + hd/2). Half-rotation
  // RoPE matches the model's RotaryEmbedding::apply convention.
  const int hd_half = hd / 2;
  const int qk_heads = n_q + n_kv;
  const int qk_pairs_total = kWmmaM * qk_heads * hd_half;
  for (int idx = threadIdx.x; idx < qk_pairs_total; idx += blockDim.x) {
    const int r  = idx / (qk_heads * hd_half);
    const int rm = idx % (qk_heads * hd_half);
    const int hi = rm / hd_half;          // head index in concatenated Q||K
    const int di = rm % hd_half;
    const int global_row = row_base + r;
    if (global_row >= N) continue;
    const int b = global_row / S;
    const int s_pos = global_row % S;

    const int head_col_offset = hi * hd;  // Q/K live in [0, (n_q+n_kv)*hd)
    const float a_val = __bfloat162float(sh_out[r * F + head_col_offset + di]);
    const float b_val = __bfloat162float(sh_out[r * F + head_col_offset + di + hd_half]);
    const float cv    = __bfloat162float(cos[(int64_t)s_pos * hd_half + di]);
    const float sv    = __bfloat162float(sin[(int64_t)s_pos * hd_half + di]);
    const float new_a = a_val * cv - b_val * sv;
    const float new_b = a_val * sv + b_val * cv;

    const bool is_q       = (hi < n_q);
    const int  h_in_dst   = is_q ? hi : (hi - n_q);
    const int  n_heads_dst = is_q ? n_q : n_kv;
    __nv_bfloat16* dst    = is_q ? q_out : k_out;
    const int64_t base    = (((int64_t)b * n_heads_dst + h_in_dst) * S + s_pos) * hd;
    dst[base + di]           = __float2bfloat16(new_a);
    dst[base + di + hd_half] = __float2bfloat16(new_b);
  }

  // Phase 3 — straight copy for V heads (no RoPE).
  const int v_col_base = (n_q + n_kv) * hd;
  const int v_total    = kWmmaM * n_kv * hd;
  for (int idx = threadIdx.x; idx < v_total; idx += blockDim.x) {
    const int r  = idx / (n_kv * hd);
    const int rm = idx % (n_kv * hd);
    const int hi = rm / hd;
    const int di = rm % hd;
    const int global_row = row_base + r;
    if (global_row >= N) continue;
    const int b = global_row / S;
    const int s_pos = global_row % S;
    const float v_val = __bfloat162float(sh_out[r * F + v_col_base + hi * hd + di]);
    const int64_t base = (((int64_t)b * n_kv + hi) * S + s_pos) * hd;
    v_out[base + di] = __float2bfloat16(v_val);
  }
}

#endif  // __CUDA_ARCH__ < 800

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_wmma_cuda(torch::Tensor x,
                           torch::Tensor w_qkv,
                           torch::Tensor cos,
                           torch::Tensor sin,
                           int64_t n_q_heads,
                           int64_t n_kv_heads,
                           int64_t head_dim) {
  TORCH_CHECK(x.is_cuda() && w_qkv.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "fused_qkv_rope_wmma_cuda: all inputs must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
              w_qkv.scalar_type() == torch::kBFloat16 &&
              cos.scalar_type() == torch::kBFloat16 &&
              sin.scalar_type() == torch::kBFloat16,
              "fused_qkv_rope_wmma_cuda: bf16 inputs required");

  c10::cuda::CUDAGuard guard(x.device());
  auto x_c   = x.contiguous();
  auto w_c   = w_qkv.contiguous();
  auto cos_c = cos.contiguous();
  auto sin_c = sin.contiguous();

  const int B  = static_cast<int>(x_c.size(0));
  const int S  = static_cast<int>(x_c.size(1));
  const int d  = static_cast<int>(x_c.size(2));
  const int n_q  = static_cast<int>(n_q_heads);
  const int n_kv = static_cast<int>(n_kv_heads);
  const int hd   = static_cast<int>(head_dim);
  const int N    = B * S;
  const int F    = (n_q + 2 * n_kv) * hd;

  TORCH_CHECK(N % kWmmaM == 0 && d % kWmmaK == 0 && F % kWmmaN == 0,
              "fused_qkv_rope_wmma_cuda: N, d, F must be multiples of 16");

  auto opts = x_c.options();
  auto q = torch::empty({B, n_q,  S, hd}, opts);
  auto k = torch::empty({B, n_kv, S, hd}, opts);
  auto v = torch::empty({B, n_kv, S, hd}, opts);

  const int grid  = N / kWmmaM;
  const size_t shmem = (size_t)kWmmaM * F * sizeof(__nv_bfloat16)
                     + (size_t)kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float);
  // ^ trailing term: per-warp scratch for wmma::store_matrix_sync.
  // Opt into >48 KB dynamic shared memory (host gate guarantees fit).
  if (shmem > 48 * 1024) {
    cudaFuncSetAttribute(fused_qkv_rope_wmma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_qkv_rope_wmma_kernel<<<grid, kThreadsPerBlock, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(cos_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(sin_c.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      N, d, S, F, n_q, n_kv, hd);
  return {q, k, v};
}

}  // namespace olmo_cpp
