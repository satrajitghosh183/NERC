/**
 * kernels/fused_lm_head_sample.cu
 *
 * Fused LM-head + Gumbel-max sampler — fast-inference [6].
 *
 * See include/olmo_cpp/backend/fused_lm_head_sample.hpp for the contract.
 *
 * ─── What this kernel collapses ───────────────────────────────────────
 *
 *   logits = W_U @ hidden          # cuBLAS GEMV: V*H FLOPs, [V]→HBM
 *   logits = logits / temperature  # elementwise: [V] HBM read+write
 *   probs  = softmax(logits)       # 2-pass reduction
 *   token  = sample(probs)         # multinomial: another kernel
 *                                  # + ~200KB D→H copy if on CPU
 *
 * Become:
 *
 *   token  = fused_kernel(hidden, W_U, T, seed, position)  # 8 bytes D→H
 *
 * Trick: argmax(logits/T + Gumbel(0,1)) is mathematically equivalent to
 * sampling from softmax(logits/T). So sampling reduces to noisy argmax,
 * which fits naturally as the GEMV's epilogue — one block-wide reduction.
 * No softmax tensor ever materialized.
 *
 * ─── Layout ───────────────────────────────────────────────────────────
 *
 * One thread per row of W_U. Grid-stride to cover V > grid_size*block_size.
 * Each thread:
 *   1. Reads W_U[row, :] (streaming, no reuse — H is small, V is huge)
 *   2. Reads hidden[:] from shared memory (loaded once per block)
 *   3. Computes the dot product → l_row
 *   4. Generates Gumbel(0,1) via Philox keyed on (seed, position, row)
 *   5. s_row = l_row / T + g_row
 *   6. Tracks the per-thread (s, row) running max
 *
 * Per-block: warp shuffle reduction → block atomicCAS into a global
 * packed-(score, index) uint64. Pack via float-to-sortable-uint flip
 * so atomicMax-on-uint64 sorts correctly.
 *
 * ─── Limitations of this version ──────────────────────────────────────
 *
 * Naive GEMV: one row per thread, loops over H sequentially. cuBLAS would
 * tile and use Tensor Cores for ~2-3x more throughput. For batch=1 GEMV
 * this is acceptable — bandwidth bound by the W_U read regardless.
 *
 * Single-threaded compare-exchange contention on the global best slot.
 * For V=50K, ~200 warps each do one atomicCAS — bounded contention.
 * If this becomes a bottleneck, replace with a tree reduction over a
 * grid-sized scratch array.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <math_constants.h>   // CUDART_INF_F
#include <cstdint>
#include <cmath>
#include <vector>

#include "olmo_cpp/backend/fused_lm_head_sample.hpp"

namespace olmo_cpp {

namespace {

// ───── Hand-rolled Philox-4x32-10 (counter-based RNG) ─────
// Stateless, parallel-friendly. Same algorithm as cuRAND's
// curandStatePhilox4_32_10_t but inlined so we don't link against
// curand. Quality is bit-identical to cuRAND.

__device__ __forceinline__ uint32_t mulhi32(uint32_t a, uint32_t b, uint32_t* lo) {
  uint64_t prod = static_cast<uint64_t>(a) * b;
  *lo = static_cast<uint32_t>(prod);
  return static_cast<uint32_t>(prod >> 32);
}

__device__ __forceinline__ void philox_round(uint32_t c[4], uint32_t k[2]) {
  uint32_t lo0, hi0 = mulhi32(0xD2511F53u, c[0], &lo0);
  uint32_t lo1, hi1 = mulhi32(0xCD9E8D57u, c[2], &lo1);
  uint32_t n0 = hi1 ^ c[1] ^ k[0];
  uint32_t n1 = lo1;
  uint32_t n2 = hi0 ^ c[3] ^ k[1];
  uint32_t n3 = lo0;
  c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint32_t position, uint32_t idx) {
  // Counter: (idx, position, 0, 0). Key: 64-bit seed split into two 32-bit halves.
  uint32_t c[4] = { idx, position, 0u, 0u };
  uint32_t k[2] = { static_cast<uint32_t>(seed),
                    static_cast<uint32_t>(seed >> 32) };
  // 10 rounds (Philox-4x32-10).
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    philox_round(c, k);
    k[0] += 0x9E3779B9u;
    k[1] += 0xBB67AE85u;
  }
  // Map c[0] to (0, 1). Top 24 bits → float in [0, 1), then shift to (0, 1].
  uint32_t bits = c[0] >> 8;       // 24 bits of randomness
  if (bits == 0) bits = 1;          // avoid u=0 (would NaN the Gumbel)
  return static_cast<float>(bits) * (1.0f / static_cast<float>(1 << 24));
}

__device__ __forceinline__ float gumbel_from_philox(uint64_t seed, uint32_t position, uint32_t idx) {
  float u = philox_uniform(seed, position, idx);
  // Gumbel(0,1) = -log(-log(u)). u is in (0, 1].
  return -__logf(-__logf(u));
}

// ───── Score/index packing for atomicMax-on-uint64 reduction ─────
//
// Want to atomicMax on (score, index), where score is float. Standard
// trick: convert float to a "sortable" uint32 such that uint comparison
// matches float comparison. Pack (sortable_uint32, index_uint32) into
// uint64. atomicMax on uint64 then gives us the highest score with its
// associated index (with score winning, ties broken by larger index).

__device__ __forceinline__ unsigned long long pack_score_idx(float s, int idx) {
  unsigned int u = __float_as_uint(s);
  // For positive floats: flip sign bit → larger float = larger uint.
  // For negative floats: flip all bits → larger (less negative) = larger uint.
  unsigned int sortable = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
  return (static_cast<unsigned long long>(sortable) << 32) |
         (static_cast<unsigned long long>(static_cast<unsigned int>(idx)));
}

__device__ __forceinline__ int unpack_idx(unsigned long long packed) {
  return static_cast<int>(packed & 0xFFFFFFFFull);
}

// Special "smallest possible" sentinel for initialization.
__device__ __forceinline__ unsigned long long pack_neg_inf() {
  // -inf as float, sortable form. Index irrelevant.
  unsigned int u = 0xFF800000u;        // -inf bits
  unsigned int sortable = ~u;          // negative → flip all bits
  return (static_cast<unsigned long long>(sortable) << 32);
}

// ───── The fused kernel ─────

template <typename WT>
__global__ void fused_lm_head_sample_kernel(
    const float* __restrict__ hidden,    // [H] FP32 (always — small, always fits)
    const WT* __restrict__ W_U,          // [V, H] FP32 or BF16
    int64_t V,
    int H,
    float inv_T,                          // 1 / temperature
    uint64_t seed,
    uint32_t position,
    const float* __restrict__ rep_pen,    // [V] per-row penalty factor, or nullptr
    unsigned long long* __restrict__ best_packed) {

  // Stage hidden into shared memory once per block.
  extern __shared__ float sh_hidden[];
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    sh_hidden[i] = hidden[i];
  }
  __syncthreads();

  // Per-thread running best.
  float local_best_score = -CUDART_INF_F;
  int   local_best_idx   = -1;

  int64_t row    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; row < V; row += stride) {
    // Dot product W_U[row] · hidden.
    const WT* w_row = W_U + row * H;
    float l = 0.0f;

    if constexpr (std::is_same<WT, float>::value) {
      // FP32 path: vectorize 4-wide.
      int h = 0;
      for (; h + 4 <= H; h += 4) {
        float4 w = *reinterpret_cast<const float4*>(w_row + h);
        float4 v = *reinterpret_cast<const float4*>(sh_hidden + h);
        l += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
      }
      for (; h < H; ++h) l += w_row[h] * sh_hidden[h];
    } else {
      // BF16 path: pair-load via __nv_bfloat162 — one 32-bit load per
      // pair vs two 16-bit loads. Same vectorization pattern as
      // lm_head_gemv.cu.
      int h = 0;
      const auto* w_row2 = reinterpret_cast<const __nv_bfloat162*>(w_row);
      for (; h + 2 <= H; h += 2) {
        __nv_bfloat162 w2 = w_row2[h >> 1];
        l += __low2float(w2)  * sh_hidden[h]
           + __high2float(w2) * sh_hidden[h + 1];
      }
      for (; h < H; ++h) l += __bfloat162float(w_row[h]) * sh_hidden[h];
    }

    // Repetition penalty on the raw logit, BEFORE temperature + Gumbel — same
    // order and float math as the CPU reference (no [V] logits materialized).
    if (rep_pen != nullptr) {
      float pf = rep_pen[row];
      if (pf != 1.0f) l = (l > 0.0f) ? (l / pf) : (l * pf);
    }

    float g = gumbel_from_philox(seed, position, static_cast<uint32_t>(row));
    float s = l * inv_T + g;

    if (s > local_best_score) {
      local_best_score = s;
      local_best_idx   = static_cast<int>(row);
    }
  }

  // Warp-level reduction: shfl_down to find warp-best.
  unsigned mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float other_s = __shfl_down_sync(mask, local_best_score, off);
    int   other_i = __shfl_down_sync(mask, local_best_idx,   off);
    if (other_s > local_best_score) {
      local_best_score = other_s;
      local_best_idx   = other_i;
    }
  }

  // Lane 0 of each warp does an atomicMax-on-uint64 into the global best.
  if ((threadIdx.x & 31) == 0 && local_best_idx >= 0) {
    unsigned long long packed = pack_score_idx(local_best_score, local_best_idx);
    atomicMax(reinterpret_cast<unsigned long long*>(best_packed), packed);
  }
}

__global__ void init_best_kernel(unsigned long long* best_packed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *best_packed = pack_neg_inf();
  }
}

__global__ void extract_token_kernel(
    const unsigned long long* __restrict__ best_packed,
    int64_t* __restrict__ out_token) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out_token = static_cast<int64_t>(unpack_idx(*best_packed));
  }
}

}  // namespace

int64_t fused_lm_head_sample_cuda(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens,
    double rep_penalty) {

  TORCH_CHECK(hidden.is_cuda() && W_U.is_cuda(),
              "fused_lm_head_sample_cuda: tensors must be on CUDA");
  TORCH_CHECK(hidden.dim() == 1, "hidden must be 1-D [H]");
  TORCH_CHECK(W_U.dim() == 2,    "W_U must be 2-D [V, H]");
  TORCH_CHECK(W_U.size(1) == hidden.size(0), "hidden dim mismatch");
  TORCH_CHECK(temperature > 0.0f, "temperature must be > 0");

  c10::cuda::CUDAGuard guard(hidden.device());

  // Force contiguity. hidden is small; W_U is huge — caller should pass
  // contiguous to avoid the silent allocation.
  auto hidden_c = hidden.contiguous().to(torch::kFloat32);
  auto W_U_c    = W_U.contiguous();
  const int64_t V = W_U_c.size(0);
  const int     H = static_cast<int>(W_U_c.size(1));
  const float inv_T = 1.0f / temperature;

  // Repetition penalty: build a [V] per-row factor (1.0 except seen tokens),
  // read once per row inside the kernel. This is a [V] float buffer (~1/H of
  // the W_U traffic — negligible), NOT a logits materialization, and only when
  // a penalty is actually requested.
  torch::Tensor rep_pen;            // empty unless used
  const float* rep_pen_ptr = nullptr;
  if (rep_penalty != 1.0 && !rep_tokens.empty()) {
    std::vector<int64_t> valid;
    valid.reserve(rep_tokens.size());
    for (int64_t t : rep_tokens) if (t >= 0 && t < V) valid.push_back(t);
    if (!valid.empty()) {
      rep_pen = torch::ones(
          {V}, torch::TensorOptions().dtype(torch::kFloat32).device(hidden.device()));
      auto ids = torch::tensor(
          valid, torch::TensorOptions().dtype(torch::kInt64).device(hidden.device()));
      rep_pen.index_put_({ids}, static_cast<float>(rep_penalty));
      rep_pen_ptr = rep_pen.data_ptr<float>();
    }
  }

  // Workspace.
  auto opts_u64 = torch::TensorOptions().dtype(torch::kInt64).device(hidden.device());
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(hidden.device());
  auto best     = torch::empty({1}, opts_u64);
  auto out_tok  = torch::empty({1}, opts_i64);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Initialize best to -inf.
  init_best_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>()));

  const int threads = 256;
  // Heuristic grid: aim for ~8x oversubscription on H100 (132 SMs * 8 = 1056).
  // Cap at the V/threads minimum.
  int blocks = static_cast<int>(std::min<int64_t>((V + threads - 1) / threads, 1024));

  // Shared memory: H floats for the staged hidden vector.
  size_t shmem_bytes = static_cast<size_t>(H) * sizeof(float);

  if (W_U_c.scalar_type() == torch::kFloat32) {
    fused_lm_head_sample_kernel<float><<<blocks, threads, shmem_bytes, stream>>>(
        hidden_c.data_ptr<float>(),
        W_U_c.data_ptr<float>(),
        V, H, inv_T, seed, position, rep_pen_ptr,
        reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>()));
  } else if (W_U_c.scalar_type() == torch::kBFloat16) {
    fused_lm_head_sample_kernel<__nv_bfloat16><<<blocks, threads, shmem_bytes, stream>>>(
        hidden_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(W_U_c.data_ptr<at::BFloat16>()),
        V, H, inv_T, seed, position, rep_pen_ptr,
        reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>()));
  } else {
    TORCH_CHECK(false, "fused_lm_head_sample_cuda: W_U dtype must be FP32 or BF16");
  }

  extract_token_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const unsigned long long*>(best.data_ptr<int64_t>()),
      out_tok.data_ptr<int64_t>());

  // One D->H of 8 bytes for the result.
  int64_t token = out_tok.cpu().item<int64_t>();
  return token;
}

}  // namespace olmo_cpp
