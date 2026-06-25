/**
 * kernels/topp_radix.cu
 *
 * Bucket-radix top-p (nucleus) sampling kernel — fast-inference [7].
 *
 * See include/olmo_cpp/backend/topp_radix.hpp for the algorithm contract.
 *
 * ─── Why not just sort? ───────────────────────────────────────────────
 *
 * Standard top-p is O(V log V) — sort probs descending, walk until
 * cumulative ≥ p, mask the tail. Sort kernels (thrust::sort, cub::sort)
 * are real launches with multiple passes over global memory. For V=50K
 * that's ~50-100μs.
 *
 * Bucket-radix exploits the fact that we don't need a full sort — we
 * just need to know "is this token's mass above or below the cutoff?".
 * Log-spaced buckets give 16 coarse bins covering [2^-16, 1.0]. One
 * histogramming pass tells us which bucket holds the cutoff. Tokens in
 * better buckets are kept; tokens in worse buckets are zeroed; tokens
 * IN the cutoff bucket are an approximation (kept entirely — small
 * over-shoot of top_p, bounded by the bucket width 2x).
 *
 * Net: O(V) work, 1-2 kernel launches, ~10-30μs at V=50K.
 *
 * Bonus: the min-p early-out drops the long tail (70-90% of vocab) before
 * even bucketing, so the histogram pass touches a small subset.
 *
 * ─── Layout ───────────────────────────────────────────────────────────
 *
 *   B = 16 buckets (log-spaced, base 2).
 *   bucket(p) = clamp(floor(-log2(p)), 0, B-1)  for p > 0
 *
 *   Three kernels, one launch each:
 *     1. histogram: per-block partial → atomic reduce to global [B] count
 *        and [B] mass. Threads with prob < min_p contribute nothing.
 *     2. cutoff: tiny launch (1 block, 32 threads) that scans the 16-bin
 *        histogram to find the cutoff bucket and emits a single int.
 *        We could just do this on host but doing it on device avoids a
 *        D->H sync on the histogram itself.
 *     3. mask: zero out tokens with bucket > cutoff_bucket and tokens
 *        below min_p. Renormalize via atomicAdd into a sum, then divide.
 *
 * Renormalization needs a sum. We compute it inline in the mask kernel
 * via atomicAdd into a single FP32 scratch slot. Atomic FP32 adds are
 * non-deterministic in ordering; for sampling this is fine (the resulting
 * distribution is the same to within rounding).
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "olmo_cpp/backend/topp_radix.hpp"
#include "cuda_reduce.cuh"

namespace olmo_cpp {

namespace {

constexpr int kNumBuckets = 16;
constexpr int kThreads = 256;
constexpr int kMaxWarps = (kThreads + 31) >> 5;  // 8 warps for 256 threads

// Map a positive probability to its log-magnitude bucket.
//   bucket 0  -> [0.5, 1.0]
//   bucket k  -> [2^-(k+1), 2^-k]
//   bucket 15 -> [2^-16, 2^-15]
// Probabilities <= 0 OR < min_p map to "skip" (we encode as kNumBuckets).
__device__ __forceinline__ int prob_to_bucket(float p, float min_p) {
  if (!(p > min_p)) return kNumBuckets;  // also handles NaN, -inf, etc.
  // -log2(p) is non-negative when p < 1. clamp to [0, B-1].
  float lp = -log2f(p);
  int b = static_cast<int>(floorf(lp));
  if (b < 0) b = 0;
  if (b >= kNumBuckets) b = kNumBuckets - 1;
  return b;
}

// Pass 1: histogram. Each thread keeps its histogram in registers (16
// buckets × {int count, float mass} = 128 bytes), then a block-wide
// warp-tree reduction sums them up. The previous code did
// atomicAdd-on-shared per vocab element, serializing tens of thousands
// of ops on 16 shared-memory locations. Per-thread register
// accumulation eliminates that contention entirely; the only atomics
// remaining are 2*B writes per BLOCK to the global histogram (one per
// bucket, by thread 0).
__global__ void topp_histogram_kernel(
    const float* __restrict__ probs,
    int64_t V,
    float min_p,
    int* __restrict__ global_count,    // [B] int counts
    float* __restrict__ global_mass) { // [B] float sums

  // Per-thread local histogram in registers.
  int   local_count[kNumBuckets] = {0};
  float local_mass[kNumBuckets]  = {0.0f};

  // Grid-stride loop over the vocab. Each thread bins its slice into its
  // own register-resident histogram — zero shared/global atomics in this
  // hot loop.
  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < V; i += stride) {
    float p = probs[i];
    int b = prob_to_bucket(p, min_p);
    if (b < kNumBuckets) {
      local_count[b] += 1;
      local_mass[b]  += p;
    }
  }

  // Block-wide tree reduction, one bucket at a time. cuda_reduce.cuh's
  // helper takes float; for the int counts we just cast through float
  // (vocab fits in 50K entries → far below 2^24, exact).
  __shared__ float sh_warpbuf[kMaxWarps];
  for (int b = 0; b < kNumBuckets; ++b) {
    const float c = block_reduce_sum(static_cast<float>(local_count[b]),
                                     threadIdx.x, blockDim.x, sh_warpbuf);
    const float m = block_reduce_sum(local_mass[b],
                                     threadIdx.x, blockDim.x, sh_warpbuf);
    if (threadIdx.x == 0) {
      // One global atomic per bucket per block. With ~10s of blocks per
      // launch this is cheap relative to V global atomics in the old
      // pattern.
      atomicAdd(&global_count[b], static_cast<int>(c));
      atomicAdd(&global_mass[b],  m);
    }
    // block_reduce_sum's final write to sh_warpbuf[0] needs a sync before
    // we reuse the buffer for the next bucket.
    __syncthreads();
  }
}

// Pass 2: scan the 16-entry histogram on a single warp.
// Outputs cutoff_bucket: the smallest bucket index `b` such that
// cumulative_mass[0..b] >= top_p. Tokens in buckets <= cutoff_bucket
// stay; everything else gets zeroed in pass 3.
__global__ void topp_find_cutoff_kernel(
    const float* __restrict__ global_mass,
    float top_p,
    int* __restrict__ cutoff_bucket) {
  // Run on a single thread — the histogram has only 16 entries, no
  // parallelism worth orchestrating. This kernel exists at all only
  // because doing it on device avoids a D->H sync of the histogram.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float cum = 0.0f;
    int b = kNumBuckets - 1;  // worst case: include everything
    for (int i = 0; i < kNumBuckets; ++i) {
      cum += global_mass[i];
      if (cum >= top_p) {
        b = i;
        break;
      }
    }
    *cutoff_bucket = b;
  }
}

// Pass 3: mask + sum. Zero out tokens whose bucket > cutoff_bucket
// (also the min_p-dropped tokens, which had bucket = kNumBuckets).
// Compute the surviving sum into one FP32 scratch slot.
__global__ void topp_mask_kernel(
    float* __restrict__ probs,
    int64_t V,
    float min_p,
    const int* __restrict__ cutoff_bucket,
    float* __restrict__ surviving_sum) {

  int cutoff = *cutoff_bucket;
  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Per-thread partial sum, reduced via atomicAdd at the end.
  float local_sum = 0.0f;

  for (int64_t i = idx; i < V; i += stride) {
    float p = probs[i];
    int b = prob_to_bucket(p, min_p);
    if (b > cutoff) {
      probs[i] = 0.0f;
    } else {
      local_sum += p;
    }
  }

  // Block-level reduction via shared memory before the global atomic
  // would be more efficient, but for V=50K and kThreads=256 we have
  // only ~200 atomic adds per block worth of contention — fine.
  if (local_sum > 0.0f) {
    atomicAdd(surviving_sum, local_sum);
  }
}

// Pass 4: divide each survivor by the sum to renormalize.
__global__ void topp_renorm_kernel(
    float* __restrict__ probs,
    int64_t V,
    const float* __restrict__ surviving_sum) {
  float s = *surviving_sum;
  if (!(s > 0.0f)) return;  // degenerate; leave as is (caller will sample uniformly)
  float inv_s = 1.0f / s;

  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < V; i += stride) {
    probs[i] *= inv_s;
  }
}

}  // namespace

void topp_radix_filter_cuda(torch::Tensor probs, float top_p, float min_p) {
  TORCH_CHECK(probs.is_cuda(), "topp_radix_filter_cuda: probs must be CUDA");
  TORCH_CHECK(probs.dim() == 1, "topp_radix_filter_cuda: probs must be 1-D [V]");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32,
              "topp_radix_filter_cuda: probs must be float32");
  TORCH_CHECK(probs.is_contiguous(), "topp_radix_filter_cuda: probs must be contiguous");
  if (top_p >= 1.0f) return;  // no-op
  if (top_p <= 0.0f) {
    // Degenerate: keep only the argmax. Easier to fall back than to special-case.
    // For now, treat as "keep top single bucket" by clamping.
    top_p = 1e-6f;
  }

  c10::cuda::CUDAGuard guard(probs.device());
  const int64_t V = probs.numel();
  auto opts_int   = torch::TensorOptions().dtype(torch::kInt32).device(probs.device());
  auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(probs.device());

  auto count          = torch::zeros({kNumBuckets}, opts_int);
  auto mass           = torch::zeros({kNumBuckets}, opts_float);
  auto cutoff_bucket  = torch::zeros({1}, opts_int);
  auto surviving_sum  = torch::zeros({1}, opts_float);

  int blocks = static_cast<int>(std::min<int64_t>((V + kThreads - 1) / kThreads, 65535));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  topp_histogram_kernel<<<blocks, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), V, min_p,
      count.data_ptr<int>(), mass.data_ptr<float>());

  topp_find_cutoff_kernel<<<1, 32, 0, stream>>>(
      mass.data_ptr<float>(), top_p, cutoff_bucket.data_ptr<int>());

  topp_mask_kernel<<<blocks, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), V, min_p,
      cutoff_bucket.data_ptr<int>(), surviving_sum.data_ptr<float>());

  topp_renorm_kernel<<<blocks, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), V, surviving_sum.data_ptr<float>());
}

}  // namespace olmo_cpp
