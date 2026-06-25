// Causal, in-place masked softmax for attention scores.
//
// scores: [BH, Sq, Sk], bf16. One block per row (Sq * BH blocks).
// scale:  multiplied into every element before the max reduction.
// causal: if true, entries where key_index > query_index are set to -inf,
//         which after exp becomes 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zwt::ops::k {

namespace {

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__device__ float warp_max(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  }
  return v;
}
__device__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__device__ float block_max(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : -INFINITY;
  if (wid == 0) v = warp_max(v);
  return v;
}
__device__ float block_sum(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

// Shared-memory variant. One bf16 read + one bf16 write per element; the
// exp values live in shmem across the sum-exp and rescale passes. Requires
// sk * sizeof(float) bytes of dynamic shared memory.
__global__ void k_masked_softmax_smem(__nv_bfloat16* scores, int64_t sq,
                                      int64_t sk, float scale, bool causal) {
  extern __shared__ float smem[];  // [sk]
  int64_t row = blockIdx.x;
  int64_t q_in_seq = row % sq;
  __nv_bfloat16* r = scores + row * sk;

  int64_t limit = causal ? (q_in_seq + 1) : sk;

  // Pass 1: read bf16 once, scale, stash scaled score in shmem, reduce max.
  float m = -INFINITY;
  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    float v = bf_to_f(r[j]) * scale;
    smem[j] = v;
    m = fmaxf(m, v);
  }
  m = block_max(m);
  __shared__ float s_m;
  if (threadIdx.x == 0) s_m = m;
  __syncthreads();
  float mv = s_m;

  // Pass 2: exp from shmem, write exp back to shmem, reduce sum.
  float ss = 0.f;
  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    float e = __expf(smem[j] - mv);
    smem[j] = e;
    ss += e;
  }
  ss = block_sum(ss);
  __shared__ float s_inv;
  if (threadIdx.x == 0) s_inv = (ss > 0.f) ? (1.0f / ss) : 0.f;
  __syncthreads();
  float inv = s_inv;

  // Pass 3: rescale from shmem, single bf16 write to HBM.
  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    r[j] = f_to_bf(smem[j] * inv);
  }
  for (int64_t j = limit + threadIdx.x; j < sk; j += blockDim.x) {
    r[j] = f_to_bf(0.f);
  }
}

// HBM fallback for rows that exceed the shared-memory budget. Three passes
// through HBM; kept for correctness on very-long-context configs.
__global__ void k_masked_softmax(__nv_bfloat16* scores, int64_t sq, int64_t sk,
                                 float scale, bool causal) {
  int64_t row = blockIdx.x;
  int64_t q_in_seq = row % sq;
  __nv_bfloat16* r = scores + row * sk;

  int64_t limit = causal ? (q_in_seq + 1) : sk;

  float m = -INFINITY;
  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    m = fmaxf(m, bf_to_f(r[j]) * scale);
  }
  m = block_max(m);
  __shared__ float s_m;
  if (threadIdx.x == 0) s_m = m;
  __syncthreads();
  float mv = s_m;

  float ss = 0.f;
  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    float e = __expf(bf_to_f(r[j]) * scale - mv);
    r[j] = f_to_bf(e);
    ss += e;
  }
  ss = block_sum(ss);
  __shared__ float s_inv;
  if (threadIdx.x == 0) s_inv = (ss > 0.f) ? (1.0f / ss) : 0.f;
  __syncthreads();
  float inv = s_inv;

  for (int64_t j = threadIdx.x; j < limit; j += blockDim.x) {
    r[j] = f_to_bf(bf_to_f(r[j]) * inv);
  }
  for (int64_t j = limit + threadIdx.x; j < sk; j += blockDim.x) {
    r[j] = f_to_bf(0.f);
  }
}

// Softmax backward, in place, scaled.
//
//   grad_scores[j] = scale * P[j] * (grad_P[j] - sum_k P[k] * grad_P[k])
//
// grad_scores_io starts as grad_P (dL/dP), is overwritten with dL/d(raw_scores).
// Values outside the causal range (where P == 0) become zero naturally.
__global__ void k_softmax_backward_scaled(__nv_bfloat16* grad_scores_io,
                                          const __nv_bfloat16* probs,
                                          int64_t sk, float scale) {
  int64_t row = blockIdx.x;
  __nv_bfloat16* g = grad_scores_io + row * sk;
  const __nv_bfloat16* p = probs + row * sk;

  // row_sum = dot(P, grad_P)
  float row_sum = 0.f;
  for (int64_t j = threadIdx.x; j < sk; j += blockDim.x) {
    row_sum += bf_to_f(p[j]) * bf_to_f(g[j]);
  }
  row_sum = block_sum(row_sum);
  __shared__ float s_sum;
  if (threadIdx.x == 0) s_sum = row_sum;
  __syncthreads();
  float sv = s_sum;

  for (int64_t j = threadIdx.x; j < sk; j += blockDim.x) {
    float pj = bf_to_f(p[j]);
    float gj = bf_to_f(g[j]);
    g[j] = f_to_bf(scale * pj * (gj - sv));
  }
}

}  // namespace

void masked_softmax_inplace_bf16(__nv_bfloat16* scores, int64_t bh, int64_t sq,
                                 int64_t sk, float scale, bool causal,
                                 cudaStream_t s) {
  int threads = sk >= 1024 ? 1024 : (sk >= 512 ? 512 : 256);
  unsigned blocks = static_cast<unsigned>(bh * sq);

  // Prefer the shmem path when the row fits comfortably. 32 KiB keeps
  // occupancy healthy on every CC we target (sm70+: 48 KiB default).
  constexpr size_t kMaxSmemBytes = 32 * 1024;
  size_t smem_bytes = static_cast<size_t>(sk) * sizeof(float);
  if (smem_bytes <= kMaxSmemBytes) {
    k_masked_softmax_smem<<<blocks, threads, smem_bytes, s>>>(
        scores, sq, sk, scale, causal);
  } else {
    k_masked_softmax<<<blocks, threads, 0, s>>>(
        scores, sq, sk, scale, causal);
  }
}

void softmax_backward_scaled_bf16(__nv_bfloat16* grad_scores_io,
                                  const __nv_bfloat16* probs,
                                  int64_t bh, int64_t sq, int64_t sk,
                                  float scale, cudaStream_t s) {
  int threads = sk >= 1024 ? 1024 : (sk >= 512 ? 512 : 256);
  unsigned blocks = static_cast<unsigned>(bh * sq);
  k_softmax_backward_scaled<<<blocks, threads, 0, s>>>(grad_scores_io, probs,
                                                       sk, scale);
}

}  // namespace zwt::ops::k
