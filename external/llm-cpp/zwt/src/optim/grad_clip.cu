// Multi-tensor FP32 sum-of-squares reduction and in-place scale.
//
// Used by global gradient clipping. Walks a device-side array of pointers +
// sizes, chunk-strides across them, reduces inside each block, atomicAdds
// to a single fp32 scalar. Scale is a second pass with the same launch shape.
//
// Two scale variants:
//   * scale_fp32_many       — alpha is a host-side float (legacy, host-sync needed)
//   * scale_fp32_many_dev   — alpha is read from a device pointer (graph-safe)
//
// The `compute_clip_scale` kernel turns sumsq+max_norm into the scale on
// device, removing the only host sync from clip-grad-norm. The whole sequence
// (sumsq → compute_clip_scale → scale_fp32_many_dev) is therefore captureable
// inside a CUDA graph.

#include <cuda_runtime.h>
#include <math_constants.h>

namespace zwt::optim::k {

namespace {

constexpr int kBlock = 256;

__device__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
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

__global__ void k_zero_scalar(float* p) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *p = 0.f;
}

// Multi-Tensor-Apply kernels. Each block owns one fixed-size CHUNK of one
// tensor. The host-side wrapper builds a (chunk_id → tensor_id, offset)
// lookup table once per call so a 250M model with mixed param sizes
// — embedding 45M, layer FFN 2.4M, per-layer norm 896 — gets one block per
// chunk regardless of underlying tensor size. The previous "one block per
// tensor" scheme either over-blocked tiny tensors (1 SM each, finishes
// instantly) or under-blocked huge tensors (256 threads tries to crush 45M
// elements in one stride-loop, GPU sits idle).
constexpr int kChunkSize = 4 * kBlock;  // 1024 elements per chunk

__global__ void k_sumsq_mta(float** ptrs, const int64_t* sizes,
                            const int* chunk_to_tensor,
                            const int64_t* chunk_to_offset,
                            float* out) {
  const int     c     = blockIdx.x;
  const int     t     = chunk_to_tensor[c];
  const int64_t off   = chunk_to_offset[c];
  const int64_t n     = sizes[t];
  const int64_t end   = (off + kChunkSize <= n) ? (off + kChunkSize) : n;
  const float* p      = ptrs[t];

  float acc = 0.f;
  for (int64_t i = off + threadIdx.x; i < end; i += blockDim.x) {
    float v = p[i];
    acc += v * v;
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

__global__ void k_scale_mta(float** ptrs, const int64_t* sizes,
                            const int* chunk_to_tensor,
                            const int64_t* chunk_to_offset,
                            float alpha) {
  const int     c   = blockIdx.x;
  const int     t   = chunk_to_tensor[c];
  const int64_t off = chunk_to_offset[c];
  const int64_t n   = sizes[t];
  const int64_t end = (off + kChunkSize <= n) ? (off + kChunkSize) : n;
  float* p          = ptrs[t];
  for (int64_t i = off + threadIdx.x; i < end; i += blockDim.x) p[i] *= alpha;
}

__global__ void k_scale_dev_mta(float** ptrs, const int64_t* sizes,
                                const int* chunk_to_tensor,
                                const int64_t* chunk_to_offset,
                                const float* alpha_dev) {
  // Read alpha; common under-the-clip case has alpha==1 → skip writes.
  const float alpha = *alpha_dev;
  if (alpha == 1.f) return;
  const int     c   = blockIdx.x;
  const int     t   = chunk_to_tensor[c];
  const int64_t off = chunk_to_offset[c];
  const int64_t n   = sizes[t];
  const int64_t end = (off + kChunkSize <= n) ? (off + kChunkSize) : n;
  float* p          = ptrs[t];
  for (int64_t i = off + threadIdx.x; i < end; i += blockDim.x) p[i] *= alpha;
}

// Legacy "one block per tensor" kernels kept as fallback for callers that
// don't precompute chunk descriptors. New code should use the MTA path.
__global__ void k_sumsq(float** ptrs, const int64_t* sizes, int n_tensors,
                        float* out) {
  int t = blockIdx.x;
  if (t >= n_tensors) return;
  float* p = ptrs[t];
  int64_t n = sizes[t];
  float acc = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    float v = p[i];
    acc += v * v;
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

__global__ void k_scale(float** ptrs, const int64_t* sizes, int n_tensors,
                        float alpha) {
  int t = blockIdx.x;
  if (t >= n_tensors) return;
  float* p = ptrs[t];
  int64_t n = sizes[t];
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) p[i] *= alpha;
}

__global__ void k_scale_dev(float** ptrs, const int64_t* sizes, int n_tensors,
                            const float* alpha_dev) {
  float alpha = *alpha_dev;
  if (alpha == 1.f) return;
  int t = blockIdx.x;
  if (t >= n_tensors) return;
  float* p = ptrs[t];
  int64_t n = sizes[t];
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) p[i] *= alpha;
}

__global__ void k_compute_clip_scale(const float* sumsq_dev, float max_norm,
                                     float* scale_dev, float* norm_out_dev) {
  // Single-thread kernel — runs once per step. Trivial cost; lets us write
  // both the scale (consumed by k_scale_dev) and the unclipped norm (pulled
  // to host at log intervals only). max_norm <= 0 ⇒ no clipping (scale=1).
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float s   = sumsq_dev[0];
    float n   = sqrtf(s);
    float sc  = 1.f;
    if (max_norm > 0.f && n > max_norm) sc = max_norm / (n + 1e-6f);
    *scale_dev    = sc;
    if (norm_out_dev) *norm_out_dev = n;
  }
}

}  // namespace

void zero_scalar_fp32(float* p, cudaStream_t s) {
  k_zero_scalar<<<1, 1, 0, s>>>(p);
}

void sumsq_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float* out, cudaStream_t s) {
  k_sumsq<<<n_tensors, kBlock, 0, s>>>(ptrs, sizes, n_tensors, out);
}

void scale_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float alpha, cudaStream_t s) {
  k_scale<<<n_tensors, kBlock, 0, s>>>(ptrs, sizes, n_tensors, alpha);
}

void scale_fp32_many_dev(float** ptrs, const int64_t* sizes, int n_tensors,
                         const float* alpha_dev, cudaStream_t s) {
  k_scale_dev<<<n_tensors, kBlock, 0, s>>>(ptrs, sizes, n_tensors, alpha_dev);
}

// Multi-Tensor-Apply variants. Caller pre-builds chunk_to_tensor /
// chunk_to_offset on host and uploads to device. n_chunks determines the
// grid size — uniform across param-size distribution.
int chunk_size_fp32() { return kChunkSize; }

void sumsq_fp32_mta(float** ptrs, const int64_t* sizes,
                    const int* chunk_to_tensor,
                    const int64_t* chunk_to_offset,
                    int n_chunks, float* out, cudaStream_t s) {
  if (n_chunks <= 0) return;
  k_sumsq_mta<<<n_chunks, kBlock, 0, s>>>(
      ptrs, sizes, chunk_to_tensor, chunk_to_offset, out);
}

void scale_fp32_mta(float** ptrs, const int64_t* sizes,
                    const int* chunk_to_tensor,
                    const int64_t* chunk_to_offset,
                    int n_chunks, float alpha, cudaStream_t s) {
  if (n_chunks <= 0) return;
  k_scale_mta<<<n_chunks, kBlock, 0, s>>>(
      ptrs, sizes, chunk_to_tensor, chunk_to_offset, alpha);
}

void scale_fp32_mta_dev(float** ptrs, const int64_t* sizes,
                        const int* chunk_to_tensor,
                        const int64_t* chunk_to_offset,
                        int n_chunks, const float* alpha_dev,
                        cudaStream_t s) {
  if (n_chunks <= 0) return;
  k_scale_dev_mta<<<n_chunks, kBlock, 0, s>>>(
      ptrs, sizes, chunk_to_tensor, chunk_to_offset, alpha_dev);
}

void compute_clip_scale(const float* sumsq_dev, float max_norm,
                        float* scale_dev, float* norm_out_dev,
                        cudaStream_t s) {
  k_compute_clip_scale<<<1, 1, 0, s>>>(sumsq_dev, max_norm, scale_dev,
                                       norm_out_dev);
}

}  // namespace zwt::optim::k
