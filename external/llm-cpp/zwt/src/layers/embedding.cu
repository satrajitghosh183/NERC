// Embedding gather / scatter-add kernels.
//
// BF16 weights, FP32 master gradient. Gather is a pure copy; scatter-add
// uses atomicAdd into fp32 grad because duplicate ids in a batch alias.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace zwt::k {

namespace {

__global__ void k_embed_gather_bf16(const int64_t* ids, const __nv_bfloat16* W,
                                    __nv_bfloat16* out, int64_t N, int64_t D) {
  int64_t i = blockIdx.x;
  if (i >= N) return;
  int64_t id = ids[i];
  const __nv_bfloat16* src = W + id * D;
  __nv_bfloat16* dst = out + i * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

__global__ void k_embed_scatter_add(const int64_t* ids,
                                    const __nv_bfloat16* grad_y,
                                    float* grad_W, int64_t N, int64_t D) {
  int64_t i = blockIdx.x;
  if (i >= N) return;
  int64_t id = ids[i];
  const __nv_bfloat16* src = grad_y + i * D;
  float* dst = grad_W + id * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    atomicAdd(dst + d, __bfloat162float(src[d]));
  }
}

}  // namespace

void embed_gather_bf16(const int64_t* ids, const __nv_bfloat16* W,
                       __nv_bfloat16* out, int64_t N, int64_t D,
                       cudaStream_t s) {
  if (N == 0) return;
  k_embed_gather_bf16<<<static_cast<unsigned>(N), 256, 0, s>>>(
      ids, W, out, N, D);
}

void embed_scatter_add(const int64_t* ids, const __nv_bfloat16* grad_y,
                       float* grad_W, int64_t N, int64_t D,
                       cudaStream_t s) {
  if (N == 0) return;
  k_embed_scatter_add<<<static_cast<unsigned>(N), 256, 0, s>>>(
      ids, grad_y, grad_W, N, D);
}

}  // namespace zwt::k
