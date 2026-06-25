// Fused multi-tensor AdamW.
//
// One kernel handles all parameters of a model in a single launch. Each block
// picks up one "chunk" of work; chunks are strided across all tensors so the
// SM occupancy is high even when per-tensor sizes are wildly uneven (which is
// always the case: embedding >> per-layer weight >> per-layer bias).
//
// Param dtype is bf16; gradient, momentum, variance are fp32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace zwt::optim::k {

namespace {

constexpr int kChunkSize = 4096;  // elements per block

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__global__ void adamw_kernel(
    void** p_ptrs, float** g_ptrs, float** m_ptrs, float** v_ptrs,
    const int64_t* sizes,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2_sqrt,
    int64_t total_chunks,
    const int64_t* chunk_tensor_idx, const int64_t* chunk_offsets) {
  int64_t chunk_id = int64_t(blockIdx.x);
  if (chunk_id >= total_chunks) return;

  int t = static_cast<int>(chunk_tensor_idx[chunk_id]);
  int64_t off = chunk_offsets[chunk_id];
  int64_t size = sizes[t];
  int64_t remain = size - off;
  int64_t work = remain < kChunkSize ? remain : kChunkSize;

  __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(p_ptrs[t]) + off;
  float* g = g_ptrs[t] + off;
  float* m = m_ptrs[t] + off;
  float* v = v_ptrs[t] + off;

  // step_size folds bc2_sqrt / bc1 into lr so we avoid per-element division.
  float step_size = lr * bc2_sqrt / bc1;
  float eps_scaled = eps * bc2_sqrt;

  for (int64_t i = threadIdx.x; i < work; i += blockDim.x) {
    float pv = bf_to_f(p[i]);
    float gv = g[i];
    float mv = beta1 * m[i] + (1.f - beta1) * gv;
    float vv = beta2 * v[i] + (1.f - beta2) * gv * gv;
    m[i] = mv;
    v[i] = vv;
    // AdamW: decoupled decay on the parameter itself.
    pv = pv - lr * wd * pv - step_size * mv / (__fsqrt_rn(vv) + eps_scaled);
    p[i] = f_to_bf(pv);
  }
}

}  // namespace

// Launcher. Caller owns the chunk plan buffers on device (set once at AdamW
// construction, reused on every step). No process-scope state here.
void adamw_multi_tensor_bf16(
    void** p_ptrs, float** g_ptrs, float** m_ptrs, float** v_ptrs,
    const int64_t* sizes,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2_sqrt,
    int64_t total_chunks,
    const int64_t* d_chunk_tensor_idx, const int64_t* d_chunk_offsets,
    cudaStream_t s) {
  if (total_chunks == 0) return;
  dim3 grid(static_cast<unsigned>(total_chunks));
  dim3 block(256);
  adamw_kernel<<<grid, block, 0, s>>>(
      p_ptrs, g_ptrs, m_ptrs, v_ptrs, sizes,
      lr, beta1, beta2, eps, wd, bc1, bc2_sqrt,
      total_chunks, d_chunk_tensor_idx, d_chunk_offsets);
}

}  // namespace zwt::optim::k
