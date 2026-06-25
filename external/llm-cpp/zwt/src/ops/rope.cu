// Rotary Position Embedding apply (half-split convention).
//
// For each (b, s, h, i) with i in [0, D/2):
//   lo = x[b, s, h, i]
//   hi = x[b, s, h, i + D/2]
//   c  = table[s, i]
//   sn = table[s, i + D/2]   (negated on backward)
//   x[b, s, h, i]        = lo * c - hi * sn
//   x[b, s, h, i + D/2]  = hi * c + lo * sn

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zwt::ops::k {

namespace {

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__global__ void k_rope_apply(__nv_bfloat16* x, const float* tab,
                             int64_t B, int64_t S, int64_t H, int64_t D,
                             bool inverse) {
  const int64_t half = D >> 1;
  // Grid: blockIdx.x = token (b*S+s)*H + h; blockIdx.y * blockDim.x + threadIdx.x = i
  int64_t token = blockIdx.x;
  int64_t i = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  if (i >= half) return;

  int64_t bs = token / H;
  int64_t s  = bs % S;

  int64_t base = token * D;
  float c  = tab[s * D + i];
  float sn = tab[s * D + i + half];
  if (inverse) sn = -sn;

  float lo = bf_to_f(x[base + i]);
  float hi = bf_to_f(x[base + i + half]);
  x[base + i]        = f_to_bf(lo * c - hi * sn);
  x[base + i + half] = f_to_bf(hi * c + lo * sn);
}

}  // namespace

void rope_apply_bf16(__nv_bfloat16* x, const float* tab,
                     int64_t B, int64_t S, int64_t H, int64_t D,
                     bool inverse, cudaStream_t s) {
  int64_t tokens = B * S * H;
  int64_t half = D >> 1;
  int block = half >= 256 ? 256 : (half >= 128 ? 128 : 64);
  dim3 grid(static_cast<unsigned>(tokens),
            static_cast<unsigned>((half + block - 1) / block));
  k_rope_apply<<<grid, block, 0, s>>>(x, tab, B, S, H, D, inverse);
}

}  // namespace zwt::ops::k
