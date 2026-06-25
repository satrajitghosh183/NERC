/**
 * src/backend/tma_loads.cpp
 *
 * Host-side TMA descriptor builder (item W).
 *
 * sm_120 (Blackwell) supports TMA via cuTensorMapEncodeTiled. We wrap
 * that into a tensor-aware helper. Non-Blackwell falls through.
 *
 * The actual rewrite of the hot kernels (paged_attention, silu_mul,
 * fused_qkv_rope, fused_ffn, rms_norm) to consume these descriptors
 * is done in separate per-kernel commits.
 */

#include "olmo_cpp/backend/tma_loads.hpp"

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  include <cuda_runtime.h>
#  include <cuda.h>
#endif

#include <cstring>

namespace olmo_cpp {

void make_tma_descriptor(const torch::Tensor& tensor,
                          int64_t tile_rows, int64_t tile_cols,
                          void* out_descriptor) {
  std::memset(out_descriptor, 0, 128);
#if (defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS))
  // cuTensorMapEncodeTiled is the CUDA Driver API entry that builds a TMA
  // descriptor. We accept it failing on pre-12.0 toolchains by leaving
  // out_descriptor as zeros (kernels then bypass the TMA path).
  if (!tensor.is_cuda() || tensor.dim() < 2) return;
  CUtensorMap* map = static_cast<CUtensorMap*>(out_descriptor);
  CUtensorMapDataType dt;
  switch (tensor.scalar_type()) {
    case torch::kBFloat16: dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; break;
    case torch::kFloat16:  dt = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;  break;
    case torch::kFloat32:  dt = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;  break;
    default: return;  // unsupported dtype
  }
  // TMA tensors support up to rank 5 (CUDA driver limit). The CUDA headers
  // don't expose a portable macro for this across the 12.x series, so pin the
  // constant rather than depend on CU_TENSOR_MAP_MAX_TENSOR_RANK (which is not
  // declared in all toolkits, e.g. 12.1).
  constexpr int kTmaMaxRank = 5;
  const int rank = static_cast<int>(tensor.dim());
  if (rank > kTmaMaxRank) return;  // leave descriptor zeroed; kernel bypasses TMA
  cuuint64_t global_dim[kTmaMaxRank] = {};
  cuuint64_t global_stride[kTmaMaxRank - 1] = {};
  cuuint32_t box_dim[kTmaMaxRank] = {};
  cuuint32_t element_stride[kTmaMaxRank] = {};
  for (int i = 0; i < rank; ++i) {
    global_dim[i] = static_cast<cuuint64_t>(tensor.size(rank - 1 - i));
    box_dim[i] = (i == 0) ? static_cast<cuuint32_t>(tile_cols)
                          : (i == 1 ? static_cast<cuuint32_t>(tile_rows) : 1);
    element_stride[i] = 1;
  }
  for (int i = 1; i < rank; ++i) {
    global_stride[i - 1] =
        static_cast<cuuint64_t>(tensor.stride(rank - 1 - i)) * tensor.element_size();
  }
  CUresult res = cuTensorMapEncodeTiled(
      map, dt, rank,
      tensor.data_ptr(),
      global_dim, global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  (void)res;
#else
  (void)tensor; (void)tile_rows; (void)tile_cols;
#endif
}

bool device_supports_tma(torch::Device device) {
#if (defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS))
  if (!device.is_cuda()) return false;
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device.index());
  return props.major >= 12;  // sm_120 = Blackwell
#else
  (void)device;
  return false;
#endif
}

}  // namespace olmo_cpp
