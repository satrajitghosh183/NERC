#pragma once

/**
 * include/olmo_cpp/backend/tma_loads.hpp
 *
 * Blackwell TMA (Tensor Memory Accelerator) async-load helpers — item W.
 *
 * On sm_120 (Blackwell), TMA does asynchronous global → shared memory
 * copies via descriptor handles. Throughput for the relevant kernels
 * (paged_attention, silu_mul, fused QKV, fused FFN, rms_norm) jumps
 * 2-3× because the loads run concurrent with the compute.
 *
 * This header declares the host-side helpers that build a TMA
 * descriptor for a given (tensor, tile shape) pair. Kernels include
 * the descriptor at launch time and issue cp.async.bulk.tensor.shared
 * (or the cuTeNSOR-managed wrapper) inside.
 *
 * The actual rewrite of the existing kernels to use TMA is a separate
 * file-by-file pass; this header is the surface they call into.
 * Gated under __CUDA_ARCH__ >= 1200 (sm_120) at the call site;
 * non-Blackwell falls through to the existing __ldg path.
 */

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#  define OLMO_HAS_TMA 1
#else
#  define OLMO_HAS_TMA 0
#endif

/// Build a TMA descriptor for `tensor` covering tile [tile_rows, tile_cols].
/// Returned descriptor is opaque (cuTensorMapEncodeTiled produces a 128-byte
/// blob that the kernel passes to cp.async.bulk.tensor).
/// Host-side helper; CUDA-only.
void make_tma_descriptor(const torch::Tensor& tensor,
                          int64_t tile_rows, int64_t tile_cols,
                          void* out_descriptor /*[128 bytes]*/);

/// Helper: returns true iff the current device supports TMA (sm_120+).
bool device_supports_tma(torch::Device device);

}  // namespace olmo_cpp
