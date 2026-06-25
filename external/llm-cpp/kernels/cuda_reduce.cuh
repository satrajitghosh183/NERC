/**
 * kernels/cuda_reduce.cuh
 *
 * Warp + block reductions used by every attention-style kernel in the
 * fast-inference path (flash_decode, paged_attention, sparse_attn_decode).
 *
 * Why this exists:
 *   The original kernels reduced thread-local Q·K dot products via
 *   `atomicAdd(&sh_score, local)`, which on H100 serializes on shared-
 *   memory bank 0 across the whole block. For our block size of 128
 *   threads that's 128 atomic ops competing on one bank — easily 10-20%
 *   of attention kernel time at long context. Warp shuffles + a small
 *   per-warp shared buffer reduces it to 5 shfl + 1 store + 1 sync.
 *
 * Usage (block size must be ≤ 1024 = 32 warps):
 *   __shared__ float warp_sums[32];   // up to 32 warps in a block
 *   float local = ...;
 *   float total = block_reduce_sum(local, threadIdx.x, blockDim.x, warp_sums);
 *   // every thread now has `total`; one __syncthreads() was used internally.
 */

#pragma once

#include <cuda_runtime.h>

namespace olmo_cpp {

__device__ __forceinline__ float warp_reduce_sum(float val) {
  // Full-warp butterfly. mask = 0xffffffff = all 32 lanes participating.
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

/// Block-wide sum reduction.
///   warp_buf : shared-memory scratch sized for max(blockDim.x / 32) warps.
///              Caller must allocate and pass via shared (e.g. extern shared
///              or a __shared__ array). 32 floats covers up to a 1024-thread
///              block, which is the CUDA hardware limit.
/// Returns the sum on every thread after one __syncthreads().
__device__ __forceinline__ float block_reduce_sum(
    float val, int tid, int block_size, float* warp_buf) {
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // Phase 1: warp-local reduction.
  val = warp_reduce_sum(val);
  if (lane == 0) warp_buf[warp] = val;
  __syncthreads();
  // Phase 2: warp 0 reduces the per-warp partials. n_warps is at most 32
  // for a 1024-thread block, fitting in one warp.
  const int n_warps = (block_size + 31) >> 5;
  if (warp == 0) {
    float v = (lane < n_warps) ? warp_buf[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_buf[0] = v;
  }
  __syncthreads();
  return warp_buf[0];
}

}  // namespace olmo_cpp
