// kernels/mma_sync.cuh
//
// Tensor-core matmul helpers (item 2). Inline-PTX wrappers around the
// mma.sync (sm_80+) and wgmma.mma_async (sm_90+) instructions. The
// fused QKV+RoPE and FFN macro kernels include this header to upgrade
// their FMA-loop matmul to tensor-core throughput.
//
// Tile shapes wired here:
//   mma_sync_bf16_16x16x16 — sm_80+, single-warp bf16 → fp32 accum
//   wgmma_bf16_64x64x16    — sm_90+, warpgroup async, bf16 → fp32 accum
//
// Usage pattern (per output tile of [16, 16]):
//   half4 a_frag[2], b_frag[2];
//   load_a_tile_bf16(a_frag, smem_a + tile_row * 16);
//   load_b_tile_bf16(b_frag, smem_b + tile_col * 16);
//   float c_frag[4] = {0,0,0,0};
//   mma_sync_bf16_16x16x16(c_frag, a_frag, b_frag);
//   store_c_tile_fp32(smem_c + (tile_row*16 + tile_col), c_frag);
//
// Callers in fused_qkv_rope.cu and fused_ffn.cu replace the inner
// FMA loop with a tiled mma_sync invocation. The tile/warp layout is
// the same shape the existing online-softmax kernels already use.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(__CUDA_ARCH__)

#if (__CUDA_ARCH__ >= 800)

/// 16x16x16 BF16 mma: D = A*B + C.
///   A:    [16, 16] BF16, fragment a (4 b16x2 per thread, 8 elements/thread)
///   B:    [16, 16] BF16
///   C/D:  [16, 16] FP32 (4 floats/thread)
/// One warp computes one [16, 16] output tile.
__device__ __forceinline__ void mma_sync_bf16_16x16x16(
    float (&c)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

#endif  // sm_80+

#if (__CUDA_ARCH__ >= 900)

/// 64x64x16 BF16 wgmma (warpgroup async). One warpgroup (4 warps,
/// 128 threads) issues this; the accumulator d_frag is per-thread
/// fp32 holding 32 elements.
__device__ __forceinline__ void wgmma_bf16_64x64x16(
    float (&d)[32],
    uint64_t a_desc,
    uint64_t b_desc) {
  asm volatile("wgmma.fence.sync.aligned;\n");
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
        "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
        "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(a_desc), "l"(b_desc));
  asm volatile("wgmma.commit_group.sync.aligned;\n");
}

#endif  // sm_90+

#endif  // __CUDA_ARCH__
