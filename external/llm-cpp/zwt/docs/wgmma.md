# Hopper WGMMA + TMA GEMM path

## What this is

A CUTLASS 3.x–driven GEMM kernel that issues `wgmma.mma_async` (the
Hopper warp-group matrix-multiply-accumulate) with TMA (Tensor Memory
Accelerator) async loads, for BF16×BF16 → BF16 with FP32 accumulate.
Used by the Q/K/V/out projections and the SwiGLU gate+up / down
projections — i.e. every `ops::gemm` call on an sm_90 device with BF16
tensors that has `M, N, K` all divisible by 8. Every transformer
projection shape we train satisfies that.

`cuBLAS` on H100 reaches ~75–85% of peak TFLOPS for BF16 GEMMs. A
well-tuned WGMMA kernel closes most of the remaining gap by:

1. Using TMA to overlap the next K-tile's global→shared load with the
   current tile's MMA — removes the load-latency bubble that cuBLAS
   pays per K-step.
2. Warp-specialized producer/consumer pairs (TMA warps feed MMA warps)
   so the MMA path never waits on address generation.
3. Larger effective tiles (128×128×64 cooperative) that amortize the
   epilogue.

Typical measured gains on H100: +5–25% over cuBLAS for Linear shapes
at M=16384, depending on N/K. The bench tool reports per-shape numbers.

## Build

Opt-in. Off by default so CPU builds and older-GPU builds (sm_80/89)
are untouched:

```bash
cmake -S . -B build \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DZWT_USE_WGMMA=ON
cmake --build build -j
```

Configure does a shallow clone of CUTLASS v3.5.1. Only the headers are
consumed — we do not invoke CUTLASS's own CMake, so no profiler/test
bloat.

WGMMA instructions live in a separate static lib (`zwt_wgmma`) compiled
with `-gencode=arch=compute_90a,code=sm_90a`. The `a` suffix is
mandatory — plain `sm_90` does not emit the WGMMA opcodes.

## Runtime dispatch

`ops::gemm(...)` internally chooses between cuBLAS and WGMMA:

```cpp
if (ZWT_DISABLE_WGMMA env unset       &&
    wgmma_available()                 &&   // sm_90 + compiled in
    a,b,c are BF16                    &&
    M,N,K multiples of 8              &&
    !(transa && transb))
    -> gemm_wgmma(...);
else
    -> cuBLAS;
```

No caller changes required — every existing Linear forward/backward
call picks up the WGMMA path automatically.

To force cuBLAS without rebuilding (for A/B benching or falling back on
a regression), set `ZWT_DISABLE_WGMMA=1` in the environment. The
dispatch reads `getenv` per call, so you can toggle mid-process.

## Layouts

Three CUTLASS instantiations cover every layout `Linear` issues:

| transa | transb | Case                            | CUTLASS LayoutA | LayoutB |
|--------|--------|---------------------------------|-----------------|---------|
| false  | true   | `Y = X @ W^T`  (forward)        | RowMajor        | ColMajor|
| false  | false  | `dX = dY @ W`  (backward)       | RowMajor        | RowMajor|
| true   | false  | `dW = dY^T @ X`(backward)       | ColMajor        | RowMajor|

The TT combination is not wired (no call site needs it) — calling it
throws.

## Test

```bash
./build/zwt_wgmma_tests
```

Compares CUTLASS-WGMMA output against cuBLAS for all three layouts on
random BF16 inputs plus a dispatch test that verifies `ops::gemm`
actually routes to WGMMA on sm_90 + BF16. Tolerance is
`sqrt(K) * 2e-2` max-abs, which catches wiring bugs (wrong stride,
swapped transpose) while staying above BF16 mma noise on large
contractions. On non-sm_90 or non-WGMMA builds the test reports SKIP.

## Bench

```bash
./build/zwt_wgmma_bench                 # default shapes
./build/zwt_wgmma_bench 16384 4096 4096 # single shape
./build/zwt_wgmma_bench --iters 50
```

Emits CSV:

```
shape,m,n,k,cublas_ms,wgmma_ms,cublas_tflops,wgmma_tflops,speedup
1B-attn-proj,16384,2048,2048,0.274,0.251,502.8,548.7,1.091
7B-ffn-up,16384,11008,4096,2.881,2.432,512.3,606.9,1.185
...
```

Each row runs the same `ops::gemm` call, toggling `ZWT_DISABLE_WGMMA`
between phases so the cuBLAS measurement is exactly what the framework
would use in production.

## Scope

Three layouts; BF16 only; `M, N, K` divisible by 8. No FP16/F32
support yet (Linear uses BF16 end-to-end). No batched WGMMA — attention
sdpa's internal Q·K^T / P·V matmuls still use cuBLAS strided batched,
because their batch dim is small and `gemm_batched` on CUTLASS adds a
full second set of instantiations. When the FlashAttention-2 CUDA
kernel lands in `flash_attn.cu`, it will issue its own WGMMA directly
rather than go through `ops::gemm`.
