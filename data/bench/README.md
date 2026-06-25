# Benchmark data

Reproducible measurements emitted by the `bench_*` executables. Regenerate with:

```
cmake --build build
OMNI_DATA_DIR=data/bench ./build/bench_trace_codec
```

## trace_codec — divergence-aware columnar codec (PLAN.md §2.1 / §11 item 1)

1,048,576 invocations/column, warp = 32, Apple M4 Pro, single-threaded, `-O3`.

| Regime (SIMT pattern)        | bits/value | ratio vs raw32 | residual-entropy bound | order-0 entropy | enc ns/val | dec ns/val |
|------------------------------|-----------:|---------------:|-----------------------:|----------------:|-----------:|-----------:|
| constant                     |      1.22  |    26.3×       |  0.00                  |  0.00           |  2.2       |  1.1       |
| quad_coherent (derivatives)  |      7.19  |     4.5×       |  5.70                  |  6.00           |  2.5       |  0.9       |
| warp_gradient (smooth)       |      6.63  |     4.8×       |  5.04                  | 16.54           |  2.5       |  0.9       |
| float_smooth (XOR-coded)     |     25.10  |     1.3×       | 19.44                  | 19.98           |  3.4       |  1.7       |
| random32 (incompressible)    |     32.03  |     1.0×       | 19.53                  | 20.00           |  3.6       |  1.6       |
| gradient + 50% divergence    |      9.57  |     3.3×       |  4.99                  | 16.54           |  7.0       |  3.7       |
| gradient + 10% divergence    |     25.94  |     1.2×       |  4.24                  | 16.54           |  2.9       |  1.4       |

**Reading the numbers.**
- On coherent SIMT data the codec lands within ~1.5 bits/value of the residual-entropy
  bound `H(value | warp base)` — the per-warp 32-bit base + 6-bit width is the overhead
  (~1.2 bits/value at warp=32).
- The per-warp **hybrid mode flag** caps the worst case: incompressible data costs raw +
  ~1 bit/warp (32.03 bpv) instead of expanding.
- Everything is **lossless** (`lossless=1` for every row).

**Known weak spots (future work, honest):**
- `float_smooth` only 1.3× — fixed-width-per-warp bit-packing on float XOR residuals is
  wasteful when one lane's XOR is large. A Gorilla-style leading/trailing-zero float
  scheme or exponent/mantissa column split would help.
- High divergence (`div10`) is dominated by the 1-bit/lane mask + per-warp base over few
  active lanes; an RLE/bitmap-index mask would reduce this.
