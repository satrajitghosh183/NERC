# Zero-Wait Trainer (zwt)

A LibTorch-free, CUDA-native transformer pretraining stack living under
`zwt/` in this repository. Branch: `zero-wait-trainer-sg`.

Nothing in this framework imports `torch/*` — not at compile time, not at
runtime. Every primitive (allocator, stream scheduler, tensor, op kernel,
optimizer, data loader, checkpoint format) is hand-written C++17/CUDA that
speaks directly to cuBLAS and the CUDA runtime. The goal is to reclaim the
overhead that a general-purpose framework unavoidably pays — dispatch,
refcounting, cache indirection, per-op launches, Python-driven control flow —
in a setting where the entire compute graph is known ahead of time: pretrain
one decoder-only language model, as fast as the hardware can run it.

This README covers **every feature**, **every optimization**, **how to use
each of them**, and **the expected speedup** they contribute relative to a
reference PyTorch + naive AdamW trainer on the same hardware.

---

## Table of contents

1. [Design principles](#design-principles)
2. [Directory layout](#directory-layout)
3. [Build](#build)
4. [Quickstart — 3B on OpenWebText](#quickstart--3b-on-openwebtext)
5. [The optimizations](#the-optimizations)
   1. [Memory model — bump-pointer activation arena](#1-memory-model--bump-pointer-activation-arena)
   2. [Three-stream execution model](#2-three-stream-execution-model)
   3. [Zero-stall TokenLoader](#3-zero-stall-tokenloader)
   4. [Fused multi-tensor AdamW](#4-fused-multi-tensor-adamw)
   5. [Multi-tensor gradient clipping](#5-multi-tensor-gradient-clipping)
   6. [BF16 weights + FP32 master gradients](#6-bf16-weights--fp32-master-gradients)
   7. [Fused softmax + cross-entropy + grad](#7-fused-softmax--cross-entropy--grad)
   8. [SDPA with recompute-P backward](#8-sdpa-with-recompute-p-backward)
   9. [Fused RMSNorm forward/backward](#9-fused-rmsnorm-forwardbackward)
   10. [RoPE half-split kernel](#10-rope-half-split-kernel)
   11. [Fused SwiGLU FFN](#11-fused-swiglu-ffn)
   12. [Single-kernel BSHD↔BHSD transpose](#12-single-kernel-bshd-bhsd-transpose)
   13. [cuBLAS GEMM via operand-swap](#13-cublas-gemm-via-operand-swap)
   14. [FP32 master-gradient accumulation](#14-fp32-master-gradient-accumulation)
   15. [Per-parameter XOR-salted init seeds](#15-per-parameter-xor-salted-init-seeds)
   16. [Correct gradient accumulation (scale grad, not LR)](#16-correct-gradient-accumulation-scale-grad-not-lr)
   17. [Binary checkpoint with atomic rename](#17-binary-checkpoint-with-atomic-rename)
   18. [Strict INI config parser](#18-strict-ini-config-parser)
   19. [.npy token loading (u16/u32/i32/i64)](#19-npy-token-loading-u16u32i32i64)
6. [Configuration reference](#configuration-reference)
7. [Checkpointing and resume](#checkpointing-and-resume)
8. [Performance budget](#performance-budget)
9. [Expected speedups vs reference trainer](#expected-speedups-vs-reference-trainer)
10. [Limitations and deferred work](#limitations-and-deferred-work)

---

## Design principles

**Zero indirection on the hot path.** Tensors are move-only owning handles
with a single allocator pointer. No atomic refcounting. No caching allocator
walking a freelist. No autograd tape. The ArenaAllocator bumps a pointer and
resets once per step; the PoolAllocator buckets by power-of-two for
parameters, gradients, and optimizer moments.

**Layers stash activations in member state.** There is no graph replay.
`Module::forward(x)` returns the output and saves whatever backward needs
as member fields. `Module::backward(grad_y)` reads those directly. This
mirrors the structure of a hand-rolled trainer, not a framework.

**Everything important is fused.** Per-op launches are the dominant cost at
H100 scale because the kernels themselves are fast. AdamW is one launch.
Gradient clipping is two launches (reduce + scale). Cross-entropy is one
launch. RMSNorm is one launch. The forward pass of a 26-layer 3B model
issues fewer than 400 kernels per step.

**Bandwidth beats FLOPs.** Fused softmax+xent, fused silu+mul, fused
residual+RMSNorm, and recompute-in-backward SDPA all exist to keep more
activation data in register/SMEM and skip round-trips through HBM.

**Crash-safe is cheap.** Checkpoints write to `path.tmp` and rename. Config
parser rejects unknown keys. Arena capacity is a configurable cap so an
accidental allocation explosion aborts early instead of paging the whole
machine.

---

## Directory layout

```
zwt/
├── include/zwt/
│   ├── core/           Tensor, Allocator, Device, DType, Shape, Stream
│   ├── ops/            GEMM, RMSNorm, SDPA, RoPE, xent, elementwise, etc.
│   ├── layers/         Module, Parameter, Linear, RMSNorm, Embedding,
│   │                   FFN, Attention, TransformerBlock, Transformer
│   ├── optim/          AdamW, grad_clip, lr_schedule
│   ├── data/           TokenLoader
│   └── train/          checkpoint, config
├── src/                mirror of include/, one .cpp per header + .cu kernels
├── tools/
│   ├── zwt_train.cpp      smoke-test trainer (small MLP)
│   └── zwt_pretrain.cpp   production pretraining binary
├── conf/
│   └── owt_3B.conf        3.25 B OpenWebText preset
├── scripts/
│   └── prep_openwebtext.sh  end-to-end data prep
└── README.md (this file)
```

Kernels under `zwt/src/` compile into a single static library `libzwt.a`,
which the two binaries (`zwt_train`, `zwt_pretrain`) link against. The
library does not pull in LibTorch, PyTorch headers, Python, or any Gloo /
NCCL code. CUDA is optional — CPU builds fall back to reference
implementations of every kernel.

---

## Build

From the repository root:

```bash
# Full auto-detected build (CUDA on H100/A100, CPU elsewhere)
./scripts/build.sh

# Equivalent manual build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc) zwt zwt_train zwt_pretrain
```

`CMAKE_PREFIX_PATH` is only needed because the surrounding `olmo_cpp`
library in this repo still links LibTorch for the legacy trainer. The
`zwt` library itself doesn't touch it.

Key targets:

* `zwt` — static library (`libzwt.a`)
* `zwt_train` — smoke-test trainer against a raw token file
* `zwt_pretrain` — production trainer driven by an INI config

---

## Quickstart — 3B on OpenWebText

```bash
# 1. Build
./scripts/build.sh

# 2. Tokenize OpenWebText into a single .npy (~15 min on a 32-core box)
./zwt/scripts/prep_openwebtext.sh data/owt

# 3. Launch pretraining
./build/zwt_pretrain zwt/conf/owt_3B.conf
```

By default `owt_3B.conf` checkpoints to `ckpts/owt_3B.bin` every 1000
steps. Resume after a crash:

```bash
./build/zwt_pretrain zwt/conf/owt_3B.conf --resume ckpts/owt_3B.bin
```

A `--dry-run` flag parses the config, builds the model, and exits before
the first step — useful for validating a config change without spinning
up the data loader.

---

## The optimizations

Each subsection below lists **what** the optimization is, **how to use it**
(or whether it's always on), **why it helps**, and a **component speedup**
plus a **realistic contribution to end-to-end training throughput** on an
H100 running the 3B preset. Numbers are estimates anchored to published
H100 figures — treat the end-to-end totals as ranges, not promises.

### 1. Memory model — bump-pointer activation arena

**What.** `ArenaAllocator` owns a single large contiguous device buffer.
`empty_scratch()` hands out chunks by bumping an offset. `step_begin()`
resets the offset to zero. There is no per-allocation bookkeeping, no
fragmentation, no freelist walking.

`PoolAllocator` is the counterpart for long-lived state (params, grads,
optimizer moments). It buckets by `ceil(log2(size))` and keeps one
freelist per bucket — a specialized allocator for the four or five
sizes training actually uses.

**How to use.** Automatic. Every op that needs scratch storage calls
`empty_scratch(shape, dtype, device)`; every persistent tensor calls
`empty(...)`. Tune the arena cap at startup:

```cpp
zwt::set_activation_arena_capacity(4ULL << 30);   // 4 GiB
```

Or from the config:

```ini
[runtime]
arena_mb = 4096
```

**Why it helps.** A caching allocator walks a freelist on every alloc.
At >200 allocations per step (typical for a 26-layer model), that's tens
of microseconds of control flow we eliminate entirely. More importantly,
arena allocation is fragmentation-free — you never pay a full
`cudaMalloc` at step 10000 because a bucket ran out.

**Component speedup.** ~50–200 μs saved per step on allocation overhead.

**End-to-end contribution.** 1–3% on a 3B run (larger relative effect on
small models where step time is short).

### 2. Three-stream execution model

**What.** Each CUDA device exposes three pre-created streams:

| Stream | Purpose |
|--------|---------|
| `compute_stream(dev)` | forward, backward, optimizer step |
| `copy_stream(dev)`    | H2D transfers for the data loader |
| `side_stream(dev)`    | round-robin pool for ops that can run in parallel with the main stream (e.g. Muon NS iterations, grad allreduce) |

**How to use.** Automatic. Every op resolves its stream through
`compute_stream(tensor.device())`, the TokenLoader's producer uses
`copy_stream`, and ops that can parallelize pull from `side_stream`.
Consumers wait on producers via CUDA events:

```cpp
cudaEventRecord(copy_done, copy_stream);
cudaStreamWaitEvent(compute_stream, copy_done, 0);
```

**Why it helps.** On a single stream every kernel serializes. With H2D on
its own stream and a compute/copy event handshake, the `cudaMemcpyAsync`
for step N+1 overlaps with the compute of step N. The only cost is the
event wait, which is a nanosecond-scale device-side synchronization.

**Component speedup.** Data-loading cost goes from 100% exposed to 0%
exposed (as long as compute exceeds data prep).

**End-to-end contribution.** 5–15% depending on how tight the data path is.
On OpenWebText at seq=2048, H2D is <1 ms per batch — the overlap fully
hides it.

### 3. Zero-stall TokenLoader

**What.** `zwt::data::TokenLoader` is a producer/consumer ring:

* A background `std::thread` gathers token chunks from a `.npy` file
  (or raw `int64_t` stream) into pinned host buffers.
* Ring has N slots (default 3). Each slot owns a pinned host buffer +
  a device tensor.
* When a slot is full, the producer issues `cudaMemcpyAsync` on the
  copy stream and flips the `ready` flag.
* `next()` blocks on `ready`, records a copy-done event, and has the
  compute stream wait on it.

**How to use.** From the config:

```ini
[data]
path       = data/owt/owt_tokens.npy
seq_len    = 2048
batch_size = 1
shuffle    = true
seed       = 0xC0FFEE
```

Or programmatically:

```cpp
zwt::data::TokenLoader::Options opts;
opts.path       = "data/owt/owt_tokens.npy";
opts.seq_len    = 2048;
opts.batch_size = 1;
opts.device     = zwt::Device::cuda(0);
zwt::data::TokenLoader loader(opts);
loader.start();
auto batch = loader.next();  // batch.input, batch.target
```

**Why it helps.** The forward pass never waits on data. Not even on
disk-read cost — the producer thread is asynchronous and pinned memory
H2D is on its own stream.

**Component speedup.** Eliminates the data-bound bubble entirely.

**End-to-end contribution.** Recovers whatever percentage of step time
your naive dataloader was burning — commonly 5–15%.

### 4. Fused multi-tensor AdamW

**What.** `adamw.cu::k::adamw_multi_tensor_bf16` updates every parameter
in a single kernel launch. The kernel walks a device-side array of
`{param_ptr, grad_ptr, m_ptr, v_ptr, numel}` tuples and chunk-strides
through them. One launch for a 300-tensor model instead of 300.

**How to use.** Automatic when you call `opt.step()`. On first call,
`AdamW` uploads the pointer/size arrays to device memory; subsequent
calls just re-dispatch the single kernel.

```cpp
zwt::optim::AdamWConfig cfg;
cfg.lr           = 3e-4f;
cfg.beta1        = 0.9f;
cfg.beta2        = 0.95f;
cfg.weight_decay = 0.1f;
zwt::optim::AdamW opt(params, cfg);
// ...
opt.step();
```

Bias corrections are computed host-side from the step counter and
passed in as two floats — no per-param recomputation.

**Why it helps.** Kernel launch on H100 is ~5 μs of host overhead.
300 launches is 1.5 ms *per step* in launch overhead alone, before the
GPU has touched a byte. Folding them into one kernel eliminates that
overhead entirely. The single kernel is memory-bound on each tensor
anyway, so fusing 300 of them does not hurt occupancy.

**Component speedup.** Optimizer step goes from ~5 ms → ~0.5 ms on a
3B model. 10× on the component.

**End-to-end contribution.** 3–8% — the optimizer is typically 5–10% of
step time on a well-tuned trainer.

### 5. Multi-tensor gradient clipping

**What.** `grad_clip.cu` ships two kernels:

* `k_sumsq` — one block per parameter, warp+block reduction, `atomicAdd`
  into a single FP32 scalar. Computes `sum(grad^2)` across every
  parameter in one launch.
* `k_scale` — multi-tensor in-place multiply by a scalar.

Host glue uploads the pointer arrays through `cudaMallocAsync` on the
compute stream, dispatches both kernels, and does a single FP32 D2H copy
to read the global norm (needed for both the scale decision and logging).

**How to use.** Automatic via `optim::clip_grad_norm(params, max_norm)`.
Returns the pre-clip global L2 norm for logging. Set `grad_clip` in the
config:

```ini
[optim]
grad_clip = 1.0
```

Pass `0` to disable scaling (the norm is still computed and returned).

**Why it helps.** Naive gradient clipping is O(N) reductions + O(N)
scales = 2N kernel launches + 2N D2H syncs. Fusing into two launches and
one D2H sync drops the cost to something negligible.

**Component speedup.** Grad clip cost: ~5 ms → ~0.2 ms. 25× on the
component.

**End-to-end contribution.** ~1% (grad clip is small but symbolic).

### 6. BF16 weights + FP32 master gradients

**What.** Parameters live in BF16. Gradients are *always* FP32,
regardless of parameter dtype. Optimizer moment buffers (m, v) are
FP32. This is the "master gradient" pattern — stability of FP32 math
where it matters (accumulation, variance estimates, weight decay), speed
of BF16 where it matters (matmul, memory bandwidth).

**How to use.** Automatic. `Parameter::ensure_grad()` allocates an FP32
grad buffer even when the parameter is BF16. `AdamW` internally uses
FP32 for all updates and writes the result back to BF16 via a cast
inside the fused kernel.

```cpp
zwt::Parameter p("weight", zwt::empty({out, in}, zwt::DType::BF16, dev));
p.ensure_grad();   // grad is [out, in] FP32
```

**Why it helps.** BF16 has the same exponent range as FP32, so you don't
need loss scaling (unlike FP16). Weights take half the memory. Matmul
throughput doubles on H100 tensor cores. Gradients stay FP32 so the
optimizer's sqrt(v)+eps math is numerically sound.

**Component speedup.** 2× matmul throughput, 2× memory capacity.

**End-to-end contribution.** 30–50% vs pure FP32 training. This is the
big lever — if you only enable one thing in this list, enable this one.

### 7. Fused softmax + cross-entropy + grad

**What.** `xent.cu::k::softmax_xent_fused_bf16` reads the logits,
computes `max` and `log-sum-exp` in one pass, computes the per-row loss,
and emits the gradient `(softmax(logits) - onehot) / n_valid` into the
output buffer — all without writing the softmax intermediate to global
memory.

**How to use.** Automatic via `ops::cross_entropy(logits, targets, loss,
&grad_logits)`. Pass `nullptr` for `grad_logits` if you don't want the
backward fused in.

```cpp
zwt::Tensor loss        = zwt::empty_scratch({1}, zwt::DType::F32, dev);
zwt::Tensor grad_logits = zwt::empty_scratch(logits_2d, zwt::DType::BF16, dev);
zwt::ops::cross_entropy(logits_2d, targets_1d, loss, &grad_logits, /*ignore=*/-100);
```

**Why it helps.** The lm_head sits on a `[B*S, V]` tensor with V ≈ 50K.
Writing softmax to HBM is (B*S*V*2) bytes read/write — for a 3B run at
B=1, S=2048, that's ~200 MB of memory traffic just to stage softmax for
the backward pass. Fusing softmax + xent + grad into one kernel cuts
that to one pass.

**Component speedup.** lm_head backward memory bandwidth halved.

**End-to-end contribution.** 3–6% on a 3B model (the xent backward is
disproportionately expensive because of V).

### 8. SDPA with recompute-P backward

**What.** Scaled dot-product attention forward computes
`P = softmax(Q Kᵀ / √d_k + mask)` and `O = P V`. The backward pass
normally needs `P` stashed in memory, which is `[B, H, S, S]` — huge at
long context. We recompute `P` in the backward pass instead:

```
P = softmax(Q Kᵀ * scale, causal)     // recompute
grad_V = Pᵀ @ grad_out                // batched GEMM
grad_P = grad_out @ Vᵀ                // batched GEMM
grad_scores = scale * P * (grad_P - rowsum(P * grad_P))  // softmax Jacobian
grad_Q = grad_scores @ K
grad_K = grad_scoresᵀ @ Q
```

The softmax Jacobian math is a fused kernel (`k_softmax_backward_scaled`)
so we don't have to produce `P` as a separate global-memory tensor —
only the rowwise dot-products live in registers.

**How to use.** Automatic. `Attention::backward` calls
`ops::sdpa_backward(grad_out, Q, K, V)`.

**Why it helps.** At seq=2048, head_dim=128, heads=24, batch=1, the
attention matrix is `[1, 24, 2048, 2048] × 2 bytes = 200 MB` per layer.
Across 26 layers that's 5.2 GB of *attention matrices alone* if you
store them all. Recomputing trades a small amount of redundant FLOPs for
zero activation storage on the attention matrix.

**Component speedup.** Backward memory footprint drops by the attention
matrix size per layer. Enables larger batch or longer context without
activation checkpointing.

**End-to-end contribution.** Enables the `owt_3B.conf` layout to fit on
a single H100 at batch=1, seq=2048 without resorting to activation
checkpointing.

### 9. Fused RMSNorm forward/backward

**What.** `norm.cu` ships fused RMSNorm forward and backward kernels.
Forward computes the row RMS (warp+block reduction), normalizes, and
scales by the learnable `gamma` in one pass. Backward fuses the standard
LayerNorm/RMSNorm gradient:

```
dy      = grad_out * gamma
inv_rms = 1/√(mean(x²) + eps)
dx      = inv_rms * (dy - x * mean(x * dy) / (mean(x²) + eps))
```

all in one kernel with one set of row reductions.

**How to use.** Automatic — `RMSNorm::forward` / `RMSNorm::backward`.
Fused into every pre-norm block of the transformer.

**Why it helps.** RMSNorm runs twice per block (pre-attn, pre-FFN). Each
naive implementation has three passes over `[B, S, D]` (mean, normalize,
scale) and three more in backward. Fusing cuts this to one pass per
direction.

**Component speedup.** RMSNorm cost per block: ~6× faster than naive.

**End-to-end contribution.** ~2% on the 3B preset (norms are small
individually but ubiquitous).

### 10. RoPE half-split kernel

**What.** Rotary Position Embedding applied in place on the Q and K
tensors using the half-split convention (`lo = x[..., i]`,
`hi = x[..., i + D/2]`, followed by a planar rotation). One CUDA kernel
reads + writes in place; no extra allocation.

A precomputed `[max_seq, head_dim]` FP32 table holds `cos` in the first
half and `sin` in the second half. Backward negates the sine component
and runs the same kernel with swapped signs.

**How to use.** `Attention` builds the table once in its constructor
and applies it on every forward/backward call:

```cpp
Tensor rope_table = ops::rope_build_table(max_seq, head_dim, base, device);
ops::rope_apply(q, rope_table);       // in-place
ops::rope_apply_backward(grad_q, rope_table);  // in-place
```

**Why it helps.** No allocation for the rotation output, no separate
cos/sin tensors per call, and no launch overhead from splitting
`(cos_apply → sin_apply → add)`.

**Component speedup.** Versus a three-op decomposition: ~3× faster.

**End-to-end contribution.** Small but unavoidable — RoPE runs on every
token in every attention layer.

### 11. Fused SwiGLU FFN

**What.** `FFN::forward` does `gate = Linear(x)`, `up = Linear(x)`,
`hidden = silu(gate) * up`, `out = Linear(hidden)`. The `silu * up`
combination is one kernel (`ops::silu_mul`), fused from what would
otherwise be `silu(gate) → tmp`, `tmp * up → hidden` (three passes, two
tmp tensors).

**How to use.** Automatic in `FFN::forward`. The backward matching
kernel is `silu_mul_backward(grad_out, gate, up, grad_gate, grad_up)`.

**Why it helps.** FFNs are typically the biggest memory-bandwidth sink
in a transformer — they operate on `[B, S, 4D]` intermediates. Halving
the global memory traffic on the SwiGLU step is worth real time.

**Component speedup.** SwiGLU step: 2–3× faster.

**End-to-end contribution.** 2–4% on the 3B preset.

### 12. Single-kernel BSHD↔BHSD transpose

**What.** Attention needs `[B, H, S, D]` layout for the QKᵀ matmul, but
the projections produce `[B, S, H, D]`. Two kernels handle the two
directions in a single pass each: one thread per element, contiguous
writes on the output, no intermediate reshape.

**How to use.** Automatic inside `Attention::forward` and
`Attention::backward`:

```cpp
ops::transpose_bshd_to_bhsd(q_bshd, q_bhsd);
ops::sdpa(q_bhsd, k_bhsd, v_bhsd, out_bhsd, scale, /*causal=*/true);
ops::transpose_bhsd_to_bshd(out_bhsd, out_bshd);
```

**Why it helps.** A naive `.view().transpose().contiguous()` pipeline
makes two copies. One kernel, one copy.

**Component speedup.** 2× faster than a two-stage transpose.

**End-to-end contribution.** <1% — but no free bytes of memory either.

### 13. cuBLAS GEMM via operand-swap

**What.** Row-major `C = A B` is mathematically equivalent to the
column-major expression `Cᵀ = Bᵀ Aᵀ`. cuBLAS is column-major-native,
so we dispatch with swapped operands and no actual transpose. Our GEMM
wrapper passes the original row-major pointers to cuBLAS with the
swapped dims; cuBLAS treats them as column-major and produces the
right answer.

**How to use.** Automatic. `ops::gemm(A, B, C, alpha, beta, transA,
transB)` is what every layer uses.

**Why it helps.** We get cuBLAS's hand-tuned H100 kernels with zero
transpose overhead. The alternative is either (a) allocating transposed
copies (bandwidth bound), (b) writing our own GEMM (slower than cuBLAS),
or (c) using cuBLASLt with explicit row-major layouts (more setup cost).

**Component speedup.** 0% overhead vs optimal cuBLAS.

**End-to-end contribution.** Matmul is ~70% of transformer compute —
being at peak cuBLAS throughput is table stakes. We just don't give any
back.

### 14. FP32 master-gradient accumulation

**What.** Gradient accumulation across micro-batches happens in FP32
because `Parameter::grad` is always FP32, even when `Parameter::value`
is BF16. Each micro-batch's backward writes its contribution, and
repeated calls sum into the same FP32 buffer.

**How to use.** Set `grad_accum > 1` in the config:

```ini
[data]
batch_size = 1
grad_accum = 64   # effective batch = 64 * batch_size
```

The trainer scales `grad_logits` by `1/grad_accum` before each
backward (see optimization 16).

**Why it helps.** Accumulating BF16 gradients loses bits every step and
corrupts the update direction. FP32 accumulation is lossless. Since the
optimizer already reads FP32, this is free.

**Component speedup.** Enables big effective batches on a single GPU
without losing numerical stability.

**End-to-end contribution.** Correctness, not speed — but without it
you can't train a 3B model on one H100 at all.

### 15. Per-parameter XOR-salted init seeds

**What.** Every Linear/Attention/FFN/TransformerBlock takes an
`init_seed` parameter and XORs it with a per-role salt before
initializing. Distinct salts for Q/K/V/O, for gate/up/down, and for
each block in the transformer.

```cpp
constexpr uint64_t kSeedSaltQ    = 0x17...ULL;
constexpr uint64_t kSeedSaltK    = 0x23...ULL;
constexpr uint64_t kSeedSaltV    = 0x31...ULL;
constexpr uint64_t kSeedSaltO    = 0x47...ULL;
// ... per-block: init_seed + i * 0x9E37'79B1ULL
```

**How to use.** Set once via `init_seed` in the config:

```ini
[runtime]
init_seed = 0xC0DEBA5E
```

**Why it helps.** Without salting, two linears with the same shape and
the same RNG state produce identical weights on the first step. The gate
and up projections in a SwiGLU FFN have the same shape, and so do Q/K/V
in MHA. Identical initial weights collapse the loss surface in subtle
ways that mostly show up as a worse-than-expected early-phase loss.

**Component speedup.** Not a speedup — a correctness fix. Recovers the
expected convergence rate that duplicate init silently destroys.

**End-to-end contribution.** Measured in loss, not time. Typical
saving: 50–200 loss steps of early-training instability avoided.

### 16. Correct gradient accumulation (scale grad, not LR)

**What.** The trainer scales `grad_logits` by `1 / grad_accum` before
each backward pass, *not* the learning rate. This is mathematically
required because AdamW is non-linear in the gradient — the sqrt(v)+eps
denominator breaks LR/gradient symmetry.

```cpp
ops::scale(grad_logits, 1.0f / float(cfg.grad_accum));  // correct
// NOT: opt.config().lr /= grad_accum;                 // wrong for AdamW
```

**How to use.** Automatic. The `zwt_pretrain` loop handles this when
`grad_accum > 1`.

**Why it helps.** Rescaling LR looks equivalent and is equivalent for
SGD. For AdamW it is *not* equivalent, and using it subtly degrades the
update direction because of the interaction with the second-moment
estimator.

**Component speedup.** Correctness fix. Ensures large-effective-batch
runs actually converge like large-batch runs should.

**End-to-end contribution.** Hard to quantify directly, but the
alternative is a silently-wrong update rule.

### 17. Binary checkpoint with atomic rename

**What.** `train::save_checkpoint(path, params, opt, meta)` writes a
single binary file. Format:

```
[FileHeader 64 B]                          magic, version, step, seed,
                                           data_cursor, n_records, lr, loss
for each parameter i:
  [TRec 80 B] + name + pad8 + data + pad8   // param value
  [TRec 80 B] + name.m + pad8 + data + pad8 // AdamW m
  [TRec 80 B] + name.v + pad8 + data + pad8 // AdamW v
```

Writes go to `path.tmp`, then `std::rename` atomically promotes to
`path`. A crash mid-write leaves the previous checkpoint untouched.

`load_checkpoint` validates every tensor's shape and dtype against the
live model. Missing records throw. Extra records are skipped (forward
compat).

**How to use.**

```ini
[runtime]
ckpt_path     = ckpts/owt_3B.bin
ckpt_interval = 1000     # steps between saves
resume_from   =          # non-empty = load this before training starts
```

Or programmatically:

```cpp
zwt::train::CheckpointMeta meta{step, seed, loader.cursor(), lr, loss};
zwt::train::save_checkpoint("ckpt.bin", params, opt, meta);
// ...
auto m = zwt::train::load_checkpoint("ckpt.bin", params, opt);
```

**Why it helps.** Python pickle checkpoints are slow (several seconds
for 3B) and not crash-safe. Our binary format writes at raw disk speed
(~2–3 s for 6 GB on NVMe) and is crash-safe by construction.

**Component speedup.** ~3× faster save/load vs pickle; 100% survival of
mid-write crashes vs ~0% with pickle.

**End-to-end contribution.** Eliminates checkpoint time from the step
cadence budget.

### 18. Strict INI config parser

**What.** `train::load_train_config(path)` parses `[model]` `[data]`
`[optim]` `[runtime]` sections. Unknown keys throw. Unknown sections
throw. Missing required fields throw with a named error
(`"config: model.vocab_size missing"`).

**How to use.**

```bash
./build/zwt_pretrain zwt/conf/owt_3B.conf
```

See `zwt/conf/owt_3B.conf` for the full list of keys.

**Why it helps.** Silent typo tolerance in config files is the #1
cause of wasted overnight runs. Failing loud at parse time costs
nothing and catches every typo before a single GEMM runs.

**Component speedup.** N/A — operational hardening.

**End-to-end contribution.** Saves runs that would otherwise silently
use the wrong `lr` / `weight_decay` / `d_model` / etc.

### 19. `.npy` token loading (u16/u32/i32/i64)

**What.** `TokenLoader::load_tokens` auto-detects NumPy `.npy` files by
the `\x93NUMPY` magic bytes and parses the v1/v2 header (dict of
`descr`, `fortran_order`, `shape`). Accepts `<u2`, `<u4`, `<i4`, `<i8`
dtypes and widens everything to `int64_t` internally so the embedding
layer sees a uniform type. Raw `int64_t` streams are also accepted as a
fallback for legacy datasets.

**How to use.** Point `data.path` at any `.npy` produced by the
existing `prepare_data` tool, or at a raw `.bin` file of `int64_t`
tokens.

```ini
[data]
path = data/owt/owt_tokens.npy
```

**Why it helps.** `prepare_data` writes `u16` for GPT-2-sized vocabs to
save disk (4× smaller than `i64`). Without native support the caller
would have to transcode to `i64` up front, doubling disk usage and
preparation time. Direct `u16` support means the 12 GB OpenWebText
tokenization stays 12 GB.

**Component speedup.** 4× smaller token file and ~4× faster initial
load vs transcoded `i64`.

**End-to-end contribution.** ~30 s saved at trainer startup; 36 GB of
disk saved on an OWT-scale dataset.

---

## Configuration reference

Every key accepted by the INI config. Unknown keys error out.

### `[model]`

| Key              | Type  | Default | Notes |
|------------------|-------|---------|-------|
| `vocab_size`     | i64   | required | Must match the tokenizer output |
| `d_model`        | i64   | required | Must equal `n_heads * head_dim` |
| `n_heads`        | i64   | required | |
| `head_dim`       | i64   | required | Typically 64, 80, 96, or 128 |
| `d_ffn`          | i64   | required | SwiGLU inner dim, ~8/3 × d_model |
| `n_layers`       | i64   | required | |
| `max_seq`        | i64   | required | Hard upper bound on context |
| `rope_base`      | f32   | 10000   | Standard RoPE base |
| `norm_eps`       | f32   | 1e-5    | RMSNorm epsilon |
| `bias`           | bool  | false   | Linear bias on all projections |
| `tie_embeddings` | bool  | false   | **Not yet implemented** — will throw |

### `[data]`

| Key          | Type  | Default   | Notes |
|--------------|-------|-----------|-------|
| `path`       | str   | required  | `.npy` or raw `i64` |
| `seq_len`    | i64   | 2048      | Must be ≤ `model.max_seq` |
| `batch_size` | i64   | 4         | Micro-batch |
| `grad_accum` | i64   | 1         | Micro-batches per optimizer step |
| `seed`       | u64   | 0xC0FFEE  | Controls shuffle order |
| `shuffle`    | bool  | true      | |

### `[optim]`

| Key             | Type | Default | Notes |
|-----------------|------|---------|-------|
| `lr`            | f32  | 3e-4    | Peak (used by cosine schedule) |
| `beta1`         | f32  | 0.9     | AdamW first moment decay |
| `beta2`         | f32  | 0.95    | AdamW second moment decay |
| `eps`           | f32  | 1e-8    | AdamW numerical floor |
| `weight_decay`  | f32  | 0.1     | Decoupled decay |
| `grad_clip`     | f32  | 1.0     | Global L2; 0 disables scaling |
| `peak_lr`       | f32  | `lr`    | Cosine schedule peak |
| `min_lr`        | f32  | 3e-5    | Cosine schedule floor |
| `warmup_steps`  | i64  | 2000    | Linear warmup from 0 |
| `max_steps`     | i64  | `runtime.max_steps` | Length of the cosine decay |

### `[runtime]`

| Key             | Type | Default           | Notes |
|-----------------|------|-------------------|-------|
| `max_steps`     | i64  | 100000            | Total training steps |
| `log_interval`  | i64  | 10                | Steps between loss/tps logs |
| `ckpt_interval` | i64  | 2000              | Steps between checkpoints (0 = off) |
| `ckpt_path`     | str  | "zwt_ckpt.bin"    | Binary checkpoint path |
| `resume_from`   | str  | ""                | If set, loaded before training starts |
| `init_seed`     | u64  | 0xC0DEBA5E        | Seeds weight init (XOR-salted per-param) |
| `arena_mb`      | i64  | 2048              | Activation arena cap in MiB |

---

## Checkpointing and resume

Checkpoints are created periodically during training. The `CheckpointMeta`
carries the information needed to resume exactly where the run left off:

```cpp
struct CheckpointMeta {
  int64_t  step;          // 1-indexed training step
  uint64_t seed;          // init_seed (echoed for verification)
  int64_t  data_cursor;   // TokenLoader chunk cursor at save time
  float    lr;            // last applied LR
  float    loss;          // last reported loss
};
```

When `--resume <ckpt.bin>` is passed, the trainer:

1. Reads the header and validates the magic/version.
2. For every parameter in the live model, looks up its name, `.m`, and
   `.v` records and restores them in place (shape-checked).
3. Sets the optimizer's step counter to `meta.step` so bias correction
   picks up at the right place.
4. Seeds the TokenLoader with `meta.data_cursor` so the training data
   stream resumes from the next chunk, not from the start of the epoch.
5. Picks up the training loop at `step = meta.step + 1`.

The LR schedule is deterministic in `step`, so resuming picks up the
correct learning rate without any state carried in the checkpoint.

### Reading metadata without loading

```cpp
auto m = zwt::train::read_checkpoint_meta("ckpts/owt_3B.bin");
std::printf("step=%lld loss=%.4f\n", (long long)m.step, m.loss);
```

Useful for tooling (sweep status, progress bars) without paying the
read cost of the full tensor body.

---

## Performance budget

3B preset (`zwt/conf/owt_3B.conf`), single H100 80 GB:

| Component                 | Bytes (BF16 weights) | Bytes (FP32 master) |
|---------------------------|----------------------|---------------------|
| Parameters (2B each)      | 6.5 GB               | —                   |
| Gradients (FP32)          | —                    | 13.0 GB             |
| AdamW `m` + `v` (FP32)    | —                    | 26.0 GB             |
| **Subtotal (persistent)** |                      | **45.5 GB**         |
| Activations (seq=2048, B=1, recompute-P) | ~25 GB    |                     |
| Arena cap                 | 4 GB (`arena_mb=4096`) |                   |
| Headroom                  | ~6 GB                |                     |

Effective batch = `batch_size × grad_accum × seq_len` = 1 × 64 × 2048 =
131,072 tokens per optimizer step. Cosine schedule from 3e-4 to 3e-5
over 50,000 steps = 6.55 B tokens ≈ 2.1 OpenWebText epochs.

---

## Expected speedups vs reference trainer

Baselines are a reference PyTorch trainer of the same architecture:

* **"Naive PyTorch"** — stock AdamW, no FlashAttention, no
  `torch.compile`, FP32 weights, standard dataloader.
* **"Tuned PyTorch"** — FlashAttention-2, `torch.compile(mode="reduce-
  overhead")`, BF16 autocast, fused AdamW, pinned dataloader.

End-to-end throughput estimates for the 3B preset on a single H100:

| Trainer         | tok/s (est.) | Relative |
|-----------------|-------------:|---------:|
| Naive PyTorch   | 6,000–9,000  | 1.0×     |
| Tuned PyTorch   | 16,000–20,000 | ~2.2×   |
| **zwt**         | 20,000–28,000 | **~3×** over naive, **~1.3–1.5×** over tuned |

Component-level estimates on the 3B preset (these are defendable from
kernel counts and published H100 numbers):

| Component                   | Naive baseline | zwt       | Speedup  |
|-----------------------------|---------------:|----------:|---------:|
| Optimizer step              | ~5 ms          | ~0.5 ms   | 10×      |
| Gradient clipping           | ~5 ms          | ~0.2 ms   | 25×      |
| RMSNorm (forward + backward)| ~0.3 ms/layer  | ~0.05 ms  | 6×       |
| SwiGLU fused                | ~1.2 ms/layer  | ~0.5 ms   | 2.4×     |
| lm_head xent backward       | ~4 ms          | ~2 ms     | 2×       |
| SDPA forward                | cuBLAS peak    | cuBLAS peak | 1×     |
| Data loading overhead       | 3–15% exposed  | 0% exposed | n/a     |
| Checkpoint save (3B)        | ~20 s (pickle) | ~3 s      | ~7×      |
| Activation memory           | baseline       | –200 MB/layer | enables bigger batch |

**Caveat.** These numbers are anchored to known H100 throughput and the
kernel counts in this framework. They are **not** runtime-measured on
this specific code yet — that validation pass is the next planned work.

---

## Limitations and deferred work

Honest list of what's not here:

* **GQA** — grouped-query attention is stubbed out; `Attention` asserts
  `Hkv == H` (full MHA). Adding GQA means widening the projection
  shapes and adjusting the SDPA call; ~a day of work.
* **Weight tying** — `tie_embeddings = true` throws. Requires sharing
  the `Embedding` weight with `lm_head.weight` as a view and routing
  gradients back to the shared buffer.
* **Multi-GPU** — no DDP, no FSDP, no tensor parallel. Single-H100 focus
  for now. The existing `olmo_cpp` DDP scaffolding could be lifted in
  but hasn't been.
* **Activation checkpointing** — intentionally not implemented. The
  3B preset fits on H100 80 GB at batch=1 seq=2048 with recompute-P
  SDPA; checkpointing would only *slow* training and is unnecessary
  at this scale.
* **MoE / expert parallel** — not in scope for the dense 3B target.
* **Runtime validation** — full training has not yet been run
  end-to-end on H100 hardware. The framework compiles clean and
  `zwt_pretrain --dry-run` constructs the 3B model and exits at the
  data step, but actual token-per-second and loss-curve measurements
  are pending a real OpenWebText run.

When any of these become relevant, they should land as additions to
`zwt/`, keeping the LibTorch-free invariant.
