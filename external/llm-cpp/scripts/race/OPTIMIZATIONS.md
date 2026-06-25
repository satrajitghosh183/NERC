# What's enabled on the race configuration

Cross-reference for `race_250m_cpp.conf`. Maps every optimization we
shipped to where it fires (or doesn't) on a 5060 Ti race run.

## Tier 1 — kernel-level, always on once `OLMO_BUILD_KERNELS=ON`

| optimization | fires from | how to disable (don't) |
|---|---|---|
| WMMA fused FFN (kernels/fused_ffn_wmma.cu)             | `fused_ffn()` for bf16 + aligned shapes **when the tiles fit in shared memory**. At the race's H=2816 the kernel needs ~270 KB > Blackwell's ~227 KB cap, so it falls back to the cuBLAS chain (fast_linear + vectorized silu_mul + fast_linear) — still fast. Fires for smaller H. | unavailable on sm < 80 |
| TMA async-load fused FFN (kernels/fused_ffn_tma.cu)    | same dispatcher; sm_90+ + shmem fit. Also falls back to cuBLAS at H=2816. | only on sm_90+ |
| WMMA fused QKV+RoPE (kernels/fused_qkv_rope_wmma.cu)   | `fused_qkv_rope()` for bf16 + aligned shapes; ~100 KB at head_dim=128, fits via cudaFuncSetAttribute opt-in on Blackwell → **fires on the race** | n/a |
| Fused LM-head + softmax-CE (kernels/fused_lm_head_ce.cu) | `transformer.cpp::forward_with_loss` calls `fused_lm_head_ce_autograd` for main + MTP heads | requires labels |
| Fused RMSNorm backward (kernels/rms_norm_backward.cu)  | `RmsNormFn::backward` (cuda_ops_autograd.cpp) routes here for bf16/fp32 CUDA tensors | n/a |
| WMMA fp32→bf16 store vectorization (B2)                | inside the WMMA kernels above | n/a |
| bf16 vectorized silu_mul (B4)                          | `silu_mul_bf16_kernel` (uint4 = 8-bf16 reads) | falls back for non-aligned tail |
| AutogradCUDA wrappers (correctness fix)                | TORCH_LIBRARY_IMPL(olmo_ops, AutogradCUDA) — rms_norm, rms_norm_add, silu_mul, apply_rope | n/a |
| Half-rotation inverse RoPE fix (A2 correctness)        | `fused_qkv_rope_autograd::backward::inverse_rope` | n/a |
| WMMA store_matrix_sync UB fix (correctness)            | per-warp shared-mem scratch in 3 WMMA kernels | n/a |

These are compiled-in. No `.conf` switch.

## Tier 2 — wired into AttentionImpl / FeedForwardImpl, fires on CUDA bf16

| optimization | where it fires |
|---|---|
| **A1** — saved gate_up (skip recompute in FFN backward)        | `FusedFFNFunction::forward` calls `fused_ffn_train` which always materializes gate_up. Backward uses the saved tensor. |
| **A4** — cached packed QKV weight (`AttentionImpl::packed_qkv_weight`) | rebuilds the cat only when w_q/w_k/w_v `_version()` changes |
| Fused QKV+RoPE in `AttentionImpl::forward` and `forward_paged` | `can_use_fused_qkv = cuda + !FP8 + !QK-norm + RoPE` |
| `fast_linear` (cuBLASLt direct) for LM head, attention out-proj, FFN backward | `fast_linear` / `fast_matmul` for any 2-D matmul with bf16/fp16/fp32 |
| **D3** — cuBLASLt descriptor cache                              | keyed by (dtype, M, N, K, opA, opB) inside cublas_direct.cpp |
| **D5** — cuBLASLt heuristic algorithm                           | `cublasLtMatmulAlgoGetHeuristic` runs once per shape, stored on the cached plan |

All conditional on the regular Transformer path. If you set `fused = 1`
in `[optimization]` you route through `FusedTransformer` / `FusedAttention`
which **lack** the QKV+RoPE and fast_linear out-proj wirings.

## Tier 3 — config-flagged

`scripts/race/configs/race_250m_cpp.conf` sets each correctly:

| flag | value | purpose |
|---|---|---|
| `[optimization] fused`             | **0** | route through standard `Transformer` / `AttentionImpl` — that's where the kernel wirings are |
| `[optimization] cuda_graph`        | **0** | OFF — capture exists only in the FusedTransformer path (fused=1, no kernel wirings) and is incompatible with the A3 fused-CE backward's .item() sync. Known follow-on. |
| `[optimization] foreach_optimizer` | 1   | use ForeachAdamW (`_foreach_*` batched ops) instead of per-param AdamW loop |
| `[optimization] gpu_data`          | 1   | whole tokenized corpus lives in VRAM (no per-step H2D copy) |
| `[optimization] zero1`             | 0   | single-GPU race; ZeRO-1 sharding only helps multi-GPU |
| `[training] bf16`                  | 1   | params + activations bf16 (50% memory drop) |
| `[training] amp`                   | 0   | mutually exclusive with bf16; we use pure bf16 |
| `[model] use_qk_norm`              | 0   | extra norms — off for raw throughput |
| `[model] use_float8`               | 0   | experimental; off |
| `[model] activation_checkpoint_mode` | none | 250M fits in VRAM; no recompute needed |

## Things deliberately off

| flag | why off |
|---|---|
| `use_amp`             | bf16 mode handles precision; AMP autocast adds dispatcher overhead |
| `use_grad_scaler`     | only needed with fp16, not bf16 |
| `async_muon`          | we use AdamW, not Muon |
| `sgp_enabled`         | SGP predictor adds overhead with no benefit at 1000 steps |
| `gated_attention`     | extra params per layer, not needed for the race |
| `sliding_window_size` | full causal — fair comparison vs Python |
| `rope_scaling_type`   | seq_len=1024 ≪ 8192 max; no scaling needed |
| `multi_res`           | DC-MRE — separate research line |

## Optimizations the C++ side can do that Python can't

Even at matched backbone + matched training hyperparameters, these are
the structural reasons C++ wins:

1. **MTP-speculative inference** — `num_mtp_heads = 3` on the C++ side
   means the chat tool drafts 2-4 tokens per verify forward instead of
   1. Python has no MTP heads, so it does linear per-token decode.

2. **cuBLASLt direct (D3/D5)** — every Linear in the forward+backward
   skips the ATen dispatcher and uses heuristic-selected algos per shape.

3. **Fused LM-head + softmax-CE (A3)** — the `[B*S, V]` logits tensor
   never materializes in training. At V=50304 and B*S=4096 this is
   ~800 MB of HBM traffic saved per step.

4. **AutogradCUDA wrapper for RMSNorm + B3 fused backward** — every
   RMSNorm backward is one fused kernel instead of ~10 ATen ops.

5. **Fused QKV+RoPE kernel** — one launch produces Q/K/V in head-major
   layout with RoPE applied (fires on the race at head_dim=128).

Note: at the race's H=2816 the fused FFN *kernel* doesn't fire (shmem
over budget); the FFN still runs as fast cuBLASLt GEMMs + the
vectorized silu_mul kernel. The fused FFN kernel is for smaller H.

CUDA graph capture is OFF (see Tier 3) — incompatible with the A3
fused-CE backward's .item() sync, and only wired on the fused=1 path.

If anything else in this list isn't firing on the actual run, the C++
side is leaving performance on the table.
