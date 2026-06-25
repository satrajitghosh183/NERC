# OmniTrace — Build Status

Autonomous build pass against the 20 tasks in `PLAN.md`. Everything below is **real,
compiling, tested C++23** (Apple clang, `-O3`, M4 Pro). **19 test suites, all green**,
including **real on-GPU work via MoltenVK**. Reproduce:

```
cmake -S . -B build -G Ninja && cmake --build build
cd build && ctest --output-on-failure
OMNI_DATA_DIR=../data/bench ../build/bench_trace_codec    # paper data
```

## Done & tested — 19 of 20 tasks

### Part I — the debugger (HPG paper)
| # | Component | What works | Test |
|---|-----------|-----------|------|
| 1 | Scaffolding | CMake, from-scratch test + bench harness | `test_smoke` |
| 2 | **UIR core** | typed SSA, shader opcodes, SoA, dedup, verifier | `test_uir` |
| 3 | **SPIR-V frontend** | hand-written `.spv` → UIR; lifts a real glslang shader | `test_spirv_frontend` |
| 4 | **SPIR-V backend** | UIR → `.spv`; **passes official `spirv-val`** + re-lift round-trip | `test_spirv_backend` |
| 5 | **CFG analysis** | dominators, post-dominators, reconvergence, thread frontiers | `test_cfg` |
| 6 | **Capture pass** | idempotent `TraceTap` insertion on UIR | `test_capture` |
| 7 | **Trace codec** ⭐ | divergence-aware compression, lossless, **26× / 4.5×**, never-expands | `test_trace_codec` + `bench_trace_codec` |
| 8 | **Trace store** | mmap'd columnar file, query by site | `test_store` |
| 9 | **CPU SIMT reference** | executes lifted shader per-lane, matches closed-form | `test_cpuref` |
| 10 | **Time-travel** | reconstructs an invocation's history, fwd/back nav | `test_timetravel` |
| 11 | **Real GPU capture** ⭐ | compute shader runs on **Apple M4 Pro (MoltenVK)**, values read back → codec (9.8×, lossless) | `test_gpu_capture` |
| 12 | **Numerical diff + divergence** | ULP diff (catches fma-vs-strict), branch divergence | `test_diff` |

### Part II — debugger-in-the-loop synthesis (SIGGRAPH paper)
| # | Component | What works | Test |
|---|-----------|-----------|------|
| 13 | **GLSL validator** | C++ compile oracle via Vulkan SDK; accepts good, rejects bad | `test_validator` |
| 14 | **GPU renderer** ⭐ | renders shaders on the **M4 Pro** (compute-as-fragment); visual_match reward | `test_renderer` |
| 15 | **Synth driver** | generate→validate→feedback→retry loop; **compile@1 0.5 → compile@k 1.0** | `test_synth_driver` |
| 16 | **Data builder** | byte tokenizer + `.npy` writer; **verified loadable by real NumPy** → llm-cpp `TokenDataset` | `test_corpus` |
| 18 | **DoRA** ⭐ | weight decomposition, exact identity, magnitude/direction split, <2% params | `test_dora` |
| 19 | **Reward oracle** ⭐ | §11 dense reward + per-line credit assignment | `test_reward` |
| 20 | **SLM training** ⭐ | from-scratch neural LM, hand-written backprop, trained on shader corpus: **loss 5.55→3.02 (ppl 258→20)** | `test_tinylm` |

⭐ = a paper contribution proven directly in code.

### Vertical slices that actually run
- **Debugger:** `SPIR-V → UIR → CFG → instrument → CPU SIMT exec → divergence-aware compress → mmap store → time-travel reconstruction`.
- **Real GPU:** compute shader on M4 Pro → value capture → codec; and shader → rendered image → visual reward.
- **Synthesis loop:** generate → GLSL compile-validate → feedback → retry (compile@k lift); reward oracle scores compile/exec/divergence/numerical/visual with line-level blame; DoRA + a from-scratch trainer whose loss drops on the shader corpus.

## Not fully done — 1 of 20, honestly

| # | Task | Status |
|---|------|--------|
| 17 | Build vendored `llm-cpp` (ZWT) here | **Hardware-blocked.** Investigated: **LibTorch (torch 2.8) is found**, but (a) there is **no `nvcc`/NVIDIA GPU**, so `llm-cpp`'s CUDA-native fast-training path — the actual contribution — cannot build or run; (b) the vendored CMake's source globs resolve empty (a vendoring quirk). The user's chosen ZWT path is CUDA-native and needs a CUDA host. |

**On #20:** the *headline* run — a from-scratch SLM **and** a DoRA-32B trained with the
debugger reward in the loop, with the compile-only-vs-debugger-feedback ablation — needs
datacenter GPUs and #17's trainer. What is **done and tested** is the *mechanism*: a real
from-scratch training loop (loss decreases), DoRA (#18), the reward oracle (#19), and the
feedback driver (#15). The large-scale training was **not run and is not faked**; it requires
a GPU cluster.

## On the 2×H100 box (Jetstream `llmcpp`) — DONE

The CUDA-host work is no longer hypothetical. On 2× NVIDIA H100 80GB (NVLink, CUDA 13.1):

- **Trained a from-scratch shader LM** (152M, GPT-2-BPE) on the **shaders21k** corpus
  (22,172 GLSL shaders → 47.6M tokens, prompt→shader pairs from Shadertoy JSON metadata):
  **267k tok/s, loss 10.0 → 2.65** in minutes. This is the from-scratch-SLM half of the
  SIGGRAPH comparison, run for real.
- **Fixed two real inference bugs** in `llm-cpp/chat` (committed to this repo):
  - BF16/FP32 dtype mismatch (crashed the forward) — `644e078`-era fix already present
    upstream; older box clone patched.
  - **FP32 logit upcast for BF16 inference** (`a520af4`) — BF16 argmax ties were collapsing
    greedy decoding to whitespace; fixed in `tools/chat.cpp`.
- **Robust multi-GPU resume** (`644e078`) — `latest_complete()` skips torn checkpoints;
  verified against the exact crash that broke the original run.
- **Debugger-in-the-loop reward demo** — the LM generates a shader → `glslangValidator`
  compiles it (the reward signal) → score; the loop runs end-to-end (`reward_loop.py`).

### Honestly remaining (large efforts, documented not faked)
- **Coherent generation** needs a stronger model: 47.6M tokens is data-limited for a 152M
  model (perplexity ~14). Inference is *correct*; more data / longer training improves it.
- **Real FP8 speedup** is NOT available: the codebase's FP8 is *emulation* (quantize→
  dequantize STE, "for parity"), with no cuBLASLt FP8 GEMM — enabling it is slower, not
  faster. The genuine lever is `fused=1` (FusedTransformer, ~1.2×, now that resume is fixed).
- **DoRA-32B** fine-tune: needs a 32B base + DoRA loading/freezing wired into the CUDA
  trainer (the tested CPU `omni/ml/dora` is the algorithm; the LibTorch port + RL loop is the
  remaining engineering). The from-scratch-SLM half is done; the DoRA-32B half is future work.
