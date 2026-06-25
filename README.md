# OmniTrace

A universal, API/OS-agnostic GPU shader debugger built from scratch in hyper-optimized
C++ — plus a **debugger-in-the-loop shader synthesis** stack. Targeting two papers:

- **HPG** — the debugger core: a universal SSA IR, a hand-written SPIR-V front/back end,
  control-flow analysis, an instrumentation/capture pass, and a **divergence-aware
  trace codec** that compresses per-invocation execution traces near the entropy bound.
- **SIGGRAPH** — an LM that *generates* shaders with OmniTrace in the loop: a from-scratch
  trainer + **DoRA** adaptation, refined by a dense reward (compile / execute / divergence /
  numerical / visual) with per-line credit assignment.

## Status

19 of 20 build tasks complete, **all with tested C++** (19 test suites green), including
**real on-GPU capture and rendering via Vulkan/MoltenVK**. See [`STATUS.md`](STATUS.md) for
the full breakdown and [`PLAN.md`](PLAN.md) for the research framing and architecture.

## Build & test

```bash
cmake -S . -B build -G Ninja && cmake --build build
cd build && ctest --output-on-failure
OMNI_DATA_DIR=../data/bench ../build/bench_trace_codec   # regenerate benchmark data
```

Requires a C++23 compiler and (optionally) the Vulkan SDK for GPU capture/rendering, which
is auto-detected.

## Layout

```
include/omni/      public headers (uir, frontends, backends, capture, trace, store,
                   cpuref, timetravel, analysis, reward, synth, gpu, ml)
src/               implementations (mirrors include/)
tests/             from-scratch test suites (no gtest)
bench/             benchmarks that emit data/ JSON+CSV
external/llm-cpp/  vendored, editable C++ trainer (CUDA-native; runs on the GPU box)
data/              sample shaders + benchmark results
```

## GPU training

`external/llm-cpp` is the CUDA-native trainer for the SIGGRAPH track. It builds and runs on
an NVIDIA GPU host (it cannot exercise its fast path on Apple silicon). The tested CPU
modules (`omni/ml/dora`, `omni/reward`, `omni/ml/tinylm`, `omni/synth/corpus`) define the
mechanism to port onto the GPU.
