# Universal Shader Debugger — Two-Paper Build Plan (HPG + SIGGRAPH)

> Working title: **OmniTrace: A Universal, API-Agnostic Time-Travel Debugger for GPU Shaders**
> Status: research synthesized + codebase maps done (2026-06-24), plan v2. From scratch in hyper-optimized C++.

## Document map — TWO papers, one shared substrate
- **PART I — HPG paper (the systems core).** Universal IR + dense per-invocation capture/
  compression. *This is the original plan below, §0–§8, unchanged.* The debugger is the whole
  contribution.
- **PART II — SIGGRAPH paper (debugger-in-the-loop shader synthesis).** §9–§15 (new). An LM
  **generates** shaders; OmniTrace is the in-the-loop oracle whose **rich** feedback
  (per-invocation traces, divergence, CPU-vs-GPU numerical diffs — not pass/fail) makes them
  better. Compares a **from-scratch SLM** (built with `llm-cpp`) vs a **DoRA-adapted 32B**.
  Reuses Part I's capture engine as the reward/feedback engine.
- **Shared substrate:** the capture spearhead (Part I §2.1) IS the reward oracle (Part II §11).
  Build Part I first; it feeds Part II.

---

## 0. The honest framing (read this first)

The vision — *"debug any shader ever written, on any API/OS, and become the industry
standard"* — is the right **product** north star and the right way to **architect** the
system. It is **not** the thing we claim in the paper. SIGGRAPH reviewers reject "we built
a tool that does everything." They accept "we invented technique **X** that makes
previously-impossible debugging **Y** real-time, and here are the numbers."

So this plan does two things at once:

1. **Architect for universality** — one API-agnostic IR core that ingests SPIR-V, DXIL,
   Metal AIR, WGSL/Naga IR, and GLSL. This protects the industry-standard ambition.
2. **Spearhead the paper with ONE defensible, measured contribution** that no existing tool
   has: **scalable, real-time, per-invocation execution-trace capture + compression that
   makes full time-travel debugging of *millions* of shader invocations practical.**

Everything below is grounded in adversarially-verified research (71 sourced claims; see
`research/` digest). Where a competitor already does part of this, it is called out
explicitly so we never over-claim novelty.

---

## 1. State of the art (verified, with sources)

| Tool | Model | API coverage | Time-travel? | Per-invocation value capture? | Killer limitation |
|---|---|---|---|---|---|
| **RenderDoc** | GPU replay + re-exec of one invocation | D3D11, D3D12, Vulkan only — **no** GL/Metal/WebGPU | No (single-invocation re-sim) | One invocation at a time | No geometry/tessellation stages; debug info is API/compiler-coupled (D3D `SetPrivateData` GUID, Vulkan `VkSetDebugUtilsObjectTagEXT`) |
| **Apple Metal Shader Debugger** | GPU frame-capture, **real** GPU values (not emulated) | Metal only (Apple platforms) | **Yes** — records full execution, forward+backward stepping, "shader code history" | Thousands of threads at once | Vendor-locked; opaque; no cross-API; capture mechanism proprietary |
| **NVBit** | Dynamic binary instrumentation at **SASS** level, dynamic recompile | NVIDIA CUDA only | No | All ISA-visible state, basic-block granularity | NVIDIA-only; compute/SASS, not graphics shader source-level |
| **Oclgrind** | **CPU interpreter** of LLVM IR (SPIR) | OpenCL only | Step/breakpoint | Yes (simulated, not real HW) | Not real hardware values; OpenCL only; slow interpreter |
| **GPUVerify** | **Static** formal verification (two-thread reduction → sequential program) | OpenCL/CUDA kernels | No (pre-execution) | No runtime values at all | Only data races + barrier divergence; no dynamic debugging |
| **Vulkan debug_printf** (validation layers) | SPIR-V **IR rewrite**: inject `SPV_KHR_non_semantic_info` + `OpExtInst NonSemantic.DebugPrintf`, write records to a GPU buffer | Vulkan only | No | Sparse printf only | Linear append buffer (default 1024 B, ~50 B/msg; resizable but **architecturally** an append log) — not designed for dense per-invocation capture of millions of threads |
| **GPU Reshape** (AMD, GDC2024) | Binary instrumentation on **GRIL**, an LLVM-inspired common IL, bi-directionally translated to/from DXIL/SPIR-V/DXBC | DX12 + Vulkan (DXIL/SPIR-V/DXBC) | No | Validation instrumentation (bounds, races), not value-capture time-travel | Integration-free & API-agnostic IL — **but aimed at validation, not source-level value-capture debugging** |

**Reconvergence/divergence background** (relevant to the CPU reference & visualization):
"Thread Frontiers" (CMU 2011) re-converge SIMT threads earlier than immediate
post-dominator (PDOM) for unstructured control flow — 1.5–633% dynamic instruction
reductions. PDOM via a predicate stack is the standard mechanism we must model.

### What this tells us
- **Universal instrumentation IR is NOT novel by itself** — GPU Reshape already did
  "write once, instrument everywhere" across DXIL/SPIR-V/DXBC. We must extend it
  (Metal AIR, WGSL, GLSL) *and* repurpose it for value capture, not claim the idea.
- **Time-travel shader debugging is NOT novel by itself** — Apple shipped it in 2018.
  But Apple is single-API, proprietary, and we don't know how it scales.
- **The unclaimed white space is the intersection + the scaling mechanism:**
  full per-invocation value-capture **time-travel** debugging, **across all APIs**,
  scaled to **millions of invocations in real time** via a novel capture/compression
  architecture, cross-checked against a **bit-exact CPU SIMT reference**. No tool does
  this. The hard, citable part is the **scaling**.

---

## 2. The gap → the contribution

> **Core claim:** Existing shader value-capture is either sparse (printf append logs),
> single-invocation (RenderDoc re-sim), or vendor-locked (Metal). None can record the
> *complete* execution history of *every* invocation in a draw/dispatch — millions of
> threads — densely enough for time-travel, cheaply enough for interactive use. We present
> the first instrumentation + on-GPU encoding scheme that does, and the first debugger that
> applies it uniformly across SPIR-V, DXIL, AIR, WGSL, and GLSL.

### Paper contributions, ranked by impact × defensibility

1. **[SPEARHEAD] Dense per-invocation trace capture & compression for SIMT.**
   A GPU-side instrumentation ABI + columnar, divergence-aware, delta/bit-packed encoding
   that captures full value histories for millions of invocations into a bounded ring with
   graceful degradation. This is the citation magnet — it's a *technique* with hard numbers
   (bytes/invocation, overhead %, threads/sec, reconstruction fidelity).

2. **Universal instrumentation IR ("UIR") with value-capture lowering.** One SSA IR that
   ingests/emits SPIR-V, DXIL, AIR, WGSL/Naga, GLSL; instrumentation written once, lowered
   everywhere. Novelty vs. GPU Reshape = broader coverage **and** value-capture/time-travel
   lowering with source-line mapping, not just validation.

3. **Bit-exact CPU SIMT reference interpreter.** A from-scratch, vectorized CPU executor of
   UIR that reproduces warp/quad semantics, derivatives, and reconvergence (PDOM + thread
   frontiers). Enables (a) divergence/correctness debugging by diffing GPU capture vs. CPU
   reference, and (b) catching driver/fast-math/IEEE-754 discrepancies — a debugging target
   no shipping tool addresses.

4. **Cross-API source mapping & time-travel UI model.** Reconstruct source-level,
   steppable, forward/backward history for any invocation from the compressed trace, with
   execution-mask divergence visualization across thread groups.

> Reviewers cite #1. #2–#4 make it a *system* and a SIGGRAPH-worthy whole. We must show #1
> beats debug_printf and matches/justifies Metal-class fidelity while being open & universal.

---

## 3. System architecture (all from-scratch C++; no slapping packages together)

```
                         ┌─────────────────────────────────────────────┐
   App's compiled shader │  FRONTENDS (parsers, hand-written)           │
   (any of):             │  SPIR-V · DXIL · AIR · WGSL/Naga IR · GLSL   │
   .spv/.dxil/.air/...   └───────────────┬─────────────────────────────┘
                                         │  lift
                              ┌──────────▼───────────┐
                              │   UIR  (our SSA IR)  │  ← instrumentation happens here ONCE
                              │  typed, CFG, debug   │
                              └──────────┬───────────┘
              ┌──────────────────────────┼───────────────────────────┐
              │ instrument (capture pass) │                           │ lower (no instr.)
   ┌──────────▼──────────┐     ┌──────────▼──────────┐     ┌──────────▼──────────┐
   │ CAPTURE LOWERING    │     │ CPU SIMT REFERENCE  │     │  BACKENDS (emit)    │
   │ inject trace-encode │     │ vectorized executor │     │ SPIR-V/DXIL/AIR/... │
   └──────────┬──────────┘     └──────────┬──────────┘     └─────────────────────┘
              │ re-emit instrumented shader            │ golden values
   ┌──────────▼──────────────────────────────────────▼──────────┐
   │ RUNTIME: API interposers (Vulkan/D3D12/Metal/WebGPU/GL)     │
   │ allocate trace ring, dispatch, read back, manage capture    │
   └──────────┬─────────────────────────────────────────────────┘
              │ compressed trace stream
   ┌──────────▼──────────┐   ┌─────────────────────┐   ┌─────────────────────┐
   │ TRACE DECODER &      │──▶│ TIME-TRAVEL ENGINE  │──▶│ UI / VISUALIZATION  │
   │ COLUMNAR STORE       │   │ reconstruct history │   │ step, masks, diffs  │
   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

**Design rule:** the IR core, capture encoder/decoder, CPU reference, and trace store know
*nothing* about any specific graphics API. APIs enter only through thin interposer shims.
That is what makes it universal — and what makes #1/#2/#3 reusable contributions.

---

## 4. From-scratch C++ component breakdown (what we build, no heavy deps)

Philosophy: **own the data structures and the hot paths.** Allowed thin/dev-only deps:
a windowing/UI layer for the demo, and the *target* APIs' own headers (Vulkan/DX/Metal) —
those are the thing we debug, not a shortcut. Everything in the contribution path is ours.

1. **`uir/` — Universal IR.** Typed SSA, CFG with dominator/post-dominator + thread-frontier
   analysis, structured-control-flow recovery, debug-info model (source line/var mapping).
   Arena-allocated, SoA instruction pool, stable handles. *No LLVM.*
2. **`frontends/` — hand-written parsers/lifters.** SPIR-V (binary word stream → UIR),
   DXIL (LLVM-bitcode reader, ours), AIR, WGSL/Naga IR, GLSL AST→UIR. Each is a lift pass
   to UIR. Start with SPIR-V (open, simplest, best ROI), add others incrementally.
3. **`backends/` — emitters** UIR → SPIR-V/DXIL/AIR/WGSL/GLSL. Round-trip fidelity tests
   from day one (lift→lower must be identity modulo IDs).
4. **`capture/` — the spearhead.** Instrumentation pass (insert value-capture taps),
   GPU-side encoder design (columnar, divergence-aware bit-packing, per-warp ring with
   atomic-free-ish claiming), and the CPU-side **decoder** + reconstruction. This is where
   the SIMD/lock-free/columnar engineering lives.
5. **`cpuref/` — vectorized SIMT interpreter.** Executes UIR on CPU with explicit
   warp/lane modeling, quad derivatives, PDOM/thread-frontier reconvergence, selectable
   IEEE-754-strict vs. GPU-fast-math modes for discrepancy hunting. Wide SIMD (AVX2/AVX-512,
   NEON on the Mac) hand-written.
6. **`store/` — columnar trace store.** Compressed, memory-mapped, query-friendly; the
   substrate the time-travel engine scrubs over. Custom allocators, delta+zigzag+RLE/bit-pack.
7. **`runtime/` — API interposers.** Layer/shim per API to inject instrumented shaders,
   bind the trace ring, and read it back. Thin by design.
8. **`timetravel/` — reconstruction + query engine.** Rebuild any invocation's stepped
   history and variable watch values from the compressed columns.
9. **`ui/` — visualization.** Source-stepping, execution-mask divergence view across thread
   groups, GPU-vs-CPU diff view. (UI lib is acceptable; the data model is ours.)
10. **`bench/` + `tests/` — evaluation harness.** Reproducible benchmarks for the paper.

---

## 5. Evaluation plan (the graphs that sell the paper)

The paper lives or dies on #1's numbers. Plan the figures *now*, build toward them:

- **Capture overhead vs. fidelity:** runtime overhead (%) and bytes/invocation vs.
  debug_printf and vs. uninstrumented baseline, across a shader corpus (Shadertoy ports,
  game-representative fragment/compute, heavy-divergence kernels).
- **Scale:** invocations/sec captured at interactive budgets; where debug_printf's append
  model collapses and ours doesn't (the headline plot).
- **Compression ratio** of the divergence-aware encoder vs. naive logging vs. generic
  (zstd-class) compression — show the domain-specific scheme wins.
- **Cross-API generality:** the *same* UIR instrumentation pass producing correct capture
  on ≥3 backends (SPIR-V/Vulkan, DXIL/D3D12, AIR/Metal) — universality as evidence.
- **Correctness via CPU reference:** N real shader bugs (incl. a driver/fast-math
  discrepancy case) localized by GPU-vs-CPU diff that existing tools miss.
- **Reconstruction fidelity:** time-travel reconstruction is lossless (or bounded-loss)
  vs. ground truth.

---

## 6. Hardest technical risks (and mitigations)

1. **Trace volume explosion** (the central risk). Full per-invocation history of millions of
   threads is enormous. *Mitigation:* divergence-aware columnar encoding, value-class
   specialization, selective/adaptive capture (region-of-interest, sampling with
   reconstruction), bounded ring with graceful degradation. **The contribution _is_ solving
   this** — derisk early with a focused prototype before building breadth.
2. **Source mapping through optimizing compilers.** Optimized IR scrambles source
   correspondence (RenderDoc leans on embedded debug info). *Mitigation:* carry debug-info
   through UIR; support both debug and optimized builds; reconstruct via the CPU reference.
3. **DXIL/AIR are under-documented / moving targets** (e.g., SPIRV-Tools removed
   `InstDebugPrintfPass` in Oct 2024 — internals shift). *Mitigation:* start with SPIR-V
   (open, stable), abstract backend specifics behind UIR, add APIs incrementally; pin
   versions; keep round-trip tests as canaries.
4. **Bit-exact CPU reference vs. real GPU.** GPUs use fast-math, FMA contraction, varied
   rounding. *Mitigation:* treat discrepancies as a *feature* (a debugging target), with
   selectable strict/fast modes; validate against vendor reference where possible.
5. **Determinism of GPU capture.** Scheduling nondeterminism complicates replay.
   *Mitigation:* capture is per-invocation and content-addressed, not order-dependent;
   record enough to reconstruct independent of warp scheduling.
6. **Scope creep (5 APIs × 4 paradigms).** *Mitigation:* the staging in §7 — one API, one
   paradigm proven end-to-end before fan-out. The paper needs depth on #1, not breadth.

---

## 7. Milestones (depth-first, derisk the spearhead first)

> Target a SIGGRAPH-style deadline ~9–11 months out. Sequence assumes the contribution must
> be *proven* before breadth is added. Dates are relative; we'll pin to the real CFP.

- **M0 — Foundations & vertical slice (weeks 1–6).** UIR skeleton; SPIR-V lift+lower with
  round-trip tests; a Vulkan interposer that can swap a shader; naive printf-style capture
  working end-to-end on one fragment shader. *Goal: prove the pipeline exists.*
- **M1 — The spearhead prototype (weeks 6–14).** First version of dense per-invocation
  capture + divergence-aware encoder + decoder. Measure overhead/bytes vs. debug_printf on a
  small corpus. *Go/no-go: do the numbers beat the append-buffer model?* This is the riskiest
  step — do it before anything else.
- **M2 — CPU SIMT reference (weeks 12–20).** Vectorized UIR interpreter; GPU-vs-CPU diff;
  first correctness/divergence bug demos. (Overlaps M1.)
- **M3 — Time-travel + UI (weeks 18–26).** Reconstruction engine, source stepping, execution
  masks, diff view. Make the demo *feel* like Metal's history debugger but open & universal.
- **M4 — Cross-API generality (weeks 24–34).** Add DXIL/D3D12 and AIR/Metal backends + their
  interposers; show the *same* instrumentation pass works on all three. (WGSL/GLSL optional
  stretch.)
- **M5 — Evaluation & writing (weeks 32–44).** Full benchmark suite, all figures, ablations,
  artifact packaging (reproducibility raises acceptance odds), paper draft, internal review.

> Parallelizable tracks once M0 lands: capture (M1) ∥ CPU reference (M2) ∥ frontends breadth.
> But **M1 gates everything** — if dense capture can't be made cheap, the contribution pivots
> (e.g., to adaptive/sampled capture as the technique). Decide at the M1 go/no-go.

---

## 8. Immediate next steps (this week)

1. **Lock the venue & deadline.** SIGGRAPH (Technical Papers) vs. SIGGRAPH Asia vs. HPG
   (High Performance Graphics — arguably a *better* fit for a systems/perf contribution like
   this, and a strong target). Pin the CFP date so M-dates become real. *(decision needed)*
2. **Stand up the repo & build system** (CMake or a hand-rolled build; C++20/23, strict
   warnings, sanitizers, a benchmark harness skeleton).
3. **Build the M0 vertical slice on SPIR-V/Vulkan** — smallest end-to-end path: lift a real
   `.spv`, instrument trivially, run, read back. Everything compounds on this.
4. **Assemble the shader corpus** for evaluation (Shadertoy ports + compute + high-divergence
   cases) so M1 numbers are credible from the start.
5. **Write the related-work table into the paper outline now** (it's already done above) so
   we always know exactly what we must out-do.

---

## Appendix A — Sources behind the SOTA table
- RenderDoc shader debugging & debug-info coupling: renderdoc.org/docs (how_debug_shader, how_shader_debug_info, shader_messages)
- Apple Metal Shader Debugger (history/time-travel, real values, divergence masks): WWDC 2018 session 608
- NVBit (SASS dynamic binary instrumentation): research.nvidia.com, 2019
- Oclgrind (LLVM-IR interpreter, plugin model): dl.acm.org/doi/10.1145/2791321.2791333
- GPUVerify (two-thread reduction, static): dl.acm.org/doi/10.1145/2743017
- Thread Frontiers (reconvergence): istc-cc.cmu.edu/publications/papers/2011/SIMD.pdf
- GPU Reshape / GRIL (universal IL, integration-free instrumentation): GDC 2024, gpuopen.com
- Vulkan debug_printf SPIR-V rewrite ABI (`NonSemantic.DebugPrintf`): KhronosGroup/Vulkan-ValidationLayers docs
- Full verified-claim digest: `research/verified_claims.json` (71 adversarially-verified claims)

*Caveats from verification: the Vulkan printf buffer is user-resizable (not hard-fixed) —
our gap argument is architectural (append-log vs. dense per-invocation), not "the buffer is
small". SPIRV-Tools' `InstDebugPrintfPass` was removed Oct 2024; treat specific pass names as
version-dependent and abstract behind UIR.*

---
---

# PART II — SIGGRAPH: Debugger-in-the-Loop Shader Synthesis

> **One-line thesis:** Code-generation feedback loops today use a *binary* signal (does it
> compile / pass a test?). We replace it with a *dense, localized* signal that only a
> debugger can produce — per-invocation execution traces, control-flow divergence, and
> bit-exact CPU-vs-GPU numerical diffs — and show it trains shader-generating models far
> better than compile-pass feedback, on both a tiny from-scratch model and a DoRA-adapted
> 32B. **OmniTrace's capture engine (Part I) is the reward engine.**

## 9. Why this is novel (and what we are NOT claiming)

- **Already exists (we build ON it, don't claim it):** the `shader_cmake` loop already does
  *inference-time* compile-error feedback (generate → compile via OpenGL → feed the exact
  error back → retry ≤3×), measured at compile@1→compile@3 = 56%→79% (Qwen-3B), 83%→93%
  (Qwen-7B). Its WRITEUP already envisions verifier-in-the-loop RL. So the *retry-on-feedback*
  pattern is prior (our own) work — the **contribution is the feedback _content_ and moving it
  into _training_**.
- **The leap:** compile-pass tells you *that* it's wrong; the **debugger tells you _where_ and
  _how_** — which invocation, which source line, what value went NaN/Inf, where warps diverged,
  how far the render is from intent. That is a *dense, per-token-attributable* training signal.
  No prior code-gen system has a per-invocation execution oracle in the training loop.
- **The empirical contribution reviewers cite:** debugger-feedback > compile-feedback,
  quantified, and it holds for BOTH a 30–350M from-scratch SLM and a DoRA-32B — i.e., the
  signal quality matters more than model scale. That is a *measured scientific claim*, not a
  tool demo.

## 10. The two models (the comparison axis)

Everything is **from scratch** in the sense the user requires: the trainer is `llm-cpp`'s own
C++ stack (no PyTorch-Python, no HuggingFace `peft`). Prefer the **ZWT (Zero-Wait Trainer)**
path — it is **LibTorch-free, CUDA-native** — to keep the contribution self-owned. We *import*
`llm-cpp` (link `libolmo_cpp.a` / `libzwt.a`, include `olmo_cpp/`), we do **not** vendor a
black-box framework.

- **Model A — from-scratch SLM.** `olmo_cpp::Transformer` + `olmo_cpp::train(model, cfg,
  train_cfg, device, callbacks)`, config in the 30M–350M range (`conf/olmo.conf`,
  `olmo_125M/250M/350M`). BPE tokenizer (`bpe_tokenizer.hpp`) or a shader-aware
  `structural_tokenizer.hpp`. Custom fused kernels (RMSNorm/SwiGLU/RoPE/fused-LM-head-CE)
  already in `kernels/`. The `Callback` hook is where we inject the debugger reward.
- **Model B — DoRA-adapted 32B.** Open code base (Qwen2.5-Coder-32B-class) imported via
  `convert_checkpoint`. **DoRA does not exist in `llm-cpp` yet — we BUILD it** (see §12). DoRA
  freezes base weights, so only adapters + magnitudes train: feasible on 1×80GB (bf16 base ≈
  64 GB) or **4-bit QDoRA** for headroom. This is the "make it perfection" adaptation: DoRA
  consistently beats LoRA at equal parameter budget.
- **Baselines for the paper:** our OWN C++ compile-only-feedback loop (the §13 `OmniSynth`
  rewrite, ablated to reward = `[compiles]`) — the bar the full debugger-feedback system must
  beat. `shader_cmake`'s published Python/Ollama numbers (Qwen-3B 56→79%, 7B 83→93% compile@k)
  are cited only as a sanity reference for the *pattern*, not run as our baseline.

## 11. The reward oracle = the capture engine (the real-math core)

This is where Part I and Part II fuse and where "improve with real math" lives. The reward is
**decomposed and per-line attributable**, computed from a captured execution trace:

```
R(shader | prompt) = w_c·[compiles] + w_e·[runs, no NaN/Inf]
                   + w_d·(1 − divergence_penalty)         # from execution masks
                   + w_n·(1 − numerical_error)            # CPU-ref vs GPU bit-diff
                   + w_v·visual_match(render, target)     # SSIM/LPIPS to target image
                   − w_p·perf_penalty                     # captured cost
```

- **Per-line credit assignment.** The trace localizes *which* source line/invocation produced
  the failing value, so reward gradient (for RL) or preference label (for DPO) attaches to the
  responsible tokens — not a flat scalar over the whole program. This is the differentiator
  vs. compile-pass RL.
- **Real-math item #1 — divergence-aware trace compression (shared with Part I §2.1).** Model
  a warp's per-lane value stream as a coherent source: across W lanes the values are highly
  correlated, so conditional entropy `H(v_lane | v_warp_base)` ≪ `H(v_lane)`. Encode
  per-warp base + per-lane residual; under SIMT coherence residuals concentrate near 0 →
  bit-pack to ≈ entropy. Target: provably approach `Σ H(v | context)` bits/value and beat both
  naive logging and generic (zstd-class) compression. Derive the bound; measure the gap.
- **Real-math item #2 — reconvergence/divergence metric.** Define divergence_penalty from the
  execution-mask lattice using PDOM vs **thread-frontier** reconvergence (CMU 2011) so the
  metric is principled, not ad hoc.
- **Real-math item #3 — numerical_error.** Bit-exact CPU SIMT reference (Part I §4 `cpuref`)
  vs GPU capture under selectable IEEE-754-strict / fast-math modes → a *quantified* ULP/relative
  error field, not "looks wrong."

## 12. DoRA — built from scratch, with the math written out

DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes a pretrained weight `W₀` into a
magnitude vector `m` and a direction matrix, updating direction with a low-rank term:

```
W' = m ⊙ ( W₀ + B·A ) / ‖ W₀ + B·A ‖_c        ( ‖·‖_c = per-column L2 norm )
   trainable: m ∈ ℝ^{1×k},  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},   r ≪ min(d,k);  W₀ frozen
```

vs LoRA's `W' = W₀ + B·A`. The magnitude/direction split lets the adapter change *direction*
without being forced to couple magnitude — empirically closer to full fine-tuning. **Build
items in `llm-cpp` (none exist today):** (a) a `DoRALinear` module wrapping frozen base GEMM +
low-rank path + per-column renorm; (b) gradient only through `m, A, B`; (c) optional 4-bit base
(QDoRA) using the existing `quant_dequant`/`int4_kv` codecs; (d) merge-back for inference. Unit
test: DoRA with `r=0` and `m=‖W₀‖_c` must equal the frozen base (identity check).

## 13. `OmniSynth` — `shader_cmake` rewritten from scratch in C++ (NO Python)

**Hard rule: we do NOT import `shader_cmake`.** It is Python (Ollama + ModernGL + pygame +
PyTorch). We read it as a *specification* of the loop that already works, discard the code, and
reimplement every piece in pure, hyper-optimized C++ as a new project component, **`synth/`**.
The only thing we take across is the **data** (GLSL text + dataset), never the code.

**C++ components we write from scratch (no package-slapping):**
- `synth/validator` — real shader compile/link validation via the actual GL/Vulkan driver
  (the target API we debug is allowed; replaces Python ModernGL `compiler.py::validate_pair`).
- `synth/renderer` — offscreen GL/Vulkan renderer: shader → image (replaces `renderer_moderngl.py`).
  Produces the per-program **ground-truth render** — the one thing NL→shader RL usually lacks.
- `synth/driver` — the generate→validate→diagnose→feedback→retry loop (replaces
  `generator.py::ShaderGenerator.generate()`), but the model runs through **`llm-cpp`'s C++
  inference** (no Ollama) and feedback is OmniTrace's structured diagnostics (not compiler text).
- `synth/bench` — compile@k / render-match harness in C++ (replaces `shader_bench.py`).
- `synth/data` — corpus builder: ingest the **shaders21k** GLSL corpus (~21k Shadertoy,
  arXiv:2211.16412 — data only), render each (target image), caption via a C++ VLM-inference
  pass, attach OmniTrace traces → `(prompt, shader, render, trace, reward_components)` tuples,
  tokenize to `.npy` for `olmo_cpp::TokenDataset` (mmap, async prefetch).

The per-shader render is the `target` in the `visual_match` reward term — solving the "no
ground truth for an NL prompt" problem honestly, all in C++.

## 14. Training & loop architecture

- **Stage 0 — SFT.** Both models supervised-fine-tuned on the `(prompt → shader)` corpus.
- **Stage 1 — inference-time debugger feedback** (C++ `synth/driver`, the from-scratch rewrite
  of `shader_cmake`'s loop): generate (via `llm-cpp` C++ inference) → OmniTrace capture/diagnose
  → feed *structured diagnostics* (not just compiler text) back → regenerate.
- **Stage 2 — training-time RL** (the contribution): GRPO / RLOO with the §11 dense reward,
  injected through `olmo_cpp`'s `Callback`. Per-line credit assignment shapes the advantage.
  DPO variant: pair higher-reward vs lower-reward generations as preferences.
- **Ablation built into the design:** same pipeline with reward = `[compiles]` only (the
  `shader_cmake` baseline) vs the full debugger reward. This single ablation IS the paper's
  headline result.

## 15. Evaluation, risks, milestones (ML track)

**Figures that sell the SIGGRAPH paper:**
- compile@k, **render-match** (SSIM/LPIPS to target), **execution-correctness** (NaN/Inf rate,
  divergence), bug-rate — for {from-scratch SLM, DoRA-32B, Ollama baselines}.
- **The ablation:** compile-only feedback vs full debugger feedback, both models — show the
  dense signal wins, and that signal quality can rival scale (small+rich ≈ big+poor).
- DoRA vs LoRA vs full-FT at equal budget (justify DoRA).
- Trace-compression bound vs achieved bits/value (ties to Part I).

**Risks specific to Part II:**
1. **DoRA-32B is a build + compute item** (not in `llm-cpp`; needs ≥80 GB or QDoRA). *Mitigate:*
   start with DoRA on a 7B (`llm-cpp` already configs to 7B) to derisk the module, then scale.
2. **RL instability / reward hacking** (model games `visual_match`). *Mitigate:* multi-term
   reward, held-out prompts, KL-to-SFT regularization, the per-line attribution as a sanity gate.
3. **Reward oracle depends on Part I capture (debugger M1).** *Mitigate:* sequence ML-M2 after
   debugger M1; until then use `shader_cmake`'s compile signal as a stand-in reward.
4. **VLM captioning quality** bounds data. *Mitigate:* cycle-consistency filtering; lean on the
   21k real renders as anchors.

**ML milestones (run AFTER/with the debugger track; reward oracle gates on debugger M1):**
- **ML-M0:** link vendored `external/llm-cpp` (ZWT path) into the project; train a 30M SLM on
  shaders21k; stand up the C++ `synth/` compile-only-feedback loop as the ablation baseline.
- **ML-M1 (wks 4–10):** full data pipeline (render + caption + trace + tokenize).
- **ML-M2 (wks 10–18, gates on debugger M1):** wire OmniTrace as the dense reward oracle;
  inference-time debugger-feedback loop beats compile-feedback loop.
- **ML-M3 (wks 14–22):** build & validate the `DoRALinear` module (identity test, then 7B, then
  32B / QDoRA); SFT both models.
- **ML-M4 (wks 20–30):** training-time RL (GRPO/RLOO) with debugger reward, both models; run the
  headline ablation.
- **ML-M5 (wks 28–40):** full comparison, ablations, DoRA-vs-LoRA, SIGGRAPH draft + artifact.

> **Purity rule (per user):** every contribution-path component is ours — OmniTrace (Part I) and
> the DoRA module, reward oracle, RL loop, and data pipeline (Part II). `llm-cpp` is from-scratch
> C++ we **vendored** (`external/llm-cpp`) and *extend with real math*, not a package we slap on.
> `shader_cmake` is **NOT** imported — it is Python and is **rewritten wholesale in C++** as
> `synth/` (§13). **No Python, no Ollama, no ModernGL/pygame anywhere in the contribution path.**
> The CPU SIMT reference, trace codec, divergence metric, and DoRA decomposition all carry
> written-out math + proofs/bounds + unit tests, not just code.

---

## Appendix B — Imported-codebase facts (verified by exploration 2026-06-24)
- **`llm-cpp`** = "OLMo-corecpp" (LibTorch-based) + **ZWT (Zero-Wait Trainer, LibTorch-free,
  CUDA-native)**. Public API: `olmo_cpp::train(model, cfg, train_cfg, device, callbacks)`,
  `TransformerConfig`, `Transformer`/`FusedTransformer`, `TokenDataset` (mmap .npy, async
  prefetch), optimizers (Muon, foreach-AdamW, 8-bit AdamW, Lion, Dion), `Callback` hook,
  backends (CUDA / SIMD-CPU / LibTorch). Builds `libolmo_cpp.a`, `olmo_kernels.so`, `libzwt.a`.
  Configs 30M→7B. Tools: `convert_checkpoint` (HF import), `export_hf`. **No LoRA/DoRA; nothing
  above 7B — both are build items.** Custom CUDA kernels in `kernels/` (fused RMSNorm, SwiGLU,
  RoPE, QKV+RoPE WMMA/TMA, fused LM-head CE, paged/flash/sparse attention, FP8/INT4).
- **`shader_cmake`** = NL→GLSL generator (Python, Ollama). Feedback loop in
  `shader_generator/generator.py::ShaderGenerator.generate()` (gen → parse → `compiler.py`
  ModernGL `validate_pair()` → feed compile error back via `ERROR_FIX_TEMPLATE` → retry ≤3).
  Measured: compile@1→@3 = 56%→79% (Qwen-3B), 83%→93% (Qwen-7B). Bench in
  `benchmarks/shader_bench.py`. Dataset: `shaders21k_repo/` (21k Shadertoy + ModernGL renderer,
  arXiv:2211.16412). No training today; WRITEUP envisions C++ verifier-in-the-loop RL (reward =
  compile_ok) — exactly what Part II builds and upgrades. **REFERENCE/SPEC ONLY — NOT imported.**
  It is Python; we reimplement its loop from scratch in C++ as `synth/` (§13) and take only the
  shaders21k DATA, never the code.
