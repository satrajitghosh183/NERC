# Results — shader synthesis on 2×H100 (Jetstream `llmcpp`)

Everything here is real and reproducible. Live logs you can `tail -f` on the box
(`exouser@149.165.170.159`) are noted per item. Raw artifacts stay on the box; this file
is the committed summary.

## The three models (the SIGGRAPH comparison) — completed run, 2026-06-25

Hands-off pipeline (`run_pipeline.sh`): train from-scratch 3.6B SLM → eval → DoRA-32B → DoRA-3B →
compare. All three are scored by the **same** harness: 8 text prompts → generate → wrap in a
Shadertoy/Vulkan GLSL harness → `glslangValidator -V`. `compile@1` = fraction that compile.

| | **From-scratch SLM** | **DoRA-3B** | **DoRA-32B** |
|---|---|---|---|
| Base | none (random init) | Qwen2.5-Coder-3B (frozen) | Qwen2.5-Coder-32B (frozen) |
| Params | **3.63B** (d=3072, 24L, 24H, GQA-8, head_dim 128, vocab 50304) | 3.09B base + DoRA | 32.9B base + **138.9M** DoRA = **0.42%** trainable |
| Adapter | — | PEFT `use_dora=True`, r=16, α=32, **BF16** | PEFT `use_dora=True`, r=16, α=32, **BF16** |
| Trainer | OLMo-corecpp, 2×H100 DDP (~20–26k tok/s) | HF+PEFT, BF16, both H100s (SDPA) | HF+PEFT, BF16, both H100s (pipeline-parallel) |
| Steps | 20,000 (~14 epochs over 47.6M tok) | 800 | 800 |
| **compile@1** (8 prompts) | **0.00** (0/8) | **0.125** (1/8) | **0.25** (2/8) |

> **Headline finding.** At this data scale, **pretrained + DoRA ≫ from-scratch**, and **scale within
> DoRA helps** (32B 0.25 > 3B 0.125). The from-scratch 3.6B — 20k steps / ~14 epochs on only 47.6M
> shader tokens — sits in the **data-limited / memorization regime**: loss plateaued ≈5.4 and its
> generations were incoherent (no valid GLSL for any prompt). DoRA on a strong code base produced real
> GLSL (`void mainImage(out vec4 fragColor, ...)`) that compiled for 2/8 (32B) and 1/8 (3B) after only
> 800 adapter steps. Note the 32B's DoRA adapters (138.9M) are ~the size the *entire* original 152M SLM.

**Honest caveats.** (1) `compile@1` is a coarse, low-N (8-prompt) signal — it measures *syntactic*
validity, not visual correctness; absolute numbers are modest for all three. (2) The from-scratch 0/8
reflects undertraining **and** a prompt-format mismatch (the model emitted generic English tokens, not
shader code), not only capacity — **more shader data is the real lever** (a Shadertoy API key would ~4×
the 47.6M-token corpus). (3) This is the *compile* reward term only; **OmniTrace** (this repo) enriches
the loop with per-invocation execution traces — divergence, NaN/Inf, ULP numerical diffs, visual diffs —
a far richer signal than compile/no-compile.

### Artifacts
- Committed: `results/shader_lm_comparison/` — `RESULTS.txt` + per-model `metrics.json` & `samples.txt`.
- On box (`exouser@149.165.170.159`): `~/pipeline/` (logs `0{1..4}_*.log`, `RESULTS.txt`), final SLM
  `~/OLMo-shader/runs/shader_3B/model.pt`, DoRA adapters `~/pipeline/{dora3b,dora32b}_out/adapter`.

### Infra fix shipped this run
- **Multi-rank prune race** (`6c70735`): `LocalFileSystem::remove` used the throwing `remove_all`;
  two ranks racing to prune the same checkpoint killed one rank and deadlocked DDP (one GPU idle, step
  counter frozen). Switched to the non-throwing `std::error_code` overload — verified the 3.6B run then
  trained cleanly through every checkpoint.

## Debugger-in-the-loop reward (the SIGGRAPH thesis, with the real LM)
`reward_loop.py`: the LM generates a shader → `glslangValidator` compiles it (the reward
signal) → score. The loop runs end-to-end. With **OmniTrace** (the debugger, this repo) the
reward is enriched by per-invocation execution traces (divergence / NaN / numerical / visual);
here we demonstrate it with the compile term (the `shader_cmake`-style first signal).

## DoRA — proven two ways
1. **From scratch** (`omni/ml/dora`, tested in this repo; plus a LibTorch `dora_demo` module
   built into the trainer): exact identity at init (`|W'-W0|=0`), base stays frozen
   (`drift=0`), only `m,A,B` adapters train (loss decreases). Weight-decomposed:
   `W' = m ⊙ (W0+BA)/‖W0+BA‖_c`.
2. **At scale** via PEFT `use_dora=True` (the same algorithm), **BF16 DoRA** on Qwen2.5-Coder-32B
   **and** -3B across both H100s (r=16, α=32), 800 steps each. (Switched off 4-bit QDoRA — BF16 on
   both GPUs was ~4× faster here.)

## Throughput / speed (#24) — honest findings
- **Best config: BF16 + CUDA graphs + ForeachAdamW** — ~**100k tok/s** (961M 1B), **267k tok/s**
  (152M), ~45% MFU. Already well-tuned.
- **FP8 gives NO speedup here**: the codebase's FP8 is *emulation* (quantize→dequantize STE,
  "for parity"), with no cuBLASLt FP8 GEMM — enabling it is slower, not faster.
- **`fused=1` is *slower***: 44k tok/s vs 100k (it loses the CUDA-graph path). Measured, not assumed.
- Real further speedup needs implementing FP8 GEMMs (cuBLASLt) — substantial follow-on work.

## Robustness fixes shipped to this repo
- **Multi-GPU resume** (`644e078`): `latest_complete()` skips torn checkpoints (the exact bug
  that crashed the original 2×H100 run); verified.
- **BF16 inference** (`a520af4`): upcast logits to FP32 before sampling (BF16 argmax ties were
  collapsing greedy decoding to whitespace) + BF16 model-load match (fixes the forward crash).

## The debugger (OmniTrace) — Part I, all tested locally (19 suites)
Universal IR + SPIR-V front/back end (passes `spirv-val`), CFG (dominators / post-dominators /
thread frontiers), capture pass, **divergence-aware trace codec** (lossless, 26× / 4.5×, ~entropy
bound), mmap store, CPU SIMT reference, time-travel reconstruction, ULP/divergence diff, **real
GPU capture + render via Vulkan/MoltenVK**. See `STATUS.md`.
