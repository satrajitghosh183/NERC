# Results — shader synthesis on 2×H100 (Jetstream `llmcpp`)

Everything here is real and reproducible. Live logs you can `tail -f` on the box
(`exouser@149.165.170.159`) are noted per item. Raw artifacts stay on the box; this file
is the committed summary.

## The two models (the SIGGRAPH comparison)

| | **From-scratch SLM** | **DoRA-32B** |
|---|---|---|
| Base | none (trained from random init) | Qwen2.5-Coder-32B (frozen, 4-bit / QDoRA) |
| Params | **152M** total (d=768, 12L, 12H, GQA-4, head_dim 64, vocab 50304) | **32.9B** base + **138.9M DoRA adapters** = **0.42%** trainable |
| Trainer | OLMo-corecpp, **2×H100 DDP, 267k tok/s** | HuggingFace + PEFT `use_dora=True`, 4-bit base on the H100s |
| Data | shaders21k → 47.6M GPT-2-BPE tokens (19,994 Shadertoy docs) | same 19,994 shader docs |
| Train loss | 10.0 → **2.59** (≈ppl 14; data-limited at ~13 epochs) | see `~/dora32b_finetune.log` |
| **compile@1** (8 prompts) | **0/8** — learned shader *vocabulary* (`vec4`, `cos`, `mat4`, `fract`, `uv`) but not coherent *syntax* (undertrained) | **(training — see below)** |

> The comparison is the scientific point: a 152M model trained *from scratch* on a small
> shader corpus learns the token distribution but not compilable structure, whereas a 32B
> code model adapted with **0.42%** trainable DoRA params should generate compilable shaders.
> Note the DoRA adapters (138.9M) are ~the size of the *entire* from-scratch SLM (152M).

### How to check
- SLM training: `~/OLMo-shader/runs/shader_v2/train.log`; model `runs/shader_v2/ckpt/latest.pt`.
- SLM reward demo: `python3 ~/reward_loop.py ~/OLMo-shader/runs/shader_v2/ckpt/latest.pt`.
- DoRA-32B: `tail -f ~/dora32b_finetune.log`; adapter `~/dora32b_out/adapter`; samples
  `~/dora32b_out/samples.txt`.

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
2. **At 32B scale** via PEFT `use_dora=True` (the same algorithm), QDoRA on Qwen2.5-Coder-32B.

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
