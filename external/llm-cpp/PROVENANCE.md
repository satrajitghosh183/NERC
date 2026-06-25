# Vendored: llm-cpp (editable copy)

This is a **vendored, editable copy** of `/Users/satrajitghosh/Projects/llm-cpp`, brought into
the shader_debugger project on 2026-06-24.

- **Edit this copy freely.** All OmniTrace ML-track changes (the DoRA module, the debugger
  reward-oracle `Callback`, RL loop hooks, shader data pipeline) happen HERE.
- **Do NOT edit the original** `/Users/satrajitghosh/Projects/llm-cpp` — it stays pristine.
- Copied: source + build config only (`src/ include/ zwt/ kernels/ tools/ cmake/ conf/
  configs/ third_party/ scripts/ docs/ CMakeLists.txt` + docs).
- Excluded (regenerable / heavy): `build/ .git/ .bench_venv/ data/ checkpoints/ runs/
  results/ benchmark_results/ olmo-python/` and all `*.so/*.a/*.o/*.npy`.

## What we will build on top (see ../../PLAN.md Part II)
- A from-scratch **`DoRALinear`** module (none exists in llm-cpp yet) — §12 of PLAN.md.
- A **debugger reward-oracle** injected via the existing `olmo_cpp` `Callback` hook — §11.
- A **GRPO/RLOO training-time loop** with the dense per-line reward — §14.
- Prefer the **ZWT (Zero-Wait Trainer)** path — LibTorch-free, CUDA-native — for purity.

## Upstream entry points (from exploration)
- `include/olmo_cpp/train.hpp` — `olmo_cpp::train(model, cfg, train_cfg, device, callbacks)`
- `include/olmo_cpp/config.hpp` — `TransformerConfig`
- `include/olmo_cpp/model/transformer.hpp` — `Transformer` / `FusedTransformer`
- `include/olmo_cpp/data/token_dataset.hpp` — `TokenDataset`
- `zwt/` — Zero-Wait Trainer (separate, LibTorch-free)
- `tools/convert_checkpoint` — HF weight import (for the 32B DoRA base)
