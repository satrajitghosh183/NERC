# QUICKSTART.md — Quick-start guide

This file is a focused walkthrough for someone who has never run the
project before, on Linux (Pop!_OS, Ubuntu, etc.) with an NVIDIA RTX 3060.
For a deeper architectural overview, see `CLAUDE.md`.

---

## TL;DR — six commands

```bash
# 1. Build (CUDA auto-detected; sm_86 = RTX 3060 is in the default arch list).
./scripts/build.sh --cuda

# 2. Fetch GPT-2 BPE files. Any method works; HF hub is easiest.
mkdir -p data/gpt2
python3 -c '
from huggingface_hub import hf_hub_download; import shutil, os
for f in ("vocab.json", "merges.txt"):
    shutil.copy(hf_hub_download(repo_id="gpt2", filename=f),
                os.path.join("data/gpt2", f))
'

# 3. Tokenize TinyStories into the .npy tensor the loader memory-maps.
./build/prepare_data \
  --download-hf  roneneldan/TinyStories \
  --output       data/tinystories_gpt2.npy \
  --vocab-file   data/gpt2/vocab.json \
  --merges-file  data/gpt2/merges.txt

# 4. Train (≈ 10 min on a 3060). profile=1 in the .conf prints a per-stage
#    timing table at the end.
mkdir -p logs checkpoints
./build/olmo_train conf/quickstart_3060.conf 2>&1 | tee logs/train.log

# 5. Dump the embedding matrix for inspection.
mkdir -p exports
./build/dump_embeddings \
  --conf  conf/quickstart_3060.conf \
  --ckpt  checkpoints/quickstart_3060.pt \
  --out   exports/embeddings \
  --vocab data/gpt2/vocab.json

# 6. (optional) print parameter names + shapes from a fresh model:
./build/dump_params
```

---

## What you'll see at the end

1. **`logs/train.log`** — full stdout, including the profile table:
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ Training Profile                                            │
   ├──────────────────────┬────────┬──────────┬──────────┬───────┤
   │ Region               │ Calls  │ Total ms │ Mean ms  │ %     │
   ├──────────────────────┼────────┼──────────┼──────────┼───────┤
   │ forward              │  2000  │  47812.1 │   23.91  │ 41.3% │
   │ backward             │  2000  │  53210.5 │   26.61  │ 45.9% │
   │ optimizer_step       │  2000  │   8941.2 │    4.47  │  7.7% │
   │ data_loading         │  2000  │   2102.7 │    1.05  │  1.8% │
   │ allreduce            │     0  │      0.0 │    0.00  │  0.0% │
   │ step_total           │  2000  │ 116066.5 │   58.03  │  ...  │
   └──────────────────────┴────────┴──────────┴──────────┴───────┘
   ```
   Plus a peak GPU memory line.

2. **`checkpoints/quickstart_3060.pt`** — model weights (LibTorch
   serialised module).

3. **`exports/embeddings.npy`** — `[50257, 256]` float32 array of input
   token embeddings. Load it with NumPy:
   ```python
   import numpy as np
   E = np.load("exports/embeddings.npy")
   print(E.shape, E.dtype)            # (50257, 256) float32
   norms = np.linalg.norm(E, axis=1)
   print("most-trained:", norms.argsort()[-10:])
   ```

4. **`exports/embeddings_summary.txt`** — already pre-computed:
   global statistics, plus the 10 highest-norm and 10 lowest-norm
   token rows (with their decoded BPE strings if the GPT-2 vocab.json
   was found).

5. **`exports/embeddings_norms.txt`** — one float per line, the L2 norm
   of each row. Useful for quick `gnuplot` plots.

---

## What the profile table tells you

`src/profiler.hpp` defines a thread-safe scope-timer. Every meaningful
stage in `src/train.cpp` is wrapped in a `ProfileScope`:

| Region            | What it measures                                       |
|-------------------|--------------------------------------------------------|
| `step_total`      | full microbatch cycle (data → fwd → bwd → opt)         |
| `data_loading`    | `DataLoader::next_batch()` only                        |
| `forward`         | `model->forward(...)` only                             |
| `backward`        | `loss.backward()` only                                 |
| `allreduce`       | DDP gradient sync (always 0 on a single GPU)           |
| `optimizer_step`  | optimizer + LR schedule + gradient clipping            |

Counts are the number of optimizer steps; mean-ms tells you the
amortised cost per step. The profiler is always on; overhead is
< 0.1% (just two `chrono::high_resolution_clock::now()` calls per
scope).

If you want finer granularity (e.g. attention vs FFN inside the
forward), add more `ProfileScope` lines in `src/model/block.cpp` or
`src/model/fused_block.cpp` — comments in those files mark the natural
boundaries.

---

## Hardware sizing

| Knob in `quickstart_3060.conf` | Default | Why                       |
|-------------------------------|---------|---------------------------|
| `d_model`                     | 256     | 30M params with vocab=50k |
| `n_layers`                    | 4       | "                         |
| `n_heads`                     | 8       | head_dim = 32             |
| `batch_size`                  | 8       | ≈ 2 GB activations        |
| `seq_len`                     | 256     | "                         |
| `steps`                       | 2000    | ~10 min on 3060           |

Peak VRAM under these settings: ~2.5 GB. The 3060 has 12 GB so there
is plenty of headroom — bump `batch_size` to 32 if you want.

To switch to a 125M model, point at `conf/olmo_125M.conf` instead;
it is also 3060-tuned (12 GB) and uses activation checkpointing.

---

## Directory tour

The repo is C++17 + LibTorch (no Python in the training path). Every
file has a docblock at the top in this format:

```
/**
 * <relative-path>
 *
 * <one-paragraph purpose>
 *
 * --- Includes from this project ---
 *   - <header>: <what's used and why>
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - <caller_path>: <which symbol used, training-pipeline context>
 *
 * --- Role in training pipeline ---
 *   <2-3 sentences>
 */
```

So you can read any file top-to-bottom and immediately know where it
sits in the pipeline. Inline comments cover every logical block in
function bodies — for the heavier files (optimizer math, MoE routing,
activation checkpointing, distributed collectives) the comments
include the formula or the collective name.

The most-read entry points:

| File                              | What it does                                  |
|-----------------------------------|-----------------------------------------------|
| `src/main.cpp`                    | CLI; parses .conf; instantiates model         |
| `src/train.cpp`                   | Training loop; where ProfileScope lives       |
| `src/model/transformer.cpp`       | Reference Transformer (one Block per layer)   |
| `src/model/fused_transformer.cpp` | Fused QKV + gate_up FFN variant               |
| `src/model/block.cpp`             | One transformer block (attn + ffn + norm)     |
| `src/model/attention.cpp`         | Plain GQA attention                           |
| `src/model/fused_attention.cpp`   | Fused-QKV attention                           |
| `src/data/token_dataset.cpp`      | Memory-maps the .npy token tensor             |
| `src/optim/muon.cpp`              | Newton-Schulz orthogonalised momentum         |
| `src/profiler.cpp`                | Always-on per-region timing                   |

---

## Inspecting embeddings — what to look for

The pre-baked `exports/embeddings_summary.txt` is the fastest signal:

* If the **bottom-10 norms** are all close to `init_std * sqrt(d_model)`
  (≈ 0.32 for d_model=256, init_std=0.02), those token rows haven't
  received many updates — they are rare in TinyStories. This is
  expected behaviour, not a bug.
* If the **top-10 norms** are dominated by frequent words ("the", " a",
  punctuation), training is healthy and the gradient signal is reaching
  the embedding table.
* If everything is near init even after 2000 steps, something is wrong
  upstream — check that `data_path` is non-empty and that the loss
  decreased in `logs/train.log`.

For deeper analysis (PCA, t-SNE, nearest-neighbour probes), load
`exports/embeddings.npy` in any Python REPL — it's just a NumPy array.

---

## Re-running individual stages

```bash
# rebuild (after editing C++):
./scripts/build.sh --cuda

# train only (after rebuild):
./build/olmo_train conf/quickstart_3060.conf

# dump embeddings only (any time after a checkpoint exists):
./build/dump_embeddings \
  --conf  conf/quickstart_3060.conf \
  --ckpt  checkpoints/quickstart_3060.pt \
  --out   exports/embeddings \
  --vocab data/gpt2/vocab.json

# print model parameter names + shapes (no checkpoint needed):
./build/dump_params

# count tokens in the .npy file:
./build/inspect_tokens --tokens data/tinystories_gpt2.npy
```

---

## Troubleshooting

* **CMake says "torch not found"** — install LibTorch via
  `pip install torch` or set `CMAKE_PREFIX_PATH` to a libtorch dir.
* **`nvcc not found`** — drop `--cuda` from the build command and use
  `./scripts/build.sh --cpu`. Training works but is much slower.
  Install CUDA Toolkit ≥ 12.0 to get GPU support
  (`apt install nvidia-cuda-toolkit` on Pop!_OS).
* **`undefined symbol: __cudaRegisterFatBinary…` at runtime** — the
  built binary expects a CUDA library that isn't on the runtime path.
  Add `LD_LIBRARY_PATH=$(python3 -c "import torch, os;
  print(os.path.dirname(torch.__file__))")/lib`.
* **Profile table prints but loss didn't decrease** — verify with
  `head logs/train.log`. If `data_path` was wrong, the loader will
  fail loudly; if the LR is too high, you'll see NaN.
