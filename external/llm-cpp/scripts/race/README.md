# Race — C++ vs Python, identical 257M backbone, on the 5060 Ti

End-to-end pipeline that builds, validates, and benchmarks the C++
implementation against the in-repo OLMo-core Python baseline at an
**identical architecture**. Both sides train OLMo-core's `llama2_271M`
shape dialed to 12 layers / 8 heads (~257M, untied embeddings); the C++
side keeps its 3 MTP heads on top (+3.2M), which is what wins inference.

## One-command run

```bash
cd <repo>         # repo root (the dir containing CMakeLists.txt + olmo-python/)
git checkout 2fast2furious && git pull
bash scripts/race/run_all.sh
```

~60–90 min on a 5060 Ti. Reads when done:

```
scripts/race/results/RESULT.md
scripts/race/results/loss_curve.png      (if matplotlib installed)
scripts/race/results/throughput.png
```

## Prerequisites (the professor installs these once)

The 5060 Ti is **Blackwell / sm_120**. That requires:

1. **CUDA Toolkit ≥ 12.8** (sm_120 support landed in 12.8):
   ```bash
   nvcc --version                       # must be ≥ 12.8
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   export CUDAToolkit_ROOT=/usr/local/cuda-12.8
   ```
2. **PyTorch with sm_120 kernels** (cu128 wheel, PyTorch ≥ 2.7):
   ```bash
   pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
   python3 -c "import torch; print(torch.cuda.get_arch_list())"   # must contain sm_120
   ```

`00_env_check.sh` enforces both — it bails with the exact fix if either
is too old, *before* the 5-minute build. The in-repo OLMo-core
(`olmo-python/`) is pip-installed automatically by phase 05.

## Phase-by-phase

Every phase logs to `scripts/race/results/<phase>/`. Skip completed
phases with `SKIP_PHASES`:

```bash
SKIP_PHASES="00 01 02 03" bash scripts/race/run_all.sh
```

| phase | script | what it does |
|---|---|---|
| 00 | `00_env_check.sh` | nvidia-smi, **enforces CUDA 12.8 + torch sm_120**, cmake/nvcc/gcc, disk |
| 01 | `01_build_cpp.sh` | `cmake -DCMAKE_CUDA_ARCHITECTURES=<gpu> -DOLMO_BUILD_KERNELS=ON` + parallel build (re-checks the Blackwell floor first) |
| 02 | `02_verify_correctness.sh` | `test_paged_kv`, `test_prefix_cache`, `test_scheduler`, `test_fused_ce`, `test_fused_qkv_rope`, and **`test_cuda_parity`** (GPU validation of the 3 correctness fixes) |
| 03 | `03_prepare_data.sh` | downloads TinyStories, GPT-2 BPE → `data/race_tokens.npy` |
| 04 | `04_train_cpp.sh` | trains C++ side (with MTP) → `metrics.csv` |
| 05 | `05_train_python.sh` | `pip install -e olmo-python`, then `torchrun olmo_train_race.py` (matched llama2_271M, no MTP) → `metrics.csv` |
| 06 | `06_infer_cpp.sh` | 5×256 tokens, paged KV + MTP speculative, median tok/s |
| 07 | `07_infer_python.sh` | 5×256 tokens, vanilla PyTorch generation, median tok/s |
| 08 | `08_analyze.py` | reads all metrics, writes `RESULT.md` + plots |

## The two sides are the same model

| | C++ (`race_250m_cpp.conf`) | Python (`olmo_train_race.py`) |
|---|---|---|
| factory | hand-matched | `TransformerConfig.llama2_271M(n_layers=12, n_heads=8)` |
| d_model | 1024 | 1024 |
| n_layers | 12 | 12 |
| n_heads | 8 (head_dim 128) | 8 (head_dim 128) |
| ffn_hidden | 2816 | 2816 |
| vocab | 50304 (GPT-2 padded) | 50304 |
| norm | RMSNorm, eps 1e-5 | RMSNorm, eps 1e-5 |
| activation | SwiGLU | SwiGLU |
| rope_theta | 10000 | 10000 |
| embeddings | untied | untied |
| **MTP heads** | **3 (+3.2M)** | **none** |
| backbone params | ~257M | ~257M |

Same lr (3e-4), weight decay (0.1), betas (0.9, 0.95), cosine schedule
with 100-step warmup, bf16, batch 32 (4 micro × 8 accum), seq 1024,
same `data/race_tokens.npy`, same seed. MTP is the only architectural
asymmetry — by design, since it's what the C++ inference side leans on.

## Which optimizations are enabled

See `OPTIMIZATIONS.md` for the full map. Short version: every kernel
optimization is compiled in (`OLMO_BUILD_KERNELS=ON`), and the config
sets the flags that matter — `fused=0` (routes through `AttentionImpl`,
which has the kernel wirings), `cuda_graph=1`, `bf16=1`,
`foreach_optimizer=1`, `gpu_data=1`.

## Python inference baseline

`python_inference_baseline.py` is a self-contained PyTorch generation
loop (KV-cached, greedy) matching the model architecture. It prints the
same `[N tokens, X tok/s]` trailer the C++ chat tool prints, so the
analyzer scrapes both identically. To benchmark vLLM / HF `generate()`
instead:

```bash
export PY_INFER_CMD="python my_vllm_runner.py"   # must accept --checkpoint --prompt
                                                  # --max-new-tokens --device
bash scripts/race/07_infer_python.sh
```

## Common failures

- **`00_env_check.sh` fails on nvcc or torch:** it prints the exact
  CUDA-12.8 / cu128 install command. Run it, re-run the phase.
- **`test_cuda_parity` fails:** a correctness fix isn't compiled in.
  Confirm `OLMO_BUILD_KERNELS=ON` and you're on commit `9e72d9a`+.
- **Phase 05 `pip install -e olmo-python` fails:** inspect the error;
  usually a missing build dep. `cd olmo-python && pip install -e .`
  surfaces it directly.
- **Data dtype mismatch in phase 05:** OLMo-core's `NumpyFSLDataset`
  expects uint16/uint32 token IDs. `03_prepare_data.sh` prints the
  dtype of `race_tokens.npy`; if OLMo-core rejects it, set the dataset
  dtype in `olmo_train_race.py`'s `NumpyFSLDatasetConfig`.
