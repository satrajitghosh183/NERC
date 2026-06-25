# OLMo-corecpp — Setup & Benchmark Guide (RTX 5060 Ti)

Build, smoke-test, and benchmark OLMo-corecpp on an RTX 5060 Ti (16 GB,
Blackwell **sm_120**), then race it against OLMo-core (training) and ollama
(inference).

> **TL;DR**
> ```bash
> ./scripts/build.sh --cuda          # 1. build
> bash scripts/race/smoke_test.sh    # 2. prove train+infer work (~2 min)
> bash scripts/race/run_all.sh       # 3. full race + RESULT.md
> ```

---

## 0. Prerequisites

| Need | Why | Check |
|------|-----|-------|
| **CUDA Toolkit ≥ 12.8** | sm_120 (Blackwell) support; older nvcc rejects `compute_120` | `nvcc --version` |
| NVIDIA driver ≥ 570 | Blackwell runtime | `nvidia-smi` |
| PyTorch (LibTorch) w/ CUDA | build + Python baseline | `python3 -c "import torch; print(torch.version.cuda)"` |
| `numpy` | race scripts read `.npy` token files | `python3 -c "import numpy"` |
| `ollama` (optional) | inference comparison | `ollama --version` |

**If your toolkit is < 12.8** and the build errors on `compute_120`, build for
Ada instead (its PTX JITs onto Blackwell — slower first launch, but runs):
```bash
cd build && cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && make -j
```

---

## 1. Build

```bash
./scripts/build.sh --cuda
```
The CUDA arch list now includes `120` (Blackwell).

> **Blackwell note:** most hand-written tensor-core kernels (WMMA/TMA FFN, fused
> QKV) are gated to `sm < 12` and **route to cuBLAS/ATen on the 5060 Ti** — that
> path is correct and fp32-accumulated, you just don't get the WGMMA kernels that
> light up on the H100. Keep `use_float8=0` until fp8 kernels exist for sm_120.

---

## 2. Smoke test — RUN THIS FIRST (~2 min)

Proves the whole stack before any real run. No big download: tokenizes a tiny
inline corpus, trains 20 steps (**asserts loss is finite and decreasing — no
NaN**), generates 16 tokens.

```bash
bash scripts/race/smoke_test.sh
```
Exits **nonzero with a clear message** on the first failure (missing binary, no
GPU, Blackwell build issue, OOM, NaN). `SMOKE TEST PASSED` ⇒ train + infer both
work on this machine.

---

## 3. Training race vs OLMo-core

```bash
bash scripts/race/run_all.sh        # → scripts/race/results/RESULT.md
```
Phases: `00` env → `01` build → `02` correctness → `03` tokenize → `04` train C++
→ `05` train Python (OLMo-core) → `06` infer C++ → `07` infer Python → `08` report.

- **VRAM auto-sizing:** phase `04` detects VRAM and picks
  `configs/race_250m_5060ti.conf` (batch 4 × grad-accum 8, `gpu_data=0`,
  `cuda_graph=0`, activation-checkpointing full — sized for 16 GB) instead of the
  H100 config. Override: `RACE_CONF=<conf> bash scripts/race/04_train_cpp.sh`.
- **Python phases (`05`,`07`) are non-fatal:** if the OLMo-core env isn't set up
  they warn and the C++ results + `RESULT.md` still complete. C++-only:
  `SKIP_PHASES="05 07" bash scripts/race/run_all.sh`.
- **OOM?** lower `batch_size` (4→2) in the 5060ti conf. **Headroom?** raise it
  (4→8, grad_accum 8→4) for ~2× throughput.

To race the OLMo-core side you need an editable install of the bundled
`olmo-python/` + deps, with its config sized to 16 GB the same way.

---

## 4. Inference race vs ollama

C++ side (MTP-speculative decoding), median tok/s over 5 trials from the phase-04
checkpoint:
```bash
bash scripts/race/06_infer_cpp.sh
```
ollama side — comparable small model, compare decode tok/s:
```bash
ollama pull llama3.2:1b
ollama run llama3.2:1b --verbose "Once upon a time" >/dev/null   # prints eval tok/s
```
Fair-number caveats: same GPU, greedy/single-stream both sides; ollama runs a
*different* model, so this is a throughput comparison, not output parity.

---

## 5. Tokenize your own corpus (optional)

`prepare_data` needs `--input` (dir of `.txt`/`.jsonl`) **or** `--download-hf`:
```bash
build/prepare_data --input data/my_corpus/ --output data/tokens.npy \
  --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --threads 8
```
Without `--vocab-file`+`--merges-file` you get the "simple" tokenizer, not GPT-2
BPE. Default cap 100 M tokens — raise with `--max-tokens`.

---

## 6. Profiling with Nsight (optional)

For head-to-head numbers prefer the **built-in CUDA-event profiler** (`profile=1`
in the conf) — it covers every training stage without Nsight's overhead skewing
results. Nsight is for one-off kernel investigations.

```bash
which nsys ncu || sudo apt install nsight-systems nsight-compute
# Drop steps to ~50 in the conf first — a full run makes a multi-GB trace.
nsys profile -o train_profile --trace=cuda,nvtx,cublas,cudnn,osrt \
  --force-overwrite=true ./build/olmo_train scripts/race/configs/race_250m_5060ti.conf
nsys stats train_profile.nsys-rep
```
GeForce cards need profiler-permissions unlocked for `ncu` (else `ERR_NVGPUCTRPERM`):
```bash
sudo tee /etc/modprobe.d/nvidia-profiler.conf <<< 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"'
sudo update-initramfs -u && sudo reboot
```
`--gpu-metrics-devices=all` may report "None of the installed GPUs are supported"
on Blackwell with older nsys — drop the flag, the timeline/kernel durations still work.

---

## Known limitations (honest notes)

- The **sm_120 build and 16 GB fit are not verified by the authors** — the smoke
  test (§2) is how you confirm them on your machine.
- The **OLMo-core Python baseline** (`05`/`07`) needs its own env + 16 GB-sized
  config; it's non-fatal in `run_all` so it can't block the C++ report.
- `cuda_graph` and `gpu_data` are **off** in the 5060 Ti config (they need
  >16 GB) — they're H100-only speedups, so C++ throughput here is below the H100.
