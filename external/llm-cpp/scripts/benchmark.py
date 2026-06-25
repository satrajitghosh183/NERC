#!/usr/bin/env python3
"""
Benchmark: OLMo-core (Python) vs llm-cpp (C++) vs cpp-llm (C++)

Measures training throughput (tok/s), forward pass latency, and memory usage
for matched model configurations across all three frameworks.

Seed control:
  - Python OLMo-core: sets torch.manual_seed() + random + numpy before training
  - llm-cpp (C++):    passes --seed <N> to binary (seeds torch + CUDA + mt19937)
  - cpp-llm (C++):    NO --seed flag available — uses torch::manual_seed() via
                       LD_PRELOAD is not feasible, so we note that cpp-llm runs
                       are non-deterministic. Throughput (tok/s) comparisons remain
                       valid; loss comparisons between cpp-llm and the others are
                       approximate.

Usage:
    python3 scripts/benchmark.py                        # all three, auto device
    python3 scripts/benchmark.py --seed 42              # reproducible run
    python3 scripts/benchmark.py --skip-install         # use existing builds
    python3 scripts/benchmark.py --model 30M            # specific model
    python3 scripts/benchmark.py --device mps           # specific device
    python3 scripts/benchmark.py --python-only
    python3 scripts/benchmark.py --cpp-only             # both C++ implementations
    python3 scripts/benchmark.py --llm-cpp-only
    python3 scripts/benchmark.py --cpp-llm-only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

ROOT = Path(__file__).parent.parent
OLMO_CORE_DIR = ROOT / "olmo-python"
BUILD_DIR = ROOT / "build"
VENV_DIR = ROOT / ".bench_venv"

# cpp-llm project (sibling directory)
CPP_LLM_DIR = Path(os.environ.get("CPP_LLM_DIR", ROOT.parent / "cpp-llm"))
CPP_LLM_BUILD_DIR = CPP_LLM_DIR / "build"

# Model configs that match between Python and both C++ implementations
MODEL_CONFIGS = {
    "30M": {
        "python_factory": "olmo2_30M",
        "cpp_config": "configs/olmo2_30M_bench.json",
        "cpp_llm_config": "configs/olmo2_30M_bench.json",
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
    },
    "100M": {
        "python_factory": "olmo2_100M",
        "cpp_config": "configs/olmo2_100M.json",
        "cpp_llm_config": "configs/olmo2_100M.json",
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
    },
}

BENCH_STEPS = 50
BATCH_SIZE = 4
SEQ_LEN = 256
WARMUP_STEPS = 5
DEFAULT_SEED = 42


@dataclass
class BenchResult:
    framework: str
    model_size: str
    device: str
    seed: Optional[int] = None
    seed_note: str = ""  # e.g. "deterministic" or "non-deterministic (no --seed flag)"
    steps: int = 0
    total_tokens: int = 0
    wall_time_s: float = 0.0
    tok_per_s: float = 0.0
    step_ms_avg: float = 0.0
    peak_memory_mb: float = 0.0
    final_loss: float = 0.0
    extra: dict = field(default_factory=dict)


def run_cmd(cmd, cwd=None, env=None, capture=True, timeout=600):
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0 and capture:
        print(f"  STDERR: {result.stderr[:500]}")
    return result


def setup_python_env():
    """Create venv and install olmo-core + torch."""
    if VENV_DIR.exists():
        print(f"  Venv already exists at {VENV_DIR}")
        return

    # Use uv if available (10-100x faster than pip), fall back to pip
    uv = shutil.which("uv")

    if uv:
        print("  Creating venv with uv (Python 3.12)...")
        run_cmd([uv, "venv", "--python", "3.12", str(VENV_DIR)], timeout=30)
        python = VENV_DIR / "bin" / "python"

        print("  Installing PyTorch + OLMo-core with uv...")
        run_cmd([uv, "pip", "install", "--python", str(python),
                 "torch", "rich"], timeout=300)
        run_cmd([uv, "pip", "install", "--python", str(python),
                 "-e", str(OLMO_CORE_DIR)], timeout=300)
    else:
        print("  uv not found, using pip (install uv for 10x faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)")
        # Use python3.12 if available (OLMo-core needs >=3.10)
        py = shutil.which("python3.12") or shutil.which("python3.14") or sys.executable
        run_cmd([py, "-m", "venv", str(VENV_DIR)])
        python = VENV_DIR / "bin" / "python"
        pip = VENV_DIR / "bin" / "pip"

        print("  Upgrading pip...")
        run_cmd([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], timeout=120)

        print("  Installing PyTorch + OLMo-core...")
        run_cmd([pip, "install", "torch", "rich"], timeout=300)
        run_cmd([pip, "install", "-e", str(OLMO_CORE_DIR)], timeout=300)

    # Verify
    result = run_cmd([python, "-c", "import olmo_core; import torch; print('OK')"])
    if result.returncode != 0:
        print("  ERROR: Failed to set up Python environment")
        sys.exit(1)
    print("  Python environment ready.")


def build_cpp():
    """Build the llm-cpp C++ project."""
    if not (BUILD_DIR / "olmo_train").exists():
        print("  Building llm-cpp project...")
        run_cmd(["cmake", "--build", str(BUILD_DIR)], cwd=ROOT, timeout=300)
    else:
        print("  llm-cpp build exists, rebuilding...")
        run_cmd(["cmake", "--build", str(BUILD_DIR)], cwd=ROOT, timeout=300)


def build_cpp_llm():
    """Build the cpp-llm C++ project."""
    if not CPP_LLM_DIR.exists():
        print(f"  cpp-llm directory not found at {CPP_LLM_DIR}")
        print(f"  Set CPP_LLM_DIR env var to override, or clone it next to llm-cpp")
        return False

    if not (CPP_LLM_BUILD_DIR / "olmo_train").exists():
        print(f"  Building cpp-llm project at {CPP_LLM_DIR}...")
        run_cmd(["cmake", "--build", str(CPP_LLM_BUILD_DIR)], cwd=CPP_LLM_DIR, timeout=300)
    else:
        print(f"  cpp-llm build exists, rebuilding...")
        run_cmd(["cmake", "--build", str(CPP_LLM_BUILD_DIR)], cwd=CPP_LLM_DIR, timeout=300)
    return True


def generate_random_data(path: Path, num_tokens: int = 500_000):
    """Generate random token data for benchmarking."""
    if path.exists():
        print(f"  Data already exists at {path}")
        return

    print(f"  Generating {num_tokens} random tokens at {path}...")
    prepare_data = BUILD_DIR / "prepare_data"
    if prepare_data.exists():
        run_cmd([prepare_data, "--random", str(num_tokens), "--output", str(path)])
    else:
        # Fallback: use numpy
        import numpy as np
        tokens = np.random.randint(0, 50257, size=num_tokens, dtype=np.uint16)
        np.save(str(path), tokens)


def bench_python(model_size: str, device: str, data_path: Path, seed: int = DEFAULT_SEED) -> Optional[BenchResult]:
    """Benchmark OLMo-core Python training."""
    config = MODEL_CONFIGS[model_size]
    python = VENV_DIR / "bin" / "python"

    if not python.exists():
        print("  Python venv not found, skipping Python benchmark")
        return None

    # Write a minimal benchmark script
    bench_script = ROOT / "scripts" / "_bench_python_inner.py"
    bench_script.write_text(f'''
import time
import sys
import random
import numpy as np
import torch

# ── Seed all RNGs (mirrors olmo_core.utils.seed_all) ──
SEED = {seed}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"SEED={{SEED}}")

# Device selection
device_str = "{device}"
if device_str == "mps":
    if not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device("mps")
elif device_str == "cuda":
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Try to import olmo_core
try:
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.config import DType

    factory = getattr(TransformerConfig, "{config['python_factory']}")
    model_config = factory(vocab_size=50304)  # padded to multiple of 128
    model = model_config.build(init_device="cpu")
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MODEL_PARAMS={{n_params}}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training loop
    batch_size = {BATCH_SIZE}
    seq_len = {SEQ_LEN}
    num_steps = {BENCH_STEPS}
    warmup = {WARMUP_STEPS}

    total_tokens = 0
    losses = []

    # Warmup
    for i in range(warmup):
        input_ids = torch.randint(0, 50304, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 50304, (batch_size, seq_len), device=device)
        out = model(input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Sync before timing
    if device_str == "mps":
        torch.mps.synchronize()
    elif device_str == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for step in range(num_steps):
        input_ids = torch.randint(0, 50304, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 50304, (batch_size, seq_len), device=device)

        out = model(input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len
        losses.append(loss.item())

    # Sync after timing
    if device_str == "mps":
        torch.mps.synchronize()
    elif device_str == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    tok_per_s = total_tokens / elapsed
    step_ms = (elapsed / num_steps) * 1000

    # Memory
    peak_mem = 0
    if device_str == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
    elif device_str == "mps":
        try:
            peak_mem = torch.mps.current_allocated_memory() / 1e6
        except:
            pass

    print(f"RESULT_STEPS={{num_steps}}")
    print(f"RESULT_TOKENS={{total_tokens}}")
    print(f"RESULT_TIME={{elapsed:.3f}}")
    print(f"RESULT_TOKS={{tok_per_s:.1f}}")
    print(f"RESULT_STEP_MS={{step_ms:.1f}}")
    print(f"RESULT_MEMORY={{peak_mem:.1f}}")
    print(f"RESULT_LOSS={{losses[-1]:.4f}}")

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
''')

    print(f"\n  Running Python benchmark ({model_size}, {device})...")
    result = run_cmd([python, str(bench_script)], cwd=ROOT, timeout=600)

    if result.returncode != 0:
        print(f"  Python benchmark failed: {result.stderr[:300]}")
        return None

    # Parse results
    bench = BenchResult(framework="OLMo-core (Python)", model_size=model_size, device=device,
                        seed=seed, seed_note="deterministic")
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_STEPS="):
            bench.steps = int(line.split("=")[1])
        elif line.startswith("RESULT_TOKENS="):
            bench.total_tokens = int(line.split("=")[1])
        elif line.startswith("RESULT_TIME="):
            bench.wall_time_s = float(line.split("=")[1])
        elif line.startswith("RESULT_TOKS="):
            bench.tok_per_s = float(line.split("=")[1])
        elif line.startswith("RESULT_STEP_MS="):
            bench.step_ms_avg = float(line.split("=")[1])
        elif line.startswith("RESULT_MEMORY="):
            bench.peak_memory_mb = float(line.split("=")[1])
        elif line.startswith("RESULT_LOSS="):
            bench.final_loss = float(line.split("=")[1])
        elif line.startswith("MODEL_PARAMS="):
            bench.extra["params"] = int(line.split("=")[1])

    # Cleanup temp script
    bench_script.unlink(missing_ok=True)
    return bench


def _parse_cpp_output(stdout: str, framework: str, model_size: str, device: str) -> BenchResult:
    """Parse training output from either C++ implementation (same output format)."""
    bench = BenchResult(framework=framework, model_size=model_size, device=device)
    bench.steps = BENCH_STEPS

    for line in stdout.splitlines():
        if "tok/s:" in line:
            # Parse: "Step 40/55  loss: 10.8447  lr: ...  step_ms: 244  tok/s: 3305  ETA: 3m"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "loss:" and i + 1 < len(parts):
                    bench.final_loss = float(parts[i + 1])
                elif p == "tok/s:" and i + 1 < len(parts):
                    bench.tok_per_s = float(parts[i + 1])
                elif p == "step_ms:" and i + 1 < len(parts):
                    bench.step_ms_avg = float(parts[i + 1])
        elif "Total tokens:" in line:
            bench.total_tokens = int(line.split(":")[1].strip())
        elif "Wall time:" in line:
            # "  Wall time: 13s (0m 13s)"
            time_str = line.split(":")[1].strip().split("s")[0]
            bench.wall_time_s = float(time_str)
        elif "Throughput:" in line:
            bench.tok_per_s = float(line.split(":")[1].strip().split()[0])

    return bench


def bench_cpp(model_size: str, device: str, data_path: Path,
              seed: int = DEFAULT_SEED, extra_flags: list = None) -> Optional[BenchResult]:
    """Benchmark llm-cpp C++ training."""
    config = MODEL_CONFIGS[model_size]
    train_bin = BUILD_DIR / "olmo_train"

    if not train_bin.exists():
        print("  llm-cpp binary not found, skipping llm-cpp benchmark")
        return None

    cpp_config = ROOT / config["cpp_config"]
    if not cpp_config.exists():
        print(f"  Config {cpp_config} not found, skipping")
        return None

    # Map device names
    cpp_device = device
    if device == "metal":
        cpp_device = "mps"

    cmd = [
        train_bin,
        "--train",
        "--config", str(cpp_config),
        "--device", cpp_device,
        "--batch-size", str(BATCH_SIZE),
        "--seq-len", str(SEQ_LEN),
        "--steps", str(BENCH_STEPS + WARMUP_STEPS),
        "--lr", "1e-4",
        "--warmup-steps", str(WARMUP_STEPS),
        "--optimizer", "adamw",
        "--seed", str(seed),
    ]

    if extra_flags:
        cmd.extend(extra_flags)

    if data_path.exists():
        cmd.extend(["--data-path", str(data_path)])

    label = "llm-cpp (C++)"
    if extra_flags:
        parts = []
        if "--fused" in extra_flags:
            parts.append("fused")
        if "--mup" in extra_flags:
            parts.append("µP")
        if "--multi-res" in extra_flags:
            parts.append("DC-MRE")
        if parts:
            label = f"llm-cpp {'+'.join(parts)} (C++)"

    print(f"\n  Running {label} benchmark ({model_size}, {device}, seed={seed})...")
    result = run_cmd(cmd, cwd=ROOT, timeout=600)

    if result.returncode != 0:
        print(f"  {label} benchmark failed: {result.stderr[:300]}")
        return None

    bench = _parse_cpp_output(result.stdout, label, model_size, device)
    bench.seed = seed
    bench.seed_note = "deterministic"
    return bench


def bench_cpp_llm(model_size: str, device: str, data_path: Path,
                  seed: int = DEFAULT_SEED) -> Optional[BenchResult]:
    """Benchmark cpp-llm C++ training.

    NOTE: cpp-llm does not support --seed or --optimizer flags.
    It always uses AdamW and non-deterministic initialization.
    Throughput comparisons are valid; loss comparisons are approximate.
    """
    config = MODEL_CONFIGS[model_size]
    train_bin = CPP_LLM_BUILD_DIR / "olmo_train"

    if not train_bin.exists():
        print(f"  cpp-llm binary not found at {train_bin}, skipping cpp-llm benchmark")
        return None

    cpp_config = CPP_LLM_DIR / config["cpp_llm_config"]
    if not cpp_config.exists():
        print(f"  Config {cpp_config} not found, skipping")
        return None

    # Map device names
    cpp_device = device
    if device == "metal":
        cpp_device = "mps"

    # cpp-llm does not support --seed or --optimizer flags; uses AdamW by default
    cmd = [
        train_bin,
        "--train",
        "--config", str(cpp_config),
        "--device", cpp_device,
        "--batch-size", str(BATCH_SIZE),
        "--seq-len", str(SEQ_LEN),
        "--steps", str(BENCH_STEPS + WARMUP_STEPS),
        "--lr", "1e-4",
        "--warmup-steps", str(WARMUP_STEPS),
    ]

    if data_path.exists():
        cmd.extend(["--data-path", str(data_path)])

    print(f"\n  Running cpp-llm benchmark ({model_size}, {device}, seed=N/A)...")
    print(f"  NOTE: cpp-llm has no --seed flag; run is non-deterministic")
    result = run_cmd(cmd, cwd=CPP_LLM_DIR, timeout=600)

    if result.returncode != 0:
        print(f"  cpp-llm benchmark failed: {result.stderr[:300]}")
        return None

    bench = _parse_cpp_output(result.stdout, "cpp-llm (C++)", model_size, device)
    bench.seed = None
    bench.seed_note = "non-deterministic (no --seed flag)"
    return bench


def print_results(results: list, seed: int):
    """Print formatted benchmark comparison."""
    print("\n" + "=" * 94)
    print("  BENCHMARK RESULTS")
    print("=" * 94)

    # Group by model size
    sizes = set(r.model_size for r in results)
    for size in sorted(sizes):
        size_results = [r for r in results if r.model_size == size]
        print(f"\n  Model: {size}")
        print(f"  Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, steps={BENCH_STEPS}, seed={seed}")
        print("  " + "-" * 90)
        print(f"  {'Framework':<30} {'tok/s':>10} {'step_ms':>10} {'wall_time':>10} {'memory_mb':>10} {'loss':>10}")
        print("  " + "-" * 90)

        py_result = None
        cpp_results = {}
        baseline_result = None  # cpp-llm as baseline C++

        for r in size_results:
            params_str = ""
            if "params" in r.extra:
                params_str = f" ({r.extra['params']/1e6:.1f}M params)"
            mem_str = f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb > 0 else "n/a"
            seed_marker = "*" if r.seed is None else ""
            print(f"  {r.framework:<30} {r.tok_per_s:>10.0f} {r.step_ms_avg:>10.1f} {r.wall_time_s:>10.1f}s {mem_str:>10} {r.final_loss:>10.4f}{seed_marker}{params_str}")

            if "Python" in r.framework:
                py_result = r
            else:
                cpp_results[r.framework] = r
            if "cpp-llm" in r.framework:
                baseline_result = r

        # Pairwise comparisons
        print("  " + "-" * 90)

        # Each framework vs Python
        if py_result and py_result.tok_per_s > 0:
            for name, cpp_r in sorted(cpp_results.items()):
                if cpp_r.tok_per_s > 0:
                    speedup = cpp_r.tok_per_s / py_result.tok_per_s
                    label = f"{name} vs Python"
                    if speedup >= 1:
                        print(f"  {label}: {speedup:.2f}x faster")
                    else:
                        print(f"  {label}: {1/speedup:.2f}x slower")

        # Each llm-cpp variant vs cpp-llm baseline
        if baseline_result and baseline_result.tok_per_s > 0:
            for name, cpp_r in sorted(cpp_results.items()):
                if "llm-cpp" in name and cpp_r.tok_per_s > 0:
                    speedup = cpp_r.tok_per_s / baseline_result.tok_per_s
                    label = f"{name} vs cpp-llm"
                    if speedup >= 1:
                        print(f"  {label}: {speedup:.2f}x faster")
                    else:
                        print(f"  {label}: {1/speedup:.2f}x slower")

        # Seed notes
        non_det = [r for r in size_results if r.seed is None]
        if non_det:
            print(f"\n  * = non-deterministic (no --seed support); loss not directly comparable")

    print("\n" + "=" * 94)

    # Save results to JSON
    results_file = ROOT / "benchmark_results.json"
    results_data = []
    for r in results:
        results_data.append({
            "framework": r.framework,
            "model_size": r.model_size,
            "device": r.device,
            "seed": r.seed,
            "seed_note": r.seed_note,
            "tok_per_s": r.tok_per_s,
            "step_ms_avg": r.step_ms_avg,
            "wall_time_s": r.wall_time_s,
            "final_loss": r.final_loss,
            "total_tokens": r.total_tokens,
            "peak_memory_mb": r.peak_memory_mb,
            "config": {
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "steps": BENCH_STEPS,
                "seed": seed,
            }
        })
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to {results_file}")


def main():
    global BENCH_STEPS, CPP_LLM_DIR, CPP_LLM_BUILD_DIR

    parser = argparse.ArgumentParser(
        description="Benchmark OLMo-core (Python) vs llm-cpp (C++) vs cpp-llm (C++)")
    parser.add_argument("--model", default="30M", choices=list(MODEL_CONFIGS.keys()),
                        help="Model size to benchmark")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, mps, cuda, or auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Global RNG seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip Python env setup and C++ builds")
    parser.add_argument("--all-variants", action="store_true",
                        help="Also benchmark llm-cpp fused, fused+µP, and DC-MRE variants")
    parser.add_argument("--bpe-vocab", type=str, default="data/gpt2/vocab.json",
                        help="BPE vocab.json for DC-MRE char trigram extraction")
    parser.add_argument("--cpp-only", action="store_true",
                        help="Only benchmark both C++ implementations")
    parser.add_argument("--python-only", action="store_true",
                        help="Only benchmark Python")
    parser.add_argument("--llm-cpp-only", action="store_true",
                        help="Only benchmark llm-cpp")
    parser.add_argument("--cpp-llm-only", action="store_true",
                        help="Only benchmark cpp-llm")
    parser.add_argument("--steps", type=int, default=BENCH_STEPS,
                        help="Number of benchmark steps")
    parser.add_argument("--cpp-llm-dir", type=str, default=None,
                        help="Path to cpp-llm project (default: ../cpp-llm)")
    args = parser.parse_args()

    BENCH_STEPS = args.steps
    if args.cpp_llm_dir:
        CPP_LLM_DIR = Path(args.cpp_llm_dir)
        CPP_LLM_BUILD_DIR = CPP_LLM_DIR / "build"

    # Determine which frameworks to run
    run_python = True
    run_llm_cpp = True
    run_cpp_llm = True

    if args.python_only:
        run_llm_cpp = False
        run_cpp_llm = False
    elif args.cpp_only:
        run_python = False
    elif args.llm_cpp_only:
        run_python = False
        run_cpp_llm = False
    elif args.cpp_llm_only:
        run_python = False
        run_llm_cpp = False

    # Auto-detect device
    device = args.device
    if device == "auto":
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            device = "mps"
        else:
            device = "cpu"

    seed = args.seed

    frameworks = []
    if run_python:
        frameworks.append("OLMo-core (Python)")
    if run_llm_cpp:
        frameworks.append("llm-cpp (C++)")
        if args.all_variants:
            frameworks.append("llm-cpp fused (C++)")
            frameworks.append("llm-cpp fused+µP (C++)")
            frameworks.append("llm-cpp DC-MRE (C++)")
            frameworks.append("llm-cpp fused+µP+DC-MRE (C++)")
    if run_cpp_llm:
        frameworks.append("cpp-llm (C++)")

    print(f"\n{'='*94}")
    print(f"  OLMo 3-Way Benchmark: {args.model} model on {device}")
    print(f"  Seed: {seed} (cpp-llm: non-deterministic)")
    print(f"  Frameworks: {', '.join(frameworks)}")
    print(f"  Steps: {BENCH_STEPS}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}")
    print(f"{'='*94}")

    # Setup
    step_num = 0
    total_setup_steps = sum([
        run_python and not args.skip_install,
        run_llm_cpp and not args.skip_install,
        run_cpp_llm and not args.skip_install,
        True,  # data generation
    ])

    if not args.skip_install:
        if run_python:
            step_num += 1
            print(f"\n[{step_num}/{total_setup_steps}] Setting up Python environment...")
            setup_python_env()
        if run_llm_cpp:
            step_num += 1
            print(f"\n[{step_num}/{total_setup_steps}] Building llm-cpp...")
            build_cpp()
        if run_cpp_llm:
            step_num += 1
            print(f"\n[{step_num}/{total_setup_steps}] Building cpp-llm...")
            if not build_cpp_llm():
                print("  WARNING: cpp-llm build failed, will skip cpp-llm benchmark")
                run_cpp_llm = False

    # Generate benchmark data
    data_path = ROOT / "data" / "bench_tokens.npy"
    step_num += 1
    print(f"\n[{step_num}/{total_setup_steps}] Preparing benchmark data...")
    generate_random_data(data_path)

    # Run benchmarks
    results = []

    if run_python:
        py_result = bench_python(args.model, device, data_path, seed=seed)
        if py_result:
            results.append(py_result)

    if run_cpp_llm:
        cpp_llm_result = bench_cpp_llm(args.model, device, data_path, seed=seed)
        if cpp_llm_result:
            results.append(cpp_llm_result)

    if run_llm_cpp:
        # Standard llm-cpp (AdamW, same as cpp-llm for fair comparison)
        cpp_result = bench_cpp(args.model, device, data_path, seed=seed)
        if cpp_result:
            results.append(cpp_result)

        # Optional: fused, fused+µP, DC-MRE variants
        if args.all_variants:
            fused_result = bench_cpp(args.model, device, data_path, seed=seed,
                                     extra_flags=["--fused"])
            if fused_result:
                results.append(fused_result)

            fused_mup_result = bench_cpp(args.model, device, data_path, seed=seed,
                                          extra_flags=["--fused", "--mup"])
            if fused_mup_result:
                results.append(fused_mup_result)

            bpe_vocab = str(ROOT / args.bpe_vocab)
            dcmre_result = bench_cpp(args.model, device, data_path, seed=seed,
                                      extra_flags=["--multi-res", "--bpe-vocab", bpe_vocab])
            if dcmre_result:
                results.append(dcmre_result)

            full_result = bench_cpp(args.model, device, data_path, seed=seed,
                                     extra_flags=["--fused", "--mup", "--multi-res",
                                                   "--bpe-vocab", bpe_vocab])
            if full_result:
                results.append(full_result)

    if results:
        print_results(results, seed=seed)
    else:
        print("\n  No benchmark results collected. Check errors above.")


if __name__ == "__main__":
    main()
