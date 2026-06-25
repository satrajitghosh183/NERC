#!/usr/bin/env python3
"""scripts/race/08_analyze.py

Reads the metrics CSVs produced by 04_train_cpp.sh, 05_train_python.sh,
06_infer_cpp.sh, 07_infer_python.sh and emits:
  - results/RESULT.md     (human-readable summary)
  - results/loss_curve.png (training loss, both sides)
  - results/throughput.png (step-time / tok/s, both sides)

If matplotlib isn't installed, plots are skipped and the report text
still lands.
"""

import argparse
import csv
import os
import statistics
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fmt(x, prec=1):
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scripts/race/results")
    args = ap.parse_args()
    root = Path(args.root)

    cpp_train = read_csv(root / "cpp_train" / "metrics.csv")
    py_train  = read_csv(root / "python_train" / "metrics.csv")
    cpp_inf   = read_csv(root / "cpp_infer" / "results.csv")
    py_inf    = read_csv(root / "python_infer" / "results.csv")

    lines = []
    push = lines.append

    push("# Race results — 250M, identical backbone (C++ has MTP, Python doesn't)\n")
    push(f"Generated from `{root}`.\n")

    # ── Training ────────────────────────────────────────────────────
    push("## Training\n")
    push("Both sides train the same 16-layer / d=1024 / ffn=2816 / vocab=50257 backbone")
    push("with tied embeddings on the same `data/race_tokens.npy` corpus, same seed,")
    push("AdamW lr=3e-4, weight decay 0.1, batch 32 (4×8 grad-accum), seq 1024.\n")

    push("| metric | C++ | Python | speedup |")
    push("| --- | --- | --- | --- |")

    def summarize_train(rows, has_step_ms=True):
        if not rows:
            return {"steps": 0, "final_loss": None, "median_tok_s": None,
                    "median_step_ms": None}
        try:
            speeds = [float(r["tok_per_s"]) for r in rows
                       if r.get("tok_per_s") and r["tok_per_s"] != "nan"]
            step_ms = ([float(r["step_ms"]) for r in rows] if has_step_ms
                       else [])
            return {
                "steps": len(rows),
                "final_loss": float(rows[-1]["loss"]),
                "median_tok_s": statistics.median(speeds) if speeds else None,
                "median_step_ms": statistics.median(step_ms) if step_ms else None,
            }
        except (KeyError, ValueError):
            return {"steps": len(rows), "final_loss": None,
                    "median_tok_s": None, "median_step_ms": None}

    c = summarize_train(cpp_train, has_step_ms=True)
    p = summarize_train(py_train, has_step_ms=False)

    push(f"| steps | {c['steps']} | {p['steps']} | — |")
    push(f"| final loss | {fmt(c['final_loss'], 4) if c['final_loss'] else '—'} "
         f"| {fmt(p['final_loss'], 4) if p['final_loss'] else '—'} | — |")
    if c['median_tok_s'] and p['median_tok_s']:
        spd = c['median_tok_s'] / p['median_tok_s']
        push(f"| median tok/s | {fmt(c['median_tok_s'])} | {fmt(p['median_tok_s'])} "
             f"| **{spd:.2f}×** |")
    else:
        push(f"| median tok/s | {fmt(c['median_tok_s']) if c['median_tok_s'] else '—'} "
             f"| {fmt(p['median_tok_s']) if p['median_tok_s'] else '—'} | — |")

    if c['median_step_ms']:
        push(f"| median step time (C++) | {fmt(c['median_step_ms'])} ms | — | — |")

    push("")

    # ── Inference ───────────────────────────────────────────────────
    push("## Inference\n")
    push("Both sides generate 256 tokens (greedy) from the same prompt, 5 trials each.")
    push("C++ uses paged KV + MTP speculative decoding. Python uses vanilla KV-cached")
    push("generation in `python_inference_baseline.py`.\n")

    def summarize_infer(rows):
        if not rows:
            return None
        speeds = [float(r["tok_per_s"]) for r in rows]
        accepts = [float(r["accept_rate"]) for r in rows] if "accept_rate" in rows[0] else None
        return {
            "trials": len(rows),
            "median": statistics.median(speeds),
            "mean":   statistics.mean(speeds),
            "min":    min(speeds),
            "max":    max(speeds),
            "median_accept": statistics.median(accepts) if accepts else None,
        }

    ci = summarize_infer(cpp_inf)
    pi = summarize_infer(py_inf)

    push("| metric | C++ | Python | speedup |")
    push("| --- | --- | --- | --- |")
    if ci and pi:
        spd = ci['median'] / pi['median']
        push(f"| median tok/s | {fmt(ci['median'])} | {fmt(pi['median'])} | **{spd:.2f}×** |")
        push(f"| mean tok/s | {fmt(ci['mean'])} | {fmt(pi['mean'])} | — |")
        push(f"| min / max tok/s | {fmt(ci['min'])} / {fmt(ci['max'])} "
             f"| {fmt(pi['min'])} / {fmt(pi['max'])} | — |")
        if ci.get('median_accept') is not None:
            push(f"| MTP accept rate (C++) | {fmt(ci['median_accept'], 0)}% | — | — |")
        push(f"| trials | {ci['trials']} | {pi['trials']} | — |")
    else:
        push("| _no inference results yet_ |  |  |  |")

    push("")

    # ── Header info ─────────────────────────────────────────────────
    push("## Run info\n")
    import platform, subprocess
    try:
        nvsmi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip()
        push(f"```\n{nvsmi}\n```\n")
    except Exception:
        pass
    push(f"Host: `{platform.node()}` — {platform.platform()}\n")

    # Write report.
    out = root / "RESULT.md"
    out.write_text("\n".join(lines))
    print(f"wrote: {out}")
    print("\n--- RESULT.md ---")
    print(out.read_text())

    # ── Plots (optional) ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not installed; skipping plots")
        return

    if cpp_train and py_train:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([int(r["step"]) for r in cpp_train], [float(r["loss"]) for r in cpp_train],
                label="C++", linewidth=2)
        ax.plot([int(r["step"]) for r in py_train], [float(r["loss"]) for r in py_train],
                label="Python", linewidth=2, linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("250M training loss — C++ vs Python")
        ax.grid(alpha=0.3)
        ax.legend()
        loss_png = root / "loss_curve.png"
        plt.savefig(loss_png, dpi=120, bbox_inches="tight")
        print(f"wrote: {loss_png}")

    if cpp_train and any("tok_per_s" in r for r in cpp_train):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([int(r["step"]) for r in cpp_train],
                [float(r["tok_per_s"]) for r in cpp_train],
                label="C++", linewidth=2)
        if py_train:
            ax.plot([int(r["step"]) for r in py_train],
                    [float(r["tok_per_s"]) for r in py_train],
                    label="Python", linewidth=2, linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("tok/s")
        ax.set_title("250M training throughput — C++ vs Python")
        ax.grid(alpha=0.3)
        ax.legend()
        tps_png = root / "throughput.png"
        plt.savefig(tps_png, dpi=120, bbox_inches="tight")
        print(f"wrote: {tps_png}")


if __name__ == "__main__":
    main()
