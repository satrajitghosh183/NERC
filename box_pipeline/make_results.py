#!/usr/bin/env python3
"""Collate all stage metrics into a comparison report."""
import os, json, glob
P = os.path.expanduser("~/pipeline")
def load(name):
    f = os.path.join(P, f"{name}_out", "metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None

rows = []
for name, label, kind in [("slm","From-scratch 3B SLM","from-scratch"),
                          ("dora3b","DoRA Qwen-Coder-3B","DoRA 3B"),
                          ("dora32b","DoRA Qwen-Coder-32B","DoRA 32B")]:
    m = load(name)
    rows.append((label, kind, m["compile_at_1"] if m else None, m["compiled"] if m else None, m["n"] if m else None))

out = []
out.append("# Shader-LM comparison — from-scratch vs DoRA (compile@k)\n")
out.append("| Model | Approach | compile@1 | compiled/n |")
out.append("|---|---|---|---|")
for label, kind, c1, comp, n in rows:
    out.append(f"| {label} | {kind} | {'%.2f'%c1 if c1 is not None else 'n/a'} | {f'{comp}/{n}' if comp is not None else 'n/a'} |")
out.append("")
out.append("compile@1 = fraction of 8 prompts whose generated shader compiles with glslangValidator")
out.append("(the debugger-in-the-loop reward signal; OmniTrace enriches it with execution traces).")
out.append("")
out.append("Sample shaders per model: ~/pipeline/<name>_out/samples.txt")
report = "\n".join(out)
print(report)
open(os.path.join(P, "RESULTS.txt"), "w").write(report + "\n")
