#!/usr/bin/env python3
"""Local shader generation on Apple Silicon (MPS) — RL-refined DoRA-3B via transformers.
Completion-format prompting (matches training). Prints the GLSL mainImage shader."""
import sys, os, re, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
prompt  = sys.argv[1]
base    = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.path.join(ROOT, "trained_models/base_qwen3b")
adapter = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else os.path.join(ROOT, "trained_models/dora/rl3b_refined")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16).to(dev)
if adapter and os.path.isdir(adapter): m = PeftModel.from_pretrained(m, adapter)
m.eval()
header = (f"// Shader: {prompt}\nvoid mainImage( out vec4 fragColor, in vec2 fragCoord )\n{{\n"
          "    vec2 uv = fragCoord / iResolution.xy;\n")
enc = tok(header, return_tensors="pt").to(dev)
g = m.generate(**enc, max_new_tokens=380, do_sample=True, temperature=0.6, top_p=0.9,
               repetition_penalty=1.15, pad_token_id=tok.eos_token_id)
full = header + tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
full = re.split(r"```|\n// Shader:", full)[0]
depth, end = 0, len(full)
for i, ch in enumerate(full):
    if ch == "{": depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0: end = i + 1; break
print(full[:end].rstrip())
