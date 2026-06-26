#!/usr/bin/env python3
"""Generate one Shadertoy shader from a prompt with the RL-refined DoRA-3B.
Uses the completion format the model was trained/RL'd on (// Shader: <desc> + forced
mainImage preamble), NOT a chat template. Prints the full GLSL shader."""
import sys, re, warnings, torch
warnings.filterwarnings("ignore")
prompt  = sys.argv[1]
base    = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "/home/exouser/models/qwen2.5-coder-3b"
adapter = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "/home/exouser/pipeline/rl3b_refined"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
if adapter and adapter != "none":
    m = PeftModel.from_pretrained(m, adapter)
m.eval()
header = (f"// Shader: {prompt}\n"
          "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n"
          "    vec2 uv = fragCoord / iResolution.xy;\n")
enc = tok(header, return_tensors="pt").to(m.device)
g = m.generate(**enc, max_new_tokens=380, do_sample=True, temperature=0.6, top_p=0.9,
               repetition_penalty=1.15, pad_token_id=tok.eos_token_id)
body = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
full = header + body
# cut at a code fence or the next "// Shader:" if the model rambles past one shader
full = re.split(r"```|\n// Shader:", full)[0]
# balance braces so we end cleanly after mainImage
depth, end = 0, len(full)
for i, ch in enumerate(full):
    if ch == "{": depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0: end = i + 1; break
print(full[:end].rstrip())
