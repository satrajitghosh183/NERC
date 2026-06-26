#!/usr/bin/env python3
"""Evaluate the from-scratch SLM via llm-cpp `chat` (the fast inference: KV-cache on,
CUDA graphs), generate + compile@k. Usage: eval_slm.py <checkpoint> <config.json> <name>"""
import subprocess, re, os, sys, tempfile, json
CKPT, CFG, NAME = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "slm"
BASE = os.path.expanduser("~/OLMo-shader")
CHAT = BASE + "/build/chat"
VOCAB = BASE + "/data/gpt2/vocab.json"; MERGES = BASE + "/data/gpt2/merges.txt"
GLSLANG = "/usr/bin/glslangValidator"
OUT = os.path.expanduser(f"~/pipeline/{NAME}_out"); os.makedirs(OUT, exist_ok=True)

PROMPTS = ["// Shader: blue fire","// Shader: ocean waves","// Shader: red plasma","// Shader: starfield sky",
           "// Shader: rainbow gradient","// Shader: noise clouds","// Shader: rotating tunnel","// Shader: water ripples"]
HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n")

def generate(prompt):
    # llm-cpp fast inference: KV cache ON (default), no MTP so no speculative.
    try:
        p = subprocess.run([CHAT,"--checkpoint",CKPT,"--config",CFG,"--vocab-file",VOCAB,
            "--merges-file",MERGES,"--device","cuda","--max-tokens","220","--no-speculative",
            "--top-k","40","--top-p","0.92","--temperature","0.8","--repetition-penalty","1.3"],
            input=prompt+"\n", capture_output=True, text=True, errors="replace", timeout=150)
    except subprocess.TimeoutExpired:
        return ""
    m = re.search(r"Model:(.*?)(\n\[[0-9]+ tokens|\nYou:|$)", p.stdout, re.S)
    return (m.group(1) if m else "").strip()

def is_meaningful(c):
    c=c.strip(); return len(c)>=15 and any(t in c for t in (";","vec","float","=","{"))
def wrap(c):
    if "mainImage" in c:
        return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy;"
                      " vec3 col=vec3(0.);\n"+c+"\n_O=vec4(col,1.); }\n")
def valid(g):
    f=tempfile.NamedTemporaryFile(suffix=".frag",delete=False,mode="w"); f.write(g); f.close()
    try: return subprocess.run([GLSLANG,"-V",f.name,"-o","/dev/null"],capture_output=True,timeout=20).returncode==0
    finally: os.unlink(f.name)

print(f"== SLM eval via llm-cpp chat | {NAME} | {os.path.basename(CKPT)} ==", flush=True)
comp=0; samples=[]
for p in PROMPTS:
    code = generate(p)
    ok = is_meaningful(code) and valid(wrap(code))
    comp += int(ok); samples.append((p, ok, code))
    print(f"  {p:28s} compiles={int(ok)}  gen[:60]={code.strip()[:60]!r}", flush=True)
res = comp/len(PROMPTS)
print(f"\n{NAME} compile@1 = {comp}/{len(PROMPTS)} = {res:.2f}", flush=True)
json.dump({"name":NAME,"model":CKPT,"compile_at_1":res,"compiled":comp,"n":len(PROMPTS)},
          open(OUT+"/metrics.json","w"))
with open(OUT+"/samples.txt","w") as f:
    for p,ok,t in samples: f.write(f"### {p}  (compiles={ok})\n{t}\n\n")
print(f"metrics -> {OUT}/metrics.json\nDONE", flush=True)
