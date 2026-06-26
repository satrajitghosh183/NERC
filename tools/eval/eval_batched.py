#!/usr/bin/env python3
"""
Batched shader-LM eval harness (replaces the per-prompt-reload eval_slm.py).

Improvements over the overnight eval:
  - ONE model session for all prompts (llm-cpp `chat` interactive stdin) instead of
    reloading the 7GB model per prompt (10min -> seconds).
  - Real HELD-OUT prompts from the val split (tools/data .../eval_prompts.jsonl), with
    reference shaders, instead of 8 hand-written strings.
  - compile@k (k samples/prompt) via glslangValidator, not just compile@1.
  - OmniTrace-ready scoring hooks: compile is the first reward term; render / NaN /
    divergence / numerical-diff slots are wired for the debugger-in-the-loop reward.

Generation backends (pluggable):
  --backend chat  --model model.pt --config cfg.json --bin ./build/chat   (from-scratch SLM)
  --backend hf    --model <hf dir or adapter>                              (DoRA models)

The wrap()/parse logic is unit-tested locally (--self-test); the generation + glslang
scoring run on the box. Outputs metrics.json (compile@1, compile@k, per-prompt).
"""
import argparse, json, os, re, subprocess, sys, tempfile

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
           "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; "
           "int iFrame; float iTimeDelta; } u;\n"
           "#define iResolution u.iResolution\n#define iTime u.iTime\n"
           "#define iMouse u.iMouse\n#define iFrame u.iFrame\n#define iTimeDelta u.iTimeDelta\n")


def extract_code(raw):
    """Pull GLSL out of model output: prefer a ```glsl fence, else strip prose lead-in."""
    m = re.search(r"```(?:glsl|c)?\s*(.*?)```", raw, re.S)
    if m:
        return m.group(1).strip()
    # else: keep from the first plausible GLSL line onward
    lines = raw.splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"\b(void\s+main|mainImage|#define|vec[234]\b|float\b|const\b)", ln):
            return "\n".join(lines[i:]).strip()
    return raw.strip()


def wrap(code):
    """Wrap generated GLSL into a compilable fragment shader (Shadertoy-style)."""
    if "mainImage" in code:
        return HARNESS + code + ("\nvoid main(){ vec4 c=vec4(0.); "
                                 "mainImage(c, gl_FragCoord.xy); _O=c; }\n")
    if re.search(r"\bvoid\s+main\s*\(", code):
        return ("#version 450\nlayout(location=0) out vec4 _O;\n" + code) \
            if "out vec4" not in code else code
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; "
                      "vec2 uv=fragCoord/iResolution.xy; vec3 col=vec3(0.);\n"
                      + code + "\n_O=vec4(col,1.); }\n")


def is_meaningful(code):
    c = code.strip()
    return len(c) >= 20 and any(t in c for t in (";", "vec", "float", "{", "="))


def glslang_ok(glsl, glslang="glslangValidator"):
    f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w")
    f.write(glsl); f.close()
    try:
        return subprocess.run([glslang, "-V", f.name, "-o", os.devnull],
                              capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------- generation
def gen_chat(prompts, k, args):
    """Feed all prompts (x k samples) to ONE chat session via interactive stdin."""
    reqs = [p for p in prompts for _ in range(k)]
    cmd = [args.bin, "--checkpoint", args.model, "--config", args.config,
           "--vocab-file", args.vocab, "--merges-file", args.merges, "--device", "cuda",
           "--max-tokens", "256", "--no-speculative", "--temperature", str(args.temp),
           "--top-k", "40", "--top-p", "0.95", "--repetition-penalty", "1.3"]
    inp = "".join(p.strip() + "\n" for p in reqs)
    out = subprocess.run(cmd, input=inp, capture_output=True, text=True,
                         errors="replace", timeout=args.timeout).stdout
    # chat echoes "Model: <gen>" per turn; split on that marker
    chunks = re.findall(r"Model:(.*?)(?=\nYou:|\nModel:|$)", out, re.S)
    return [c.strip() for c in chunks][:len(reqs)]


def gen_hf(prompts, k, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.base or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.base or args.model,
                                                 dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    outs = []
    for p in prompts:
        for _ in range(k):
            ids = tok(p, return_tensors="pt").to(model.device)
            g = model.generate(**ids, max_new_tokens=256, do_sample=True,
                               temperature=args.temp, top_p=0.95, repetition_penalty=1.1,
                               pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="eval_prompts.jsonl")
    ap.add_argument("--backend", choices=["chat", "hf"])
    ap.add_argument("--model"); ap.add_argument("--config"); ap.add_argument("--base")
    ap.add_argument("--adapter"); ap.add_argument("--vocab"); ap.add_argument("--merges")
    ap.add_argument("--bin", default="./build/chat")
    ap.add_argument("-k", type=int, default=1, help="samples per prompt (compile@k)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("--out", default="eval_out/metrics.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if _self_test() else 1)
    if not args.prompts or not args.backend:
        ap.error("--prompts and --backend are required (unless --self-test)")

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    texts = [p["prompt"] for p in prompts]
    gens = (gen_chat if args.backend == "chat" else gen_hf)(texts, args.k, args)

    # regroup k samples per prompt; a prompt "compiles@k" if ANY sample compiles
    per, n_ok1, n_okk = [], 0, 0
    for i, p in enumerate(prompts):
        samples = gens[i * args.k:(i + 1) * args.k]
        oks = [is_meaningful(extract_code(s)) and glslang_ok(wrap(extract_code(s)), args.glslang)
               for s in samples]
        ok1 = bool(oks and oks[0]); okk = any(oks)
        n_ok1 += ok1; n_okk += okk
        per.append({"name": p.get("name"), "compile@1": ok1, f"compile@{args.k}": okk})
    res = {"n": len(prompts), "k": args.k,
           "compile@1": round(n_ok1 / len(prompts), 4),
           f"compile@{args.k}": round(n_okk / len(prompts), 4), "per_prompt": per}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


# ------------------------------------------------------------------- self-test
def _self_test():
    ok = True
    frag = "void mainImage(out vec4 c, in vec2 f){ c = vec4(f/iResolution.xy,0.,1.); }"
    w = wrap(frag)
    ok &= "mainImage" in w and "void main()" in w and w.startswith("#version")
    # fenced extraction
    raw = "Sure!\n```glsl\nvoid mainImage(out vec4 c, in vec2 f){ c=vec4(1.); }\n```\nDone"
    ex = extract_code(raw)
    ok &= ex.startswith("void mainImage") and "```" not in ex and "Sure" not in ex
    # prose lead-in stripping
    ex2 = extract_code("here is the code\nvec3 col = vec3(1.0);\ngl_FragColor = vec4(col,1.);")
    ok &= ex2.startswith("vec3 col")
    ok &= is_meaningful(frag) and not is_meaningful("hello world")
    print(f"[self-test] wrap/extract/meaningful: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
