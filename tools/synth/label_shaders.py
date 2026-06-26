#!/usr/bin/env python3
"""
Compile-label generated shaders -> the broken/not-broken supervised dataset, and append the
compiling ones to the corpus.

Each candidate is wrapped in a Shadertoy/Vulkan harness and run through glslangValidator. The
verdict (compiles + error text) is the supervised label; this is the same signal OmniTrace's
reward oracle uses, so the labeled set doubles as training data for a compile/verifier model
AND the broken+error rows feed the debugger-in-the-loop repair loop.

  python3 label_shaders.py --in ~/shader_data/synth/gen.jsonl \
      --labeled ~/shader_data/synth/labeled.jsonl \
      --corpus-out ~/shader_data/synth/synth_corpus.txt --workers 32

Outputs:
  labeled.jsonl     {prompt, code, compiles: bool, error: str}   (supervised broken/not-broken)
  synth_corpus.txt  compiling shaders in `// Shader: <prompt>` instruction format (feed to build_corpus --extra or concat)
"""
import argparse, json, os, re, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
           "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; "
           "int iFrame; float iTimeDelta; } u;\n"
           "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n"
           "#define iFrame u.iFrame\n#define iTimeDelta u.iTimeDelta\n")
GLSLANG = os.environ.get("GLSLANG", "glslangValidator")


def wrap(code):
    if "mainImage" in code:
        return HARNESS + code + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    if re.search(r"\bvoid\s+main\s*\(", code):
        return code if code.lstrip().startswith("#version") else "#version 450\n" + code
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy;"
                      " vec3 col=vec3(0.);\n" + code + "\n_O=vec4(col,1.); }\n")


def label_one(rec):
    code = rec["code"]
    glsl = wrap(code)
    f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w")
    f.write(glsl); f.close()
    try:
        p = subprocess.run([GLSLANG, "-V", f.name, "-o", os.devnull],
                           capture_output=True, text=True, errors="replace", timeout=20)
        ok = p.returncode == 0
        err = "" if ok else (p.stdout + p.stderr)[:600]
    except Exception as e:
        ok, err = False, f"timeout/err: {e}"
    finally:
        os.unlink(f.name)
    return {"prompt": rec.get("prompt", ""), "code": code, "compiles": ok, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--labeled", default="labeled.jsonl")
    ap.add_argument("--corpus-out", default="synth_corpus.txt")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    print(f"[label] {len(recs)} candidates -> glslangValidator x{args.workers}", flush=True)
    for d in (args.labeled, args.corpus_out):
        os.makedirs(os.path.dirname(os.path.abspath(d)), exist_ok=True)

    ok = 0
    with open(args.labeled, "w") as lf, open(args.corpus_out, "w") as cf, \
            ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(label_one, recs, chunksize=16)):
            lf.write(json.dumps(r) + "\n")
            if r["compiles"]:
                ok += 1
                name = r["prompt"][:80] or "untitled"
                cf.write(f"// Shader: {name}\n{r['code'].rstrip()}\n<|endoftext|>\n")
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(recs)}  compile-rate {ok/(i+1):.1%}", flush=True)

    rate = ok / len(recs) if recs else 0
    print(f"\n[label] compiles {ok}/{len(recs)} = {rate:.1%}")
    print(f"[label] supervised set -> {args.labeled}")
    print(f"[label] {ok} compiling shaders (corpus) -> {args.corpus_out}")
    json.dump({"n": len(recs), "compiles": ok, "compile_rate": round(rate, 4)},
              open(os.path.splitext(args.labeled)[0] + "_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
