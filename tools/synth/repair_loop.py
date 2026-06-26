#!/usr/bin/env python3
"""
Debugger-in-the-loop repair pipeline: take shaders that failed to compile,
feed each one plus its GLSL compiler error to a local LLM (via vLLM) to fix,
then re-check whether the fix compiles.

The output is both training data (error->fix pairs) and a demonstration of
feedback-driven repair — the same oracle signal OmniTrace uses at inference time.

  ~/vllm_env/bin/python repair_loop.py \
      --in ~/shader_data/synth/labeled.jsonl \
      --model ~/models/qwen2.5-coder-32b \
      --out ~/shader_data/synth/repaired.jsonl

Notes:
- Input is a labeled.jsonl produced by label_shaders.py; only compiles==false rows
  are processed (compiling shaders are already fine).
- All broken shaders are sent to the LLM in a single batched llm.chat call.
- Compile re-check is parallelised via ProcessPoolExecutor (same harness and
  glslangValidator subprocess as label_shaders.py).
- repaired.jsonl (error->fix pairs) is direct training data for a debugger model;
  the repair success rate is the key metric of the feedback loop.
"""

import argparse, json, os, re, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor

# --- harness + wrap (copied verbatim from label_shaders.py) -------------------
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


# --- LLM prompt ---------------------------------------------------------------
CODE_FENCE = re.compile(r"```(?:glsl|c)?\s*(.*?)```", re.S)

SYS_REPAIR = (
    "You are a GLSL shader debugging expert. Fix the shader so it compiles. "
    "Output only the corrected ```glsl code block."
)


def build_conversation(broken_code, error_msg):
    """Return a chat conversation list for a single broken shader."""
    user = (
        f"The following GLSL shader fails to compile:\n\n"
        f"```glsl\n{broken_code}\n```\n\n"
        f"Compiler error:\n{error_msg}\n\n"
        f"Please provide the corrected shader."
    )
    return [
        {"role": "system", "content": SYS_REPAIR},
        {"role": "user",   "content": user},
    ]


def extract_fix(text):
    """Extract the first ```glsl ... ``` block from model output."""
    m = CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


# --- compile re-check (parallelised via ProcessPoolExecutor) ------------------

def check_one(fixed_code):
    """Wrap and compile-check a single fixed shader. Returns bool."""
    glsl = wrap(fixed_code)
    f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w")
    f.write(glsl)
    f.close()
    try:
        p = subprocess.run(
            [GLSLANG, "-V", f.name, "-o", os.devnull],
            capture_output=True, text=True, errors="replace", timeout=20,
        )
        ok = p.returncode == 0
    except Exception:
        ok = False
    finally:
        os.unlink(f.name)
    return ok


# --- self-test (no vllm, no glslang) -----------------------------------------

def self_test():
    """Verify prompt construction, wrap(), and extract_fix() on hardcoded data.
    Does NOT import vllm or invoke glslangValidator."""

    broken = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec3(1.0);\n"   # type error: vec3 assigned to vec4 out
        "}"
    )
    error = (
        "ERROR: 0:2: 'assign' :  cannot convert from "
        "'const 3-component vector of float' to "
        "'out 4-component vector of float'"
    )

    # --- prompt construction --------------------------------------------------
    conv = build_conversation(broken, error)
    assert len(conv) == 2,                         "expected 2 messages"
    assert conv[0]["role"] == "system",            "first message must be system"
    assert "GLSL shader debugging expert" in conv[0]["content"], \
        "system message must mention role"
    assert conv[1]["role"] == "user",              "second message must be user"
    assert broken in conv[1]["content"],           "user message must contain broken code"
    assert error in conv[1]["content"],            "user message must contain error text"
    print("[self-test] build_conversation(): OK")

    # --- wrap() ---------------------------------------------------------------
    wrapped = wrap(broken)
    assert "#version 450" in wrapped,   "wrap() must inject #version 450"
    assert "mainImage" in wrapped,      "wrap() must preserve mainImage"
    assert "void main()" in wrapped,    "wrap() must add void main()"
    assert "_O" in wrapped,             "wrap() must route output to _O"

    # snippet without mainImage falls through to the bare-snippet branch
    snippet = "col = vec3(sin(uv.x), cos(uv.y), 0.5);"
    wrapped_snip = wrap(snippet)
    assert "void main()" in wrapped_snip,  "bare snippet must be wrapped in main"
    assert snippet in wrapped_snip,        "bare snippet content must be preserved"
    print("[self-test] wrap(): OK")

    # --- extract_fix() --------------------------------------------------------
    model_resp = (
        "Sure, here is the fixed shader:\n"
        "```glsl\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(1.0);\n"
        "}\n"
        "```"
    )
    fixed = extract_fix(model_resp)
    assert "vec4(1.0)" in fixed,      "extract_fix() must pull corrected code"
    assert "```" not in fixed,        "extract_fix() must strip fence markers"
    assert "Sure" not in fixed,       "extract_fix() must strip prose preamble"

    # also handles a bare response without fences
    bare_resp = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(0.); }"
    fixed_bare = extract_fix(bare_resp)
    assert "mainImage" in fixed_bare, "extract_fix() must pass through bare code"
    print("[self-test] extract_fix(): OK")

    print("[self-test] PASSED")


# --- main pipeline ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Debugger-in-the-loop: LLM-repair broken shaders and re-compile."
    )
    ap.add_argument("--in",      dest="inp",     default=None,
                    help="labeled.jsonl from label_shaders.py (required unless --self-test)")
    ap.add_argument("--model",   default=None,
                    help="path/name of the HF model to load via vLLM (required unless --self-test)")
    ap.add_argument("--out",     default="repaired.jsonl",
                    help="output repaired.jsonl (default: repaired.jsonl)")
    ap.add_argument("--tp",      type=int,   default=2,
                    help="tensor-parallel GPUs (default: 2)")
    ap.add_argument("--temp",    type=float, default=0.5,
                    help="sampling temperature (default: 0.5)")
    ap.add_argument("--workers", type=int,   default=32,
                    help="parallel workers for glslang re-check (default: 32)")
    ap.add_argument("--self-test", action="store_true",
                    help="run internal self-test and exit (no model/glslang needed)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.inp:
        ap.error("--in is required")
    if not args.model:
        ap.error("--model is required")

    # --- load broken shaders --------------------------------------------------
    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    broken = [r for r in recs if not r.get("compiles", True)]
    print(f"[repair] {len(recs)} total records, {len(broken)} failed to compile -> LLM repair",
          flush=True)
    if not broken:
        print("[repair] nothing to repair — all shaders compile. Exiting.")
        return

    # --- build chat conversations and run LLM (vllm import guarded here) ------
    convs = [build_conversation(r["code"], r.get("error", "")) for r in broken]

    from vllm import LLM, SamplingParams  # noqa: import guarded inside run path
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    sp  = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=640)

    print(f"[repair] running llm.chat on {len(convs)} broken shaders ...", flush=True)
    outs = llm.chat(convs, sp)
    fixed_codes = [extract_fix(o.outputs[0].text) for o in outs]

    # --- parallel compile re-check --------------------------------------------
    print(f"[repair] re-checking {len(fixed_codes)} fixed shaders "
          f"(glslangValidator x{args.workers}) ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        compile_results = list(ex.map(check_one, fixed_codes, chunksize=16))

    # --- write repaired.jsonl -------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_ok = 0
    with open(args.out, "w") as f:
        for r, fixed, ok in zip(broken, fixed_codes, compile_results):
            if ok:
                n_ok += 1
            f.write(json.dumps({
                "prompt":         r.get("prompt", ""),
                "broken":         r["code"],
                "error":          r.get("error", ""),
                "fixed":          fixed,
                "fixed_compiles": ok,
            }) + "\n")

    rate = n_ok / len(broken) if broken else 0.0
    print(f"\n[repair] repaired {n_ok}/{len(broken)} = {rate:.1%}", flush=True)
    print(f"[repair] output -> {args.out}", flush=True)

    stats_path = os.path.splitext(args.out)[0] + "_stats.json"
    json.dump(
        {"n_broken": len(broken), "repaired_ok": n_ok, "repair_rate": round(rate, 4)},
        open(stats_path, "w"),
        indent=2,
    )
    print(f"[repair] stats  -> {stats_path}", flush=True)


if __name__ == "__main__":
    main()
