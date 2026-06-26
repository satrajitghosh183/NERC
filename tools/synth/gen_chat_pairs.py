#!/usr/bin/env python3
"""
Instruction/chat pair generation for shader-LM fine-tuning, via vLLM (batched, on the H100s).

Pure completion data (prompt, code) gets the model to write shaders on demand, but a
conversational SFT mix is needed to unlock instruction-following: "fix this bug", "explain
what this renders", "make it blue".  This script generates that mix from five task types that
together cover the main shader-assistance use-cases.  Output is one JSON object per line with
the mizuamedesu/shader-sft schema:

    {"messages": [{"role":"system","content":...},{"role":"user","content":...},
                  {"role":"assistant","content":...}]}

Task types (balanced across --n total pairs):
  WRITE    — given a text description, write a complete Shadertoy shader.
  FIX      — given a (slightly broken) shader, return the corrected version.
  EXPLAIN  — given a shader, describe what it renders and the techniques used.
  MODIFY   — given a shader plus a requested change, return the modified shader.
  OPTIMIZE — given a shader, return a cleaner / more-efficient version.

WRITE prompts are generated combinatorially (concept x style x technique, same vocabulary as
gen_shaders.py).  FIX / EXPLAIN / MODIFY / OPTIMIZE prompts seed from --seed-corpus, a jsonl
where each line is {"prompt":..., "code":...} (e.g. ~/shader_data/corpus_v4/train.jsonl).
If no corpus is provided, those four task types fall back to a small built-in shader stub so
the script is still runnable.

Run in the isolated vLLM venv:
  ~/vllm_env/bin/python gen_chat_pairs.py --model ~/models/qwen2.5-coder-32b \\
      --n 8000 --seed-corpus ~/shader_data/corpus_v4/train.jsonl \\
      --out ~/shader_data/synth/chat_pairs.jsonl

Verify prompt construction without a model:
  python3 gen_chat_pairs.py --self-test
"""
import argparse, json, random, re, os, sys

# ── vocabulary (same as gen_shaders.py) ──────────────────────────────────────
CONCEPTS = [
    "a mandelbrot fractal", "a julia set", "a raymarched sphere", "raymarched terrain",
    "ocean waves", "a roaring fire", "swirling plasma", "a rotating tunnel", "a starfield",
    "drifting noise clouds", "a voronoi cell pattern", "a kaleidoscope", "a lava lamp",
    "a neon grid", "a glowing orb", "metaballs", "a fractal flame", "an aurora borealis",
    "rippling water", "a checkerboard floor in perspective", "a rotating cube", "a torus knot",
    "a swirling galaxy", "falling rain", "digital matrix rain", "truchet tiles",
    "a sierpinski triangle", "domain-warped fbm noise", "underwater caustics",
    "soft shadows on a sphere", "a sunset sky gradient", "rolling hills", "a spinning vinyl record",
    "a pulsing heart", "electric lightning", "a hexagon grid", "concentric ripples",
    "a flickering candle", "northern stars", "a clockwork of gears", "marble texture",
    "wood grain", "a glass refraction", "smoke plumes", "a wireframe landscape",
    "bouncing balls", "a spiral vortex", "stained glass", "a melting blob", "fireworks",
]
STYLES = [
    "psychedelic", "minimal", "photorealistic", "retro 80s", "glowing neon", "pastel",
    "high-contrast", "monochrome", "dreamy", "vibrant", "dark and moody", "iridescent",
    "cel-shaded", "watercolor", "cyberpunk",
]
TECH = [
    "using raymarching", "using signed distance fields", "using fbm noise",
    "using domain warping", "with simple 2D math", "using polar coordinates",
    "with ray-sphere intersection", "using trigonometric color palettes",
    "with a distance-field metric", "using value noise", "with simplex-style noise",
    "using analytic derivatives for lighting",
]

# Requested changes used in MODIFY tasks
MODIFY_REQUESTS = [
    "make the primary color blue", "make it animate faster",
    "add pulsing animation tied to iTime", "double the spatial frequency",
    "add a vignette around the edges", "make it monochrome",
    "add mouse interaction using iMouse", "increase the brightness by 50%",
    "invert the colors", "add a subtle scanline overlay",
    "make the motion slower and more dreamlike", "add a horizontal mirror symmetry",
    "swap the foreground and background colors", "make it loop with a period of exactly 4 seconds",
    "add a glowing bloom effect around bright areas",
]

# ── system prompts per task type ──────────────────────────────────────────────
SYS_WRITE = (
    "You are a Shadertoy GLSL expert. When given a description, output ONE complete, "
    "compilable Shadertoy shader with a `void mainImage(out vec4 fragColor, in vec2 fragCoord)` "
    "function. Use only Shadertoy built-in uniforms (iResolution, iTime, iMouse, iFrame). "
    "Do NOT use textures or iChannel inputs. Output ONLY a single ```glsl code block."
)
SYS_FIX = (
    "You are a Shadertoy GLSL debugger. When given a GLSL shader that may contain bugs, "
    "identify any issues and return the fully corrected shader. Output ONLY a single "
    "```glsl code block containing the fixed, compilable shader."
)
SYS_EXPLAIN = (
    "You are a Shadertoy GLSL expert and technical writer. When given a shader, explain "
    "clearly what it renders visually and describe the key techniques and algorithms used. "
    "Be concise but thorough. Do not output code — output plain prose only."
)
SYS_MODIFY = (
    "You are a Shadertoy GLSL expert. When given a shader and a requested change, apply the "
    "change and return the complete modified shader. Output ONLY a single ```glsl code block "
    "containing the full updated, compilable shader."
)
SYS_OPTIMIZE = (
    "You are a Shadertoy GLSL expert focused on performance and code quality. When given a "
    "shader, return a cleaner, more efficient version: remove redundant computations, simplify "
    "math where possible, and improve readability. Preserve the visual output exactly. "
    "Output ONLY a single ```glsl code block."
)

CODE_FENCE = re.compile(r"```(?:glsl|c)?\s*(.*?)```", re.S)

# ── fallback stub for when no corpus is provided ──────────────────────────────
STUB_SHADER = """\
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    fragColor = vec4(col, 1.0);
}"""

STUB_PROMPT = "a smooth cycling color gradient"


# ── corpus loading ────────────────────────────────────────────────────────────
def load_corpus(path, n, rng):
    """Return up to n (prompt, code) tuples sampled from a jsonl corpus file."""
    if path is None:
        return [(STUB_PROMPT, STUB_SHADER)] * n
    rows = []
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get("prompt", "")
                c = obj.get("code", "")
                if p and c and "mainImage" in c:
                    rows.append((p, c))
    except FileNotFoundError:
        print(f"[warn] seed corpus not found: {path} — using stub shader", flush=True)
        return [(STUB_PROMPT, STUB_SHADER)] * n
    if not rows:
        print("[warn] corpus is empty or has no valid rows — using stub shader", flush=True)
        return [(STUB_PROMPT, STUB_SHADER)] * n
    if len(rows) >= n:
        return rng.sample(rows, n)
    # allow repeats if corpus is smaller than needed
    return [rng.choice(rows) for _ in range(n)]


# ── prompt builders ───────────────────────────────────────────────────────────
def write_user(rng):
    desc = f"{rng.choice(STYLES)} {rng.choice(CONCEPTS)} {rng.choice(TECH)}"
    return f"Write a Shadertoy shader: {desc}."


def fix_user(prompt, code):
    return (
        f"The following Shadertoy GLSL shader (originally described as \"{prompt}\") "
        f"may contain bugs. Find and fix any issues so it compiles and runs correctly.\n\n"
        f"```glsl\n{code}\n```"
    )


def explain_user(prompt, code):
    return (
        f"Explain what the following Shadertoy GLSL shader renders and the techniques it uses.\n\n"
        f"```glsl\n{code}\n```"
    )


def modify_user(prompt, code, change, rng):
    return (
        f"Given this Shadertoy GLSL shader (originally: \"{prompt}\"), apply the following "
        f"change and return the complete modified shader.\n\n"
        f"Requested change: {change}\n\n"
        f"```glsl\n{code}\n```"
    )


def optimize_user(prompt, code):
    return (
        f"Optimize the following Shadertoy GLSL shader for performance and code clarity "
        f"while preserving its visual output exactly.\n\n"
        f"```glsl\n{code}\n```"
    )


# ── conversation builders ─────────────────────────────────────────────────────
TASK_TYPES = ["WRITE", "FIX", "EXPLAIN", "MODIFY", "OPTIMIZE"]

def build_conversations(n, corpus_path, seed, rng):
    """
    Build n (system, user_text, task_type) triples, balanced across task types.
    Returns a list of dicts: {"system": str, "user": str, "task": str}
    """
    # Allocate roughly equal counts per task type; last type absorbs any remainder
    per_type = n // len(TASK_TYPES)
    counts = {t: per_type for t in TASK_TYPES}
    remainder = n - per_type * len(TASK_TYPES)
    for t in TASK_TYPES[:remainder]:
        counts[t] += 1

    # Load corpus entries for seed-based tasks
    seed_n = counts["FIX"] + counts["EXPLAIN"] + counts["MODIFY"] + counts["OPTIMIZE"]
    corpus = load_corpus(corpus_path, seed_n, rng)
    corpus_iter = iter(corpus)

    convs = []

    for _ in range(counts["WRITE"]):
        convs.append({"system": SYS_WRITE, "user": write_user(rng), "task": "WRITE"})

    for _ in range(counts["FIX"]):
        p, c = next(corpus_iter)
        convs.append({"system": SYS_FIX, "user": fix_user(p, c), "task": "FIX"})

    for _ in range(counts["EXPLAIN"]):
        p, c = next(corpus_iter)
        convs.append({"system": SYS_EXPLAIN, "user": explain_user(p, c), "task": "EXPLAIN"})

    for _ in range(counts["MODIFY"]):
        p, c = next(corpus_iter)
        change = rng.choice(MODIFY_REQUESTS)
        convs.append({"system": SYS_MODIFY, "user": modify_user(p, c, change, rng), "task": "MODIFY"})

    for _ in range(counts["OPTIMIZE"]):
        p, c = next(corpus_iter)
        convs.append({"system": SYS_OPTIMIZE, "user": optimize_user(p, c), "task": "OPTIMIZE"})

    rng.shuffle(convs)
    return convs


def extract(text):
    m = CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


# ── self-test (no vllm) ───────────────────────────────────────────────────────
def self_test():
    """
    Construct one example prompt for each task type using a tiny hardcoded corpus
    and print them.  No model or vllm import needed.
    """
    rng = random.Random(42)
    stub_corpus = [(STUB_PROMPT, STUB_SHADER)]

    print("=" * 70)
    print("SELF-TEST: one example per task type (no model required)")
    print("=" * 70)

    examples = [
        ("WRITE",   SYS_WRITE,   write_user(rng)),
        ("FIX",     SYS_FIX,     fix_user(*stub_corpus[0])),
        ("EXPLAIN", SYS_EXPLAIN, explain_user(*stub_corpus[0])),
        ("MODIFY",  SYS_MODIFY,  modify_user(stub_corpus[0][0], stub_corpus[0][1],
                                              rng.choice(MODIFY_REQUESTS), rng)),
        ("OPTIMIZE",SYS_OPTIMIZE,optimize_user(*stub_corpus[0])),
    ]

    for task, sys_prompt, user_prompt in examples:
        print(f"\n{'─'*70}")
        print(f"TASK: {task}")
        print(f"{'─'*70}")
        print(f"[SYSTEM]\n{sys_prompt}\n")
        print(f"[USER]\n{user_prompt}\n")

    print("=" * 70)
    print("Self-test PASSED: all 5 task types produced valid prompts.")
    print("=" * 70)


# ── main (generation path — imports vllm here only) ───────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate instruction/chat SFT pairs for shader-LM fine-tuning."
    )
    ap.add_argument("--model", help="Path or HF id of the model to serve via vLLM (required for generation)")
    ap.add_argument("--n", type=int, default=8000, help="Total chat pairs to generate")
    ap.add_argument("--seed-corpus", dest="seed_corpus", default=None,
                    help="Path to jsonl with {prompt, code} rows used for FIX/EXPLAIN/MODIFY/OPTIMIZE")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel GPU count")
    ap.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    ap.add_argument("--out", default="chat_pairs.jsonl", help="Output jsonl path")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="Print example prompts for each task type and exit (no model needed)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model:
        ap.error("--model is required for generation (omit only with --self-test)")

    # vllm import is deferred so --self-test works with plain python3
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    rng = random.Random(args.seed)
    convs = build_conversations(args.n, args.seed_corpus, args.seed, rng)
    print(f"[gen] built {len(convs)} conversations across {len(TASK_TYPES)} task types", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=768, n=1)

    # Build vllm message lists: [[system, user], ...]
    message_lists = [
        [{"role": "system", "content": c["system"]},
         {"role": "user",   "content": c["user"]}]
        for c in convs
    ]

    print(f"[gen] running vLLM inference on {len(message_lists)} conversations…", flush=True)
    outputs = llm.chat(message_lists, sp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept = 0
    skipped = 0
    with open(args.out, "w") as f:
        for conv, out in zip(convs, outputs):
            assistant_text = out.outputs[0].text.strip()
            # For code tasks require non-empty output; EXPLAIN tasks accept any non-empty prose
            if not assistant_text:
                skipped += 1
                continue
            record = {
                "messages": [
                    {"role": "system",    "content": conv["system"]},
                    {"role": "user",      "content": conv["user"]},
                    {"role": "assistant", "content": assistant_text},
                ],
                "_task": conv["task"],  # metadata; drop before final training if needed
            }
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"[gen] wrote {kept} pairs ({skipped} skipped) -> {args.out}", flush=True)
    # Print per-task breakdown
    from collections import Counter
    # Re-read to count (kept count is all that matters; this is informational)
    task_counts = Counter(c["task"] for c in convs)
    for t in TASK_TYPES:
        print(f"  {t}: {task_counts[t]} requested", flush=True)


if __name__ == "__main__":
    main()
