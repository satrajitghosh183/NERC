#!/usr/bin/env python3
"""
Synthetic shader generation with a code LLM via vLLM (batched, on the H100s).

Real shader data is finite (~hundreds of M tokens, all of Shadertoy + GitHub). Synthetic
generation is the way past that ceiling: have a strong code model write fresh Shadertoy-style
GLSL from diverse prompts, then compile-filter (label_shaders.py) so only valid shaders enter
the corpus. The (prompt, code) pairs also seed the broken/not-broken verifier dataset.

Run in the isolated vLLM venv (so its torch can't disturb the trainer's):
  ~/vllm_env/bin/python gen_shaders.py --model ~/models/qwen2.5-coder-32b \
      --n-prompts 8000 --k 2 --out ~/shader_data/synth/gen.jsonl

Notes:
- The model is Qwen (an HF arch) so it runs via vLLM, not our llm-cpp inference (which serves
  our own from-scratch model). tensor_parallel_size=2 shards it across both H100s.
- Diversity comes from a combinatorial prompt generator (concept x style x technique); raise
  --n-prompts / --k for more volume. Generation is compute-bound, not data-bound.
"""
import argparse, json, random, re, os

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

SYS = ("You are a Shadertoy GLSL expert. Output ONE complete, compilable Shadertoy shader with "
       "a `void mainImage(out vec4 fragColor, in vec2 fragCoord)` function. Use only Shadertoy "
       "built-in uniforms (iResolution, iTime, iMouse, iFrame). Do NOT use textures or iChannel "
       "inputs. Output ONLY a single ```glsl code block and nothing else.")

CODE_FENCE = re.compile(r"```(?:glsl|c)?\s*(.*?)```", re.S)


def make_prompts(n, seed=0):
    rng = random.Random(seed)
    out = set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        out.add(f"{rng.choice(STYLES)} {rng.choice(CONCEPTS)} {rng.choice(TECH)}")
    return sorted(out)


def extract(text):
    m = CODE_FENCE.search(text)
    code = (m.group(1) if m else text).strip()
    return code if "mainImage" in code else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=8000)
    ap.add_argument("--k", type=int, default=2, help="samples per prompt")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    prompts = make_prompts(args.n_prompts, args.seed)
    print(f"[gen] {len(prompts)} unique prompts x {args.k} = {len(prompts)*args.k} shaders", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=args.max_tokens, n=args.k)
    convs = [[{"role": "system", "content": SYS},
              {"role": "user", "content": f"Write a Shadertoy shader: {p}."}] for p in prompts]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    outs = llm.chat(convs, sp)
    kept = 0
    with open(args.out, "w") as f:
        for p, o in zip(prompts, outs):
            for comp in o.outputs:
                code = extract(comp.text)
                if code and len(code) > 60:
                    f.write(json.dumps({"prompt": p, "code": code}) + "\n")
                    kept += 1
    print(f"[gen] wrote {kept} candidate shaders -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
