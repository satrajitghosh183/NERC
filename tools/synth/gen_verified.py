#!/usr/bin/env python3
"""
Compiler-in-the-loop shader generation — the debugger-in-the-loop, applied to data.

Instead of generating N shaders blind and discarding the ~88% that don't compile, this keeps
the model loaded and loops: generate -> compile (glslangValidator) -> for every prompt that
failed, feed the code + the compiler error back and ask for a fix -> compile again -> repeat
up to --rounds. Each round rescues more of the "wasted" generations, so the yield per prompt
climbs toward 1.0 instead of sitting at ~0.12.

  ~/vllm_env/bin/python gen_verified.py --model ~/models/qwen2.5-coder-32b \
      --n-prompts 9000 --rounds 4 --out verified.txt

Outputs verified.txt (compiling shaders in `// Shader:` instruction format) + a stats line.
"""
import argparse, json, os, re, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

# reuse the prompt generator + wrap/compile from the sibling scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_shaders import make_prompts, extract, SYS  # noqa: E402
from label_shaders import wrap, GLSLANG             # noqa: E402

CODE_FENCE = re.compile(r"```(?:glsl|c)?\s*(.*?)```", re.S)


def compile_one(code):
    """Return (ok, error_text) for a mainImage shader via glslangValidator."""
    glsl = wrap(code)
    f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w"); f.write(glsl); f.close()
    try:
        p = subprocess.run([GLSLANG, "-V", f.name, "-o", os.devnull],
                           capture_output=True, text=True, errors="replace", timeout=20)
        return (p.returncode == 0, "" if p.returncode == 0 else (p.stdout + p.stderr)[:400])
    except Exception as e:
        return (False, f"timeout/err: {e}")
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=9000)
    ap.add_argument("--rounds", type=int, default=4, help="generate + (rounds-1) repair passes")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="verified.txt")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=512)

    prompts = make_prompts(args.n_prompts, args.seed)
    # state per prompt: the latest (code, error); None until first attempt
    pending = {p: None for p in prompts}
    verified = {}   # prompt -> compiling code

    for rnd in range(args.rounds):
        todo = list(pending.keys())
        if not todo:
            break
        convs = []
        for p in todo:
            last = pending[p]
            if last is None:                                   # first attempt: write it
                user = f"Write a Shadertoy shader: {p}."
            else:                                              # repair: code + compiler error
                code, err = last
                user = (f"This GLSL shader does not compile. Fix it so it compiles; output the "
                        f"complete corrected shader.\n```glsl\n{code}\n```\nCompiler error:\n{err}")
            convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": user}])

        outs = llm.chat(convs, sp)
        codes = []
        for p, o in zip(todo, outs):
            m = CODE_FENCE.search(o.outputs[0].text)
            codes.append((p, (m.group(1) if m else o.outputs[0].text).strip()))

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(compile_one, [c for _, c in codes], chunksize=16))

        fixed = 0
        for (p, code), (ok, err) in zip(codes, results):
            if ok and len(code) > 40:
                verified[p] = code; del pending[p]; fixed += 1
            else:
                pending[p] = (code, err)
        print(f"[round {rnd}] {fixed} newly compiled · {len(verified)} total · {len(pending)} still failing "
              f"({len(verified)/len(prompts):.0%} yield)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for p, code in verified.items():
            f.write(f"// Shader: {p}\n{code.rstrip()}\n<|endoftext|>\n")
    print(f"\n[verified] {len(verified)}/{len(prompts)} prompts -> compiling shaders "
          f"({len(verified)/len(prompts):.0%}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
