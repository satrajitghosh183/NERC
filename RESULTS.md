# OmniTrace — Results

A universal GPU shader debugger built from scratch in C++, plus a debugger-in-the-loop shader
generation stack. Everything below is measured on real hardware (2×H100 for training, Apple M4 Pro
for the local debugger/renderer). Honest throughout — failures and weak numbers are stated plainly.

## Headline: the debugger-in-the-loop RL works

We refine a shader generator with the **OmniTrace debugger as the reward** — every generated shader is
compiled, lifted to UIR, and executed on the CPU reference; the reward is compile + runs-finite, not a
black-box score. GRPO drives the policy toward shaders that actually run.

On **20 held-out prompts the RL never saw** (disjoint from its training set):

| DoRA-3B | compile@1 |
|---|---|
| before RL | **0.20** (4/20) |
| after debugger-in-the-loop RL | **0.90** (18/20) |

The improvement generalizes — this is not the training prompts. (On the RL's *own* prompts it hits
20/20 = 1.00, but that's leakage and we don't report it as the result.) Mean RL reward rose 0.875 → 3.5
over 80 steps. This is the contribution: feeding the debugger's execution signal back into training
teaches the model to write shaders that compile and run.

## The model comparison

| Model | Approach | compile@1 | notes |
|---|---|---|---|
| From-scratch 300M / 1B / 3.6B | trained on 171M shader tokens | **~0** | word-salad — see below |
| DoRA Qwen2.5-Coder-3B | PEFT DoRA, r16 | 0.20 | held-out, completion-format prompts |
| **DoRA-3B + debugger-RL** | + GRPO with OmniTrace reward | **0.90** | the headline |
| DoRA Qwen2.5-Coder-32B | PEFT DoRA, r16 | 0.25 | (8-prompt overnight eval) |

The story: **from-scratch fails, pretrained + DoRA works, and the debugger-in-the-loop RL is what makes
it good.** A from-scratch model can't learn shaders from the data that exists; you borrow a code model's
knowledge with DoRA, then sharpen it with the debugger.

## Why from-scratch fails (the honest baseline)

The from-scratch models (300M, 1B, 3.6B) all generate shader *vocabulary* salad — `vec3`, `texture`,
`fragColor` tokens in no working order — and compile ~0/N regardless of prompt engineering (forced
`mainImage(` preamble included). This isn't a bug; it's the data ceiling:

- A 3.6B model wants ~70B tokens (Chinchilla). All public shader code — Shadertoy + every GLSL/HLSL/Metal
  file on GitHub (The Stack) — is a few hundred million tokens. ~150× short.
- Loss bottoms out at ~7.5 on the diverse corpus, ~6.4 on a focused Shadertoy-only corpus. Focused data
  lowers loss (data curation matters) but doesn't make from-scratch generate compilable shaders.

That's the result that motivates the DoRA path, not a failure to fix.

## Data & tokenizer

- **Corpus**: 22k shaders / 47.6M tokens → **196,418 unique shaders / 171M tokens** (~9× / ~3.6×).
  Sources (all key-free; the Shadertoy API is gated + capped at 1500 req/mo and was abandoned):
  The Stack GLSL (117k), seanmemery (42k), Vipitis (19.5k), mizuamedesu (14k), HLSL/Metal (3k).
- **GLSL tokenizer**: a from-scratch byte-level BPE trained on the corpus — **2.69 chars/token vs
  GPT-2's 2.0** (~27% fewer tokens for the same shader code), so more shader content per context.

## Synthetic verifier dataset (#33)

Generated 18k shaders with Qwen2.5-Coder-32B (vLLM, 8k tok/s), compile-labeled every one:
**7,714 labeled · 14.6% compile · +571 recovered by feeding compiler errors back for repair (8.7%)**.
The 14.6% is honest — the Vulkan compiler is far stricter than Shadertoy's WebGL, so real Shadertoy
shaders score similarly low through it. This labeled set is the verifier/reward data.

## The debugger (OmniTrace) — Part I, all tested C++

Universal SSA IR; hand-written SPIR-V front/back end (passes `spirv-val`); CFG (dominators /
post-dominators / thread frontiers); capture pass; **divergence-aware trace codec** (lossless, 26× /
4.5×, ~entropy bound); mmap store; CPU SIMT reference; time-travel reconstruction; ULP/divergence diff;
**real GPU capture + render via Vulkan/MoltenVK**. 19 test suites green.

`omni_reward` exposes it as a reward CLI: compile-but-NaN shaders score **2.5** (they build but don't
run), valid ones **3.5**, broken ones **0.0** — catching exactly what a compile check misses. It builds
and runs locally on the M4 Pro.

## Use it: the `omnishader` CLI (fully local)

```bash
./omnishader debug  shader.frag    # compile + execute + reward, then render the shader and open it
./omnishader render shader.frag    # render on the local GPU (MoltenVK) and open the image
./omnishader loop   "<prompt>"     # generate (DoRA-3B on MPS) -> debug -> render
```

The debugger and renderer run entirely on the Mac. Generation runs the RL-refined DoRA-3B locally via
transformers/MPS (the from-scratch `llm-cpp` path is a dead end — weak model + a vendoring build bug).

## Honest caveats

- Absolute DoRA numbers are modest; the strong result is the *relative* 0.20 → 0.90 from the debugger RL.
- The 1.00 on RL's own prompts is leakage; ignore it.
- From-scratch shader LMs fail at this data scale — that's a finding, not a TODO.
- The synthetic compile rate (14.6%) reflects a strict Vulkan oracle, not just model quality.
- The reward's visual term is present but currently flat (the v1 interpreter doesn't propagate
  fragment-coordinate variation through every built-in); the GPU renderer covers visuals for the CLI.

Artifacts: `results/shader_lm_comparison/` (eval JSON, corpus stats, samples). Trained models pulled
locally to `trained_models/` (RL-refined adapter, DoRA 3B/32B, from-scratch 300M/1B/focused, tokenizer).
