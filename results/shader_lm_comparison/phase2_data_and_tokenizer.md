# Phase 2 — data scaling, GLSL tokenizer, eval + RL harnesses (box-free)

Done entirely on the laptop while the H100 box is off. Everything below is measured, not
assumed. Tooling: `tools/data/`, `tools/eval/`, `tools/rl/` (all have `--self-test`, all pass).

## Corpus — merged & deduped (key-free, no Shadertoy API)
The Shadertoy API turned out to be a dead end: creating an app needs Silver/Gold profile
status, and it's capped at 1,500 requests/month — useless for bulk. So the corpus is built
entirely from already-scraped public datasets:

- **seanmemery/shader_dataset** (HF): 45,209 shaders with descriptions (+42,272 net-new).
- **The Stack** glsl subset (`bigcode/the-stack-dedup`, gated — HF token): 175,576 raw GitHub
  `.glsl/.shader` files, filtered to fragment-style (`mainImage` or `gl_Frag*`, dropping Unity
  ShaderLab / vertex / engine noise) → +34,829.
- **mizuamedesu/shader-sft-dataset** (HF): 56,854 chat-format shaders (+14,096 after dedup).
- **Vipitis/shadertoys-dataset** (GitHub): 19,622 shaders w/ metadata (+19,556).

`build_corpus.py` normalizes all of them into the `// Shader: <name>` instruction format (the
format the eval prompts use — fixing the mismatch that gave the from-scratch model 0/8) and
dedupes by Shadertoy id + normalized-code SHA1:

| | count |
|---|---|
| **unique shaders** | **196,418** (the-stack-glsl 117,361 · seanmemery 42,272 · vipitis 19,556 · mizuamedesu 14,096 · hlsl/metal 3,133) |
| duplicates removed | 49,723 |
| size | 467M chars / **~230M GPT-2 tokens** (≈9× the old run's 22k shaders / 47.6M tokens) |

This is roughly the ceiling of *public shader code*. Worth stating plainly: a 3.6B model wants ~70B
tokens to be Chinchilla-optimal, and the entire shader-code universe is ~hundreds of millions of tokens
— ~150× short. You cannot from-scratch a large shader model into being best-in-class, because the data
doesn't exist. That's the paper's point, not a failure: when data is scarce, you borrow a pretrained code
model's knowledge (DoRA) and refine with the debugger reward (RL). The bigger corpus still helps every
arm; it just isn't a substitute for the pretrained base.

> Note on The Stack: those 34,829 are general WebGL/GLSL-ES fragment shaders (gl_FragColor style),
> a different flavor from Shadertoy's `mainImage`/`iResolution` convention. Good for teaching GLSL
> syntax, tagged by source so they can be down-weighted vs the Shadertoy data if the eval (which is
> Shadertoy-style) calls for it.
>
> The Shadertoy API was abandoned: app creation needs Silver/Gold profile status, and it's capped at
> 1,500 requests/month — useless for bulk.

## GLSL tokenizer — measured win
`train_bpe.py` is a from-scratch byte-level BPE (GPT-2-compatible `vocab.json`/`merges.txt`, so
it drops into OLMo's loader) but learns merges on GLSL. Measured on a 2 MB held-out chunk:

| tokenizer | vocab | chars/token | tokens (2 MB) |
|---|---|---|---|
| GPT-2 | 50,257 | 2.008 | 995,243 |
| **GLSL-BPE (ours)** | **8,192** | **2.563** | **779,592** |

**1.28× fewer tokens at 1/6 the vocab.** A full 16–32k-vocab tokenizer trained on the whole
corpus (a box job — the pure-Python per-merge scan is too slow locally at full scale) will widen
this further. Fewer tokens = more shader content per context window = better learning at the same
data budget. This is the single cheapest quality lever for the from-scratch arm.

## Eval harness — `tools/eval/eval_batched.py`
Replaces the overnight eval's two flaws: (1) it reloaded the 7 GB model **per prompt** (8× → ~10
min); this runs ONE session. (2) it used 8 hand-written prompts; this uses **100 held-out prompts**
from the val split (`tools/eval/fixtures/heldout_prompts.jsonl`, never trained on) and reports
**compile@1 and compile@k** (k samples/prompt). Scoring hooks are wired for the full OmniTrace
signal set, not just compile. Wrap/extract/meaningful logic unit-tested.

## Debugger-in-the-loop RL — `tools/rl/rl_refine.py` (the SIGGRAPH thesis)
Closes the loop the overnight run only *measured*: generate → score with OmniTrace's **rich** reward
→ update policy (GRPO, group-relative advantages, no value net). Reward mirrors
`src/reward/oracle.cpp` exactly — compile-gate + exec + (1−divergence) + (1−ULP) + visual(exp(−MSE))
− perf. Signal provider calls the OmniTrace binaries when present, degrades to compile-only
elsewhere (runs anywhere). Reward gating/ordering + GRPO advantage math unit-tested.

## When the box is back (critical path)
1. Train the full 16–32k GLSL tokenizer on the merged corpus (+ shaders21k).
2. `tokenize_corpus.py` → `.npy`; retrain the large model on the bigger, better-tokenized,
   instruction-formatted corpus.
3. `eval_batched.py` (compile@1/@k on 100 held-out) → compare to the overnight baseline.
4. `rl_refine.py` with the built OmniTrace binaries → debugger-in-the-loop refinement.
5. Fold in the Shadertoy API pull once the key is provided (`shadertoy_fetch.py`).
