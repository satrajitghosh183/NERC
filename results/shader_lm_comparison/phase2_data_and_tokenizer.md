# Phase 2 — data scaling, GLSL tokenizer, eval + RL harnesses (box-free)

Done entirely on the laptop while the H100 box is off. Everything below is measured, not
assumed. Tooling: `tools/data/`, `tools/eval/`, `tools/rl/` (all have `--self-test`, all pass).

## Corpus — merged & deduped (key-free, no Shadertoy API)
The Shadertoy API turned out to be a dead end: creating an app needs Silver/Gold profile
status, and it's capped at 1,500 requests/month — useless for bulk. So the corpus is built
entirely from already-scraped public datasets:

- **seanmemery/shader_dataset** (HF): 45,209 shaders with descriptions — the big find, almost
  no overlap with the others (+42,272 net-new).
- **mizuamedesu/shader-sft-dataset** (HF): 56,854 chat-format shaders.
- **Vipitis/shadertoys-dataset** (GitHub, `data/annotated/`): 19,622 shaders w/ metadata.

`build_corpus.py` normalizes all of them into the `// Shader: <name>` instruction format (the
format the eval prompts use — fixing the mismatch that gave the from-scratch model 0/8) and
dedupes by Shadertoy id + normalized-code SHA1:

| | count |
|---|---|
| **unique shaders** | **75,924** (seanmemery 42,272 · vipitis 19,556 · mizuamedesu 14,096) |
| duplicates removed | 48,729 (mizuamedesu & vipitis heavily overlap — both scrape Shadertoy) |
| size | 257M chars / **~128M GPT-2 tokens** (≈2.7× the old run's 47.6M) |

> The next lever is **The Stack**'s GLSL subset (`bigcode/the-stack-dedup`, `data/glsl/`), which
> is gated — it needs the HF account to accept the dataset terms + a token. That could add
> tens of thousands more GLSL files from GitHub.

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
