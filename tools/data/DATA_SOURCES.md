# Shader corpus — data sources & pickup notes

Status: **paused** (H100 box powered off by user; resume later). All tooling below is
committed and ready to run the moment the box is back. Goal: grow the corpus well beyond
the current 47.6M tokens (~22k shaders21k docs) so the large from-scratch model has enough
to learn from, and unify everything into the `// Shader: <name>` instruction format.

## Shadertoy official API (needs a key — user creates it, tied to their login)
- API docs: https://www.shadertoy.com/api
- Create app key (sign in → "Create new app"): https://www.shadertoy.com/myapps
- Ingester: `tools/data/shadertoy_fetch.py` (stdlib-only, rate-limited, resumable, dedupes).
  Run: `export SHADERTOY_KEY=<key> && python3 tools/data/shadertoy_fetch.py --out ~/shader_data/api`

## Key-free corpora to pull (user-provided — "use a lot of this")
1. **github.com/Vipitis/shadertoys-dataset** — canonical Shadertoy-via-API scrape + builder
   (the source behind ShaderEval). `git clone`, then map its records into our instruction format.
2. **huggingface.co/datasets/mizuamedesu/shader-sft-dataset** — shader SFT dataset, already in
   prompt→code form; `hf download --repo-type dataset mizuamedesu/shader-sft-dataset`. Highest
   value: already instruction-formatted, minimal reshaping.
3. **huggingface.co/Vipitis/santacoder-finetuned-Shadertoys-fine** — this is a *model* (SantaCoder
   finetuned on Shadertoys-fine); useful as (a) a strong baseline/comparison point and (b) a pointer
   to its training dataset `Vipitis/Shadertoys-fine`. Pull the dataset, not the model, for corpus.

## Pickup checklist (when box is back)
- [ ] Resume box; `ssh -i ~/.ssh/id_nerc_box exouser@149.165.170.159`.
- [ ] Pull sources 1–3 (+ API if key provided) → dedupe by code hash against shaders21k.
- [ ] Re-extract ALL to `// Shader: <name>` instruction format (fixes the prompt mismatch that
      gave the from-scratch model 0/8).
- [ ] Train GLSL BPE: `python3 tools/data/train_bpe.py --corpus ~/shader_data --vocab-size 16384
      --out ~/shader_data/glsl_bpe` (self-test passes; verify pretokenizer parity with OLMo's
      BPETokenizer on the box before retraining).
- [ ] Retokenize → .npy; retrain the large model on the bigger corpus.
- [ ] Then: batched eval (N=50–100, compile@1/@5/@10) + debugger-in-the-loop RL (GRPO/RLOO with
      OmniTrace's rich reward). See tasks #27–#32.
