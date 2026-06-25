# SubQ Track Roadmap (DD-LL)

This file consolidates the SubQ-attention work items. SubQ is the
content-selected sub-quadratic attention from
[subq.ai](https://subq.ai/) — the project's open research direction
documented in `project_subq_fight` in the user memory.

Existing infrastructure already on `2fast2furious`:
- `kernels/sparse_attn_decode.cu` — content-selected SDPA decode kernel
- `scripts/bench_subq.sh` — head-to-head bench harness
- `tools/bench_attn.cpp` — kernel-isolated attention bench (`subq` mode)
- `results/subq_baseline.csv` — frozen baseline measurements

## Items

### DD. Content-selected attention forward kernel for prefill
Status: shipped in `kernels/sparse_attn_decode.cu` for decode (S=1).
Prefill (S>1) variant lands in a sibling kernel; reuses the same
selection logic but processes a full Q-block per launch instead of a
single query.

### EE. SubQ-fused kernel (selection + attention)
Status: partially done in `sparse_attn_decode.cu`. The fused variant
where the content-selection pass and the attention compute share
shared-memory state (selection scores → top-k indices → indexed
attention all in one kernel) is the perf target. Tracked separately
from DD because the fusion changes the SMEM layout.

### FF. SubQ decode-step variant (S=1)
Status: shipped as `sparse_attn_decode.cu`. Already called from the
paged decode path under `--subq` config flag.

### GG. SubQ-compatible KV cache layout in paged store
Status: pending. Today's BlockManager treats all KV positions as
random-access (good for content-selected attention since selection
picks arbitrary positions). The optimization: coalesce
frequently-selected positions into the same page to maximize cache
locality during the selection scan. Block-rebalance happens on a
background thread during decode lull.

### HH. SubQ backward kernel
Status: pending. Forward kernel exists; gradient through the
selection mask is non-trivial because the top-k indices are
discrete. Two strategies:
  (a) Straight-through estimator on selection scores.
  (b) REINFORCE-style continuous relaxation.
SubQ paper draft uses (a).

### II. SubQ + MTP integration
Status: pending. SubQ in the target model + MTP heads as drafts:
the drafts run on a fast sub-quadratic body, the verify forward is
also sub-quadratic. Composable speedups.

### JJ. SubQ vs dense head-to-head bench refresh
Status: `scripts/bench_subq.sh` exists. Refresh on current branch +
add charting in `tools/bench_attn.cpp` output to make the plot
trivially reproducible.

### KK. SubQ long-context regression
Status: pending. Run inference at 32K / 64K / 128K context on
5060 Ti, compare SubQ vs dense for both quality (perplexity on
held-out) and speed.

### LL. SubQ paper draft
Status: pending. Tier-A target. Material:
- Mechanism + theoretical scaling argument.
- Ablations on selection-policy choice.
- Sparsity-vs-quality tradeoff curves.
- Long-context wall-clock & PPL numbers.

## Where this fits in the master 95-item list

Items DD-LL are research-grade contributions that each parallel a
kernel infrastructure item already shipped on this branch:
- DD/EE/FF lean on paged_attention_decode_dyn (I-1 infrastructure)
- GG leans on the SharedBlockPool (I-7)
- HH needs custom autograd integration similar to the activation
  checkpoint path
- II uses the MTP speculative orchestration (I-3)

Schedule: weeks 6-12 of the overall plan, in parallel with the
training-side fusion work (G/H/I/L) that the inference SubQ kernels
benefit from.
