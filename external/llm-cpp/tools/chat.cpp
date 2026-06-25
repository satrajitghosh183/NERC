/**
 * tools/chat.cpp
 *
 * Interactive CLI for sampling from a trained OLMo C++ model. Loads the
 * Transformer + tokenizer, then in a REPL loop reads "You:" prompts on
 * stdin, encodes them, runs autoregressive decoding (with optional KV
 * cache and MTP speculative decoding), and streams decoded tokens to
 * stdout as they are generated. Per-turn it also prints a stats line
 * with tokens/sec. No files are written.
 *
 * Example:
 *   ./build/chat --checkpoint checkpoints/125M.pt \
 *                --config configs/olmo2_125M.json \
 *                --vocab-file data/gpt2/vocab.json \
 *                --merges-file data/gpt2/merges.txt
 *
 * --- Flags ---
 *   --checkpoint            torch::save'd .pt file produced by training
 *   --config                JSON config used to build the model topology
 *   --vocab-file            GPT-2 vocab.json
 *   --merges-file           GPT-2 merges.txt
 *   --structural-config     optional structural tokenizer config dir
 *   --device                mps | cuda | cpu (default: auto-detect)
 *   --max-tokens            cap on generated tokens per turn (default 128)
 *   --temperature           sampling temperature; <=0 means greedy (0.8)
 *   --top-k                 top-k cutoff, 0 disables (default 50)
 *   --top-p                 nucleus sampling cutoff, 1.0 disables (0.9)
 *   --repetition-penalty    >1.0 penalises tokens already in context (1.1)
 *   --legacy-decode         decode token 33 as space (older tokenizer)
 *   --no-kv-cache           force full-context recompute every step
 *   --no-speculative        disable MTP speculative decoding path
 *
 * --- Build target ---
 *   chat (CMakeLists.txt:517). Links the static `olmo_cpp` library
 *   (model + tokenizer + backends), LibTorch, and nlohmann/json.
 *   Compiled with -O3 -march=native; HAS_NLOHMANN_JSON is defined when
 *   JSON support was found at configure time (required to load configs).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp               : load_config_from_json()
 *   - olmo_cpp/model/transformer.hpp    : Transformer module class
 *   - olmo_cpp/model/kv_cache.hpp       : per-layer KV cache + snapshot/rollback
 *   - olmo_cpp/backend/cuda_graph.hpp   : CUDA graph capture (future use here)
 *   - olmo_cpp/data/bpe_tokenizer.hpp   : GPT-2 BPE
 *   - olmo_cpp/data/structural_tokenizer.hpp : optional structural tokenizer
 *   - olmo_cpp/backend/cuda_backend.hpp : enable fused CUDA kernels on GPU
 *   - olmo_cpp/backend/simd_backend.hpp : enable SIMD CPU kernels on CPU
 *
 * --- Reads / Writes ---
 *   - reads:  checkpoint .pt, config .json, vocab.json, merges.txt,
 *             optional structural-config dir
 *   - writes: nothing — generation is streamed to stdout.
 *
 * --- Role in workflow ---
 *   Used after training (`olmo_train conf/...`) to qualitatively eyeball
 *   model behaviour and benchmark inference throughput. With MTP heads
 *   enabled in the config, this is also where the speculative decoding
 *   acceptance rate is observed.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/paged_kv_cache.hpp"
#if defined(OLMO_HAS_CUDA_KERNELS) || defined(USE_CUDA)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "olmo_cpp/backend/cuda_graph.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/data/structural_tokenizer.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include "olmo_cpp/backend/topp_radix.hpp"
#include "olmo_cpp/backend/gpu_sample.hpp"
#include "olmo_cpp/backend/fused_lm_head_sample.hpp"
#include "olmo_cpp/backend/lm_head_gemv.hpp"
#include "olmo_cpp/backend/persistent_decode.hpp"
#include "olmo_cpp/generate/draft_model_speculative.hpp"
#include "olmo_cpp/generate/tree_speculative.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <optional>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

/// Resolve the runtime device given a user preference string.
/// Falls through to CPU if MPS/CUDA was requested but isn't available.
/// On Apple Silicon "metal" is treated as an alias for "mps".
torch::Device select_device(const std::string& preferred) {
  if (preferred == "mps" || preferred == "metal") {
#ifdef __APPLE__
    if (torch::mps::is_available())
      return torch::Device(torch::kMPS);
#endif
  }
  if (preferred == "cuda") {
    if (torch::cuda::is_available())
      return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

/// Old checkpoints omit the mtp_heads submodule; match TransformerImpl's
/// "only register when non-empty" rule so torch::load does not fail.
void align_mtp_config_with_checkpoint(olmo_cpp::TransformerConfig& cfg,
                                      const std::string& checkpoint_path) {
  if (cfg.num_mtp_heads <= 0) return;
  torch::serialize::InputArchive archive;
  archive.load_from(checkpoint_path, torch::kCPU);
  for (const auto& key : archive.keys()) {
    if (key == "mtp_heads") return;
  }
  std::cerr << "Note: checkpoint has no mtp_heads weights; "
               "loading with num_mtp_heads=0 (speculative decoding disabled)\n";
  cfg.num_mtp_heads = 0;
}

/// Bring a tensor to host via *pinned* (page-locked) memory. Pinned pages
/// skip the driver's pageable-staging copy, ~2x faster D->H transfer on
/// PCIe. Always returns a fresh writable owned buffer. On CPU/MPS this
/// degrades to .contiguous().clone() — pinning is CUDA-specific.
/// (fast-inference [12d])
torch::Tensor to_pinned_host(const torch::Tensor& t) {
  if (t.is_cuda()) {
    auto src = t.contiguous();
    auto dst = torch::empty(src.sizes(),
                            torch::TensorOptions()
                                .dtype(src.dtype())
                                .device(torch::kCPU)
                                .pinned_memory(true));
    dst.copy_(src, /*non_blocking=*/true);
    torch::cuda::synchronize();
    return dst;
  }
  return t.contiguous().clone();
}

/// H2: single-token decode input without a per-token alloc + 8-byte H->D.
/// Reuses one [1,1] int64 device buffer and fills the token via fill_, which
/// bakes the scalar into a fill kernel (no host buffer copy). Stream ordering
/// guarantees the forward reads it before the next iteration's fill_; safe
/// because the returned tensor is consumed by the forward immediately.
/// (Replaces `torch::tensor({tok}).unsqueeze(0).to(device)` in the decode
/// loops. The paged CUDA-graph branch keeps its own static capture buffer.)
torch::Tensor decode_input_buf(int64_t tok, torch::Device device) {
  static thread_local torch::Tensor buf;
  if (!buf.defined() || buf.device() != device) {
    buf = torch::empty({1, 1},
                       torch::TensorOptions().dtype(torch::kInt64).device(device));
  }
  buf.fill_(tok);
  return buf;
}

// =====================================================================
// FAST-INFERENCE ROADMAP — beat TensorRT-LLM at one config.
// Branch: fast-inference. Target: Llama/OLMo-class 1B–7B on H100,
//   batch=1 decode latency. Numbers assume V≈50K, H100, bf16.
// =====================================================================
//
// Strategy: don't fight TRT-LLM on its home turf (general transformer
// inference, all GPUs, all batches). Specialize ruthlessly to ONE
// config and win there. ~10% from each item below; combined ~2-3x
// vs TRT-LLM at this niche is the realistic ceiling.
//
// Order is execution order, not just impact order. Earlier items
// unlock later items (e.g. paged KV unlocks CUDA graphs).
// =====================================================================
//
// PHASE 1 — UNBLOCK (weeks 1-4)
// ---------------------------------------------------------------------
//
// [1] Paged KV cache.   Prerequisite for almost everything else.
//     Current concat-based cache (chat.cpp around L655 forward call)
//     reallocates each step → shape changes → CUDA graphs impossible,
//     batching impossible, long context O(L) memcpy per step.
//     Build: fixed-size page allocator (e.g. 16 tokens/page), per-
//     request page table, kernel-side gather via page indices.
//     Reference: vLLM PagedAttention paper. ~3 weeks of focused work.
//     Files to touch: include/olmo_cpp/model/kv_cache.hpp,
//                     src/model/attention.cpp,
//                     new kernels/paged_attention.cu
//
// [2] CUDA graphs around the decode step.   ~10-30% latency.
//     Capture the [1,1]-input forward pass once, replay each step.
//     Eliminates ~500 launches × ~3-5μs of driver overhead per token.
//     Blocked on [1] (shapes must be stable).
//     Call site: tools/chat.cpp:650 (the KV-cache decode loop).
//     Reference: cudaGraphCreate / cudaGraphLaunch. ~1 week.
//
// [3] Bench harness vs TRT-LLM.   Without this you don't know if you
//     won. Pick: Llama-3-8B (or OLMo equivalent), H100 SXM, FP8/BF16,
//     128-token prompt → 256-token decode, batch=1. Measure: TPOT
//     (time per output token) and TTFT (time to first token).
//     Build: a script that runs both engines on the same prompts
//     and emits a comparison table. ~3 days.
//     Lives under: scripts/bench_vs_trtllm.sh
//
// PHASE 2 — KERNEL DOMINANCE (weeks 5-12)
// ---------------------------------------------------------------------
//
// [4] FlashAttention-2/3 decode kernel.   ~30-50% on attention.
//     Decode attention is bandwidth-bound on KV reads. Need the
//     decode variant (single Q vs many K,V) — different from the
//     training attention. Options: port FA-3, fork FlashInfer's
//     batch_decode_with_paged_kv_cache, or write fresh on top of
//     CUTLASS. ~4 weeks.
//     Reference: Tri Dao FA-3 paper, FlashInfer source.
//     Files: new kernels/decode_attention.cu, integrate via IBackend.
//
// [5] Custom LM-head GEMV.   ~20-40% on LM head step.
//     cuBLAS GEMM is tuned for square matmuls, not [1,H]·[H,V] GEMV.
//     Hand-written GEMV with split-K reduction across the V dim,
//     using TMA on H100 for W_U streaming. ~2 weeks.
//     Files: new kernels/lm_head_gemv.cu
//
// [6] Fused LM-head + sampling kernel via Gumbel-max trick.   ~5-10%.
//     sample(softmax(l/T)) ≡ argmax_i (l_i/T + g_i), g_i~Gumbel(0,1).
//     Fold into [5]: stream W_U rows, dot with hidden, generate
//     Gumbel via Philox(seed,position,vocab_idx) [or curand_uniform
//     in device API], reduce argmax, emit one int64.
//     Eliminates: [V] logits write to HBM, the multi-kernel
//     softmax/topk/sort chain, AND the 200KB D->H copy at chat.cpp
//     around L132. Phase 1: greedy + temperature only.
//     Switches std::mt19937 -> Philox; samples differ from CPU path
//     even with same seed — document this and gate behind a config flag.
//     Files: kernels/lm_head_gemv.cu (extend [5])
//     Call site: tools/chat.cpp around L545, L564, L222 (speculative).
//
// [7] Bucket-radix top-p kernel.   For when top-p is needed.
//     16 log-spaced bins, single pass histogram, scan to find cutoff
//     bucket, sort just that bucket. O(V) vs O(V log V). Min-p early
//     drop (probs < 2^-16) cuts ~70% of vocab outright.
//     Replaces CPU sort at L148-165. Fuses with [6].
//     Files: kernels/topp_radix.cu. ~1 week.
//
// PHASE 3 — QUANTIZATION (weeks 13-18)
// ---------------------------------------------------------------------
//
// [8] FP8 weight-only quantization.   ~1.5x decode throughput.
//     H100 has native FP8. Quantize weights to FP8 E4M3, keep
//     activations in BF16, dequant in the GEMM/GEMV epilogue.
//     Touches every Linear layer; biggest hit on LM head and
//     attention out_proj. Use TransformerEngine as a reference,
//     but write kernels specialized to this model. ~2-3 weeks.
//     Quality: typically <0.5% perplexity regression on calibrated
//     sets. No retraining needed.
//
// [9] INT4 weight-only quantization (AWQ-style).   ~2x on top of [8].
//     Group-quantized INT4 (g=128) with FP16 scales. Custom GEMV
//     kernel that dequants in registers. Loses ~1-2% perplexity,
//     gainable back with group-aware fine-tune. ~3-4 weeks.
//     Reference: AWQ paper, llama.cpp Q4_K_M.
//
// PHASE 4 — RESEARCH WINS (weeks 19-26)
// ---------------------------------------------------------------------
//
// [10] Speculative decoding overhaul.   1.5-3x at unchanged quality.
//      MTP path exists at speculative_decode_step (around L189) but
//      is suboptimal:
//      (a) batch the k draft sample calls (currently k separate
//          .cpu() round-trips at L322-329 — already TODO'd inline).
//      (b) tune draft length k dynamically from running accept rate.
//      (c) try a tiny separate draft model (TinyLlama 1B drafting
//          for 7B target) — typically higher acceptance than MTP.
//      (d) combine with EAGLE-2 / lookahead decoding for tree-style
//          drafting. This is publishable.
//
// [11] Persistent decode kernel.   The killer.
//      One kernel launched at startup, runs forever, polls a memory
//      queue for work. CPU writes "next token + KV ptr"; GPU runs
//      the full decode step (forward + sample) and writes the result.
//      ZERO kernel launches per token after init.
//      Saves ~500 launches × ~3μs = ~1.5ms/token of pure overhead
//      (which is huge if your forward is already optimized below 5ms).
//      Reference: TensorRT-LLM in-flight batching kernel,
//                 FlashInfer persistent kernel.
//      ~3-4 weeks. This is what gets you from "tied with TRT-LLM"
//      to "beating it."
//
// PHASE 5 — POLISH (ongoing)
// ---------------------------------------------------------------------
//
// [12] Smaller wins worth a pass:
//      - Pinned host buffers for D->H copies (cudaHostAlloc) —
//        irrelevant once [6] lands (no copies left).
//      - Fuse repetition_penalty into the sampling kernel (currently
//        a separate CPU loop at chat.cpp:105 that materializes a clone).
//      - Pre-tokenize prompt on a worker thread while model loads.
//      - bf16 end-to-end (model already supports it; verify config).
//      - Capture full conversation context once across turns instead
//        of re-encoding per turn (chat loop in main()).
//      - Replace torch::Tensor on the inference path with raw CUDA
//        pointers + a tiny shape struct — saves PyTorch dispatcher
//        overhead in C++ (~1-2μs per op, real at high token rates).
//
// EXPLICITLY NOT DOING (sunk-cost traps):
// ---------------------------------------------------------------------
//   - Rewriting from scratch. The model code, tokenizer, and existing
//     CUDA kernels (rms_norm.cu, silu_mul.cu, rope.cu) are correct
//     and reusable. Carve and replace, don't rebuild.
//   - General-purpose engine (multi-arch, multi-GPU, multi-precision).
//     Specialize to ONE target config; that's how you beat TRT-LLM.
//   - Continuous batching unless we go server-side. Single-stream
//     latency is the win condition; batching is a different game.
//   - top-p sort fusion. Use [7] (bucket-radix) instead.
//   - Standalone GPU multinomial replacement. Subsumed by [6].
//
// SUCCESS CRITERION:
// ---------------------------------------------------------------------
//   Median TPOT (time per output token) on Llama-3-8B (or equivalent),
//   H100 SXM, batch=1, 128→256 tokens, FP8: lower than TensorRT-LLM
//   v0.16+ at the same TTFT and same output quality (PPL within 1%).
//   Bench harness from [3] is the source of truth.
//
// ESTIMATED TIMELINE:
//   Single researcher full-time: 5-6 months to TRT-LLM-competitive,
//   8-10 months to clearly beating it on this niche.
//   With one collaborator: cut by ~30%.
// =====================================================================

/// Sample from logits with rep-penalty, temperature, top-k, and top-p
/// (nucleus) filtering, all fused into a single host-side preprocessing pass.
///
/// Owns one CPU clone of the logits and modifies it in place. The previous
/// `apply_repetition_penalty` step is gone — rep_tokens scatter directly into
/// this buffer, and the temperature scale is the same dense walk. One walk,
/// not two; one allocation, not two. (fast-inference [12c])
///
/// TODO(fast-inference [6]): for greedy/temperature-only (top_k<=0 &&
/// top_p>=1.0), replace this entire function with a single fused kernel
/// call inside the LM-head GEMV (roadmap item [6], depends on [5]).
/// The .cpu() below is the costliest line — 200 KB D->H/token.
/// For top-p path see roadmap item [7] (bucket-radix kernel) — keep this
/// CPU implementation only as a debug fallback once [7] lands.
int64_t sample_logits(torch::Tensor logits, double temperature,
                      int64_t top_k, double top_p,
                      const std::vector<int64_t>& rep_tokens,
                      double rep_penalty,
                      std::mt19937& gen) {
  // ── GPU-resident fast path (fast-inference [6]/[7], wired live) ──
  // Everything stays on-device through the shared sampler; only the token id
  // (8 bytes) crosses D->H, vs the full [V] (~200 KB/token) the host path
  // below copies. Greedy (temperature 0, the race path) is deterministic and
  // identical to the host path; sampled output uses torch's CUDA generator.
  if (logits.is_cuda()) {
    (void)gen;
    return olmo_cpp::gpu_sample(logits, temperature, top_k, top_p,
                                rep_tokens, rep_penalty);
  }

  // ── Host fallback (CPU / MPS) ──
  // Bring to host via pinned memory (faster D->H on CUDA) and own the
  // resulting buffer so we can mutate it in place. (fast-inference [12d])
  auto logits_cpu = to_pinned_host(logits);
  int64_t vocab_size = logits_cpu.size(0);
  auto* p = logits_cpu.data_ptr<float>();

  // FUSED pre-pass over the host buffer:
  //   (a) rep penalty: sparse scatter into already-seen token ids
  //   (b) temperature: dense scale across the vocab
  // Order matters: rep penalty divides existing logits, so applying it
  // before temperature is mathematically equivalent to applying it after
  // (just a constant factor swap) and lets us touch the same memory once.
  if (rep_penalty != 1.0 && !rep_tokens.empty()) {
    const float pen = static_cast<float>(rep_penalty);
    for (int64_t id : rep_tokens) {
      if (id < 0 || id >= vocab_size) continue;
      const float s = p[id];
      p[id] = (s > 0.0f) ? (s / pen) : (s * pen);
    }
  }

  // Greedy: rep-penalty-modified argmax. Skip temperature/sampling pipeline.
  if (temperature <= 0.0) {
    return logits_cpu.argmax(-1).item<int64_t>();
  }

  if (temperature != 1.0) {
    const float inv_t = 1.0f / static_cast<float>(temperature);
    for (int64_t i = 0; i < vocab_size; ++i) p[i] *= inv_t;
  }

  // Top-k filtering. NOTE: this reassigns logits_cpu to a new tensor (the
  // where() output), so the `p` pointer above becomes stale here. That's
  // fine because we don't need the original buffer anymore.
  if (top_k > 0 && top_k < vocab_size) {
    auto [topk_vals, topk_indices] = logits_cpu.topk(top_k);
    auto threshold = topk_vals.index({topk_vals.size(0) - 1}).item<float>();
    logits_cpu = torch::where(logits_cpu < threshold,
                              torch::full_like(logits_cpu, -std::numeric_limits<float>::infinity()),
                              logits_cpu);
  }

  auto probs = torch::softmax(logits_cpu, -1);

  // Top-p (nucleus) via bucket-radix histogram (O(V) vs O(V log V) sort).
  // (fast-inference [7] — wired live)
  if (top_p < 1.0) {
    olmo_cpp::topp_radix_filter_cpu(probs, static_cast<float>(top_p), /*min_p=*/1.0e-5f);
  }

  // Sampling: cumulative-probability + std::uniform_real_distribution +
  // upper_bound binary search. The previous code allocated a fresh
  // std::vector<double>(vocab_size) per token, copied float→double across
  // the whole vocab, then constructed std::discrete_distribution which
  // re-walks the vector to build its own probability table. For 50257
  // vocab that's two ~200 KB allocs + a hash-map's worth of internal
  // bookkeeping every token. The cumulative-search route walks the vocab
  // ONCE into a thread-local scratch buffer (reused across tokens), then
  // does a single binary search to draw the sample.
  static thread_local std::vector<float> cdf_buf;
  cdf_buf.resize(static_cast<size_t>(vocab_size));
  auto* p_probs = probs.data_ptr<float>();
  float running = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    running += p_probs[i];
    cdf_buf[static_cast<size_t>(i)] = running;
  }
  // Guard against an all-zero distribution (shouldn't happen after softmax,
  // but top-p / top-k filtering can degenerate at extreme settings).
  if (running <= 0.0f) return logits_cpu.argmax(-1).item<int64_t>();
  std::uniform_real_distribution<float> u(0.0f, running);
  const float r = u(gen);
  const auto it = std::upper_bound(cdf_buf.begin(),
                                   cdf_buf.begin() + vocab_size, r);
  return std::distance(cdf_buf.begin(), it);
}

/// Speculative decoding with KV cache: draft k tokens via MTP heads, verify in batch.
///
/// Algorithm (per step):
///   1. Run backbone incrementally (1 token) → get hidden state + update KV cache
///   2. Main LM head → token t+1
///   3. MTP heads → draft tokens t+2..t+k+1 (no forward pass, just projections)
///   4. Snapshot KV cache, then verify [main_token, drafts...] in one forward pass
///   5. Accept matching tokens, rollback KV cache to first rejection point
///
/// Speedup: O(1) draft (projections only) + O(k) verify vs O(k) full forwards.
/// With k=3 MTP heads: up to 4 tokens from 2 short forward passes instead of 4 full passes.
int64_t speculative_decode_step(
    olmo_cpp::Transformer& model,
    std::vector<int64_t>& all_tokens,
    olmo_cpp::KVCache& kv_cache,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double repetition_penalty,
    std::mt19937& rng,
    olmo_cpp::BPETokenizer& tokenizer,
    int64_t& total_drafted,
    int64_t& total_accepted,
    int64_t max_drafts,    // (fast-inference [10b]) caller's dynamic cap
    bool use_custom_lm_head_gemv) {  // (fast-inference [11])

  // Honor the dynamic cap: never exceed the model's MTP head count, but
  // allow the caller to draft fewer when running acceptance is poor.
  int64_t num_drafts = model->num_mtp_heads();
  if (max_drafts > 0 && max_drafts < num_drafts) num_drafts = max_drafts;
  int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());
  torch::NoGradGuard no_grad;

  // Step 1: Incremental backbone forward (1 token) → hidden state
  int64_t last_token = all_tokens.back();
  auto input = decode_input_buf(last_token, device);  // H2: reused buffer, no per-token H->D
  auto hidden = model->forward_backbone(input, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif

  // hidden: [1, 1, d_model] → last position
  auto last_hidden = hidden.select(1, 0);  // [1, d_model]

  // Step 2: Main head prediction for position t+1.
  // (fast-inference [11]) Optionally route the LM-head projection through
  // the custom GEMV kernel; behaviorally identical to apply_lm_head.
  // (fast-inference [12d]) sample_logits transfers via pinned memory.
  torch::Tensor main_logits;
  if (use_custom_lm_head_gemv) {
    auto h = last_hidden.squeeze(0);  // [d_model]
    main_logits = olmo_cpp::lm_head_gemv(h, model->lm_head_weight());
  } else {
    main_logits = model->apply_lm_head(last_hidden.unsqueeze(1))
                       .squeeze(0).squeeze(0);
  }
  int64_t main_token = sample_logits(main_logits, temperature, top_k, top_p,
                                     all_tokens, repetition_penalty, rng);

  if (main_token == eos_id) {
    all_tokens.push_back(main_token);
    return 1;
  }

  // Step 3: Draft tokens from MTP heads (just projections — no backbone forward!)
  auto draft_logits_list = model->forward_mtp_draft(last_hidden);

  std::vector<int64_t> draft_tokens;
  draft_tokens.reserve(num_drafts);

  auto temp_tokens = all_tokens;
  temp_tokens.push_back(main_token);

  // Draft sampling stays on-device. sample_logits routes CUDA logits through
  // the GPU-resident sampler, so there's NO [k, V] device->host copy — only
  // each drafted token id (8 bytes) returns. The loop is sequential because
  // each draft's rep_penalty depends on the previous (temp_tokens grows).
  for (int64_t k = 0; k < num_drafts; ++k) {
    int64_t draft_tok = sample_logits(draft_logits_list[k], temperature, top_k, top_p,
                                      temp_tokens, repetition_penalty, rng);
    draft_tokens.push_back(draft_tok);
    temp_tokens.push_back(draft_tok);
    if (draft_tok == eos_id) break;
  }

  total_drafted += static_cast<int64_t>(draft_tokens.size());

  // Step 4: Snapshot KV cache, then verify [main_token, draft_tokens...] in one pass
  auto snap = kv_cache.snapshot();

  // Build verification input: [main_token, draft_0, draft_1, ...]
  std::vector<int64_t> verify_seq;
  verify_seq.reserve(1 + draft_tokens.size());
  verify_seq.push_back(main_token);
  for (auto dt : draft_tokens) verify_seq.push_back(dt);

  auto verify_input = torch::tensor(
      at::IntArrayRef(verify_seq.data(), verify_seq.size()),
      torch::kInt64).unsqueeze(0).to(device);

  // This extends the KV cache with the verify sequence
  auto verify_logits = model->forward(verify_input, c10::nullopt, -100, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif
  // Verify on-device: one argmax over the vocab for every verify position,
  // then bring back only the [k+1] chosen ids — NOT the [k+1, V] logits
  // (~800 KB at V=50304). (fast-inference [12d], GPU-resident)
  auto model_choices = verify_logits.select(0, 0).argmax(-1).to(torch::kCPU).contiguous();  // [k+1]
  const int64_t* mc = model_choices.data_ptr<int64_t>();

  // Step 5: Accept tokens — verify_logits[0][0] predicts what comes after main_token
  // verify_logits[0][k] predicts what comes after draft_tokens[k-1]
  all_tokens.push_back(main_token);
  std::vector<uint32_t> accepted_toks = {static_cast<uint32_t>(main_token)};
  int64_t accepted = 1;
  int64_t drafts_accepted = 0;

  for (int64_t k = 0; k < static_cast<int64_t>(draft_tokens.size()); ++k) {
    if (draft_tokens[k] == eos_id) {
      all_tokens.push_back(eos_id);
      accepted++;
      drafts_accepted++;
      break;
    }

    // verify_logits[0][k] is the logits after processing verify_seq[k]
    // which predicts the token at position verify_seq[k+1]
    // So position k in verify_logits predicts what should come after draft_tokens[k-1]
    // (or after main_token when k=0)
    int64_t model_choice = mc[k];   // GPU argmax computed once, above

    if (model_choice == draft_tokens[k]) {
      all_tokens.push_back(draft_tokens[k]);
      accepted_toks.push_back(static_cast<uint32_t>(draft_tokens[k]));
      accepted++;
      drafts_accepted++;
    } else {
      // Reject: use model's choice instead
      all_tokens.push_back(model_choice);
      accepted_toks.push_back(static_cast<uint32_t>(model_choice));
      accepted++;
      break;
    }
  }

  total_accepted += drafts_accepted;

  // Rollback KV cache: keep snap + accepted tokens worth of new entries
  // snap was the length before verification. We added (1 + draft_tokens.size()) entries.
  // We want to keep snap + accepted entries.
  int64_t desired_len = snap + accepted;
  if (desired_len < kv_cache.seq_len()) {
    kv_cache.rollback(desired_len);
  }

  // Print all accepted tokens
  std::cout << tokenizer.decode(accepted_toks) << std::flush;

  return accepted;
}

/// Tree-shaped speculative decoding step (Medusa/EAGLE-style).
///
/// Drafts a width-`fanout` tree of depth `depth` from the MTP heads
/// of the target model, runs ONE tree-attention verify forward, and
/// walks the longest matching root-to-leaf path. Each MTP head k
/// supplies top-`fanout` candidates for depth k+1 of the tree.
///
/// Why tree over linear chain: the linear-chain spec accepts the
/// longest prefix of ONE drafted sequence. The tree carries multiple
/// alternatives per depth, so the verify can match any of them — the
/// expected-accepted-length is strictly ≥ the linear case at the
/// same MTP depth. Cost is one additional forward_backbone on the
/// accepted prefix (since forward_tree is stateless and doesn't
/// update the KV cache).
///
/// Returns the count of new tokens appended to `all_tokens`.
int64_t tree_decode_step(
    olmo_cpp::Transformer& model,
    std::vector<int64_t>& all_tokens,
    olmo_cpp::KVCache& kv_cache,
    torch::Device device,
    olmo_cpp::BPETokenizer& tokenizer,
    int64_t fanout,
    int64_t depth,
    int64_t& total_drafted,
    int64_t& total_accepted) {
  torch::NoGradGuard no_grad;
  int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());

  // Step 1: Incremental backbone forward (1 token) to obtain the seed
  // hidden state. Mirrors the speculative path's invariant of caching
  // up through last_token before drafting.
  int64_t last_token = all_tokens.back();
  auto input = decode_input_buf(last_token, device);  // H2: reused buffer, no per-token H->D
  auto hidden = model->forward_backbone(input, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif
  auto last_hidden = hidden.select(1, 0);   // [1, d_model]

  // Step 2: Tree-speculative orchestrator. Builds the tree, runs the
  // stateless forward_tree verify, walks the accepted path. Always
  // returns at least one token (the root's argmax prediction).
  auto accepted = olmo_cpp::tree_speculative_step(
      model, last_hidden, last_token, fanout, depth, device);
  if (accepted.empty()) return 0;

  // Stats: every non-root node in the tree is a draft candidate.
  // For fanout f, depth d the tree has (1 + f + f² + ... + fᵈ) nodes;
  // drafts = nodes − 1.
  int64_t drafts_this_step = 0;
  {
    int64_t pow_term = 1;
    for (int64_t d = 1; d <= depth; ++d) {
      pow_term *= fanout;
      drafts_this_step += pow_term;
    }
  }
  total_drafted   += drafts_this_step;
  total_accepted  += static_cast<int64_t>(accepted.size());

  // Step 3: forward_tree is stateless — re-run the accepted prefix
  // through forward_backbone to materialize KV cache entries for
  // the new tokens. We could fuse this with the tree verify if the
  // verify path also wrote into the cache, but tree masks complicate
  // cache-write logic and the recompute on a short accepted prefix
  // is cheap relative to the verify forward.
  auto accept_input = torch::tensor(
      at::IntArrayRef(accepted.data(), accepted.size()),
      torch::kInt64).unsqueeze(0).to(device);
  model->forward_backbone(accept_input, &kv_cache);
#ifdef __APPLE__
  if (device.is_mps()) torch::mps::synchronize();
#endif

  // Step 4: commit + stream the accepted tokens.
  std::vector<uint32_t> accepted_u32;
  accepted_u32.reserve(accepted.size());
  for (auto t : accepted) {
    all_tokens.push_back(t);
    accepted_u32.push_back(static_cast<uint32_t>(t));
    if (t == eos_id) break;
  }
  std::cout << tokenizer.decode(accepted_u32) << std::flush;
  return static_cast<int64_t>(accepted_u32.size());
}

}  // namespace

int main(int argc, char** argv) {
  // -----------------------------------------------------------------
  // Phase 1: parse CLI flags. Defaults below match the docblock above.
  // -----------------------------------------------------------------
  std::string checkpoint_path, config_path, vocab_path, merges_path;
  std::string device_pref = "auto";
  int64_t max_tokens = 128;
  double temperature = 0.8;
  int64_t top_k = 50;
  double top_p = 0.9;
  double repetition_penalty = 1.1;
  bool legacy_decode = false;
  bool use_kv_cache = true;
  bool use_speculative = true;
  // Tree-shaped speculative decoding (Medusa/EAGLE-style). When set,
  // overrides the linear-chain spec path: drafts a fanout-`tree_fanout`
  // × depth-`tree_depth` tree from the MTP heads and verifies it in
  // ONE forward via forward_tree. Requires the model to have MTP heads.
  bool use_tree_decode = false;
  int64_t tree_fanout = 2;
  int64_t tree_depth  = 0;   // 0 = use model->num_mtp_heads()
  // fast-inference [6]: fused LM-head + Gumbel-max sampler. Bypasses
  // sample_logits entirely. Requires top_k=0, top_p=1.0, rep_penalty=1.0
  // (the kernel doesn't support those filters yet). Off by default
  // because it changes sampling semantics (Philox RNG, not std::mt19937).
  bool use_fused_sampler = false;
  uint64_t fused_seed = 0xC0FFEEULL;  // override-able via --fused-seed
  // fast-inference [11]: route LM-head projection through the custom
  // GEMV kernel instead of cuBLAS-via-Linear. Behaviorally identical
  // (same logits), perf only.
  bool use_custom_lm_head_gemv = false;
  // fast-inference [13]: route the (greedy/temp-only) sampler through
  // the persistent-decode handle. Currently a stub that delegates
  // synchronously to fused_lm_head_sample; demonstrates the API path.
  bool use_persistent_decode = false;
  // fast-inference [17]: two-model speculative decoding. When both
  // a draft checkpoint and config are supplied, switches the
  // speculative path to use a separate small model instead of MTP.
  std::string draft_checkpoint_path;
  std::string draft_config_path;
  std::string structural_config;

  // fast-inference [1]: paged KV cache (BlockManager-backed) on the
  // non-speculative decode path. Stable layout precondition for whole-step
  // CUDA graph capture (item [2]) and continuous batching (item [3]).
  bool use_paged_kv = false;
  int64_t paged_page_size = 16;
  int64_t paged_max_seq   = 2048;
  // fast-inference [6]: chunked prefill. Split a long prompt into
  // chunks of this size and feed each through forward_paged. 0 disables
  // (single full-length prefill).
  int64_t prefill_chunk_size = 0;
  // fast-inference [1]: capture decode step into a CUDA graph and
  // replay each subsequent step. Requires --paged-kv and a CUDA device;
  // automatically uses make_paged_kv_cache_graph_safe so K/V writes go
  // through the dyn write kernel and replays land in the right slot.
  bool use_cuda_graph = false;
  int64_t cuda_graph_warmup_steps = 2;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
    else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (arg == "--vocab-file" && i + 1 < argc) vocab_path = argv[++i];
    else if (arg == "--merges-file" && i + 1 < argc) merges_path = argv[++i];
    else if (arg == "--structural-config" && i + 1 < argc) structural_config = argv[++i];
    else if (arg == "--device" && i + 1 < argc) device_pref = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoll(argv[++i]);
    else if (arg == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
    else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoll(argv[++i]);
    else if (arg == "--top-p" && i + 1 < argc) top_p = std::stod(argv[++i]);
    else if (arg == "--repetition-penalty" && i + 1 < argc) repetition_penalty = std::stod(argv[++i]);
    else if (arg == "--legacy-decode") legacy_decode = true;
    else if (arg == "--no-kv-cache") use_kv_cache = false;
    else if (arg == "--no-speculative") use_speculative = false;
    else if (arg == "--tree-decode") use_tree_decode = true;
    else if (arg == "--tree-fanout" && i + 1 < argc) tree_fanout = std::stoll(argv[++i]);
    else if (arg == "--tree-depth" && i + 1 < argc) tree_depth = std::stoll(argv[++i]);
    else if (arg == "--fused-sampler") use_fused_sampler = true;
    else if (arg == "--fused-seed" && i + 1 < argc)
      fused_seed = std::stoull(argv[++i]);
    else if (arg == "--custom-lm-head-gemv") use_custom_lm_head_gemv = true;
    else if (arg == "--persistent-decode") use_persistent_decode = true;
    else if (arg == "--draft-checkpoint" && i + 1 < argc)
      draft_checkpoint_path = argv[++i];
    else if (arg == "--draft-config" && i + 1 < argc)
      draft_config_path = argv[++i];
    else if (arg == "--paged-kv") use_paged_kv = true;
    else if (arg == "--page-size" && i + 1 < argc)
      paged_page_size = std::stoll(argv[++i]);
    else if (arg == "--paged-max-seq" && i + 1 < argc)
      paged_max_seq = std::stoll(argv[++i]);
    else if (arg == "--prefill-chunk-size" && i + 1 < argc)
      prefill_chunk_size = std::stoll(argv[++i]);
    else if (arg == "--cuda-graph") use_cuda_graph = true;
    else if (arg == "--cuda-graph-warmup" && i + 1 < argc)
      cuda_graph_warmup_steps = std::stoll(argv[++i]);
  }

  if (checkpoint_path.empty() || config_path.empty() || vocab_path.empty() || merges_path.empty()) {
    std::cerr << "Usage: chat --checkpoint <path> --config <path> "
                 "--vocab-file <path> --merges-file <path>\n"
              << "\nOptions:\n"
              << "  --device <mps|cuda|cpu>     (default: auto)\n"
              << "  --max-tokens <n>            (default: 128)\n"
              << "  --temperature <float>       (default: 0.8)\n"
              << "  --top-k <n>                 (default: 50, 0=disabled)\n"
              << "  --top-p <float>             (default: 0.9, 1.0=disabled)\n"
              << "  --repetition-penalty <float>(default: 1.1, 1.0=disabled)\n"
              << "  --legacy-decode            (for checkpoints trained with old tokenizer)\n"
              << "  --no-kv-cache              (slower, avoids MPS memory issues)\n"
              << "  --no-speculative           (disable MTP speculative decoding)\n"
              << "  --tree-decode              (tree-shaped speculative via forward_tree;\n"
              << "                              requires MTP heads; overrides linear spec)\n"
              << "  --tree-fanout <n>          (children per node in the draft tree; default 2)\n"
              << "  --tree-depth <n>           (tree depth; default = num MTP heads)\n"
              << "  --fused-sampler            (use fused LM-head + Gumbel-max sampler;\n"
              << "                              forces top_k=0, top_p=1.0, rep_penalty=1.0;\n"
              << "                              Philox RNG, not std::mt19937)\n"
              << "  --fused-seed <uint64>      (RNG seed for the fused sampler)\n"
              << "  --custom-lm-head-gemv      (use custom GEMV kernel for the LM head)\n"
              << "  --persistent-decode        (route greedy/temp sampling through the\n"
              << "                              persistent-decode handle; stub for now)\n"
              << "  --draft-checkpoint <path>  (draft model .pt for two-model speculative)\n"
              << "  --draft-config <path>      (draft model JSON config)\n"
              << "  --paged-kv                 (use paged KV cache on non-speculative decode;\n"
              << "                              required precondition for whole-step CUDA graphs)\n"
              << "  --page-size <n>            (paged KV page size in tokens; default 16)\n"
              << "  --paged-max-seq <n>        (paged KV cap on cached seq length; default 2048)\n"
              << "  --prefill-chunk-size <n>   (chunked prefill: split prompt into N-token\n"
              << "                              chunks; 0 = full prefill in one call; only\n"
              << "                              effective with --paged-kv)\n"
              << "  --cuda-graph               (capture decode step into a CUDA graph and\n"
              << "                              replay; requires --paged-kv on a CUDA device)\n"
              << "  --cuda-graph-warmup <n>    (decode steps to run eagerly before capture;\n"
              << "                              default 2 — lets caching allocator settle)\n";
    return 1;
  }

  // -----------------------------------------------------------------
  // Phase 2: pick a device. "auto" prefers GPU on each platform.
  // -----------------------------------------------------------------
  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }

  auto device = select_device(device_pref);

  // Switch the IBackend implementation in olmo_cpp so model code dispatches
  // to fused CUDA kernels (GPU) or vectorized SIMD kernels (CPU).
  if (device.is_cuda()) {
    olmo_cpp::use_cuda_backend();
  } else if (device.is_cpu()) {
    olmo_cpp::use_simd_backend();
  }

  bool is_mps = device.is_mps();

  // MPS + KV cache is unstable — force no-kv-cache on MPS
  if (is_mps && use_kv_cache) {
    std::cout << "Note: KV cache disabled on MPS for stability. Using full-context mode.\n";
    use_kv_cache = false;
  }

  // (fast-inference [6]) Fused sampler only fires in the KV-cache decode
  // path and only supports the unfiltered greedy/temperature regime.
  // Force the conflicting params off (with a notice) so the user gets the
  // speedup they asked for instead of a silent fallback.
  if (use_fused_sampler) {
    if (!use_kv_cache) {
      std::cout << "Note: --fused-sampler requires KV cache; disabling --no-kv-cache override.\n";
      use_kv_cache = true;
    }
    if (top_k != 0 || top_p < 1.0) {
      std::cout << "Note: --fused-sampler does not support top_k/top_p"
                   " — forcing top_k=0, top_p=1.0 (repetition_penalty IS fused in).\n";
      top_k = 0;
      top_p = 1.0;
    }
    if (use_speculative) {
      std::cout << "Note: --fused-sampler is incompatible with speculative decoding; disabling.\n";
      use_speculative = false;
    }
    std::cout << "Using fused LM-head + Gumbel-max sampler (Philox seed=" << fused_seed << ").\n";
  }

  try {
#ifdef HAS_NLOHMANN_JSON
    // -----------------------------------------------------------------
    // Phase 3: build the model from JSON config, load weights, move to
    // the chosen device, and switch into eval mode (disables dropout).
    // -----------------------------------------------------------------
    auto cfg = olmo_cpp::load_config_from_json(config_path);
    cfg.validate();
    align_mtp_config_with_checkpoint(cfg, checkpoint_path);

    olmo_cpp::Transformer model(cfg);
    // Remap serialized tensors to CPU — checkpoints may embed MPS/CUDA
    // device tags from the training host; we move to `device` next.
    // bf16-trained checkpoints store BF16 weights; an fp32 model mismatches
    // storage size on load, so try fp32 then fall back to a BF16 model.
    try {
      torch::load(model, checkpoint_path, torch::kCPU);
    } catch (const c10::Error&) {
      model = olmo_cpp::Transformer(cfg);
      model->to(torch::kBFloat16);
      torch::load(model, checkpoint_path, torch::kCPU);
      if (device.type() != torch::kCUDA) model->to(torch::kFloat32);
    }

    // (fast-inference [17]) Optional draft model for two-model speculative.
    std::unique_ptr<olmo_cpp::Transformer> draft_model;
    if (!draft_checkpoint_path.empty() && !draft_config_path.empty()) {
      auto draft_cfg = olmo_cpp::load_config_from_json(draft_config_path);
      draft_cfg.validate();
      draft_model = std::make_unique<olmo_cpp::Transformer>(draft_cfg);
      torch::load(*draft_model, draft_checkpoint_path, torch::kCPU);
      (*draft_model)->to(device);
      (*draft_model)->eval();
      std::cout << "Loaded draft model from " << draft_checkpoint_path
                << " (two-model speculative enabled)\n";
    } else if (!draft_checkpoint_path.empty() || !draft_config_path.empty()) {
      std::cout << "Note: --draft-checkpoint and --draft-config must both be set"
                   " for two-model speculative; ignoring partial spec.\n";
    }
    model->to(device);
    model->eval();

    // Check if model has MTP heads
    bool has_mtp = model->has_mtp();
    bool do_speculative = use_speculative && has_mtp;

    if (has_mtp) {
      std::cout << "Model has " << model->num_mtp_heads()
                << " MTP heads for speculative decoding"
                << (do_speculative ? " (enabled)" : " (disabled)") << "\n";
    }

    olmo_cpp::BPETokenizer bpe_tokenizer;
    if (!bpe_tokenizer.load(vocab_path, merges_path)) {
      std::cerr << "Failed to load tokenizer\n";
      return 1;
    }
    bpe_tokenizer.set_legacy_decode(legacy_decode);

    // Optional structural tokenizer
    std::unique_ptr<olmo_cpp::StructuralTokenizer> struct_tok;
    if (!structural_config.empty()) {
      struct_tok = std::make_unique<olmo_cpp::StructuralTokenizer>();
      if (!struct_tok->load(structural_config, vocab_path, merges_path)) {
        std::cerr << "Warning: failed to load structural tokenizer, using BPE\n";
        struct_tok.reset();
      } else {
        std::cout << "Using structural tokenizer (vocab_size=" << struct_tok->vocab_size() << ")\n";
      }
    }

    // Tokenizer interface lambdas. encode_text_into appends into a caller-
    // owned buffer; the REPL hoists a single scratch vector out of the loop
    // so we don't reallocate every turn (BPE path skips the allocation
    // entirely; struct_tok path still allocates inside its own encode).
    // (fast-inference [12b])
    auto encode_text_into = [&](const std::string& text, std::vector<uint32_t>& out) {
      if (struct_tok) {
        auto ids = struct_tok->encode(text);
        // Remove trailing EOS
        if (!ids.empty() && ids.back() == struct_tok->eos_id()) ids.pop_back();
        out.insert(out.end(), ids.begin(), ids.end());
      } else {
        bpe_tokenizer.encode_append(text, out);
      }
    };
    auto decode_tokens = [&](const std::vector<uint32_t>& ids) -> std::string {
      if (struct_tok) return struct_tok->decode(ids);
      return bpe_tokenizer.decode(ids);
    };
    auto& tokenizer = bpe_tokenizer;  // for eos_id() access

    // -----------------------------------------------------------------
    // Phase 4: REPL loop — read prompt, generate response, repeat.
    // -----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::cout << "OLMo Chat (type 'quit' to exit, 'reset' to clear context)\n" << std::endl;

    // Persistent conversation tokens — grows each turn with both the
    // user's prompt and the model's response. Each turn's prefill re-
    // processes the whole conversation (no cross-turn KV reuse yet —
    // that needs paged KV, fast-inference [1]). (fast-inference [12a])
    constexpr int64_t kMaxContext = 2048;
    std::vector<int64_t> all_tokens;
    all_tokens.reserve(static_cast<size_t>(kMaxContext));
    // Reused scratch buffer for the per-turn tokenizer output. (fast-inference [12b])
    std::vector<uint32_t> prompt_ids;
    prompt_ids.reserve(256);

    while (true) {
      std::cout << "You: ";
      std::string prompt;
      if (!std::getline(std::cin, prompt)) break;
      if (prompt == "quit" || prompt == "exit" || prompt == "q") break;
      if (prompt == "reset" || prompt == "clear") {
        all_tokens.clear();
        std::cout << "[context cleared]\n" << std::endl;
        continue;
      }
      if (prompt.empty()) continue;

      prompt_ids.clear();
      encode_text_into(prompt, prompt_ids);

      if (prompt_ids.empty()) {
        std::cout << "Model: (empty)\n" << std::endl;
        continue;
      }

      // Append new turn's tokens to running conversation.
      for (auto id : prompt_ids) {
        all_tokens.push_back(static_cast<int64_t>(id));
      }

      // Trim oldest tokens if conversation would overflow context budget.
      // Keep enough headroom for max_tokens of generation.
      int64_t budget = kMaxContext - max_tokens;
      if (budget < 1) budget = 1;
      if (static_cast<int64_t>(all_tokens.size()) > budget) {
        int64_t to_drop = static_cast<int64_t>(all_tokens.size()) - budget;
        all_tokens.erase(all_tokens.begin(), all_tokens.begin() + to_drop);
      }

      int64_t prompt_len = static_cast<int64_t>(all_tokens.size());
      int64_t max_total = prompt_len + max_tokens;
      if (max_total > kMaxContext) max_total = kMaxContext;

      std::cout << "Model: " << std::flush;

      auto gen_start = std::chrono::steady_clock::now();
      int64_t tokens_generated = 0;

      // (fast-inference [17]) When a draft model is loaded, route the
      // speculative path through the two-model implementation instead of
      // the MTP-head version. Both share the chat loop's state.
      if (do_speculative && draft_model) {
        olmo_cpp::DraftModelSpeculativeState dms(
            &model, draft_model.get(),
            model->n_layers(), (*draft_model)->n_layers(), device);

        // Prefill both models so their KV caches are warm.
        {
          auto prefill_input = torch::tensor(
              at::IntArrayRef(all_tokens.data(), all_tokens.size()),
              torch::kInt64).unsqueeze(0).to(device);
          torch::NoGradGuard no_grad;
          model->forward(prefill_input, c10::nullopt, -100, &dms.target_kv);
          (*draft_model)->forward(prefill_input, c10::nullopt, -100, &dms.draft_kv);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
        }

        while (static_cast<int64_t>(all_tokens.size()) < max_total) {
          if (!all_tokens.empty() && all_tokens.back() == static_cast<int64_t>(tokenizer.eos_id()))
            break;
          int64_t accepted = olmo_cpp::draft_model_speculative_step(
              dms, all_tokens, device, temperature, top_k, top_p,
              repetition_penalty, rng, tokenizer);
          tokens_generated += accepted;
          // Stream the accepted tail.
          int64_t start = static_cast<int64_t>(all_tokens.size()) - accepted;
          std::vector<uint32_t> emit;
          for (int64_t i = start; i < static_cast<int64_t>(all_tokens.size()); ++i) {
            emit.push_back(static_cast<uint32_t>(all_tokens[i]));
          }
          std::cout << decode_tokens(emit) << std::flush;
        }

        auto gen_end_dms = std::chrono::steady_clock::now();
        double gen_s = std::chrono::duration<double>(gen_end_dms - gen_start).count();
        double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);
        double ar = dms.total_drafted > 0
                        ? 100.0 * dms.total_accepted / dms.total_drafted : 0.0;
        std::cout << "\n[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s, "
                  << "draft-model-spec, " << std::setprecision(0) << ar << "% accepted]\n"
                  << std::endl;
        continue;
      }

      if (do_speculative) {
        // === MTP Speculative Decoding with KV Cache ===
        // Either linear-chain (speculative_decode_step) or tree-shaped
        // (tree_decode_step). Tree variant gated on --tree-decode.
        const bool tree_mode = use_tree_decode && has_mtp;
        olmo_cpp::KVCache spec_kv(model->n_layers());
        {
          auto prefill_input = torch::tensor(
              at::IntArrayRef(all_tokens.data(), all_tokens.size()),
              torch::kInt64).unsqueeze(0).to(device);
          torch::NoGradGuard no_grad;
          model->forward_backbone(prefill_input, &spec_kv);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
        }

        int64_t total_drafted = 0, total_accepted = 0;
        // Effective tree depth: caller's override or num_mtp_heads.
        const int64_t effective_tree_depth =
            tree_depth > 0 ? tree_depth : model->num_mtp_heads();

        // (fast-inference [10b]) Dynamic draft length: tune k from running
        // acceptance rate. Start at 1 (most conservative — k=1 still gets
        // a 2-token-from-1-step speedup if accepted) and ramp toward
        // num_mtp_heads when the running accept rate is high. Drop back
        // toward 1 when it craters.
        //
        //   accept_rate >= 0.7  → ramp up (k += 1, capped at MTP heads)
        //   accept_rate <= 0.3  → ramp down (k -= 1, floored at 1)
        //   in between          → hold steady
        //
        // Re-evaluated every kAdjustEvery steps so we have a stable
        // window of samples (avoids thrashing on noise).
        const int64_t mtp_max = model->num_mtp_heads();
        int64_t dyn_k = std::min<int64_t>(1, mtp_max);  // start small
        const int64_t kAdjustEvery = 4;
        int64_t steps_since_adjust = 0;
        int64_t prev_drafted = 0, prev_accepted = 0;

        while (static_cast<int64_t>(all_tokens.size()) < max_total) {
          if (!all_tokens.empty() && all_tokens.back() == static_cast<int64_t>(tokenizer.eos_id()))
            break;

          int64_t accepted;
          if (tree_mode) {
            accepted = tree_decode_step(
                model, all_tokens, spec_kv, device, tokenizer,
                tree_fanout, effective_tree_depth,
                total_drafted, total_accepted);
          } else {
            accepted = speculative_decode_step(
                model, all_tokens, spec_kv, device, temperature, top_k, top_p,
                repetition_penalty, rng, tokenizer, total_drafted, total_accepted,
                /*max_drafts=*/dyn_k,
                /*use_custom_lm_head_gemv=*/use_custom_lm_head_gemv);
          }
          tokens_generated += accepted;

          if (++steps_since_adjust >= kAdjustEvery) {
            int64_t window_drafted  = total_drafted  - prev_drafted;
            int64_t window_accepted = total_accepted - prev_accepted;
            if (window_drafted > 0) {
              double rate = static_cast<double>(window_accepted) / window_drafted;
              if (rate >= 0.7 && dyn_k < mtp_max) ++dyn_k;
              else if (rate <= 0.3 && dyn_k > 1) --dyn_k;
            }
            prev_drafted = total_drafted;
            prev_accepted = total_accepted;
            steps_since_adjust = 0;
          }
        }

        auto gen_end_spec = std::chrono::steady_clock::now();
        double gen_s = std::chrono::duration<double>(gen_end_spec - gen_start).count();
        double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);
        double accept_rate = total_drafted > 0 ? 100.0 * total_accepted / total_drafted : 0.0;

        std::cout << "\n[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s, "
                  << (tree_mode ? "tree-spec" : "speculative")
                  << ", " << std::setprecision(0) << accept_rate << "% accepted]\n"
                  << std::endl;
        continue;  // skip the generic stats block below
      } else if (use_kv_cache && use_paged_kv) {
        // fast-inference [1]: paged KV cache. Replaces the concat KVCache
        // on the decode path. Memory layout is stable (per-layer page pools
        // pre-allocated by BlockManager), unblocking whole-step CUDA graph
        // capture (item [2]) and continuous batching (item [3]).
        const int64_t n_kv_heads = cfg.get_n_kv_heads();
        const int64_t head_dim   = cfg.get_head_dim();
        const int64_t max_pages  = (paged_max_seq + paged_page_size - 1) / paged_page_size;
        // dtype: match the model's first parameter, which already reflects
        // any post-load .to(dtype) the user did. Fallback to fp32.
        auto model_dtype = torch::kFloat32;
        if (!model->parameters().empty()) {
          model_dtype = model->parameters()[0].dtype().toScalarType();
        }
        const bool graph_mode = use_cuda_graph && device.is_cuda();
        auto paged = graph_mode
            ? olmo_cpp::make_paged_kv_cache_graph_safe(
                  model->n_layers(), n_kv_heads, head_dim,
                  paged_page_size, max_pages, device, model_dtype)
            : olmo_cpp::make_paged_kv_cache(
                  model->n_layers(), n_kv_heads, head_dim,
                  paged_page_size, max_pages, device, model_dtype);

        // Prefill — optionally chunked. fast-inference [6].
        //
        // Without chunking: one forward_paged([1, prompt_len]) call. With
        // chunking: ceil(prompt_len / chunk_size) calls, each writing its
        // K/V into the page pool and using the previously-written prefix
        // via paged->materialize() inside attention. The chunked-prefill
        // mask in AttentionImpl::forward_paged is what makes this safe.
        //
        // The chunk size dial gates the trade-off between (a) one-shot
        // prefill peak memory / latency and (b) interleaving prefill
        // with concurrent decode steps in a future continuous-batching
        // scheduler (fast-inference [2]). Today it just exercises the
        // path; in a multi-request setting it's the lever that prevents
        // a long prompt from starving running decodes.
        {
          torch::NoGradGuard no_grad;
          const int64_t chunk = (prefill_chunk_size > 0 && prefill_chunk_size < prompt_len)
                                    ? prefill_chunk_size
                                    : prompt_len;
          torch::Tensor logits;
          for (int64_t off = 0; off < prompt_len; off += chunk) {
            const int64_t this_chunk = std::min(chunk, prompt_len - off);
            auto input = torch::from_blob(all_tokens.data() + off, {1, this_chunk},
                                          torch::TensorOptions().dtype(torch::kInt64))
                             .clone()
                             .to(device);
            logits = model->forward_paged(input, paged.get());
          }
          // Sample from the last chunk's last position only.
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
          all_tokens.push_back(next_id);
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
        // Incremental decode: feed one token at a time. In CUDA-graph
        // mode, we run a few eager warmup steps first (so the caching
        // allocator settles), then capture forward_paged into a graph
        // and replay it each subsequent step. The captured graph relies
        // on the dyn paged_attention kernel + dyn K/V write kernel
        // wired earlier (both read their seq_len from the cache's stable
        // device-side n_tokens scalar at launch time), and on the
        // PagedKVCache's external_advance mode — we move the cursor
        // OUTSIDE the captured region between replays.
        torch::NoGradGuard no_grad;
        int64_t step = prompt_len + 1;

        // Eager pre-capture region: run kWarmupSteps (or fewer if EOS /
        // budget) so any first-touch JIT or allocator setup happens
        // before capture.
        const int64_t warmup_budget = graph_mode ? cuda_graph_warmup_steps : 0;
        int64_t warm_done = 0;
        while (step < max_total && warm_done < warmup_budget) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto input = decode_input_buf(last_token, device);  // H2: reused buffer, no per-token H->D
          auto logits = model->forward_paged(input, paged.get());
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
          step++;
          warm_done++;
        }

#if defined(OLMO_HAS_CUDA_KERNELS) || defined(USE_CUDA)
        // Capture path. Only entered when on a CUDA device and the user
        // asked for --cuda-graph. The captured graph is the entire decode
        // forward (every layer's projections, QK-norm, RoPE, paged write,
        // paged attention, FFN, LM head) replayed as one launch.
        if (graph_mode && step < max_total &&
            all_tokens.back() != static_cast<int64_t>(tokenizer.eos_id())) {
          // Stable input buffer the captured launch reads from. We
          // .fill_() this between replays — same storage address, new
          // value, kernel sees the update.
          auto static_input = torch::empty(
              {1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
          static_input.fill_(all_tokens.back());

          // Switch the cache to external-advance mode. From here on,
          // paged->append no longer bumps logical_len_; we do it.
          paged->set_external_advance(true);

          // Advance once for the to-be-captured step. The captured K/V
          // write kernel will read this n_tokens and put its data at slot
          // (n_tokens - 1).
          paged->advance_cursor(1);

          // Capture on a side stream so we don't trample the default
          // stream's allocator state.
          auto cap_stream = c10::cuda::getStreamFromPool();
          c10::cuda::CUDAStreamGuard guard(cap_stream);

          at::cuda::CUDAGraph graph;
          torch::Tensor captured_logits;
          graph.capture_begin();
          captured_logits = model->forward_paged(static_input, paged.get());
          graph.capture_end();

          // The capture run IS the first real decode step under graph
          // mode: it just wrote one token's K/V and produced its logits.
          {
            auto next_logits = captured_logits.select(1, 0).squeeze(0);
            int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                            all_tokens, repetition_penalty, rng);
            all_tokens.push_back(next_id);
            if (next_id != static_cast<int64_t>(tokenizer.eos_id())) {
              std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
              std::cout << decode_tokens(tok_to_decode) << std::flush;
              tokens_generated++;
              step++;
            } else {
              step = max_total;  // exit
            }
          }

          // Replay loop. Each iteration: bump cursor, write new token
          // into the static input, replay graph, sample.
          while (step < max_total) {
            int64_t last_token = all_tokens.back();
            if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
            static_input.fill_(last_token);
            paged->advance_cursor(1);
            graph.replay();
            auto next_logits = captured_logits.select(1, 0).squeeze(0);
            int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                            all_tokens, repetition_penalty, rng);
            all_tokens.push_back(next_id);
            if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
            std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
            std::cout << decode_tokens(tok_to_decode) << std::flush;
            tokens_generated++;
            step++;
          }
        } else
#endif
        // Eager fallback: same behavior as before, no graph capture.
        for (; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto input = decode_input_buf(last_token, device);  // H2: reused buffer, no per-token H->D
          auto logits = model->forward_paged(input, paged.get());
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      } else if (use_kv_cache) {
        // KV cache path: prefill + incremental decode
        // CUDA graphs capture the decode step (always [1,1] input) for zero launch overhead
        // TODO(fast-inference [1]+[2]): this loop is THE hot path.
        //   Step 1: replace concat KV cache with paged KV ([1]) — current
        //   reshape every step blocks everything below.
        //   Step 2: cudaGraphCapture this loop body once, cudaGraphLaunch
        //   every iteration ([2]). Combined with fused sampler ([6]) and
        //   custom decode attention ([4]), this loop becomes:
        //     1 graph replay + 8B D->H per token. ~3-5x faster than today.
        //   Eventually replace the whole loop with a persistent kernel ([11]).
        olmo_cpp::KVCache kv_cache(model->n_layers());
        // Position counter for the fused sampler's Philox sequence.
        // Each generated token gets a unique (seed, position) pair; resetting
        // here so a fresh turn starts at 0.
        uint32_t fused_pos = 0;
        // (fast-inference [13]) Optionally route greedy/temp sampling
        // through the persistent-decode handle. Currently a synchronous
        // stub; future real implementation runs a long-lived CUDA kernel.
        std::unique_ptr<olmo_cpp::PersistentDecode> pd;
        if (use_persistent_decode && use_fused_sampler) {
          pd = std::make_unique<olmo_cpp::PersistentDecode>(
              cfg.vocab_size, cfg.d_model, device, fused_seed);
        }
        auto sample_fused = [&](torch::Tensor h) -> int64_t {
          float t = static_cast<float>(temperature > 0 ? temperature : 1.0);
          if (pd) {
            int slot = pd->enqueue(h, model->lm_head_weight(), t, fused_pos++);
            return pd->poll(slot);
          }
          // Repetition penalty is fused into the kernel (applied to each row's
          // logit before temperature + Gumbel) — still zero [V] materialization.
          return olmo_cpp::fused_lm_head_sample(
              h, model->lm_head_weight(), t, fused_seed, fused_pos++,
              all_tokens, repetition_penalty);
        };
        {
          auto input = torch::from_blob(all_tokens.data(), {1, prompt_len},
                                        torch::TensorOptions().dtype(torch::kInt64))
                           .clone()
                           .to(device);
          torch::NoGradGuard no_grad;
          int64_t next_id;
          if (use_fused_sampler) {
            auto hidden = model->forward_backbone(input, &kv_cache);
            auto last_hidden = hidden.select(1, hidden.size(1) - 1).squeeze(0);
            next_id = sample_fused(last_hidden);  // [6] / [13] routing
          } else {
            auto logits = model->forward(input, c10::nullopt, -100, &kv_cache);
            auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
            next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                    all_tokens, repetition_penalty, rng);
          }
          all_tokens.push_back(next_id);
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
        // Note: CUDA graph capture for KV-cache decode would require static
        // KV cache tensors (pre-allocated max-length buffers). The current
        // concat-based KV cache changes shape each step, so we use direct
        // execution. For full CUDA graph support, see the paged attention
        // implementation (future work).
        for (int64_t step = prompt_len + 1; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto input = decode_input_buf(last_token, device);  // H2: reused buffer, no per-token H->D
          torch::NoGradGuard no_grad;
          int64_t next_id;
          if (use_fused_sampler) {
            auto hidden = model->forward_backbone(input, &kv_cache);
            auto last_hidden = hidden.select(1, hidden.size(1) - 1).squeeze(0);
            next_id = sample_fused(last_hidden);  // [6] / [13] routing
          } else {
            auto logits = model->forward(input, c10::nullopt, -100, &kv_cache);
            auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
            next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                    all_tokens, repetition_penalty, rng);
          }
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      } else {
        // No KV cache: feed full sequence each step
        // TODO(perf): this is the slow fallback path — O(L^2) attention recompute
        // per step. Don't optimize this loop directly; the fix is just "use the
        // KV cache path above." Keep as a correctness reference / debug mode.
        for (int64_t step = prompt_len; step < max_total; ++step) {
          int64_t last_token = all_tokens.back();
          if (last_token == static_cast<int64_t>(tokenizer.eos_id())) break;
          auto cur_seq_len = static_cast<int64_t>(all_tokens.size());
          // Clone token data to a fresh tensor (avoid dangling pointer after push_back)
          auto input_cpu = torch::tensor(
              at::IntArrayRef(all_tokens.data(), static_cast<size_t>(cur_seq_len)),
              torch::kInt64).unsqueeze(0);
          auto input = input_cpu.to(device);
          torch::NoGradGuard no_grad;
          auto logits = model->forward(input, c10::nullopt, -100, nullptr);
#ifdef __APPLE__
          if (is_mps) torch::mps::synchronize();
#endif
          // Sample on-device (GPU-resident sampler) — only the token id (8 B)
          // returns, not the [V] logits. (fast-inference [12d])
          auto next_logits = logits.select(1, logits.size(1) - 1).squeeze(0);
          int64_t next_id = sample_logits(next_logits, temperature, top_k, top_p,
                                          all_tokens, repetition_penalty, rng);
          // Release GPU tensors after sampling (next_logits views into logits).
          logits.reset();
          input.reset();
          all_tokens.push_back(next_id);
          if (next_id == static_cast<int64_t>(tokenizer.eos_id())) break;
          std::vector<uint32_t> tok_to_decode = {static_cast<uint32_t>(next_id)};
          std::cout << decode_tokens(tok_to_decode) << std::flush;
          tokens_generated++;
        }
      }

      auto gen_end = std::chrono::steady_clock::now();
      double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
      double tok_per_s = tokens_generated / (gen_s > 0 ? gen_s : 1);

      std::cout << "\n";
      if (do_speculative) {
        std::cout << "[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s, speculative]\n";
      } else {
        std::cout << "[" << tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tok_per_s << " tok/s]\n";
      }
      std::cout << std::endl;
    }
#else
    std::cerr << "Chat requires nlohmann/json\n";
    return 1;
#endif
    return 0;
  } catch (const std::exception& e) {
    // Phase 5: surface any LibTorch / IO / config error with a clear
    // "Error:" prefix instead of letting the binary terminate via a
    // raw uncaught exception.
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
