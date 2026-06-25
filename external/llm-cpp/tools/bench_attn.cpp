/**
 * tools/bench_attn.cpp
 *
 * Attention-kernel-isolated micro-benchmark for the SubQ head-to-head.
 *
 * Why this exists separately from bench_chat:
 *   bench_chat times a full transformer forward (embeddings, every layer,
 *   LM head, sampler). SubQ's headline "52x over FlashAttention-2 at 1M
 *   tokens" is an attention-only number. To compare apples-to-apples we
 *   need a tool that allocates Q/K/V, runs only the attention op, and
 *   reports the kernel time. No model load, no embeddings, no LM head.
 *
 * Modes:
 *   dense   — torch::scaled_dot_product_attention. On CUDA + recent torch
 *             this dispatches to FlashAttention-2 internally; on CPU it
 *             runs a math fallback. This is the SubQ baseline.
 *   fa2     — explicit FA-2 path. Prefill currently falls through to dense
 *             (we don't have a standalone FA-2 prefill kernel in this repo
 *             yet). Decode uses olmo_cpp::flash_decode.
 *   fa3     — stub. On Hopper we'll wire FA-3 via libtorch SDPA backend
 *             selection or a vendored kernel; until then we mark the row
 *             "fa3_stub" and run dense.
 *   sparse  — Phase 1: NSA-style content-selected attention. Not
 *             implemented yet; emits an error if requested.
 *
 * Stages:
 *   prefill — one forward over Q,K,V of shape [B,H,T,D].
 *   decode  — loop of N single-query forwards. Each step reads a growing
 *             KV cache. Mimics the real decode pattern.
 *
 * Output: one CSV row per (mode, stage, seq_len) tuple.
 *   columns: mode,stage,B,H,Hkv,T,D,dtype,iters,fwd_ms_mean,fwd_ms_p50,
 *            fwd_ms_p99,tok_per_s,hbm_GB,device,note
 *
 * Local macOS use: built without CUDA, runs at small T on CPU. Real
 * numbers come from H100 — see scripts/bench_subq.sh.
 *
 * Usage:
 *   ./build/bench_attn \
 *       --mode dense --stage prefill \
 *       --seq-lens 1024,4096,16384 \
 *       --batch 1 --n-heads 32 --n-kv-heads 8 --head-dim 128 \
 *       --device auto --dtype bf16 \
 *       --warmup 3 --iters 5 \
 *       --output results/bench_attn.csv
 */

#include "olmo_cpp/backend/flash_decode.hpp"
#include "olmo_cpp/backend/sparse_attn.hpp"

#include <ATen/Context.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <ATen/mps/MPSStream.h>
#endif

namespace {

torch::Device select_device(const std::string& pref) {
  if (pref == "cuda" && torch::cuda::is_available()) return torch::Device(torch::kCUDA);
#ifdef __APPLE__
  if ((pref == "mps" || pref == "metal") && torch::mps::is_available())
    return torch::Device(torch::kMPS);
#endif
  return torch::Device(torch::kCPU);
}

torch::Dtype select_dtype(const std::string& s, torch::Device dev) {
  if (s == "bf16") return torch::kBFloat16;
  if (s == "fp16") return torch::kFloat16;
  if (s == "fp32") return torch::kFloat32;
  // auto: bf16 on cuda, fp32 elsewhere (CPU bf16 SDPA is shaky in older torch).
  return dev.is_cuda() ? torch::kBFloat16 : torch::kFloat32;
}

void device_sync(torch::Device dev) {
  if (dev.is_cuda()) torch::cuda::synchronize();
#ifdef __APPLE__
  if (dev.is_mps()) torch::mps::synchronize();
#endif
}

double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  size_t idx = static_cast<size_t>(p * (xs.size() - 1));
  return xs[idx];
}

double mean(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  return std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
}

std::vector<int64_t> parse_csv_int64(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoll(item));
  }
  return out;
}

// Element count -> bytes for a dtype. Used for crude HBM accounting.
size_t dtype_bytes(torch::Dtype d) {
  switch (d) {
    case torch::kBFloat16:
    case torch::kFloat16: return 2;
    case torch::kFloat32: return 4;
    case torch::kFloat64: return 8;
    default: return 4;
  }
}

struct BenchRow {
  std::string mode, stage, dtype, device, note;
  int64_t B, H, Hkv, T, D, iters;
  double fwd_ms_mean, fwd_ms_p50, fwd_ms_p99, tok_per_s, hbm_gb;
};

void write_csv_header(std::ostream& os) {
  os << "mode,stage,B,H,Hkv,T,D,dtype,iters,"
        "fwd_ms_mean,fwd_ms_p50,fwd_ms_p99,tok_per_s,hbm_GB,device,note\n";
}

void write_csv_row(std::ostream& os, const BenchRow& r) {
  os << r.mode << "," << r.stage << "," << r.B << "," << r.H << "," << r.Hkv
     << "," << r.T << "," << r.D << "," << r.dtype << "," << r.iters << ","
     << r.fwd_ms_mean << "," << r.fwd_ms_p50 << "," << r.fwd_ms_p99 << ","
     << r.tok_per_s << "," << r.hbm_gb << "," << r.device << "," << r.note
     << "\n";
}

// ---------------------------------------------------------------------------
// SDPA backend selector — RAII guard around at::globalContext()'s per-backend
// toggles. Lets us force libtorch's SDPA dispatcher to a specific backend so
// the fa2 / fa3 rows in our CSV are real, not implicit.
//
// fa2  -> FLASH only      (libtorch's bundled FlashAttention is v2)
// fa3  -> CUDNN only      (cuDNN's SDPA path is FA-3 on Hopper for bf16/fp16)
// dense-> all enabled     (default; libtorch picks)
//
// On CPU the toggles still apply but every backend except MATH no-ops, so
// fa2/fa3 rows fall through to MATH locally and we'll only see the
// distinction on H100. That's fine — bench_subq.sh knows.
// ---------------------------------------------------------------------------
struct SDPABackendGuard {
  bool flash, mem_eff, math, cudnn;
  SDPABackendGuard() {
    auto& ctx = at::globalContext();
    flash   = ctx.userEnabledFlashSDP();
    mem_eff = ctx.userEnabledMemEfficientSDP();
    math    = ctx.userEnabledMathSDP();
    cudnn   = ctx.userEnabledCuDNNSDP();
  }
  ~SDPABackendGuard() {
    auto& ctx = at::globalContext();
    ctx.setSDPUseFlash(flash);
    ctx.setSDPUseMemEfficient(mem_eff);
    ctx.setSDPUseMath(math);
    ctx.setSDPUseCuDNN(cudnn);
  }
  static void apply(bool flash, bool mem_eff, bool math, bool cudnn) {
    auto& ctx = at::globalContext();
    ctx.setSDPUseFlash(flash);
    ctx.setSDPUseMemEfficient(mem_eff);
    ctx.setSDPUseMath(math);
    ctx.setSDPUseCuDNN(cudnn);
  }
  // Force flash-only (FA-2 in libtorch). Math kept as a fallback so CPU
  // doesn't crash — on CUDA, flash is preferred when eligible.
  static void force_flash() { apply(true,  false, true,  false); }
  // Force cuDNN-only (FA-3 on Hopper bf16). Math fallback for CPU.
  static void force_cudnn() { apply(false, false, true,  true ); }
  // Default: everything on, libtorch picks.
  static void all_on()     { apply(true,  true,  true,  true ); }
};

// ---------------------------------------------------------------------------
// Mode implementations. Each takes pre-allocated tensors and returns the
// forward time in milliseconds for one call (or one full decode loop).
// ---------------------------------------------------------------------------

// torch SDPA — picks whatever backend libtorch has compiled (FlashAttention
// on H100 + recent torch, math fallback on CPU). Backend selection is
// controlled by the surrounding SDPABackendGuard configuration.
torch::Tensor sdpa_forward(const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& v, bool is_causal) {
  // q,k,v: [B,H,T,D]. ATen handles GQA via head broadcast when Hk < Hq if we
  // call F::scaled_dot_product_attention; the C++ surface here doesn't, so
  // expand kv heads to match q heads if needed.
  auto q_h = q.size(1);
  auto k_h = k.size(1);
  torch::Tensor k_use = k, v_use = v;
  if (k_h != q_h) {
    int64_t group = q_h / k_h;
    k_use = k.repeat_interleave(group, /*dim=*/1);
    v_use = v.repeat_interleave(group, /*dim=*/1);
  }
  return at::scaled_dot_product_attention(q_h ? q : q, k_use, v_use,
                                          /*attn_mask=*/{},
                                          /*dropout_p=*/0.0,
                                          is_causal);
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "dense";
  std::string stage = "prefill";
  std::string seq_lens_csv = "1024,4096,16384";
  int64_t batch = 1;
  int64_t n_heads = 32;
  int64_t n_kv_heads = 8;
  int64_t head_dim = 128;
  int64_t decode_steps = 64;
  std::string device_pref = "auto";
  std::string dtype_pref = "auto";
  int64_t warmup = 3;
  int64_t iters = 5;
  uint64_t seed = 42;
  // Sparse-mode knobs: NSA-style block-mean scoring, top-k blocks per
  // (B, kv_head). Selection is shared across the GQA group.
  int64_t block_size = 64;
  int64_t top_k = 64;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if      (a == "--mode"        && i + 1 < argc) mode = next();
    else if (a == "--stage"       && i + 1 < argc) stage = next();
    else if (a == "--seq-lens"    && i + 1 < argc) seq_lens_csv = next();
    else if (a == "--batch"       && i + 1 < argc) batch = std::stoll(next());
    else if (a == "--n-heads"     && i + 1 < argc) n_heads = std::stoll(next());
    else if (a == "--n-kv-heads"  && i + 1 < argc) n_kv_heads = std::stoll(next());
    else if (a == "--head-dim"    && i + 1 < argc) head_dim = std::stoll(next());
    else if (a == "--decode-steps"&& i + 1 < argc) decode_steps = std::stoll(next());
    else if (a == "--device"      && i + 1 < argc) device_pref = next();
    else if (a == "--dtype"       && i + 1 < argc) dtype_pref = next();
    else if (a == "--warmup"      && i + 1 < argc) warmup = std::stoll(next());
    else if (a == "--iters"       && i + 1 < argc) iters = std::stoll(next());
    else if (a == "--seed"        && i + 1 < argc) seed = std::stoull(next());
    else if (a == "--block-size"  && i + 1 < argc) block_size = std::stoll(next());
    else if (a == "--top-k"       && i + 1 < argc) top_k = std::stoll(next());
    else if (a == "--output"      && i + 1 < argc) output_path = next();
    else if (a == "--help" || a == "-h") {
      std::cerr <<
        "Usage: bench_attn --mode {dense|fa2|fa3|sparse} "
        "--stage {prefill|decode}\n"
        "                  --seq-lens 1K,4K,16K --batch B --n-heads H "
        "--n-kv-heads Hk\n"
        "                  --head-dim D --decode-steps N "
        "[--device auto|cuda|mps|cpu] [--dtype auto|bf16|fp16|fp32]\n"
        "                  [--warmup W] [--iters I] [--output csv]\n";
      return 0;
    }
  }

  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }
  auto device = select_device(device_pref);
  auto dtype = select_dtype(dtype_pref, device);

  if (n_kv_heads <= 0) n_kv_heads = n_heads;
  if (n_heads % n_kv_heads != 0) {
    std::cerr << "n_heads (" << n_heads << ") must be divisible by n_kv_heads ("
              << n_kv_heads << ")\n";
    return 1;
  }

  std::vector<int64_t> seq_lens = parse_csv_int64(seq_lens_csv);
  if (seq_lens.empty()) { std::cerr << "no seq lens\n"; return 1; }

  if (mode != "dense" && mode != "fa2" && mode != "fa3" && mode != "sparse") {
    std::cerr << "unknown mode: " << mode << "\n"; return 1;
  }
  if (stage != "prefill" && stage != "decode") {
    std::cerr << "unknown stage: " << stage << "\n"; return 1;
  }
  // sparse prefill is supported via olmo_cpp::sparse_attn_prefill.

  torch::manual_seed(seed);

  // Apply SDPA backend selection for the duration of the run. The guard
  // restores the previous toggles on exit so the bench process leaves the
  // global context unchanged.
  SDPABackendGuard sdp_guard;
  std::string mode_note;
  if      (mode == "fa2")    { SDPABackendGuard::force_flash(); mode_note = "sdpa_flash_only"; }
  else if (mode == "fa3")    { SDPABackendGuard::force_cudnn(); mode_note = "sdpa_cudnn_only"; }
  else if (mode == "dense")  { SDPABackendGuard::all_on();      mode_note = "sdpa_default"; }
  else /* sparse */          { /* no SDPA toggle — sparse path is hand-rolled */ ;
                               mode_note = "nsa_block_top_k"; }

  std::ofstream csv;
  if (!output_path.empty()) {
    csv.open(output_path);
    write_csv_header(csv);
  }
  write_csv_header(std::cerr);

  for (int64_t T : seq_lens) {
    auto opts = torch::TensorOptions().dtype(dtype).device(device);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    BenchRow row{};
    row.mode = mode; row.stage = stage; row.dtype = dtype_pref == "auto"
        ? (dtype == torch::kBFloat16 ? "bf16"
           : dtype == torch::kFloat16 ? "fp16" : "fp32")
        : dtype_pref;
    {
      std::stringstream ss; ss << device;
      row.device = ss.str();
    }
    row.note = mode_note;
    row.B = batch; row.H = n_heads; row.Hkv = n_kv_heads;
    row.T = T; row.D = head_dim; row.iters = iters;

    // HBM accounting: Q [B,H,T,D] + K,V [B,Hkv,T,D] + O [B,H,T,D].
    size_t bytes = dtype_bytes(dtype);
    double hbm_gb = static_cast<double>(
        (size_t)batch * n_heads * T * head_dim * bytes
      + 2 * (size_t)batch * n_kv_heads * T * head_dim * bytes
      + (size_t)batch * n_heads * T * head_dim * bytes) / (1024.0 * 1024.0 * 1024.0);
    row.hbm_gb = hbm_gb;

    std::vector<double> step_ms;
    step_ms.reserve(static_cast<size_t>(iters));

    if (stage == "prefill") {
      auto q = torch::randn({batch, n_heads,    T, head_dim}, opts);
      auto k = torch::randn({batch, n_kv_heads, T, head_dim}, opts);
      auto v = torch::randn({batch, n_kv_heads, T, head_dim}, opts);

      auto run_once = [&]() -> torch::Tensor {
        // dense / fa2 / fa3 all dispatch through SDPA — the difference is
        // which backends the SDPABackendGuard left enabled. The libtorch
        // dispatcher then picks among the eligible set. On H100 + bf16:
        //   dense  -> Flash (libtorch's preferred)
        //   fa2    -> Flash (forced)
        //   fa3    -> cuDNN (forced; FA-3 path)
        // On CPU all three fall to MATH.
        if (mode == "sparse") {
          return olmo_cpp::sparse_attn_prefill(
              q, k, v, sm_scale, block_size, top_k);
        }
        return sdpa_forward(q, k, v, /*is_causal=*/true);
      };

      for (int64_t w = 0; w < warmup; ++w) {
        auto out = run_once();
        device_sync(device);
        (void)out;
      }
      torch::NoGradGuard ng;
      for (int64_t i = 0; i < iters; ++i) {
        device_sync(device);
        auto t0 = std::chrono::steady_clock::now();
        auto out = run_once();
        device_sync(device);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        step_ms.push_back(ms);
        (void)out;
      }
      row.fwd_ms_mean = mean(step_ms);
      row.fwd_ms_p50  = percentile(step_ms, 0.50);
      row.fwd_ms_p99  = percentile(step_ms, 0.99);
      // tokens-per-second for prefill = batch*T tokens consumed per call.
      row.tok_per_s = (row.fwd_ms_mean > 0)
          ? 1000.0 * static_cast<double>(batch) * T / row.fwd_ms_mean
          : 0.0;
    } else {
      // decode: T = KV cache length; we run N single-query steps that read
      // the full prefix. We do NOT grow the cache here — every step reads
      // exactly T positions, which is the worst-case decode cost. Real
      // decode would grow T by 1 per step; the cost difference is ~0.5x
      // averaged over the run and not worth the bookkeeping for this bench.
      // Allocate kv at full T; q at len 1.
      auto q = torch::randn({batch, n_heads,    1, head_dim}, opts);
      auto k = torch::randn({batch, n_kv_heads, T, head_dim}, opts);
      auto v = torch::randn({batch, n_kv_heads, T, head_dim}, opts);

      auto run_step = [&]() -> torch::Tensor {
        if (mode == "dense" || mode == "fa3") {
          // SDPA with the backend guard's toggles applied.
          return sdpa_forward(q, k, v, /*is_causal=*/false);
        } else if (mode == "sparse") {
          // NSA-style content-selected attention — block-mean scoring,
          // top-k blocks per (B, kv_head), shared across the GQA group.
          return olmo_cpp::sparse_attn_decode(
              q, k, v, sm_scale, block_size, top_k);
        } else {  // fa2: explicit decode kernel (B=1 only for now)
          if (batch != 1) {
            // flash_decode's current entry-point is single-batch. Until the
            // batched kernel lands, fall through to dense for B>1 so the
            // sweep doesn't fail.
            return sdpa_forward(q, k, v, /*is_causal=*/false);
          }
          // Reshape to flash_decode's expected layout:
          //   q: [Hq, D]            (single query, single batch)
          //   k: [T, Hkv, D], v: [T, Hkv, D]
          auto qr = q.squeeze(0).squeeze(1);  // [H, D]
          auto kr = k.squeeze(0).transpose(0, 1);  // [T, Hkv, D]
          auto vr = v.squeeze(0).transpose(0, 1);
          auto out = olmo_cpp::flash_decode(qr, kr, vr, sm_scale);  // [H, D]
          return out.unsqueeze(0).unsqueeze(2);  // [1, H, 1, D]
        }
      };

      for (int64_t w = 0; w < warmup; ++w) {
        auto out = run_step();
        device_sync(device);
        (void)out;
      }
      torch::NoGradGuard ng;
      // Time `iters` outer reps, each rep = decode_steps single-token forwards.
      for (int64_t i = 0; i < iters; ++i) {
        device_sync(device);
        auto t0 = std::chrono::steady_clock::now();
        for (int64_t s = 0; s < decode_steps; ++s) {
          auto out = run_step();
          (void)out;
        }
        device_sync(device);
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Per-step time: total / decode_steps.
        step_ms.push_back(total_ms / static_cast<double>(decode_steps));
      }
      row.fwd_ms_mean = mean(step_ms);
      row.fwd_ms_p50  = percentile(step_ms, 0.50);
      row.fwd_ms_p99  = percentile(step_ms, 0.99);
      // tokens-per-second for decode = batch tokens per step.
      row.tok_per_s = (row.fwd_ms_mean > 0)
          ? 1000.0 * static_cast<double>(batch) / row.fwd_ms_mean
          : 0.0;
    }

    write_csv_row(std::cerr, row);
    if (csv.is_open()) write_csv_row(csv, row);
  }

  if (csv.is_open()) std::cerr << "Wrote " << output_path << "\n";
  return 0;
}
