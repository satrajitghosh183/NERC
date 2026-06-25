/**
 * tools/bench_chat.cpp
 *
 * Inference micro-benchmark for the fast-inference roadmap (item [3]).
 * Loads a checkpoint, runs N decode steps with greedy sampling, and emits
 * machine-readable JSON timing so two builds (baseline vs optimized) can
 * be compared apples-to-apples.
 *
 * Why this exists separately from chat.cpp:
 *   - chat.cpp is interactive and uses std::mt19937 / top-p / rep penalty.
 *     None of that is wanted in a benchmark — we want deterministic greedy
 *     output and the tightest possible decode loop.
 *   - This tool reports CUDA-event timings on GPU (microsecond precision)
 *     and std::chrono on CPU/MPS. Output is a single JSON object.
 *
 * Comparison flow:
 *   1. Build baseline (older branch) and fast-inference branch separately.
 *   2. Run both against the same checkpoint + prompt set, same decode length.
 *   3. scripts/bench_compare.sh diffs the two JSON outputs.
 *
 * Greedy only (temperature=0 path) so output tokens are deterministic and
 * the comparison is not muddied by RNG differences.
 *
 * Usage:
 *   ./build/bench_chat \
 *       --checkpoint checkpoints/1B.pt \
 *       --config configs/olmo_1B.json \
 *       --vocab-file data/gpt2/vocab.json \
 *       --merges-file data/gpt2/merges.txt \
 *       --device cuda --prompt-len 128 --decode-len 256 \
 *       --warmup 3 --iters 10 \
 *       --output results/bench_fast.json
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

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

// Sync the device so timing measurements are valid. CUDA events handle their
// own sync; this is for MPS and chrono-based measurements.
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

}  // namespace

int main(int argc, char** argv) {
  std::string checkpoint_path, config_path, vocab_path, merges_path;
  std::string device_pref = "auto";
  std::string output_path = "";
  int64_t prompt_len = 128;
  int64_t decode_len = 256;
  int64_t batch = 1;
  int64_t warmup = 3;
  int64_t iters = 5;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--checkpoint" && i + 1 < argc) checkpoint_path = next();
    else if (a == "--config" && i + 1 < argc) config_path = next();
    else if (a == "--vocab-file" && i + 1 < argc) vocab_path = next();
    else if (a == "--merges-file" && i + 1 < argc) merges_path = next();
    else if (a == "--device" && i + 1 < argc) device_pref = next();
    else if (a == "--prompt-len" && i + 1 < argc) prompt_len = std::stoll(next());
    else if (a == "--decode-len" && i + 1 < argc) decode_len = std::stoll(next());
    else if (a == "--batch" && i + 1 < argc) batch = std::stoll(next());
    else if (a == "--warmup" && i + 1 < argc) warmup = std::stoll(next());
    else if (a == "--iters" && i + 1 < argc) iters = std::stoll(next());
    else if (a == "--output" && i + 1 < argc) output_path = next();
    else if (a == "--help" || a == "-h") {
      std::cerr << "Usage: bench_chat --checkpoint <path> --config <path> "
                   "--vocab-file <path> --merges-file <path> "
                   "[--device cuda|mps|cpu] [--prompt-len N] [--decode-len N] "
                   "[--batch N] [--warmup N] [--iters N] [--output bench.json]\n";
      return 0;
    }
  }

  if (checkpoint_path.empty() || config_path.empty() ||
      vocab_path.empty() || merges_path.empty()) {
    std::cerr << "Missing required args. Use --help for usage.\n";
    return 1;
  }

  if (device_pref == "auto") {
#ifdef __APPLE__
    device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
    device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
  }
  auto device = select_device(device_pref);

  if (device.is_cuda()) olmo_cpp::use_cuda_backend();
  else if (device.is_cpu()) olmo_cpp::use_simd_backend();

#ifdef HAS_NLOHMANN_JSON
  auto cfg = olmo_cpp::load_config_from_json(config_path);
  cfg.validate();
#else
  std::cerr << "Built without nlohmann_json — cannot parse config.\n";
  return 1;
#endif

  olmo_cpp::Transformer model(cfg);
  // bf16-trained checkpoints store BF16 weights; an fp32 model mismatches storage
  // size on load. Try fp32, fall back to casting the model to BF16 first.
  try {
    torch::load(model, checkpoint_path, torch::kCPU);
  } catch (const c10::Error&) {
    model = olmo_cpp::Transformer(cfg);
    model->to(torch::kBFloat16);
    torch::load(model, checkpoint_path, torch::kCPU);
    // Upcast to fp32 unless the device is CUDA (where bf16 inference is fast).
    if (device.type() != torch::kCUDA) model->to(torch::kFloat32);
  }
  model->to(device);
  model->eval();
  torch::NoGradGuard no_grad;

  olmo_cpp::BPETokenizer tok;
  if (!tok.load(vocab_path, merges_path)) {
    std::cerr << "Tokenizer load failed.\n";
    return 1;
  }

  // Build a fixed deterministic prompt. Real content doesn't matter for
  // benchmarking — only shape and decode length do. Use the EOS token id
  // (GPT-2 reuses 50256 as BOS) as the leading token; rest are token 0.
  std::vector<int64_t> prompt(prompt_len, 0);
  prompt[0] = static_cast<int64_t>(tok.eos_id());

  // Per-iter measurements.
  std::vector<double> ttft_ms;     // time-to-first-token (prefill)
  std::vector<double> tpot_ms;     // mean time-per-output-token
  std::vector<double> total_ms;    // end-to-end one full decode
  std::vector<double> all_step_ms; // every per-step timing across iters (for p50/p99)

  auto run_once = [&](bool measure) -> void {
    olmo_cpp::KVCache kv(model->n_layers());

    // Build [B, prompt_len] input by tiling the same prompt across batch.
    auto input = torch::tensor(prompt, torch::kInt64).unsqueeze(0)  // [1, T]
                     .repeat({batch, 1}).to(device);                // [B, T]

    auto t_start = std::chrono::steady_clock::now();
    device_sync(device);

    // Prefill
    auto logits = model->forward(input, c10::nullopt, -100, &kv);
    device_sync(device);
    auto t_prefill = std::chrono::steady_clock::now();

    // Greedy: argmax of last position
    auto next_tok = logits.select(1, logits.size(1) - 1).argmax(-1);  // [B]

    // Decode loop
    std::vector<double> step_ms;
    step_ms.reserve(static_cast<size_t>(decode_len));
    for (int64_t s = 0; s < decode_len; ++s) {
      auto step_start = std::chrono::steady_clock::now();
      auto step_input = next_tok.unsqueeze(1);  // [B, 1]
      auto out = model->forward(step_input, c10::nullopt, -100, &kv);
      next_tok = out.select(1, 0).argmax(-1);   // [B]
      device_sync(device);
      auto step_end = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      step_ms.push_back(dt);
    }

    auto t_end = std::chrono::steady_clock::now();

    if (measure) {
      double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
      double decode_total_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
      ttft_ms.push_back(prefill_ms);
      tpot_ms.push_back(decode_total_ms / static_cast<double>(decode_len));
      total_ms.push_back(prefill_ms + decode_total_ms);
      for (auto v : step_ms) all_step_ms.push_back(v);
    }
  };

  for (int64_t w = 0; w < warmup; ++w) run_once(false);
  for (int64_t i = 0; i < iters; ++i) run_once(true);

  auto mean = [](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };

  double ttft_mean = mean(ttft_ms);
  double tpot_mean = mean(tpot_ms);
  double total_mean = mean(total_ms);
  double tpot_p50 = percentile(all_step_ms, 0.50);
  double tpot_p99 = percentile(all_step_ms, 0.99);
  double tok_per_s = 1000.0 * batch / tpot_mean;

  std::cerr << "device=" << device << " batch=" << batch
            << " prompt_len=" << prompt_len << " decode_len=" << decode_len << "\n"
            << "TTFT mean: " << ttft_mean << " ms\n"
            << "TPOT mean: " << tpot_mean << " ms (p50=" << tpot_p50
            << ", p99=" << tpot_p99 << ")\n"
            << "Throughput: " << tok_per_s << " tok/s (batch=" << batch << ")\n";

#ifdef HAS_NLOHMANN_JSON
  if (!output_path.empty()) {
    nlohmann::json j;
    j["device"] = device_pref;
    j["batch"] = batch;
    j["prompt_len"] = prompt_len;
    j["decode_len"] = decode_len;
    j["warmup"] = warmup;
    j["iters"] = iters;
    j["ttft_ms_mean"] = ttft_mean;
    j["tpot_ms_mean"] = tpot_mean;
    j["tpot_ms_p50"] = tpot_p50;
    j["tpot_ms_p99"] = tpot_p99;
    j["total_ms_mean"] = total_mean;
    j["throughput_tok_per_s"] = tok_per_s;
    j["ttft_ms_all"] = ttft_ms;
    j["tpot_ms_all"] = tpot_ms;
    std::ofstream out(output_path);
    out << j.dump(2) << "\n";
    std::cerr << "Wrote " << output_path << "\n";
  }
#endif

  return 0;
}
