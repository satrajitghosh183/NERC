// numerical_audit — measure end-to-end numerical drift between the zwt
// training precision (bf16 storage, fp32 compute) and a pure-fp32 reference.
//
// What it does:
//   1. Loads a train config (reuses the .conf format the trainer consumes).
//   2. Builds *two* Transformers from the same config + seed: one at the
//      training dtype (bf16 on CUDA, f32 on CPU build), one at DType::F32.
//      Same init seed + same kaiming RNG path means the fp32 model's
//      weights are exactly the bf16 model's weights *before* bf16 rounding.
//   3. Feeds both the same synthetic input and prints logit drift.
//
// The CSV row on stdout is:
//   tensor,max_abs_err,mean_rel_err,lhs_norm,rhs_norm,rows
//
// Per-layer comparison is not wired yet — the tool captures the final logits
// only. Extending to layer boundaries means exposing intermediates on
// Transformer, which is a separate refactor. This is the MVP that lands the
// measurement methodology.
//
// Build: zwt_numerical_audit (added to CMakeLists). The CPU build compiles
// cleanly; meaningful runs need the CUDA build (the CPU paths don't support
// bf16 forward end-to-end).

#include "zwt/core/allocator.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/transformer.hpp"
#include "zwt/train/config.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

namespace {

struct CliArgs {
  std::string config_path;
  int64_t     seq_len    = 64;
  int64_t     batch_size = 2;
  uint64_t    input_seed = 0xABCD'1234ULL;
};

CliArgs parse_cli(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    std::string k = s.substr(0, eq);
    std::string v = (eq == std::string::npos) ? std::string{}
                                              : s.substr(eq + 1);
    if      (k == "--config")     a.config_path = v;
    else if (k == "--seq")        a.seq_len     = std::stoll(v);
    else if (k == "--batch")      a.batch_size  = std::stoll(v);
    else if (k == "--input-seed") a.input_seed  = std::stoull(v, nullptr, 0);
    else throw std::runtime_error("numerical_audit: unknown arg: " + k);
  }
  if (a.config_path.empty()) {
    throw std::runtime_error("numerical_audit: --config=PATH required");
  }
  return a;
}

// Bring a tensor's contents onto the host as fp32 regardless of its source
// dtype. This is the common ground on which we compute drift metrics.
std::vector<float> tensor_to_host_f32(const Tensor& t) {
  const size_t n = static_cast<size_t>(t.numel());
  std::vector<float> out(n);

  std::vector<uint8_t> raw(t.nbytes());
#ifdef USE_CUDA
  if (t.device().is_cuda()) {
    cudaMemcpy(raw.data(), t.data(), t.nbytes(), cudaMemcpyDeviceToHost);
  } else {
    std::memcpy(raw.data(), t.data(), t.nbytes());
  }
#else
  std::memcpy(raw.data(), t.data(), t.nbytes());
#endif

  switch (t.dtype()) {
    case DType::F32:
      std::memcpy(out.data(), raw.data(), n * sizeof(float));
      break;
    case DType::BF16: {
      const uint16_t* bf = reinterpret_cast<const uint16_t*>(raw.data());
      for (size_t i = 0; i < n; ++i) {
        uint32_t u = static_cast<uint32_t>(bf[i]) << 16;
        std::memcpy(&out[i], &u, 4);
      }
      break;
    }
    default:
      throw std::runtime_error("numerical_audit: unsupported dtype in audit");
  }
  return out;
}

struct Drift {
  float  max_abs  = 0.f;
  float  mean_rel = 0.f;
  float  lhs_norm = 0.f;
  float  rhs_norm = 0.f;
  size_t n        = 0;
};

Drift measure(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("numerical_audit: tensor size mismatch");
  }
  Drift d;
  d.n = a.size();
  double l = 0.0, r = 0.0, rel_acc = 0.0;
  size_t rel_n = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = std::fabs(a[i] - b[i]);
    if (diff > d.max_abs) d.max_abs = diff;
    l += double(a[i]) * a[i];
    r += double(b[i]) * b[i];
    float denom = std::max(std::fabs(a[i]), std::fabs(b[i]));
    if (denom > 1e-8f) { rel_acc += diff / denom; ++rel_n; }
  }
  d.lhs_norm = static_cast<float>(std::sqrt(l));
  d.rhs_norm = static_cast<float>(std::sqrt(r));
  d.mean_rel = (rel_n > 0) ? static_cast<float>(rel_acc / double(rel_n)) : 0.f;
  return d;
}

void print_csv_header() {
  std::printf("tensor,max_abs_err,mean_rel_err,lhs_norm,rhs_norm,n\n");
}

void print_csv_row(const std::string& name, const Drift& d) {
  std::printf("%s,%.6g,%.6g,%.6g,%.6g,%zu\n",
              name.c_str(), d.max_abs, d.mean_rel,
              d.lhs_norm, d.rhs_norm, d.n);
}

Tensor make_input_tokens(int64_t batch, int64_t seq, int64_t vocab,
                         uint64_t seed, Device dev) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, vocab - 1);
  const size_t n = static_cast<size_t>(batch * seq);
  std::vector<int64_t> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = dist(rng);

  Tensor t = empty({batch, seq}, DType::I64, dev);
#ifdef USE_CUDA
  if (dev.is_cuda()) {
    cudaMemcpy(t.data(), host.data(), n * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    return t;
  }
#endif
  std::memcpy(t.data(), host.data(), n * sizeof(int64_t));
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliArgs cli = parse_cli(argc, argv);
    train::TrainConfig cfg = train::load_train_config(cli.config_path);

#ifdef USE_CUDA
    Device dev = Device::cuda(0);
    DType  train_dtype = DType::BF16;
#else
    Device dev = Device::cpu();
    DType  train_dtype = DType::F32;
    std::fprintf(stderr,
        "numerical_audit: CPU build — both models run at fp32; audit is "
        "trivial here. Build with CUDA for a meaningful comparison.\n");
#endif

    size_t arena_bytes = static_cast<size_t>(std::max<int64_t>(cfg.arena_mb, 256)) << 20;
    set_activation_arena_capacity(arena_bytes);

    // Same init_seed feeds kaiming's PCG, so the fp32 model holds the exact
    // values the bf16 model would have before rounding. Drift we measure
    // therefore includes both weight-rounding and compute-precision loss.
    Transformer bf_model(cfg.model, train_dtype, dev, cfg.init_seed);
    Transformer fp_model(cfg.model, DType::F32,   dev, cfg.init_seed);

    Tensor tokens = make_input_tokens(cli.batch_size, cli.seq_len,
                                      cfg.model.vocab_size,
                                      cli.input_seed, dev);

    Tensor bf_logits = bf_model.forward(tokens);
    Tensor fp_logits = fp_model.forward(tokens);

    std::vector<float> a = tensor_to_host_f32(bf_logits);
    std::vector<float> b = tensor_to_host_f32(fp_logits);
    Drift d = measure(a, b);

    print_csv_header();
    print_csv_row("final_logits", d);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "numerical_audit: %s\n", e.what());
    return 1;
  }
}
