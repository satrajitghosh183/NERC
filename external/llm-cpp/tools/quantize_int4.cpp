/**
 * tools/quantize_int4.cpp
 *
 * Post-training INT4 (AWQ-style) weight-only quantizer (item T).
 *
 * Reads a trained .pt checkpoint, walks every Linear-class weight tensor
 * (bandwidth-bound layers: attention Q/K/V/out, FFN gate_up/w1/w3/w2,
 * and the LM head), quantizes each to INT4 with per-group FP16 scales
 * via the existing quantize_int4_awq() routine, and writes a sidecar
 * .int4.pt archive that the inference path consumes through fp8_gemv /
 * int4_gemv (kernels in I-4).
 *
 * No calibration data is required by AWQ as we've implemented it — the
 * group-wise abs-max scale is computed from the weight tensor alone.
 * (The full AWQ paper uses activation statistics; that's a follow-on
 * refinement; the structure here is identical so plugging in
 * activation-aware scaling later is a one-function swap.)
 *
 * Weights that should stay BF16 (embeddings, LayerNorm/RMSNorm gains)
 * are left untouched and written through to the sidecar.
 *
 * Usage:
 *   ./build/quantize_int4 \
 *     --in   checkpoints/125M.pt \
 *     --out  checkpoints/125M.int4.pt \
 *     --config configs/olmo2_125M.json \
 *     [--group-size 128] [--keep-pattern "lm_head|norm|embed"]
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/nn/quant.hpp"

#include <torch/torch.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace {

bool should_keep_fp(const std::string& name, const std::regex& keep_re) {
  return std::regex_search(name, keep_re);
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path, out_path, conf_path;
  std::string keep_pattern = "(^|\\.)(norm|embed|tokens?_embed|role_embed|char_embed|phrase_embed|lm_head\\.w_out)";
  int64_t group_size = 128;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if      (a == "--in"          && i+1 < argc) in_path        = argv[++i];
    else if (a == "--out"         && i+1 < argc) out_path       = argv[++i];
    else if (a == "--config"      && i+1 < argc) conf_path      = argv[++i];
    else if (a == "--group-size"  && i+1 < argc) group_size     = std::stoll(argv[++i]);
    else if (a == "--keep-pattern"&& i+1 < argc) keep_pattern   = argv[++i];
    else if (a == "-h" || a == "--help") {
      std::cerr << "Usage: quantize_int4 --in <ckpt.pt> --out <ckpt.int4.pt> "
                   "--config <config.json> [--group-size N] [--keep-pattern regex]\n";
      return 0;
    }
  }
  if (in_path.empty() || out_path.empty() || conf_path.empty()) {
    std::cerr << "missing --in / --out / --config\n";
    return 1;
  }

  std::regex keep_re(keep_pattern);

  try {
    // Load the model + checkpoint to walk its named_parameters().
    auto cfg = olmo_cpp::load_config_from_json(conf_path);
    cfg.validate();
    olmo_cpp::Transformer model(cfg);
    torch::load(model, in_path);
    model->eval();

    torch::serialize::OutputArchive out_arch;

    size_t n_total = 0, n_quantized = 0, n_kept = 0;
    int64_t bytes_orig = 0, bytes_quant = 0;

    for (auto& named : model->named_parameters()) {
      const auto& name = named.key();
      auto p = named.value().detach().contiguous();
      ++n_total;

      const int64_t orig_bytes = p.numel() * p.element_size();
      bytes_orig += orig_bytes;

      // Skip patterns that must stay in fp/bf16: norms, embeddings,
      // optionally the LM head depending on the keep-pattern. The
      // weight-only quant trick relies on bandwidth-bound matmuls; small
      // tensors or per-token lookups don't benefit and may lose accuracy.
      const bool keep_fp = should_keep_fp(name, keep_re);
      const bool quantizable = (p.dim() == 2)
                                && (p.size(0) > 1) && (p.size(1) > 1)
                                && (p.size(1) % group_size == 0)
                                && (p.size(1) % 2 == 0);
      if (keep_fp || !quantizable) {
        out_arch.write(name, p);
        bytes_quant += orig_bytes;
        ++n_kept;
        continue;
      }

      // AWQ-style INT4 quantization. Per-channel × per-group scales.
      // Force CPU + fp32 + contiguous; the saved checkpoint may have been on
      // CUDA and the AWQ path assumes a vanilla CPU fp32 layout.
      auto p_f32 = p.to(torch::kCPU).to(torch::kFloat32).contiguous();
      auto q = olmo_cpp::quantize_int4_awq(p_f32, group_size);
      out_arch.write(name + ".int4.weight", q.weight);
      out_arch.write(name + ".int4.scales", q.scales);
      out_arch.write(name + ".int4.group_size",
                     torch::tensor(q.group_size, torch::kInt64));
      ++n_quantized;
      bytes_quant += q.weight.numel() * q.weight.element_size()
                    + q.scales.numel() * q.scales.element_size();
    }

    out_arch.save_to(out_path);

    const double ratio = bytes_orig > 0 ? static_cast<double>(bytes_quant) / bytes_orig : 0.0;
    std::cout << "INT4 quantize complete\n"
              << "  in:           " << in_path  << "\n"
              << "  out:          " << out_path << "\n"
              << "  params total: " << n_total << "\n"
              << "  quantized:    " << n_quantized << "\n"
              << "  kept fp:      " << n_kept << "\n"
              << "  size:         " << bytes_orig/1024.0/1024.0 << " MiB -> "
                                    << bytes_quant/1024.0/1024.0 << " MiB"
              << " (" << ratio*100 << "%)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
