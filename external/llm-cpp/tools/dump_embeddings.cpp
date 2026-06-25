/**
 * tools/dump_embeddings.cpp
 *
 * Load a trained checkpoint, extract the input token-embedding matrix, and
 * write it to disk in three forms so the user can inspect it however they
 * like:
 *
 *   1. <out>.npy            — raw [vocab_size, d_model] float32 tensor
 *                             (loadable in Python with numpy.load).
 *   2. <out>_norms.txt      — one float per line, the L2 norm of each row.
 *                             Cheap human-readable sanity check that
 *                             training actually moved the weights.
 *   3. <out>_summary.txt    — global stats (min/max/mean/std), top-10 and
 *                             bottom-10 rows by norm with their token-id,
 *                             plus optional decoded-text if a vocab.json
 *                             file was supplied.
 *
 * Example:
 *   ./build/dump_embeddings \
 *     --conf  conf/quickstart_3060.conf \
 *     --ckpt  checkpoints/model.pt \
 *     --out   exports/embeddings \
 *     --vocab data/gpt2/vocab.json   # optional, gives token strings
 *
 * Producing:
 *   exports/embeddings.npy
 *   exports/embeddings_norms.txt
 *   exports/embeddings_summary.txt
 *
 * --- Build target ---
 *   dump_embeddings (CMakeLists.txt). Links olmo_cpp + LibTorch + cnpy.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/common/config_ini.hpp : parse the same .conf used to train
 *   - olmo_cpp/config.hpp            : TransformerConfig (must match the
 *                                       checkpoint's architecture exactly)
 *   - olmo_cpp/model/transformer.hpp : Transformer model
 *   - olmo_cpp/model/fused_transformer.hpp : FusedTransformer model
 *   - third_party/cnpy/cnpy.h        : write .npy file format
 *
 * --- Reads / Writes ---
 *   - reads:  <conf>, <ckpt>, optional <vocab.json>
 *   - writes: <out>.npy, <out>_norms.txt, <out>_summary.txt
 *
 * --- Role in workflow ---
 *   Post-training inspection tool. Run once after `olmo_train` finishes to
 *   look at what the model has learned about each input token. Shipping
 *   this is part of the "newcomer-friendly" workflow — embeddings are the
 *   easiest part of a transformer to reason about visually.
 */

#include "olmo_cpp/common/config_ini.hpp"
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"

#include "cnpy.h"

#include <torch/torch.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

/// Parse a TransformerConfig from a .conf file, reading only the fields we
/// need to instantiate the model in the exact shape it was trained in.
/// We only read what affects parameter shapes; runtime knobs (lr, steps,
/// optimizer) are irrelevant for dumping weights.
olmo_cpp::TransformerConfig load_model_config(const std::string& conf_path) {
  ConfigINI model_ini(conf_path, "model");
  ConfigINI opt_ini(conf_path, "optimization");

  olmo_cpp::TransformerConfig cfg;
  // Required architectural fields.
  model_ini.get("d_model",    cfg.d_model);
  model_ini.get("vocab_size", cfg.vocab_size);
  model_ini.get("n_layers",   cfg.n_layers);
  model_ini.get("n_heads",    cfg.n_heads);

  // Optional fields with the same defaults as src/main.cpp.
  cfg.n_kv_heads               = model_ini.get_or<int64_t>("n_kv_heads", -1);
  cfg.head_dim                 = model_ini.get_or<int64_t>("head_dim", -1);
  cfg.rope_theta               = model_ini.get_or<int64_t>("rope_theta", 500000);
  cfg.layer_norm_eps           = model_ini.get_or<double>("layer_norm_eps", 1e-6);
  cfg.init_std                 = model_ini.get_or<double>("init_std", 0.02);
  cfg.use_qk_norm              = model_ini.get_or<bool>("use_qk_norm", true);
  cfg.hidden_size_multiple_of  = model_ini.get_or<int64_t>("hidden_size_multiple_of", 256);
  cfg.hidden_size_multiplier   = model_ini.get_or<double>("hidden_size_multiplier", 1.5);
  cfg.num_mtp_heads            = model_ini.get_or<int64_t>("num_mtp_heads", 0);

  // Multi-res DC-MRE: if enabled, the embedding lives under a different name.
  cfg.use_multi_res            = opt_ini.get_or<bool>("multi_res", false);

  // Pull along the BPE vocab path because the multi-res embedding uses it
  // to size its sub-codebooks.
  ConfigINI data_ini(conf_path, "data");
  cfg.bpe_vocab_path           = data_ini.get_or<std::string>("bpe_vocab", "");

  cfg.validate();
  return cfg;
}

/// Try to load a vocab.json file in the GPT-2 BPE format (token-string ->
/// integer id). Returns an id->string lookup vector indexed by id, or empty
/// if the file can't be parsed. We tolerate failure silently because the
/// vocab is optional.
std::vector<std::string> load_vocab(const std::string& path) {
  std::vector<std::string> out;
#ifdef HAS_NLOHMANN_JSON
  if (path.empty()) return out;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "warning: could not open vocab file: " << path << "\n";
    return out;
  }
  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception& e) {
    std::cerr << "warning: could not parse vocab JSON: " << e.what() << "\n";
    return out;
  }
  // GPT-2 vocab.json maps "token" -> id. Build the inverse lookup.
  int64_t max_id = 0;
  for (auto it = j.begin(); it != j.end(); ++it) {
    int64_t id = it.value().get<int64_t>();
    if (id > max_id) max_id = id;
  }
  out.assign(max_id + 1, std::string());
  for (auto it = j.begin(); it != j.end(); ++it) {
    int64_t id = it.value().get<int64_t>();
    out[id] = it.key();
  }
#else
  (void)path;
  std::cerr << "warning: built without nlohmann/json; vocab decoding disabled\n";
#endif
  return out;
}

/// Pull the embedding weight tensor out of the model. For plain Embedding
/// modules the parameter key is "embeddings.weight"; for the multi-res
/// DC-MRE module the *semantic* sub-embedding (the closest analogue to a
/// classical embedding) is registered as "multi_res_embed.semantic.weight".
/// Returns a CPU float32 tensor of shape [vocab_size, d_model].
torch::Tensor extract_embedding(torch::nn::Module& model, bool use_multi_res) {
  const std::string key = use_multi_res
      ? "multi_res_embed.semantic.weight"
      : "embeddings.weight";

  for (const auto& p : model.named_parameters()) {
    if (p.key() == key) {
      // detach() so we don't carry any autograd state, then to(CPU,float32)
      // for portable dumping.
      return p.value().detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    }
  }
  throw std::runtime_error(
      "embedding parameter not found in model (looked for '" + key + "')");
}

/// Argument record. Positional-free parsing so the user can pass flags in
/// any order — easier for a first-time user.
struct Args {
  std::string conf;     // .conf file used during training
  std::string ckpt;     // .pt checkpoint path
  std::string out;      // output prefix (no extension)
  std::string vocab;    // optional vocab.json
};

/// Tiny CLI parser. We don't use a heavy library because this binary only
/// has four flags and we want zero extra dependencies.
Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if      (k == "--conf")  a.conf  = need("--conf");
    else if (k == "--ckpt")  a.ckpt  = need("--ckpt");
    else if (k == "--out")   a.out   = need("--out");
    else if (k == "--vocab") a.vocab = need("--vocab");
    else if (k == "-h" || k == "--help") {
      std::cout << "Usage: dump_embeddings --conf <conf> --ckpt <ckpt> "
                   "--out <prefix> [--vocab <vocab.json>]\n";
      std::exit(0);
    }
    else throw std::runtime_error("unknown flag: " + k);
  }
  if (a.conf.empty() || a.ckpt.empty() || a.out.empty()) {
    throw std::runtime_error(
        "required flags: --conf <conf> --ckpt <ckpt> --out <prefix>");
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    // Phase 1: rebuild the model in the same shape it was saved in.
    auto cfg = load_model_config(args.conf);
    ConfigINI opt_ini(args.conf, "optimization");
    bool use_fused = opt_ini.get_or<bool>("fused", false);

    std::cout << "Loading model from " << args.ckpt << "\n"
              << "  d_model=" << cfg.d_model
              << "  vocab_size=" << cfg.vocab_size
              << "  fused=" << use_fused
              << "  multi_res=" << cfg.use_multi_res << "\n";

    // Phase 2: load weights into the model. Both Transformer and
    // FusedTransformer share the embedding key, so the load path is the
    // same; we only branch on which class to instantiate.
    torch::Tensor embed;
    if (use_fused) {
      auto model = olmo_cpp::FusedTransformer(cfg);
      torch::load(model, args.ckpt);
      embed = extract_embedding(*model, cfg.use_multi_res);
    } else {
      auto model = olmo_cpp::Transformer(cfg);
      torch::load(model, args.ckpt);
      embed = extract_embedding(*model, cfg.use_multi_res);
    }

    const int64_t V = embed.size(0);
    const int64_t D = embed.size(1);
    std::cout << "Embedding tensor: [" << V << ", " << D << "] "
              << embed.dtype() << "\n";

    // Phase 3: ensure output directory exists.
    auto out_path = std::filesystem::path(args.out);
    if (out_path.has_parent_path()) {
      std::filesystem::create_directories(out_path.parent_path());
    }

    // Phase 4a: write raw matrix as .npy. cnpy expects a contiguous buffer
    // and a shape vector. embed is already contiguous & on CPU & float32.
    {
      std::string npy_path = args.out + ".npy";
      std::vector<size_t> shape = { static_cast<size_t>(V),
                                     static_cast<size_t>(D) };
      const float* data = embed.data_ptr<float>();
      cnpy::npy_save(npy_path, data, shape, "w");
      std::cout << "wrote " << npy_path << "\n";
    }

    // Phase 4b: write per-row L2 norms (cheap signal of which tokens have
    // received gradient updates — under-trained tokens stay near init).
    auto norms = embed.norm(2, /*dim=*/1).contiguous();
    {
      std::string norms_path = args.out + "_norms.txt";
      std::ofstream of(norms_path);
      const float* nptr = norms.data_ptr<float>();
      for (int64_t i = 0; i < V; ++i) of << nptr[i] << "\n";
      std::cout << "wrote " << norms_path << "\n";
    }

    // Phase 4c: write summary (global stats + top/bottom tokens by norm).
    auto vocab = load_vocab(args.vocab);
    {
      std::string sum_path = args.out + "_summary.txt";
      std::ofstream of(sum_path);
      auto mn = embed.min().item<float>();
      auto mx = embed.max().item<float>();
      auto me = embed.mean().item<float>();
      auto sd = embed.std().item<float>();
      auto nm_min = norms.min().item<float>();
      auto nm_max = norms.max().item<float>();
      auto nm_me  = norms.mean().item<float>();

      of << "Embedding summary\n";
      of << "  shape           : [" << V << ", " << D << "]\n";
      of << "  value min/max   : " << mn << " / " << mx << "\n";
      of << "  value mean/std  : " << me << " / " << sd << "\n";
      of << "  row-norm min/max: " << nm_min << " / " << nm_max << "\n";
      of << "  row-norm mean   : " << nm_me << "\n\n";

      // Sort token ids by norm ascending so [0] is smallest, [V-1] biggest.
      std::vector<int64_t> idx(V);
      std::iota(idx.begin(), idx.end(), 0);
      const float* nptr = norms.data_ptr<float>();
      std::sort(idx.begin(), idx.end(),
                [nptr](int64_t a, int64_t b) { return nptr[a] < nptr[b]; });

      auto print_block = [&](const char* title, int64_t lo, int64_t hi) {
        of << title << "\n";
        for (int64_t i = lo; i < hi; ++i) {
          int64_t id = idx[i];
          of << "  id=" << id << "  norm=" << nptr[id];
          if (id < static_cast<int64_t>(vocab.size()) && !vocab[id].empty()) {
            of << "  token=" << vocab[id];
          }
          of << "\n";
        }
      };

      // Bottom-10 (likely under-trained / near init).
      int64_t k = std::min<int64_t>(10, V);
      print_block("Bottom-10 tokens by row-norm (least-trained):", 0, k);
      // Top-10 (most "active" tokens).
      print_block("Top-10 tokens by row-norm (most-trained):", V - k, V);

      std::cout << "wrote " << sum_path << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dump_embeddings: " << e.what() << "\n";
    return 1;
  }
}
