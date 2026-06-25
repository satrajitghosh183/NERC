/**
 * tools/benchmark_tokenizer.cpp
 *
 * CLI that benchmarks two tokenizers head-to-head on a real corpus:
 *   1. GPT-2 BPE                — the reference baseline.
 *   2. Structural tokenizer     — this project's pattern-aware tokenizer
 *                                 that emits "structural" tokens for
 *                                 frequent code/prose snippets.
 * For each tokenizer it measures:
 *   - bytes/token (compression)
 *   - tokens/KB
 *   - vocabulary utilization (unique IDs seen / vocab size)
 *   - encode speed in MB/s
 * It also prints a per-domain (code/prose/data) comparison so you can see
 * where the structural tokenizer wins or loses. Pure stdout — no files
 * written to disk.
 *
 * Example:
 *   ./build/benchmark_tokenizer \
 *     --bpe-vocab data/gpt2/vocab.json --bpe-merges data/gpt2/merges.txt \
 *     --structural-config data/structural_tokenizer/ \
 *     --corpus src/ \
 *     --corpus data/tinystories_raw/
 *
 * --- Build target ---
 *   benchmark_tokenizer (CMakeLists.txt:580). Standalone executable —
 *   does NOT link the full `olmo_cpp` library or LibTorch. It directly
 *   compiles `src/data/bpe_tokenizer.cpp` and
 *   `src/data/structural_tokenizer.cpp`. Compiled with -O3 -march=native.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/bpe_tokenizer.hpp:        GPT-2 BPE encoder/decoder
 *   - olmo_cpp/data/structural_tokenizer.hpp: pattern-aware tokenizer
 *
 * --- Reads / Writes ---
 *   - reads:  vocab.json, merges.txt, structural-config dir, every file
 *             under each --corpus dir (with allowed extensions, <=1 MB)
 *   - writes: nothing to disk; results go to stdout.
 *
 * --- Role in workflow ---
 *   Used to evaluate the structural tokenizer before deciding whether
 *   to switch the training pipeline over from raw BPE. Run after
 *   `mine_patterns` has produced a structural tokenizer config; before
 *   `prepare_data --structural-config ...`.
 */

#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/data/structural_tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/// Slurp a whole file into a std::string. Returns "" if the file cannot
/// be opened (so callers should treat empty content as "skip"). Uses an
/// ostringstream to avoid hand-managing the buffer.
static std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/// One aggregated row of stats per tokenizer (or per domain). Counters
/// accumulate across many files; helper methods derive the final metrics
/// (bytes/token, vocab utilization, encode throughput).
struct BenchResult {
  std::string name;
  int64_t total_bytes = 0;
  int64_t total_tokens = 0;
  int64_t files = 0;
  double encode_ms = 0;
  std::set<uint32_t> unique_tokens;
  uint32_t vocab_size = 0;

  // For structural tokenizer breakdown
  int64_t pattern_tokens = 0;
  int64_t atom_tokens = 0;
  int64_t bpe_tokens = 0;

  double bytes_per_token() const {
    return total_tokens > 0 ? static_cast<double>(total_bytes) / total_tokens : 0;
  }
  double tokens_per_kb() const {
    return total_bytes > 0 ? static_cast<double>(total_tokens) / (total_bytes / 1024.0) : 0;
  }
  double vocab_utilization() const {
    return vocab_size > 0 ? 100.0 * unique_tokens.size() / vocab_size : 0;
  }
  double encode_speed_mbps() const {
    return encode_ms > 0 ? (total_bytes / 1e6) / (encode_ms / 1000.0) : 0;
  }
};

/// Walk every --corpus directory recursively and collect candidate files
/// for the benchmark. Filters: must be a regular file, extension must be
/// in the whitelist (text/code/data formats), and size <= 1 MB so a few
/// huge dumps cannot dominate the result. Stops at `max_files` total.
static std::vector<std::string> collect_files(const std::vector<std::string>& dirs, int max_files) {
  static const std::set<std::string> valid_exts = {
    ".txt", ".py", ".c", ".cpp", ".h", ".hpp", ".cc",
    ".java", ".js", ".ts", ".json", ".jsonl", ".md", ".rst", ".text",
  };

  std::vector<std::string> files;
  for (auto& dir : dirs) {
    if (!fs::exists(dir)) continue;
    for (auto& entry : fs::recursive_directory_iterator(dir)) {
      if (!entry.is_regular_file()) continue;
      auto ext = entry.path().extension().string();
      if (!valid_exts.count(ext)) continue;
      if (entry.file_size() > 1000000) continue;  // skip >1MB files
      files.push_back(entry.path().string());
      if (static_cast<int>(files.size()) >= max_files) break;
    }
    if (static_cast<int>(files.size()) >= max_files) break;
  }
  return files;
}

int main(int argc, char** argv) {
  // -----------------------------------------------------------------
  // Phase 1: parse CLI flags.
  //   --bpe-vocab/-merges:      GPT-2 BPE inputs (REQUIRED).
  //   --structural-config:      directory describing structural tokenizer
  //                             (optional; if absent we benchmark BPE only).
  //   --corpus:                 may be passed multiple times to merge dirs.
  //   --max-files:              cap on total files scanned.
  //   --verbose/-v:             print per-file progress.
  // -----------------------------------------------------------------
  std::string bpe_vocab, bpe_merges, structural_config;
  std::vector<std::string> corpus_dirs;
  int max_files = 500;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--bpe-vocab" && i + 1 < argc) bpe_vocab = argv[++i];
    else if (arg == "--bpe-merges" && i + 1 < argc) bpe_merges = argv[++i];
    else if (arg == "--structural-config" && i + 1 < argc) structural_config = argv[++i];
    else if (arg == "--corpus" && i + 1 < argc) corpus_dirs.push_back(argv[++i]);
    else if (arg == "--max-files" && i + 1 < argc) max_files = std::atoi(argv[++i]);
    else if (arg == "--verbose" || arg == "-v") verbose = true;
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: benchmark_tokenizer [OPTIONS]\n\n"
                << "  --bpe-vocab <path>          GPT-2 vocab.json\n"
                << "  --bpe-merges <path>         GPT-2 merges.txt\n"
                << "  --structural-config <dir>   Structural tokenizer config directory\n"
                << "  --corpus <dir>              Corpus directory (can repeat)\n"
                << "  --max-files <n>             Max files (default: 500)\n"
                << "  --verbose / -v              Show per-file results\n";
      return 0;
    }
  }

  if (bpe_vocab.empty() || bpe_merges.empty()) {
    std::cerr << "Error: --bpe-vocab and --bpe-merges are required\n";
    return 1;
  }
  if (corpus_dirs.empty()) {
    std::cerr << "Error: at least one --corpus directory is required\n";
    return 1;
  }

  // -----------------------------------------------------------------
  // Phase 2: load tokenizers.
  // The structural tokenizer internally uses BPE as a fallback for any
  // text it cannot match against a structural pattern, so it needs the
  // same vocab/merges files passed to it.
  // -----------------------------------------------------------------
  olmo_cpp::BPETokenizer bpe;
  if (!bpe.load(bpe_vocab, bpe_merges)) {
    std::cerr << "Error: failed to load BPE tokenizer\n";
    return 1;
  }

  olmo_cpp::StructuralTokenizer structural;
  bool has_structural = false;
  if (!structural_config.empty()) {
    has_structural = structural.load(structural_config, bpe_vocab, bpe_merges);
    if (!has_structural) {
      std::cerr << "Warning: failed to load structural tokenizer, BPE-only benchmark\n";
    }
  }

  // -----------------------------------------------------------------
  // Phase 3: collect candidate files from the corpus dirs.
  // -----------------------------------------------------------------
  auto files = collect_files(corpus_dirs, max_files);
  std::cout << "Collected " << files.size() << " files from " << corpus_dirs.size() << " directories\n\n";

  if (files.empty()) {
    std::cerr << "Error: no files found in corpus directories\n";
    return 1;
  }

  // -----------------------------------------------------------------
  // Phase 4: run both tokenizers over every file and accumulate stats.
  // Two top-level totals (BPE / Structural) plus per-domain breakdowns
  // keyed on extension class (code / prose / data).
  // -----------------------------------------------------------------
  BenchResult bpe_result;
  bpe_result.name = "BPE (GPT-2)";
  bpe_result.vocab_size = bpe.vocab_size();

  BenchResult struct_result;
  struct_result.name = "Structural";
  struct_result.vocab_size = structural.vocab_size();

  // Per-domain breakdown
  std::map<std::string, BenchResult> bpe_by_domain;
  std::map<std::string, BenchResult> struct_by_domain;

  for (auto& fpath : files) {
    std::string content = read_file(fpath);
    if (content.empty()) continue;

    // Detect domain
    auto ext = fs::path(fpath).extension().string();
    std::string domain;
    if (ext == ".py" || ext == ".c" || ext == ".cpp" || ext == ".h" || ext == ".hpp" ||
        ext == ".java" || ext == ".js" || ext == ".ts" || ext == ".cc")
      domain = "code";
    else if (ext == ".json" || ext == ".jsonl")
      domain = "data";
    else
      domain = "prose";

    int64_t bytes = static_cast<int64_t>(content.size());

    // -- BPE timing/encoding for this file. We measure wall-clock around
    //    the encode() call only, not the file read, to keep the MB/s
    //    metric meaningful.
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      auto ids = bpe.encode(content);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      // Strip the trailing EOS sentinel so it does not skew the per-file
      // bytes/token ratio.
      if (!ids.empty() && ids.back() == bpe.eos_id()) ids.pop_back();

      bpe_result.total_bytes += bytes;
      bpe_result.total_tokens += ids.size();
      bpe_result.encode_ms += ms;
      bpe_result.files++;
      for (auto id : ids) bpe_result.unique_tokens.insert(id);

      auto& bd = bpe_by_domain[domain];
      bd.total_bytes += bytes;
      bd.total_tokens += ids.size();
      bd.files++;
    }

    // -- Structural tokenizer encoding (only if the user passed
    //    --structural-config). last_stats() returns a breakdown of how
    //    each token was produced (pattern match / atom / BPE fallback).
    if (has_structural) {
      auto t0 = std::chrono::high_resolution_clock::now();
      auto ids = structural.encode(content);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      // Remove EOS
      if (!ids.empty() && ids.back() == structural.eos_id()) ids.pop_back();

      auto& stats = structural.last_stats();
      struct_result.total_bytes += bytes;
      struct_result.total_tokens += ids.size();
      struct_result.encode_ms += ms;
      struct_result.files++;
      struct_result.pattern_tokens += stats.pattern_tokens;
      struct_result.atom_tokens += stats.atom_tokens;
      struct_result.bpe_tokens += stats.bpe_tokens;
      for (auto id : ids) struct_result.unique_tokens.insert(id);

      auto& sd = struct_by_domain[domain];
      sd.total_bytes += bytes;
      sd.total_tokens += ids.size();
      sd.files++;
    }

    if (verbose) {
      std::cout << fpath << " [" << domain << "] " << bytes << "B → BPE:" << bpe_result.total_tokens;
      if (has_structural) std::cout << " Struct:" << struct_result.total_tokens;
      std::cout << "\n";
    }
  }

  // -----------------------------------------------------------------
  // Phase 5: print formatted results to stdout.
  // -----------------------------------------------------------------
  auto print_result = [](const BenchResult& r) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total bytes:       " << r.total_bytes << "\n";
    std::cout << "  Total tokens:      " << r.total_tokens << "\n";
    std::cout << "  Files:             " << r.files << "\n";
    std::cout << "  Bytes/token:       " << r.bytes_per_token() << "\n";
    std::cout << "  Tokens/KB:         " << r.tokens_per_kb() << "\n";
    std::cout << "  Unique tokens:     " << r.unique_tokens.size()
              << " / " << r.vocab_size
              << " (" << r.vocab_utilization() << "% util)\n";
    std::cout << "  Encode speed:      " << r.encode_speed_mbps() << " MB/s\n";
    std::cout << "  Encode time:       " << r.encode_ms << " ms\n";
  };

  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TOKENIZER BENCHMARK RESULTS\n";
  std::cout << std::string(60, '=') << "\n";

  std::cout << "\n--- " << bpe_result.name << " ---\n";
  print_result(bpe_result);

  if (has_structural) {
    std::cout << "\n--- " << struct_result.name << " ---\n";
    print_result(struct_result);
    std::cout << "  Pattern tokens:    " << struct_result.pattern_tokens
              << " (" << (100.0 * struct_result.pattern_tokens / std::max(struct_result.total_tokens, int64_t(1)))
              << "%)\n";
    std::cout << "  Atom tokens:       " << struct_result.atom_tokens
              << " (" << (100.0 * struct_result.atom_tokens / std::max(struct_result.total_tokens, int64_t(1)))
              << "%)\n";
    std::cout << "  BPE fallback:      " << struct_result.bpe_tokens
              << " (" << (100.0 * struct_result.bpe_tokens / std::max(struct_result.total_tokens, int64_t(1)))
              << "%)\n";

    // Compression comparison
    double reduction = 100.0 * (1.0 - static_cast<double>(struct_result.total_tokens) / bpe_result.total_tokens);
    std::cout << "\n--- Comparison ---\n";
    std::cout << "  Token reduction:   " << std::setprecision(1) << reduction << "%"
              << " (" << bpe_result.total_tokens << " → " << struct_result.total_tokens << ")\n";
    std::cout << "  BPE bytes/tok:     " << std::setprecision(2) << bpe_result.bytes_per_token() << "\n";
    std::cout << "  Struct bytes/tok:  " << struct_result.bytes_per_token() << "\n";

    // Per-domain comparison
    std::cout << "\n--- Per-Domain Comparison ---\n";
    std::cout << std::left << std::setw(10) << "Domain"
              << std::right << std::setw(12) << "BPE tok"
              << std::setw(12) << "Struct tok"
              << std::setw(10) << "Reduction"
              << std::setw(12) << "BPE b/t"
              << std::setw(12) << "Struct b/t" << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (auto& [domain, bd] : bpe_by_domain) {
      auto& sd = struct_by_domain[domain];
      double red = bd.total_tokens > 0 ?
        100.0 * (1.0 - static_cast<double>(sd.total_tokens) / bd.total_tokens) : 0;
      double bpe_bpt = bd.total_tokens > 0 ?
        static_cast<double>(bd.total_bytes) / bd.total_tokens : 0;
      double struct_bpt = sd.total_tokens > 0 ?
        static_cast<double>(sd.total_bytes) / sd.total_tokens : 0;

      std::cout << std::left << std::setw(10) << domain
                << std::right << std::setw(12) << bd.total_tokens
                << std::setw(12) << sd.total_tokens
                << std::setw(9) << std::fixed << std::setprecision(1) << red << "%"
                << std::setw(12) << std::setprecision(2) << bpe_bpt
                << std::setw(12) << struct_bpt << "\n";
    }
  }

  std::cout << "\n" << std::string(60, '=') << "\n";

  return 0;
}
