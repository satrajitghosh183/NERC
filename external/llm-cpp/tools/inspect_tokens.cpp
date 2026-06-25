/**
 * tools/inspect_tokens.cpp
 *
 * Single-tokenizer (GPT-2 BPE) introspection CLI. Given a string, a
 * file, or a directory tree, it shows the GPT-2 pre-tokenization
 * chunks, the token IDs, the decoded textual form of each token, and
 * compression metrics (bytes/token, tokens/line, vocab utilization,
 * top-50 most frequent tokens). Output is purely stdout — no files
 * written.
 *
 * Examples:
 *   # Single string
 *   ./build/inspect_tokens --text "for (int i = 0; i < n; i++)" \
 *     --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt
 *
 *   # Single file with aggregate stats
 *   ./build/inspect_tokens --file src/train.cpp \
 *     --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --stats
 *
 *   # Whole directory tree, aggregate only
 *   ./build/inspect_tokens --dir data/tinystories_raw/ \
 *     --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt --stats
 *
 * --- Flags ---
 *   --text <s>       inspect this exact string
 *   --file <p>       inspect this single file
 *   --dir <p>        recurse into directory; only aggregate stats printed
 *   --vocab-file     GPT-2 vocab.json (REQUIRED)
 *   --merges-file    GPT-2 merges.txt (REQUIRED)
 *   --stats          print aggregate stats (always on for --dir)
 *   --max-files <n>  cap on files when using --dir (default 100)
 *
 * --- Build target ---
 *   inspect_tokens (CMakeLists.txt:557). Standalone executable — does
 *   not link LibTorch. Compiles tools/inspect_tokens.cpp +
 *   src/data/bpe_tokenizer.cpp directly. -O3 -march=native.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/bpe_tokenizer.hpp: GPT-2 BPE tokenizer used here.
 *
 * --- Reads / Writes ---
 *   - reads:  vocab.json, merges.txt, plus whatever --text/--file/--dir
 *             points at.
 *   - writes: nothing — all output is stdout.
 *
 * --- Role in workflow ---
 *   Diagnostic aid for sanity-checking tokenization behaviour before
 *   running `prepare_data` or training. Especially useful when
 *   debugging tokenizer regressions or evaluating BPE compression on
 *   a new corpus.
 */

#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>

namespace fs = std::filesystem;

/// Running aggregate of tokenization stats across one or many files.
/// `token_freq` is used to compute vocabulary utilization and the
/// "top-N most frequent tokens" table.
struct CompressionStats {
  int64_t total_bytes = 0;
  int64_t total_tokens = 0;
  int64_t total_lines = 0;
  int64_t files_processed = 0;
  std::map<uint32_t, int64_t> token_freq;  // token_id -> count

  double bytes_per_token() const {
    return total_tokens > 0 ? static_cast<double>(total_bytes) / total_tokens : 0;
  }
  double tokens_per_line() const {
    return total_lines > 0 ? static_cast<double>(total_tokens) / total_lines : 0;
  }
};

/// Read whole file into memory; empty string on open failure.
static std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/// Tokenize one piece of text. If `verbose` is true, dumps the
/// pre-tokenized chunks, token IDs, decoded tokens, and bytes/token to
/// stdout (capped at first 50 chunks/IDs to keep output readable).
/// If `stats` is non-null, accumulates aggregates into it.
static void inspect_text(olmo_cpp::BPETokenizer& tok, const std::string& text,
                          bool verbose, CompressionStats* stats) {
  auto chunks = tok.get_pre_tokenized_chunks(text);
  auto ids = tok.encode(text);
  // Remove trailing EOS for display
  if (!ids.empty() && ids.back() == tok.eos_id()) {
    ids.pop_back();
  }

  int64_t byte_count = static_cast<int64_t>(text.size());
  int64_t token_count = static_cast<int64_t>(ids.size());

  if (stats) {
    stats->total_bytes += byte_count;
    stats->total_tokens += token_count;
    stats->total_lines += std::count(text.begin(), text.end(), '\n') + 1;
    for (auto id : ids) {
      stats->token_freq[id]++;
    }
  }

  if (verbose) {
    std::cout << "Input (" << byte_count << " bytes): \"";
    // Print first 200 chars
    if (text.size() > 200) {
      std::cout << text.substr(0, 200) << "...";
    } else {
      std::cout << text;
    }
    std::cout << "\"\n";

    // Pre-tokenized chunks
    std::cout << "Pre-tokenized chunks (" << chunks.size() << "): [";
    for (size_t i = 0; i < chunks.size() && i < 50; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << "\"" << chunks[i] << "\"";
    }
    if (chunks.size() > 50) std::cout << ", ... +" << (chunks.size() - 50) << " more";
    std::cout << "]\n";

    // Token IDs
    std::cout << "Token IDs (" << token_count << "): [";
    for (size_t i = 0; i < ids.size() && i < 50; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << ids[i];
    }
    if (ids.size() > 50) std::cout << ", ... +" << (ids.size() - 50) << " more";
    std::cout << "]\n";

    // Decoded tokens
    std::cout << "Decoded:   [";
    for (size_t i = 0; i < ids.size() && i < 50; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << "\"" << tok.decode_token(ids[i]) << "\"";
    }
    if (ids.size() > 50) std::cout << ", ... +" << (ids.size() - 50) << " more";
    std::cout << "]\n";

    // Compression
    double bpt = token_count > 0 ? static_cast<double>(byte_count) / token_count : 0;
    std::cout << "Compression: " << std::fixed;
    std::cout.precision(2);
    std::cout << bpt << " bytes/token (" << token_count << " tokens for "
              << byte_count << " bytes)\n";
    std::cout << std::endl;
  }
}

/// Pretty-print the aggregate stats table: totals, ratios, vocab usage,
/// and the top-50 token-frequency leaderboard.
static void print_stats(const CompressionStats& stats, const olmo_cpp::BPETokenizer& tok) {
  std::cout << "\n=== Compression Statistics ===\n";
  std::cout << "  Files processed: " << stats.files_processed << "\n";
  std::cout << "  Total bytes:     " << stats.total_bytes << "\n";
  std::cout << "  Total tokens:    " << stats.total_tokens << "\n";
  std::cout << "  Total lines:     " << stats.total_lines << "\n";
  std::cout.precision(2);
  std::cout << std::fixed;
  std::cout << "  Bytes/token:     " << stats.bytes_per_token() << "\n";
  std::cout << "  Tokens/line:     " << stats.tokens_per_line() << "\n";

  // Vocabulary utilization
  int64_t unique_tokens = static_cast<int64_t>(stats.token_freq.size());
  std::cout << "  Unique tokens:   " << unique_tokens << " / " << tok.vocab_size()
            << " (" << (100.0 * unique_tokens / tok.vocab_size()) << "% utilization)\n";

  // Top-50 most frequent tokens
  std::vector<std::pair<int64_t, uint32_t>> freq_list;
  for (auto& [id, count] : stats.token_freq) {
    freq_list.push_back({count, id});
  }
  std::sort(freq_list.rbegin(), freq_list.rend());

  std::cout << "\n  Top 50 tokens:\n";
  std::cout << "  " << std::string(60, '-') << "\n";
  std::cout << "  " << std::left << std::setw(8) << "ID"
            << std::setw(20) << "Token"
            << std::right << std::setw(10) << "Count"
            << std::setw(10) << "%" << "\n";
  std::cout << "  " << std::string(60, '-') << "\n";

  for (size_t i = 0; i < 50 && i < freq_list.size(); ++i) {
    auto [count, id] = freq_list[i];
    std::string token_str = tok.decode_token(id);
    // Escape special chars for display
    std::string display;
    for (char c : token_str) {
      if (c == '\n') display += "\\n";
      else if (c == '\t') display += "\\t";
      else if (c == '\r') display += "\\r";
      else display += c;
    }
    if (display.size() > 18) display = display.substr(0, 15) + "...";

    double pct = 100.0 * count / stats.total_tokens;
    std::cout << "  " << std::left << std::setw(8) << id
              << std::setw(20) << ("\"" + display + "\"")
              << std::right << std::setw(10) << count
              << std::setw(9) << std::fixed << std::setprecision(1) << pct << "%\n";
  }
  std::cout << "  " << std::string(60, '-') << "\n";
  std::cout << "==============================\n";
}

int main(int argc, char** argv) {
  // -----------------------------------------------------------------
  // Phase 1: parse CLI flags. Exactly one of --text/--file/--dir must
  // be supplied; vocab-file and merges-file are always required.
  // -----------------------------------------------------------------
  std::string text_input;
  std::string file_path;
  std::string dir_path;
  std::string vocab_file;
  std::string merges_file;
  bool show_stats = false;
  int max_files = 100;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--text" && i + 1 < argc) {
      text_input = argv[++i];
    } else if (arg == "--file" && i + 1 < argc) {
      file_path = argv[++i];
    } else if (arg == "--dir" && i + 1 < argc) {
      dir_path = argv[++i];
    } else if (arg == "--vocab-file" && i + 1 < argc) {
      vocab_file = argv[++i];
    } else if (arg == "--merges-file" && i + 1 < argc) {
      merges_file = argv[++i];
    } else if (arg == "--stats") {
      show_stats = true;
    } else if (arg == "--max-files" && i + 1 < argc) {
      max_files = std::atoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: inspect_tokens [OPTIONS]\n\n"
                << "  --text <string>       Inspect a text string\n"
                << "  --file <path>         Inspect a file\n"
                << "  --dir <path>          Inspect all files in directory\n"
                << "  --vocab-file <path>   GPT-2 vocab.json\n"
                << "  --merges-file <path>  GPT-2 merges.txt\n"
                << "  --stats               Show aggregate statistics\n"
                << "  --max-files <n>       Max files in --dir mode (default: 100)\n";
      return 0;
    }
  }

  if (vocab_file.empty() || merges_file.empty()) {
    std::cerr << "Error: --vocab-file and --merges-file are required\n";
    return 1;
  }
  if (text_input.empty() && file_path.empty() && dir_path.empty()) {
    std::cerr << "Error: one of --text, --file, or --dir is required\n";
    return 1;
  }

  // -----------------------------------------------------------------
  // Phase 2: load the GPT-2 BPE tokenizer once.
  // -----------------------------------------------------------------
  olmo_cpp::BPETokenizer tok;
  if (!tok.load(vocab_file, merges_file)) {
    std::cerr << "Error: failed to load tokenizer from " << vocab_file << " + " << merges_file << "\n";
    return 1;
  }
  std::cout << "Loaded tokenizer: vocab_size=" << tok.vocab_size()
            << ", eos_id=" << tok.eos_id() << "\n\n";

  CompressionStats stats;

  // -----------------------------------------------------------------
  // Phase 3: dispatch on input mode (--text / --file / --dir).
  // -----------------------------------------------------------------
  if (!text_input.empty()) {
    inspect_text(tok, text_input, true, &stats);
    if (show_stats) print_stats(stats, tok);
  } else if (!file_path.empty()) {
    std::string content = read_file(file_path);
    if (content.empty()) {
      std::cerr << "Error: could not read " << file_path << "\n";
      return 1;
    }
    std::cout << "File: " << file_path << " (" << content.size() << " bytes)\n\n";
    inspect_text(tok, content, true, &stats);
    stats.files_processed = 1;
    if (show_stats) print_stats(stats, tok);
  } else if (!dir_path.empty()) {
    int file_count = 0;
    std::vector<std::string> extensions = {".txt", ".py", ".c", ".cpp", ".h", ".hpp",
                                            ".java", ".js", ".ts", ".json", ".jsonl",
                                            ".md", ".rst", ".text"};
    for (auto& entry : fs::recursive_directory_iterator(dir_path)) {
      if (!entry.is_regular_file()) continue;
      auto ext = entry.path().extension().string();
      bool match = extensions.empty();
      for (auto& e : extensions) {
        if (ext == e) { match = true; break; }
      }
      if (!match) continue;
      if (file_count >= max_files) break;

      std::string content = read_file(entry.path().string());
      if (content.empty()) continue;

      std::cout << "[" << (file_count + 1) << "] " << entry.path().string()
                << " (" << content.size() << " bytes)\n";
      inspect_text(tok, content, false, &stats);
      stats.files_processed++;
      file_count++;
    }
    std::cout << "\nProcessed " << file_count << " files\n";
    print_stats(stats, tok);
  }

  return 0;
}
