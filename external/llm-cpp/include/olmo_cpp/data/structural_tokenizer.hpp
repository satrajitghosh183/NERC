#pragma once

/**
 * include/olmo_cpp/data/structural_tokenizer.hpp
 *
 * Public interface for the *Structural* tokenizer — an experimental
 * tokenizer that segments code/prose into (1) structural pattern templates,
 * (2) identifier atoms (camelCase / snake_case decomposed), (3) numeric
 * atoms, and (4) a BPE fallback. It builds a 61002-entry vocabulary on top
 * of stock GPT-2 BPE so the BPE id space is preserved unchanged for
 * downstream model embedding sharing.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/bpe_tokenizer.hpp: embedded BPETokenizer used for the
 *     fallback path and for the [0, 50000) id range.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - tools/mine_patterns.cpp: trains the patterns the tokenizer consumes.
 *   - tools/inspect_tokens.cpp / tools/benchmark_tokenizer.cpp: stats &
 *     side-by-side comparisons against pure BPE.
 *
 * --- Role in training pipeline ---
 *   Optional alternative to BPETokenizer at prepare_data time. Aim is to
 *   compress code-heavy corpora more aggressively (more bytes per token)
 *   without breaking the model's existing BPE-id embedding table.
 */

#include "olmo_cpp/data/bpe_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

/**
 * Structural Tokenizer — a novel tokenizer that separates structural patterns
 * from content, achieving better compression than pure BPE for code and prose.
 *
 * Vocabulary layout (61002 tokens total):
 *   [0, 49999]     — BPE fallback tokens (standard GPT-2 subwords)
 *   [50000, 54999]  — Structure templates (code loops, conditionals, functions, prose frames)
 *   [55000, 57999]  — Identifier atoms (i, j, len, self, model, etc.)
 *   [58000, 59999]  — Numeric literal atoms (digits, common constants)
 *   [60000, 60999]  — Domain patterns (JSON, XML, CSV)
 *   61001           — EOS
 *
 * Encode pipeline (4 stages):
 *   1. Domain detection (heuristic: code vs prose vs data)
 *   2. Structural pattern matching (greedy longest-match → template token + slot contents)
 *   3. Identifier/numeric atom encoding (camelCase/snake_case split → atom IDs)
 *   4. BPE fallback (anything unmatched goes through standard BPE)
 */
class StructuralTokenizer {
 public:
  StructuralTokenizer() = default;

  /// Load from config directory containing:
  ///   - config.json (vocabulary ranges, settings)
  ///   - mined_patterns.json or seed_patterns.json (structural patterns)
  ///   - identifiers.json (identifier atoms)
  ///   - BPE vocab.json + merges.txt (fallback)
  bool load(const std::string& config_dir,
            const std::string& bpe_vocab_path,
            const std::string& bpe_merges_path);

  /// Encode text to token IDs. Appends EOS at end.
  std::vector<uint32_t> encode(const std::string& text);

  /// Decode token IDs back to text.
  std::string decode(const std::vector<uint32_t>& ids);

  /// Total vocabulary size (61002).
  uint32_t vocab_size() const { return total_vocab_size_; }

  /// EOS token ID (61001).
  uint32_t eos_id() const { return eos_id_; }

  /// Decode a single token to its string representation.
  std::string decode_token(uint32_t id) const;

  /// Get compression stats from last encode call.
  struct EncodeStats {
    int pattern_tokens = 0;   // tokens emitted as structural patterns
    int atom_tokens = 0;      // tokens emitted as identifier/numeric atoms
    int bpe_tokens = 0;       // tokens emitted via BPE fallback
    int total_tokens = 0;
    int input_bytes = 0;
    double bytes_per_token() const {
      return total_tokens > 0 ? static_cast<double>(input_bytes) / total_tokens : 0;
    }
  };
  const EncodeStats& last_stats() const { return last_stats_; }

 private:
  // ─── Vocabulary ranges ───
  static constexpr uint32_t BPE_BASE        = 0;
  static constexpr uint32_t BPE_END         = 50000;
  static constexpr uint32_t PATTERN_BASE    = 50000;
  static constexpr uint32_t PATTERN_END     = 55000;
  static constexpr uint32_t IDENT_BASE      = 55000;
  static constexpr uint32_t IDENT_END       = 58000;
  static constexpr uint32_t NUMERIC_BASE    = 58000;
  static constexpr uint32_t NUMERIC_END     = 60000;
  static constexpr uint32_t DOMAIN_BASE     = 60000;
  static constexpr uint32_t DOMAIN_END      = 61000;
  static constexpr uint32_t EOS_ID          = 61001;
  static constexpr uint32_t TOTAL_VOCAB     = 61002;

  // ─── Pattern matching ───
  struct Pattern {
    uint32_t id;
    std::string name;
    std::string pattern;            // template with {SLOT} placeholders
    std::vector<std::string> slots; // slot types (IDENT, EXPR, TYPE, etc.)
    std::string domain;
    // Compiled matching data
    std::vector<std::string> literal_parts; // text between slots
    int priority = 0;                       // longer patterns match first
  };

  // ─── Domain detection ───
  enum class Domain { CODE, PROSE, DATA };
  Domain detect_domain(const std::string& text) const;

  // ─── Encode stages ───
  // Stage 1: Try to match structural patterns on a line
  bool try_pattern_match(const std::string& line, std::vector<uint32_t>& out);

  // Stage 2: Try to encode a word as an identifier atom
  bool try_atom_encode(const std::string& word, std::vector<uint32_t>& out);

  // Stage 3: Try to encode a number as a numeric atom
  bool try_numeric_encode(const std::string& num_str, std::vector<uint32_t>& out);

  // Stage 4: BPE fallback
  void bpe_fallback(const std::string& text, std::vector<uint32_t>& out);

  // Split camelCase/snake_case identifiers into atoms
  std::vector<std::string> split_identifier(const std::string& ident) const;

  // Compile a pattern string into literal_parts for matching
  void compile_pattern(Pattern& p);

  // Try to match a compiled pattern against text starting at pos
  // Returns end position if matched, or std::string::npos if not
  size_t match_pattern(const Pattern& p, const std::string& text, size_t pos,
                       std::vector<std::string>& slot_values) const;

  // ─── Data ───
  BPETokenizer bpe_;                              // fallback tokenizer
  std::vector<Pattern> patterns_;                  // all structural patterns
  std::unordered_map<std::string, uint32_t> atom_map_;     // identifier → atom ID
  std::vector<std::string> atom_names_;                     // atom ID → identifier
  std::unordered_map<std::string, uint32_t> numeric_map_;  // number string → numeric ID
  std::vector<std::string> numeric_names_;                  // numeric ID → number string

  // Reverse decode maps
  std::unordered_map<uint32_t, std::string> id_to_string_; // for all custom tokens

  uint32_t total_vocab_size_ = TOTAL_VOCAB;
  uint32_t eos_id_ = EOS_ID;

  EncodeStats last_stats_;
};

}  // namespace olmo_cpp
