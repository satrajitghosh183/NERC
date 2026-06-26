#pragma once

/**
 * include/olmo_cpp/data/bpe_tokenizer.hpp
 *
 * Public interface for a self-contained GPT-2 style Byte-Pair Encoding (BPE)
 * tokenizer. The implementation lives in src/data/bpe_tokenizer.cpp and uses
 * GPT-2's standard two-file format: a JSON vocabulary mapping byte-encoded
 * strings to integer ids (vocab.json) and a text file of ranked merge rules
 * (merges.txt). The tokenizer reproduces GPT-2's byte->unicode mapping so all
 * 256 raw bytes survive a JSON round-trip.
 *
 * --- Includes from this project ---
 *   - (none — only STL); the tokenizer is intentionally dependency-light so
 *     it can be used from prepare_data, chat, inspect_tokens, and the main
 *     training binary without pulling in libtorch.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/structural_tokenizer.cpp: embedded as the BPE-fallback path
 *     when structural pattern / atom encoding fails.
 *   - tools/prepare_data.cpp, tools/chat.cpp, tools/inspect_tokens.cpp:
 *     load vocab.json + merges.txt and call encode/decode.
 *
 * --- Role in training pipeline ---
 *   Converts raw text into the integer token-id stream that prepare_data
 *   serialises into the .npy files consumed by TokenDataset. At inference
 *   time it converts model output ids back to text. Constructed once per
 *   process and used in single-thread mode.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>

namespace olmo_cpp {

/// GPT-2 style BPE tokenizer. Load vocab.json + merges.txt for full compatibility.
/// Includes proper pre-tokenization (GPT-2 regex pattern) so spaces are merged
/// with following words, not left as standalone "B" tokens.
///
/// State after load(): vocab_ (token -> id), id_to_token_ (id -> token),
/// merges_ (ordered list of pair-merges), merge_ranks_ (pair -> priority),
/// bytes_to_unicode_ / unicode_to_byte_ (GPT-2 byte transparency mapping).
class BPETokenizer {
 public:
  BPETokenizer() = default;

  /// Load from GPT-2 format: vocab.json (token->id) and merges.txt (merge pairs).
  /// vocab.json is a flat object mapping {byte-encoded-string -> integer id}.
  /// merges.txt has a header line ("#version: 0.2") followed by space-separated
  /// pairs `a b` where `a + b` is the merged token; line index = merge rank.
  /// Returns false on error (file missing, JSON parse failure, or the build
  /// did not include nlohmann::json — see HAS_NLOHMANN_JSON in the .cpp).
  bool load(const std::string& vocab_path, const std::string& merges_path);

  /// Encode text to token IDs. Adds EOS (50256 for GPT-2) at end.
  std::vector<uint32_t> encode(const std::string& text);

  /// Encode and append to output (no trailing EOS). Use this when packing
  /// many documents into one stream and caller controls separator tokens.
  void encode_append(const std::string& text, std::vector<uint32_t>& out);

  /// Decode token IDs to text. Stops early at EOS. Reverses GPT-2's
  /// byte-encoding step so non-ASCII bytes round-trip exactly.
  std::string decode(const std::vector<uint32_t>& ids);
  std::string decode(const std::vector<int64_t>& ids);

  /// For models trained with the old tokenizer (spaces encoded as "B" id 33).
  /// When true, decode token 33 as space instead of "B". Compatibility shim
  /// for early olmo-cpp checkpoints; leave off for upstream HF GPT-2 weights.
  void set_legacy_decode(bool v) { legacy_decode_ = v; }
  bool legacy_decode() const { return legacy_decode_; }

  /// Number of distinct token strings in vocab_.
  uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }
  /// End-of-text id (50256 for stock GPT-2; loaded from "<|endoftext|>").
  uint32_t eos_id() const { return eos_id_; }

  /// Expose pre-tokenization chunks for inspection (what BPE sees before merging).
  /// Useful from tools/inspect_tokens to debug pre-tokenizer behaviour.
  std::vector<std::string> get_pre_tokenized_chunks(const std::string& text);

  /// Decode a single token ID to its raw string representation. Returns the
  /// byte-encoded form (still in unicode space), unlike decode() which fully
  /// reverses the byte mapping.
  std::string decode_token(uint32_t id) const;

 private:
  /// Build the GPT-2 byte<->unicode bijection (256 distinct printable
  /// codepoints, one per byte). Filled into bytes_to_unicode_ /
  /// unicode_to_byte_ during load().
  void init_bytes_to_unicode();

  /// GPT-2 pre-tokenization: split text into chunks that respect word boundaries.
  /// Each chunk is BPE-encoded independently, preventing cross-word merges.
  /// Implements the ASCII-fast path of the GPT-2 regex; see top of .cpp.
  std::vector<std::string> pre_tokenize(const std::string& text);

  /// Apply BPE merges to a pre-tokenized chunk. Uses a doubly-linked list +
  /// priority queue (rank-ordered) so each merge is O(log N).
  std::vector<uint32_t> bpe_encode_chunk(const std::string& chunk);

  std::unordered_map<std::string, uint32_t> vocab_;       // token string -> id
  std::vector<std::string> id_to_token_;                  // id -> token (for decode)
  std::vector<std::pair<std::string, std::string>> merges_;  // ordered merges
  std::unordered_map<std::string, int> merge_ranks_;      // "a\0b" -> priority rank (lower = higher priority)
  std::unordered_map<std::string, std::string> bytes_to_unicode_;  // 1-byte string -> printable utf-8
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_; // reverse mapping for decode
  uint32_t eos_id_ = 50256;                               // default GPT-2 EOS, overwritten by load()
  bool legacy_decode_ = false;                            // decode token 33 as space (for old-trained checkpoints)

  // Special tokens (e.g. <|im_start|>, <|im_end|>, <|endoftext|>) are matched
  // ATOMICALLY in encode_append — emitted as their single id, never BPE-split —
  // so ChatML templates round-trip. Detected from the vocab during load() (keys
  // of the form <|...|>), sorted longest-first for greedy matching.
  std::vector<std::pair<std::string, uint32_t>> special_tokens_;

 public:
  /// Look up a special token's id by its literal string (e.g. "<|im_start|>").
  /// Returns -1 if absent. Used by chat to set stop tokens / build templates.
  int64_t special_id(const std::string& tok) const {
    auto it = vocab_.find(tok);
    return it == vocab_.end() ? -1 : static_cast<int64_t>(it->second);
  }
};

}  // namespace olmo_cpp
