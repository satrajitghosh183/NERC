#pragma once

/**
 * include/olmo_cpp/data/simple_tokenizer.hpp
 *
 * Header-only word/punctuation-level tokenizer for tiny test corpora and
 * local experiments. Lower-cases the input, splits on whitespace and a
 * fixed punctuation set, and assigns ids on the fly (open vocabulary —
 * unknown tokens get added to vocab_ rather than being mapped to <unk>).
 *
 * --- Includes from this project ---
 *   - (none — STL only). Kept header-only so the non-load_vocab/save_vocab
 *     paths can be inlined into tools without linking the data library.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. Used historically by early
 *   smoke tests and tutorials; production training uses BPETokenizer.
 *
 * --- Role in training pipeline ---
 *   Acts as a zero-config tokenizer for unit tests and quick experiments
 *   where the GPT-2 BPE files are unavailable. Not used on the OLMo
 *   pretraining critical path.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace olmo_cpp {

/// Simple tokenizer: whitespace + punctuation split. Builds vocab from corpus.
/// Use for local text when GPT-2 compatibility is not required.
///
/// Reserves ids 0..2 for special tokens (<pad>, <eos>, <unk>) so that user
/// vocabulary starts at id 3. Open-vocabulary: encode() will mint new ids
/// for unseen words (so train and eval encoders MUST share a SimpleTokenizer
/// instance or a saved vocab file via save_vocab/load_vocab).
class SimpleTokenizer {
 public:
  static constexpr uint32_t kPadId = 0;        // padding for batches
  static constexpr uint32_t kEosId = 1;        // end-of-sequence sentinel
  static constexpr uint32_t kUnkId = 2;        // unknown (rarely used because we auto-add)
  static constexpr uint32_t kFirstUserId = 3;  // first id available to real vocab

  /// Constructs the tokenizer with the three reserved special tokens
  /// pre-populated; next_id_ is advanced past them so vocab grows from 3.
  SimpleTokenizer() : next_id_(kFirstUserId) {
    vocab_["<pad>"] = kPadId;
    vocab_["<eos>"] = kEosId;
    vocab_["<unk>"] = kUnkId;
  }

  /// Tokenize text and return token IDs. Adds <eos> at end.
  std::vector<uint32_t> encode(const std::string& text) {
    std::vector<uint32_t> ids;
    // Case-fold the entire string up-front; SimpleTokenizer is case-insensitive.
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Walk one past the end so the trailing token (if any) is flushed at i==len.
    std::string token;
    for (size_t i = 0; i <= lower.size(); ++i) {
      char c = (i < lower.size()) ? lower[i] : '\0';
      // Boundary characters: whitespace, sentinel, or punctuation. They
      // close the current token but are themselves *not* emitted as ids
      // (cf. BPE which preserves them).
      if (std::isspace(static_cast<unsigned char>(c)) || c == '\0' ||
          is_punct(c)) {
        if (!token.empty()) {
          ids.push_back(get_or_add(token));
          token.clear();
        }
        if (c == '\0') break;
      } else {
        token += c;
      }
    }
    ids.push_back(kEosId);  // terminator so model learns sequence boundaries
    return ids;
  }

  /// Tokenize text and append to output vector (no trailing EOS).
  void encode_append(const std::string& text, std::vector<uint32_t>& out) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string token;
    for (size_t i = 0; i <= lower.size(); ++i) {
      char c = (i < lower.size()) ? lower[i] : '\0';
      if (std::isspace(static_cast<unsigned char>(c)) || c == '\0' ||
          is_punct(c)) {
        if (!token.empty()) {
          out.push_back(get_or_add(token));
          token.clear();
        }
        if (c == '\0') break;
      } else {
        token += c;
      }
    }
  }

  /// Total number of distinct tokens (including the 3 specials).
  uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }

  /// Load vocab from file: one token per line, id = line number (0-based).
  /// Implemented in src/data/simple_tokenizer.cpp.
  bool load_vocab(const std::string& path);

  /// Save vocab to file for inspection. Tokens are written in id order so
  /// load_vocab reads back the same id assignments.
  bool save_vocab(const std::string& path) const;

 private:
  /// ASCII punctuation set treated as token boundaries. Note this is
  /// deliberately small; characters like '_' are kept inside tokens.
  static bool is_punct(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' ||
           c == '"' || c == '\'' || c == '(' || c == ')' || c == '-' || c == '/';
  }

  /// Look up `token` in vocab_; if not present, assign next_id_ and bump.
  /// This is the "open-vocabulary" path that grows the tokenizer at use time.
  uint32_t get_or_add(const std::string& token) {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) return it->second;
    uint32_t id = next_id_++;
    vocab_[token] = id;
    return id;
  }

  std::unordered_map<std::string, uint32_t> vocab_;  // token -> id
  uint32_t next_id_;                                 // next free id (monotone)
};

}  // namespace olmo_cpp
