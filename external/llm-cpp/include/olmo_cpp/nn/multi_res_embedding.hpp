#pragma once

/**
 * include/olmo_cpp/nn/multi_res_embedding.hpp
 *
 * Declares the Dual-Codebook Multi-Resolution Embedding (DC-MRE), a drop-in
 * replacement for nn::Embedding that fuses four orthogonal sources of token
 * information into a single d_model vector via additive composition:
 *
 *   1. Semantic       — standard learned token embedding.
 *   2. Syntactic role — a tiny codebook indexed by a precomputed token->role map
 *                       (WORD/PROPER/NUMBER_LIKE/PUNCT/CONTINUATION/STRUCTURE/...).
 *   3. Morphology     — character trigram hashes pooled per token (catches
 *                       prefixes/suffixes/roots without growing vocab).
 *   4. Phrase context — local 3-gram hash over (left, center, right) token IDs.
 *
 * Streams 2-4 use low-rank embeddings projected up to d_model, so the added
 * parameter count and FLOPs are small. Lookup tables are built once at
 * construction time from the BPE vocab and cached as buffers.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: nn::Module, nn::Embedding, nn::Linear, TORCH_MODULE.
 *   - <string>: BPE vocab path argument.
 *   - <vector>: included for transitive use (no direct std::vector here).
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/nn/multi_res_embedding.cpp: implementation of every method.
 *   - src/model/transformer.cpp / fused_transformer.cpp: TransformerImpl can
 *     instantiate MultiResEmbedding instead of plain nn::Embedding when
 *     enabled by config (see TransformerConfig::use_multi_res_embedding).
 *
 * --- Role in training pipeline ---
 *   First module touched by every forward pass: token IDs go in, hidden
 *   states come out before the first attention block. Replacing the standard
 *   embedding with DC-MRE is one of the structural changes that lets us
 *   exploit the StructuralTokenizer's role-aware vocab partitioning.
 */

#include <torch/torch.h>
#include <string>
#include <vector>

namespace olmo_cpp {

/// Configuration for dual-codebook multi-resolution embedding (DC-MRE).
/// Enhances the structural tokenizer's structure/content separation
/// by propagating it into the embedding space with:
///   1. Semantic embedding   — standard token lookup
///   2. Syntactic embedding  — role-based (structure vs content vs identifier vs numeric)
///   3. Morphological embedding — character trigram features (captures prefixes/suffixes/roots)
///   4. Phrase context embedding — local 3-gram hash (captures syntactic context)
///
/// All four streams are summed (additive fusion) — zero concat overhead.
/// All lookups are batched tensor ops — fully GPU-saturating.
struct MultiResConfig {
  // Dual codebook: syntactic roles derived from structural tokenizer vocab ranges
  int64_t num_syntactic_roles = 10;  // WORD, PROPER, NUMBER_LIKE, PUNCT, CONTINUATION,
                                     // STRUCTURE, IDENTIFIER, NUMERIC, DOMAIN, OTHER
  int64_t role_embed_dim = 64;       // low-rank → projected up to d_model

  // Multi-resolution: character trigrams (morphological)
  bool enable_char_trigrams = true;
  int64_t char_trigram_buckets = 4096;   // hash buckets for char trigram embedding
  int64_t max_trigrams_per_token = 10;   // max trigrams extracted per BPE token
  int64_t char_embed_dim = 64;          // low-rank → projected up to d_model

  // Multi-resolution: phrase context (local syntactic)
  bool enable_phrase_context = true;
  int64_t phrase_buckets = 8192;     // hash buckets for phrase trigram embedding
  int64_t phrase_embed_dim = 64;     // low-rank → projected up to d_model

  // Structural tokenizer vocab ranges (must match StructuralTokenizer)
  int64_t bpe_end = 50000;
  int64_t pattern_base = 50000;
  int64_t pattern_end = 55000;
  int64_t ident_base = 55000;
  int64_t ident_end = 58000;
  int64_t numeric_base = 58000;
  int64_t numeric_end = 60000;
  int64_t domain_base = 60000;
  int64_t domain_end = 61000;
};

/// Syntactic role IDs for the dual codebook.
enum class SyntacticRole : int64_t {
  WORD = 0,         // lowercase BPE token (English words)
  PROPER_NOUN = 1,  // starts with uppercase
  NUMBER_LIKE = 2,  // contains digits
  PUNCT = 3,        // punctuation only
  CONTINUATION = 4, // whitespace-prefixed (BPE continuation)
  STRUCTURE = 5,    // structural template (50000-54999)
  IDENTIFIER = 6,   // identifier atom (55000-57999)
  NUMERIC = 7,      // numeric atom (58000-59999)
  DOMAIN_PATTERN = 8,  // domain pattern (60000-60999)
  OTHER = 9         // fallback
};

/// Dual-Codebook Multi-Resolution Embedding.
///
/// Drop-in replacement for nn::Embedding. Takes token IDs → [B, S, d_model].
/// Internally computes four additive streams:
///   e_final = e_semantic + proj_role(e_role) + proj_char(pool(e_char)) + proj_phrase(e_phrase)
///
/// All precomputed maps (role_map, char_trigram_map) are registered as buffers
/// so they move with .to(device) and live on GPU alongside model weights.
class MultiResEmbeddingImpl : public torch::nn::Module {
 public:
  /// Construct with vocab size, embedding dim, config, and BPE vocab path
  /// (required for char trigram extraction and syntactic role classification).
  MultiResEmbeddingImpl(int64_t vocab_size, int64_t d_model,
                        const MultiResConfig& config = {},
                        const std::string& bpe_vocab_path = "");

  /// Forward: token_ids [B, S] → embeddings [B, S, d_model]
  torch::Tensor forward(torch::Tensor token_ids);

  /// Access the underlying semantic embedding weight (for init_weights, tying, etc.)
  torch::Tensor& semantic_weight() { return token_embed_->weight; }

  int64_t vocab_size() const { return vocab_size_; }
  int64_t d_model() const { return d_model_; }

 private:
  // Build precomputed maps at construction time
  void build_role_map();
  void build_char_trigram_map(const std::string& bpe_vocab_path);

  int64_t vocab_size_;
  int64_t d_model_;
  MultiResConfig config_;

  // Stream 1: Semantic (standard token embedding)
  torch::nn::Embedding token_embed_{nullptr};

  // Stream 2: Dual codebook — syntactic role
  torch::nn::Embedding role_embed_{nullptr};   // [num_roles, role_dim]
  torch::nn::Linear role_proj_{nullptr};       // [role_dim, d_model]
  // Buffer: vocab_size → role_id mapping (precomputed from vocab ranges)
  torch::Tensor role_map_;  // [vocab_size] int64

  // Stream 3: Character trigrams (morphological features)
  torch::nn::Embedding char_embed_{nullptr};   // [char_buckets, char_dim]
  torch::nn::Linear char_proj_{nullptr};       // [char_dim, d_model]
  // Buffers: precomputed char trigram indices and mask
  torch::Tensor char_trigram_map_;   // [vocab_size, max_trigrams] int64
  torch::Tensor char_trigram_count_; // [vocab_size] float — number of valid trigrams
  // Precomputed [1, 1, T] range used to build the trigram-count mask in forward.
  // Registered as a buffer so it moves with the module and never reallocates.
  torch::Tensor trigram_range_;

  // Stream 4: Phrase context (local syntactic features)
  torch::nn::Embedding phrase_embed_{nullptr}; // [phrase_buckets, phrase_dim]
  torch::nn::Linear phrase_proj_{nullptr};     // [phrase_dim, d_model]
};

TORCH_MODULE(MultiResEmbedding);

}  // namespace olmo_cpp
