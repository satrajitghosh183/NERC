/**
 * src/nn/multi_res_embedding.cpp
 *
 * ─── What an "embedding" is ────────────────────────────────────────
 *
 * Every transformer starts the forward pass by mapping each input
 * token id (an integer) to a continuous d_model-wide vector. The
 * canonical way is a single big lookup table:
 *
 *     class Embedding(vocab_size, d_model):
 *       weight: [vocab_size, d_model]      // learned parameters
 *       def forward(ids):                  // ids: [B, S]
 *         return weight[ids]               // result: [B, S, d_model]
 *
 * That's `torch::nn::Embedding`, used in src/model/transformer.cpp
 * when cfg.use_multi_res is false.
 *
 * ─── What DC-MRE is (this file) ────────────────────────────────────
 *
 * "DC-MRE" = **D**ual-**C**odebook **M**ulti-**R**esolution **E**mbedding.
 * The motivation: in a 50k-vocabulary BPE, 90% of the tokens occur very
 * rarely.  Their rows in the plain Embedding table never get enough
 * gradient signal to learn anything useful.  DC-MRE addresses this by
 * decomposing each token's representation into THREE parallel codebook
 * lookups whose results are summed:
 *
 *   semantic_emb[id]            (the conventional row, vocab_size × D)
 *   + char_emb[trigram_bucket]  (bucketed by sub-token trigrams,
 *                                 ~few-thousand buckets — heavy parameter
 *                                 sharing across rare tokens)
 *   + phrase_emb[phrase_bucket] (bucketed at the super-token level)
 *
 * Rare tokens still benefit because most of their gradient lands in
 * the (shared) char- and phrase-buckets, which other tokens with
 * similar substrings or contexts are also updating.
 *
 * The trigram and phrase bucket lookups are precomputed at construction
 * time from the BPE vocabulary file (hence the include of
 * bpe_tokenizer.hpp) so that the forward pass is just three index_select
 * calls + a sum — no string-processing on the hot path.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/nn/multi_res_embedding.hpp : MultiResEmbedding declaration.
 *   - olmo_cpp/data/bpe_tokenizer.hpp     : used at construction to
 *     enumerate the token strings so we can compute char trigrams.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp: when cfg.use_multi_res, the Transformer
 *     replaces torch::nn::Embedding with this MultiResEmbedding.
 *   - tools/dump_embeddings.cpp: extracts the .semantic sub-codebook
 *     when --multi_res is in effect (because the "semantic" stream is
 *     the closest analogue to a classical embedding row).
 *
 * --- Role in training pipeline ---
 *   First op in the forward pass when DC-MRE is enabled. Output shape
 *   matches a plain Embedding: [B, S, d_model]. Constructed once at
 *   model init.
 */
#include "olmo_cpp/nn/multi_res_embedding.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace olmo_cpp {

// ── djb2 hash for character trigrams ──────────────────────────────────
static inline uint32_t djb2_trigram(char a, char b, char c) {
  uint32_t h = 5381;
  h = h * 33 + static_cast<uint8_t>(a);
  h = h * 33 + static_cast<uint8_t>(b);
  h = h * 33 + static_cast<uint8_t>(c);
  return h;
}

// ── Classify a BPE token string into a syntactic role ────────────────
static SyntacticRole classify_bpe_token(const std::string& tok) {
  if (tok.empty()) return SyntacticRole::OTHER;

  // Check for whitespace prefix (BPE continuation token — starts with Ġ or space)
  bool has_space_prefix = false;
  size_t start = 0;
  if (tok.size() >= 2 && static_cast<uint8_t>(tok[0]) == 0xC4 &&
      static_cast<uint8_t>(tok[1]) == 0xA0) {
    // UTF-8 Ġ (U+0120) — GPT-2's representation of leading space
    has_space_prefix = true;
    start = 2;
  } else if (tok[0] == ' ') {
    has_space_prefix = true;
    start = 1;
  }

  // After stripping prefix, analyze the content
  std::string content = tok.substr(start);
  if (content.empty()) return SyntacticRole::CONTINUATION;

  bool all_alpha = true;
  bool all_lower = true;
  bool has_digit = false;
  bool has_alpha = false;
  bool starts_upper = false;
  bool all_punct = true;

  for (size_t i = 0; i < content.size(); ++i) {
    char c = content[i];
    if (std::isalpha(static_cast<unsigned char>(c))) {
      has_alpha = true;
      all_punct = false;
      if (i == 0 && std::isupper(static_cast<unsigned char>(c))) starts_upper = true;
      if (!std::islower(static_cast<unsigned char>(c))) all_lower = false;
    } else if (std::isdigit(static_cast<unsigned char>(c))) {
      has_digit = true;
      all_alpha = false;
      all_punct = false;
    } else {
      all_alpha = false;
      all_lower = false;
    }
  }

  if (has_digit) return SyntacticRole::NUMBER_LIKE;
  if (all_punct) return SyntacticRole::PUNCT;
  if (has_space_prefix && has_alpha) return SyntacticRole::CONTINUATION;
  if (starts_upper && has_alpha) return SyntacticRole::PROPER_NOUN;
  if (all_alpha && all_lower) return SyntacticRole::WORD;
  return SyntacticRole::OTHER;
}

// ─────────────────────────────────────────────────────────────────────

MultiResEmbeddingImpl::MultiResEmbeddingImpl(
    int64_t vocab_size, int64_t d_model,
    const MultiResConfig& config,
    const std::string& bpe_vocab_path)
    : vocab_size_(vocab_size), d_model_(d_model), config_(config) {

  // Stream 1: Semantic embedding (standard)
  token_embed_ = register_module(
      "token_embed", torch::nn::Embedding(vocab_size, d_model));

  // Stream 2: Dual codebook — syntactic role embedding
  role_embed_ = register_module(
      "role_embed",
      torch::nn::Embedding(config.num_syntactic_roles, config.role_embed_dim));
  role_proj_ = register_module(
      "role_proj",
      torch::nn::Linear(torch::nn::LinearOptions(config.role_embed_dim, d_model).bias(false)));
  build_role_map();

  // Stream 3: Character trigrams (morphological)
  if (config.enable_char_trigrams) {
    char_embed_ = register_module(
        "char_embed",
        torch::nn::Embedding(config.char_trigram_buckets, config.char_embed_dim));
    char_proj_ = register_module(
        "char_proj",
        torch::nn::Linear(torch::nn::LinearOptions(config.char_embed_dim, d_model).bias(false)));
    build_char_trigram_map(bpe_vocab_path);
    // [1, 1, T] int64 — shared mask template. Registered as a buffer so the
    // module's .to(device) call places it on the correct device; forward can
    // then use it directly without per-call allocation.
    trigram_range_ = register_buffer(
        "trigram_range",
        torch::arange(config.max_trigrams_per_token, torch::kLong)
            .unsqueeze(0).unsqueeze(0));
  }

  // Stream 4: Phrase context
  if (config.enable_phrase_context) {
    phrase_embed_ = register_module(
        "phrase_embed",
        torch::nn::Embedding(config.phrase_buckets, config.phrase_embed_dim));
    phrase_proj_ = register_module(
        "phrase_proj",
        torch::nn::Linear(torch::nn::LinearOptions(config.phrase_embed_dim, d_model).bias(false)));
  }
}

void MultiResEmbeddingImpl::build_role_map() {
  // Precompute: for each token ID, assign a syntactic role based on vocab range.
  // This is a [vocab_size] int64 tensor registered as a buffer.
  auto map = torch::full({vocab_size_}, static_cast<int64_t>(SyntacticRole::OTHER),
                         torch::kLong);
  auto map_acc = map.accessor<int64_t, 1>();

  for (int64_t id = 0; id < vocab_size_; ++id) {
    if (id < config_.bpe_end) {
      // BPE range — will be refined by char analysis if vocab is available.
      // Default to WORD for now; build_char_trigram_map refines these.
      map_acc[id] = static_cast<int64_t>(SyntacticRole::WORD);
    } else if (id >= config_.pattern_base && id < config_.pattern_end) {
      map_acc[id] = static_cast<int64_t>(SyntacticRole::STRUCTURE);
    } else if (id >= config_.ident_base && id < config_.ident_end) {
      map_acc[id] = static_cast<int64_t>(SyntacticRole::IDENTIFIER);
    } else if (id >= config_.numeric_base && id < config_.numeric_end) {
      map_acc[id] = static_cast<int64_t>(SyntacticRole::NUMERIC);
    } else if (id >= config_.domain_base && id < config_.domain_end) {
      map_acc[id] = static_cast<int64_t>(SyntacticRole::DOMAIN_PATTERN);
    }
  }

  role_map_ = register_buffer("role_map", map);
}

void MultiResEmbeddingImpl::build_char_trigram_map(const std::string& bpe_vocab_path) {
  int64_t T = config_.max_trigrams_per_token;
  auto tri_map = torch::zeros({vocab_size_, T}, torch::kLong);
  auto tri_count = torch::zeros({vocab_size_}, torch::kFloat);
  auto map_acc = tri_map.accessor<int64_t, 2>();
  auto count_acc = tri_count.accessor<float, 1>();

  // Also refine BPE role assignments if vocab is available
  auto role_acc = role_map_.accessor<int64_t, 1>();

  if (!bpe_vocab_path.empty()) {
    // Load BPE vocabulary to get token strings.
    // The load() function needs both vocab and merges paths.
    // Infer merges path from vocab path: same directory, merges.txt
    BPETokenizer bpe;
    std::string merges_path;
    {
      auto slash = bpe_vocab_path.rfind('/');
      if (slash != std::string::npos) {
        merges_path = bpe_vocab_path.substr(0, slash + 1) + "merges.txt";
      } else {
        merges_path = "merges.txt";
      }
    }
    bool loaded = bpe.load(bpe_vocab_path, merges_path);

    if (loaded) {
      int64_t bpe_vocab = std::min(vocab_size_, static_cast<int64_t>(bpe.vocab_size()));
      for (int64_t id = 0; id < bpe_vocab; ++id) {
        std::string tok_str = bpe.decode_token(static_cast<uint32_t>(id));
        if (tok_str.empty()) continue;

        // Refine syntactic role for BPE tokens
        role_acc[id] = static_cast<int64_t>(classify_bpe_token(tok_str));

        // Extract character trigrams
        int64_t t = 0;
        for (size_t i = 0; i + 2 < tok_str.size() && t < T; ++i) {
          uint32_t h = djb2_trigram(tok_str[i], tok_str[i + 1], tok_str[i + 2]);
          map_acc[id][t] = static_cast<int64_t>(h % config_.char_trigram_buckets);
          t++;
        }
        // For short tokens (1-2 chars), use char-level hashes
        if (t == 0) {
          uint32_t h = 5381;
          for (char c : tok_str) h = h * 33 + static_cast<uint8_t>(c);
          map_acc[id][0] = static_cast<int64_t>(h % config_.char_trigram_buckets);
          t = 1;
        }
        count_acc[id] = static_cast<float>(t);
      }

      std::cout << "MultiRes: loaded BPE vocab (" << bpe_vocab
                << " tokens), computed char trigrams and syntactic roles\n";
    } else {
      throw std::runtime_error(
          "MultiRes: failed to load BPE vocab from " + bpe_vocab_path +
          " — check that vocab.json and merges.txt exist");
    }
  } else {
    throw std::runtime_error(
        "MultiRes: bpe_vocab_path is required when char trigrams are enabled");
  }

  char_trigram_map_ = register_buffer("char_trigram_map", tri_map);
  char_trigram_count_ = register_buffer("char_trigram_count", tri_count);
}

torch::Tensor MultiResEmbeddingImpl::forward(torch::Tensor token_ids) {
  // token_ids: [B, S] int64
  auto B = token_ids.size(0);
  auto S = token_ids.size(1);

  // ── Stream 1: Semantic embedding ──
  auto e = token_embed_->forward(token_ids);  // [B, S, d_model]

  // ── Stream 2: Dual codebook — syntactic role ──
  {
    // role_map_ is [vocab_size] on same device as model (moves via buffer)
    // Gather role IDs for each token: [B, S]
    auto flat_ids = token_ids.reshape(-1);                    // [B*S]
    auto role_ids = role_map_.index_select(0, flat_ids)
                        .reshape({B, S});                     // [B, S]
    auto e_role = role_embed_->forward(role_ids);             // [B, S, role_dim]
    e = e + role_proj_->forward(e_role);                      // [B, S, d_model]
  }

  // ── Stream 3: Character trigrams (morphological) ──
  if (config_.enable_char_trigrams && char_embed_) {
    int64_t T = config_.max_trigrams_per_token;
    auto flat_ids = token_ids.reshape(-1);                    // [B*S]

    // Gather precomputed trigram indices: [B*S, T] → [B, S, T]
    auto tri_ids = char_trigram_map_.index_select(0, flat_ids)
                       .reshape({B, S, T});                   // [B, S, T]

    // Lookup char embeddings: [B, S, T, char_dim]
    auto e_chars = char_embed_->forward(tri_ids);

    // Gather counts and cast to the activation dtype so the final division
    // stays in that precision. If counts remains FP32 while e_chars is BF16,
    // `e_char / safe_count` promotes e_char to FP32, and char_proj_ (whose
    // weight is BF16 in pure-BF16 mode) then sees an FP32 input and errors.
    auto counts = char_trigram_count_.index_select(0, flat_ids)
                      .reshape({B, S}).to(e_chars.dtype());   // [B, S]

    // Mask is built from the precomputed [1, 1, T] range buffer and the
    // per-token counts; comparison is bool, cast to the activation dtype.
    (void)T;
    auto mask = (trigram_range_ < counts.unsqueeze(-1))
                    .to(e_chars.dtype()).unsqueeze(-1);       // [B, S, T, 1]

    // Masked mean pooling: sum valid trigram embeddings / count
    auto e_char = (e_chars * mask).sum(2);                    // [B, S, char_dim]
    auto safe_count = counts.clamp_min(1.0).unsqueeze(-1);    // [B, S, 1]
    e_char = e_char / safe_count;

    e = e + char_proj_->forward(e_char);                      // [B, S, d_model]
  }

  // ── Stream 4: Phrase context (local syntactic) ──
  if (config_.enable_phrase_context && phrase_embed_) {
    // Hash local context: (token[i-1] * P1 + token[i] * P2 + token[i+1] * P3) % phrase_buckets.
    // token_ids is already int64 so we skip the redundant dtype cast.
    auto left = torch::roll(token_ids, 1, 1);                // token[i-1]
    auto right = torch::roll(token_ids, -1, 1);              // token[i+1]

    // Pure arithmetic — maps to a single CUDA kernel
    auto phrase_hash = (left * 6291469 + token_ids * 12582917 + right * 25165843)
                           % config_.phrase_buckets;
    phrase_hash = phrase_hash.abs();                           // ensure non-negative

    auto e_phrase = phrase_embed_->forward(phrase_hash);       // [B, S, phrase_dim]
    e = e + phrase_proj_->forward(e_phrase);                   // [B, S, d_model]
  }

  return e;
}

}  // namespace olmo_cpp
