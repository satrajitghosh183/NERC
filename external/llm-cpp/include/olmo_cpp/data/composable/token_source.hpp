#pragma once

/**
 * include/olmo_cpp/data/composable/token_source.hpp
 *
 * Second stage of the composable data pipeline: a TokenSource produces a
 * flat stream of token ids with no notion of document boundaries — what
 * concat-and-chunk pretraining wants to consume. Concrete implementations
 * adapt a DocumentSource (DocumentTokenSource), restrict to a contiguous
 * range (SlicedTokenSource — used for train/val splits), or interleave
 * multiple TokenSources weighted by ratio (MixingTokenSource).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/document_source.hpp: DocumentTokenSource
 *     wraps a DocumentSource.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/instance_source.cpp: ConcatAndChunkInstanceSource
 *     pulls tokens from a TokenSource to fill its rolling buffer.
 *
 * --- Role in training pipeline ---
 *   The "linearize" step between document-boundary-aware sources and the
 *   chunk-into-seq_len logic. total_tokens() lets schedulers compute
 *   epoch length without consuming the stream.
 */

#include "olmo_cpp/data/composable/document_source.hpp"
#include <vector>
#include <memory>

namespace olmo_cpp {

/// Abstract token source: produces streams of tokens (no document boundaries).
/// total_tokens() may return -1 to indicate "unknown / streaming".
class TokenSource {
 public:
  virtual ~TokenSource() = default;
  virtual bool has_next() const = 0;
  virtual int64_t next() = 0;
  virtual void reset() = 0;
  virtual int64_t total_tokens() const = 0;
};

/// Flattens documents into a token stream. Pulls one Document at a time
/// from the upstream DocumentSource and yields its tokens in order.
class DocumentTokenSource : public TokenSource {
 public:
  explicit DocumentTokenSource(std::unique_ptr<DocumentSource> doc_source);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::unique_ptr<DocumentSource> doc_source_;
  Document current_doc_;       // active document being drained
  size_t token_cursor_ = 0;    // offset within current_doc_.tokens
  bool doc_loaded_ = false;    // false until the first next() call lands a doc
  int64_t total_tokens_;       // -1 = unknown (we don't pre-scan)
};

/// Sliced token source: takes a slice [start, end) of tokens. Fast-forwards
/// past the first `start` tokens at construction (and at every reset()).
/// Useful for carving an explicit train/val split out of a single source.
class SlicedTokenSource : public TokenSource {
 public:
  SlicedTokenSource(std::unique_ptr<TokenSource> source, int64_t start, int64_t end);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::unique_ptr<TokenSource> source_;
  int64_t start_, end_, pos_ = 0;  // pos_ counts ABSOLUTE position in source
};

/// Mixing token source: interleaves tokens from multiple sources by ratio.
/// Each next() call samples a source via the precomputed CDF and pulls one
/// token from it. Note this mixes at *token* granularity — patterns like
/// "stay in the same source for a few tokens then switch" are not modelled.
class MixingTokenSource : public TokenSource {
 public:
  MixingTokenSource(std::vector<std::unique_ptr<TokenSource>> sources,
                    std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  int64_t next() override;
  void reset() override;
  int64_t total_tokens() const override;
 private:
  std::vector<std::unique_ptr<TokenSource>> sources_;
  std::vector<double> cumulative_ratios_;
  std::mt19937 rng_;
  int64_t seed_;
};

}  // namespace olmo_cpp
