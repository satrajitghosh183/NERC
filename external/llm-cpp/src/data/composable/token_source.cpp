/**
 * src/data/composable/token_source.cpp
 *
 * Adapter layer in the composable data pipeline (see
 * document_source.cpp's docblock for the full architecture).
 *
 * A TokenSource is a unified interface for "things that produce a
 * stream of token ids", with concrete implementations for:
 *
 *   - a single .npy file (FileTokenSource),
 *   - a SourceMixture combining several TokenSources at weighted
 *     ratios (so multi-corpus pretraining drops in cleanly),
 *   - a DocumentSource adapter (so the document-aware pipeline can
 *     pretend its docs are just a flat token stream when the
 *     consumer doesn't care about boundaries).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/token_source.hpp : declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/instance_source.cpp: instance sources read
 *     from a TokenSource.
 *
 * --- Role in training pipeline ---
 *   Active only in the composable data path. Inactive in the
 *   quickstart's flow.
 */
#include "olmo_cpp/data/composable/token_source.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// DocumentTokenSource
// ---------------------------------------------------------------------------

DocumentTokenSource::DocumentTokenSource(
    std::unique_ptr<DocumentSource> doc_source)
    : doc_source_(std::move(doc_source)), total_tokens_(0) {
  // Estimate total tokens by summing across all documents.
  // We'll do a full pass to count, then reset.
  // For large datasets this is expensive; callers can override total_tokens().
  // For now, we set total_tokens_ = -1 to indicate unknown and compute lazily
  // if num_documents is available we can estimate.
  // Actually, let's just set to -1 (unknown) and not do a full pass.
  total_tokens_ = -1;
}

bool DocumentTokenSource::has_next() const {
  if (doc_loaded_ && token_cursor_ < current_doc_.tokens.size()) {
    return true;
  }
  return doc_source_->has_next();
}

int64_t DocumentTokenSource::next() {
  // If we need a new document, load one
  while (!doc_loaded_ || token_cursor_ >= current_doc_.tokens.size()) {
    if (!doc_source_->has_next()) {
      throw std::runtime_error("DocumentTokenSource: no more tokens");
    }
    current_doc_ = doc_source_->next();
    token_cursor_ = 0;
    doc_loaded_ = true;

    // Skip empty documents
    if (!current_doc_.tokens.empty()) {
      break;
    }
  }

  if (token_cursor_ >= current_doc_.tokens.size()) {
    throw std::runtime_error("DocumentTokenSource: no more tokens");
  }

  return current_doc_.tokens[token_cursor_++];
}

void DocumentTokenSource::reset() {
  doc_source_->reset();
  current_doc_ = Document{};
  token_cursor_ = 0;
  doc_loaded_ = false;
}

int64_t DocumentTokenSource::total_tokens() const {
  return total_tokens_;
}

// ---------------------------------------------------------------------------
// SlicedTokenSource
// ---------------------------------------------------------------------------

SlicedTokenSource::SlicedTokenSource(std::unique_ptr<TokenSource> source,
                                     int64_t start, int64_t end)
    : source_(std::move(source)), start_(start), end_(end) {
  if (start < 0 || end < start) {
    throw std::runtime_error(
        "SlicedTokenSource: invalid range [" + std::to_string(start) + ", " +
        std::to_string(end) + ")");
  }
  // Skip the first `start` tokens
  for (int64_t i = 0; i < start_ && source_->has_next(); ++i) {
    source_->next();
  }
  pos_ = start_;
}

bool SlicedTokenSource::has_next() const {
  return pos_ < end_ && source_->has_next();
}

int64_t SlicedTokenSource::next() {
  if (!has_next()) {
    throw std::runtime_error("SlicedTokenSource: no more tokens in slice");
  }
  ++pos_;
  return source_->next();
}

void SlicedTokenSource::reset() {
  source_->reset();
  // Re-skip the first `start` tokens
  for (int64_t i = 0; i < start_ && source_->has_next(); ++i) {
    source_->next();
  }
  pos_ = start_;
}

int64_t SlicedTokenSource::total_tokens() const {
  return end_ - start_;
}

// ---------------------------------------------------------------------------
// MixingTokenSource
// ---------------------------------------------------------------------------

MixingTokenSource::MixingTokenSource(
    std::vector<std::unique_ptr<TokenSource>> sources,
    std::vector<double> ratios, int64_t seed)
    : sources_(std::move(sources)),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {
  if (sources_.empty()) {
    throw std::runtime_error("MixingTokenSource: no sources provided");
  }
  if (sources_.size() != ratios.size()) {
    throw std::runtime_error(
        "MixingTokenSource: sources and ratios size mismatch");
  }

  // Normalize and compute cumulative ratios
  double sum = std::accumulate(ratios.begin(), ratios.end(), 0.0);
  if (sum <= 0.0) {
    throw std::runtime_error(
        "MixingTokenSource: ratios must sum to positive value");
  }

  cumulative_ratios_.resize(ratios.size());
  double cumulative = 0.0;
  for (size_t i = 0; i < ratios.size(); ++i) {
    cumulative += ratios[i] / sum;
    cumulative_ratios_[i] = cumulative;
  }
  cumulative_ratios_.back() = 1.0;
}

bool MixingTokenSource::has_next() const {
  for (const auto& s : sources_) {
    if (s->has_next()) {
      return true;
    }
  }
  return false;
}

int64_t MixingTokenSource::next() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Try to pick a source by ratio
  for (size_t attempt = 0; attempt < sources_.size() * 10; ++attempt) {
    double r = dist(rng_);
    size_t idx = 0;
    while (idx < cumulative_ratios_.size() - 1 &&
           r > cumulative_ratios_[idx]) {
      ++idx;
    }

    if (sources_[idx]->has_next()) {
      return sources_[idx]->next();
    }
  }

  // Fallback: any source with tokens
  for (auto& s : sources_) {
    if (s->has_next()) {
      return s->next();
    }
  }

  throw std::runtime_error("MixingTokenSource: all sources exhausted");
}

void MixingTokenSource::reset() {
  rng_.seed(static_cast<unsigned>(seed_));
  for (auto& s : sources_) {
    s->reset();
  }
}

int64_t MixingTokenSource::total_tokens() const {
  int64_t total = 0;
  for (const auto& s : sources_) {
    int64_t t = s->total_tokens();
    if (t < 0) return -1;  // unknown
    total += t;
  }
  return total;
}

}  // namespace olmo_cpp
