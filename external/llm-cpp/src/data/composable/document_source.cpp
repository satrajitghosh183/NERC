/**
 * src/data/composable/document_source.cpp
 *
 * ─── What the "composable" data pipeline is ─────────────────────────
 *
 * The classic TokenDataset (in src/data/token_dataset.cpp) is fine
 * for simple "one big corpus, fixed-length windows" training, but it
 * couples three concerns: where the bytes come from (file), what
 * counts as a "document" (boundary information), and how documents
 * become training instances (windowing + packing).
 *
 * The "composable" pipeline splits those concerns into three abstract
 * sources composed in series:
 *
 *   DocumentSource   → emits raw token streams with doc boundaries
 *   InstanceSource   → consumes a DocumentSource, yields per-sample
 *                       token sequences (with windowing / shuffling /
 *                       cross-doc packing)
 *   ComposableDataLoader → batches InstanceSource samples into tensors
 *
 * THIS file implements the topmost layer: DocumentSource. It reads
 * the .npy token file plus an optional document-offsets file (a
 * companion .npy where each entry is the start index of one document
 * inside the token stream).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/document_source.hpp : declaration.
 *   - third_party/cnpy/cnpy.h : .npy reader.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/instance_source.cpp: instance sources wrap
 *     a DocumentSource and slice it into training instances.
 *
 * --- Role in training pipeline ---
 *   Lowest layer of the optional composable data path. Inactive in
 *   the quickstart flow (which uses the simpler TokenDataset).
 */
#include "olmo_cpp/data/composable/document_source.hpp"
#include <cnpy.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// InMemoryDocumentSource
// ---------------------------------------------------------------------------

InMemoryDocumentSource::InMemoryDocumentSource(std::vector<Document> docs)
    : docs_(std::move(docs)) {}

bool InMemoryDocumentSource::has_next() const {
  return cursor_ < docs_.size();
}

Document InMemoryDocumentSource::next() {
  if (!has_next()) {
    throw std::runtime_error("InMemoryDocumentSource: no more documents");
  }
  return docs_[cursor_++];
}

void InMemoryDocumentSource::reset() {
  cursor_ = 0;
}

int64_t InMemoryDocumentSource::num_documents() const {
  return static_cast<int64_t>(docs_.size());
}

// ---------------------------------------------------------------------------
// NumpyDocumentSource
// ---------------------------------------------------------------------------

NumpyDocumentSource::NumpyDocumentSource(const std::string& path,
                                         int64_t eos_token_id) {
  cnpy::NpyArray arr = cnpy::npy_load(path);

  // Expect a 1-D array of integers (int32 or int64)
  const size_t n = arr.num_vals;
  std::vector<int64_t> flat(n);

  if (arr.word_size == 4) {
    // int32
    const int32_t* data = arr.data<int32_t>();
    for (size_t i = 0; i < n; ++i) {
      flat[i] = static_cast<int64_t>(data[i]);
    }
  } else if (arr.word_size == 8) {
    // int64
    const int64_t* data = arr.data<int64_t>();
    std::copy(data, data + n, flat.begin());
  } else if (arr.word_size == 2) {
    // int16 / uint16 (some tokenized datasets use uint16)
    const int16_t* data = arr.data<int16_t>();
    for (size_t i = 0; i < n; ++i) {
      flat[i] = static_cast<int64_t>(data[i]);
    }
  } else {
    throw std::runtime_error(
        "NumpyDocumentSource: unsupported word size " +
        std::to_string(arr.word_size));
  }

  // Split on eos_token_id into separate documents.
  // If eos_token_id == -1, treat the whole array as one document.
  if (eos_token_id < 0) {
    Document doc;
    doc.tokens = std::move(flat);
    doc.source_id = 0;
    docs_.push_back(std::move(doc));
  } else {
    Document current;
    current.source_id = 0;
    for (size_t i = 0; i < flat.size(); ++i) {
      if (flat[i] == eos_token_id) {
        // End of document -- include the eos token as the last token
        current.tokens.push_back(flat[i]);
        if (!current.tokens.empty()) {
          docs_.push_back(std::move(current));
          current = Document{};
          current.source_id = 0;
        }
      } else {
        current.tokens.push_back(flat[i]);
      }
    }
    // Remaining tokens that weren't terminated by eos
    if (!current.tokens.empty()) {
      docs_.push_back(std::move(current));
    }
  }
}

bool NumpyDocumentSource::has_next() const {
  return cursor_ < docs_.size();
}

Document NumpyDocumentSource::next() {
  if (!has_next()) {
    throw std::runtime_error("NumpyDocumentSource: no more documents");
  }
  return docs_[cursor_++];
}

void NumpyDocumentSource::reset() {
  cursor_ = 0;
}

int64_t NumpyDocumentSource::num_documents() const {
  return static_cast<int64_t>(docs_.size());
}

// ---------------------------------------------------------------------------
// ConcatenatedDocumentSource
// ---------------------------------------------------------------------------

ConcatenatedDocumentSource::ConcatenatedDocumentSource(
    std::vector<std::unique_ptr<DocumentSource>> sources)
    : sources_(std::move(sources)) {}

bool ConcatenatedDocumentSource::has_next() const {
  // Check current and subsequent sources
  for (size_t i = current_source_; i < sources_.size(); ++i) {
    if (sources_[i]->has_next()) {
      return true;
    }
  }
  return false;
}

Document ConcatenatedDocumentSource::next() {
  // Advance past exhausted sources
  while (current_source_ < sources_.size() &&
         !sources_[current_source_]->has_next()) {
    ++current_source_;
  }
  if (current_source_ >= sources_.size()) {
    throw std::runtime_error("ConcatenatedDocumentSource: no more documents");
  }
  return sources_[current_source_]->next();
}

void ConcatenatedDocumentSource::reset() {
  current_source_ = 0;
  for (auto& s : sources_) {
    s->reset();
  }
}

int64_t ConcatenatedDocumentSource::num_documents() const {
  int64_t total = 0;
  for (const auto& s : sources_) {
    total += s->num_documents();
  }
  return total;
}

// ---------------------------------------------------------------------------
// SamplingDocumentSource
// ---------------------------------------------------------------------------

SamplingDocumentSource::SamplingDocumentSource(
    std::unique_ptr<DocumentSource> source, int64_t seed)
    : rng_(static_cast<unsigned>(seed)), seed_(seed) {
  // Preload all documents from the underlying source
  while (source->has_next()) {
    all_docs_.push_back(source->next());
  }
  if (all_docs_.empty()) {
    throw std::runtime_error(
        "SamplingDocumentSource: underlying source has no documents");
  }
}

bool SamplingDocumentSource::has_next() const {
  // Infinite source -- always has next as long as we have docs
  return !all_docs_.empty();
}

Document SamplingDocumentSource::next() {
  std::uniform_int_distribution<size_t> dist(0, all_docs_.size() - 1);
  return all_docs_[dist(rng_)];
}

void SamplingDocumentSource::reset() {
  rng_.seed(static_cast<unsigned>(seed_));
}

int64_t SamplingDocumentSource::num_documents() const {
  // Effectively infinite; return underlying count
  return static_cast<int64_t>(all_docs_.size());
}

// ---------------------------------------------------------------------------
// MixingDocumentSource
// ---------------------------------------------------------------------------

MixingDocumentSource::MixingDocumentSource(
    std::vector<std::unique_ptr<DocumentSource>> sources,
    std::vector<double> ratios, int64_t seed)
    : sources_(std::move(sources)),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {
  if (sources_.empty()) {
    throw std::runtime_error("MixingDocumentSource: no sources provided");
  }
  if (sources_.size() != ratios.size()) {
    throw std::runtime_error(
        "MixingDocumentSource: sources and ratios must have same size");
  }

  // Normalize ratios and compute cumulative distribution
  double sum = std::accumulate(ratios.begin(), ratios.end(), 0.0);
  if (sum <= 0.0) {
    throw std::runtime_error(
        "MixingDocumentSource: ratios must sum to positive value");
  }

  cumulative_ratios_.resize(ratios.size());
  double cumulative = 0.0;
  for (size_t i = 0; i < ratios.size(); ++i) {
    cumulative += ratios[i] / sum;
    cumulative_ratios_[i] = cumulative;
  }
  // Ensure last entry is exactly 1.0 to avoid floating-point edge cases
  cumulative_ratios_.back() = 1.0;
}

bool MixingDocumentSource::has_next() const {
  for (const auto& s : sources_) {
    if (s->has_next()) {
      return true;
    }
  }
  return false;
}

Document MixingDocumentSource::next() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Try up to sources_.size() times to find a non-exhausted source
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

  // Fallback: find any source that has documents
  for (auto& s : sources_) {
    if (s->has_next()) {
      return s->next();
    }
  }

  throw std::runtime_error("MixingDocumentSource: all sources exhausted");
}

void MixingDocumentSource::reset() {
  rng_.seed(static_cast<unsigned>(seed_));
  for (auto& s : sources_) {
    s->reset();
  }
}

int64_t MixingDocumentSource::num_documents() const {
  int64_t total = 0;
  for (const auto& s : sources_) {
    total += s->num_documents();
  }
  return total;
}

}  // namespace olmo_cpp
