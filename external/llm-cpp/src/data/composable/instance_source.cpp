/**
 * src/data/composable/instance_source.cpp
 *
 * Middle layer of the composable data pipeline (see
 * document_source.cpp's docblock for the full architecture). An
 * InstanceSource consumes a DocumentSource and emits per-sample
 * sequences ready to be batched, applying:
 *
 *   - windowing            (slice each document into seq_len chunks),
 *   - cross-document packing (stitch short docs together to fill
 *                              a window — saves padding),
 *   - per-epoch shuffling.
 *
 * Several variants implement different windowing policies (fixed
 * window, document-respecting, span-corrupt for masked-LM, etc.).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/instance_source.hpp : declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/data_loader.cpp: ComposableDataLoader wraps
 *     an InstanceSource and produces batched tensors.
 *
 * --- Role in training pipeline ---
 *   Active only in the composable data path. Inactive in the
 *   quickstart's flow.
 */
#include "olmo_cpp/data/composable/instance_source.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// ConcatAndChunkInstanceSource
// ---------------------------------------------------------------------------

ConcatAndChunkInstanceSource::ConcatAndChunkInstanceSource(
    std::unique_ptr<TokenSource> source, int64_t seq_len)
    : source_(std::move(source)), seq_len_(seq_len) {
  if (seq_len <= 0) {
    throw std::runtime_error(
        "ConcatAndChunkInstanceSource: seq_len must be positive");
  }
}

bool ConcatAndChunkInstanceSource::has_next() const {
  // We need seq_len + 1 tokens to produce one instance (input + 1 shifted label)
  // Check buffer size plus whether source has more
  return (static_cast<int64_t>(buffer_.size()) >= seq_len_ + 1) ||
         source_->has_next();
}

Instance ConcatAndChunkInstanceSource::next() {
  // Fill buffer until we have at least seq_len + 1 tokens
  while (static_cast<int64_t>(buffer_.size()) < seq_len_ + 1 &&
         source_->has_next()) {
    buffer_.push_back(source_->next());
  }

  if (static_cast<int64_t>(buffer_.size()) < seq_len_ + 1) {
    throw std::runtime_error(
        "ConcatAndChunkInstanceSource: not enough tokens for an instance");
  }

  Instance inst;
  inst.input_ids.assign(buffer_.begin(), buffer_.begin() + seq_len_);
  inst.labels.assign(buffer_.begin() + 1, buffer_.begin() + seq_len_ + 1);

  // Remove the consumed tokens (keep the last token as overlap for next chunk)
  buffer_.erase(buffer_.begin(), buffer_.begin() + seq_len_);

  return inst;
}

void ConcatAndChunkInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
}

// ---------------------------------------------------------------------------
// PackingInstanceSource
// ---------------------------------------------------------------------------

PackingInstanceSource::PackingInstanceSource(
    std::unique_ptr<DocumentSource> source, int64_t seq_len,
    int64_t pad_token_id, int64_t eos_token_id)
    : source_(std::move(source)),
      seq_len_(seq_len),
      pad_token_id_(pad_token_id),
      eos_token_id_(eos_token_id) {
  if (seq_len <= 0) {
    throw std::runtime_error(
        "PackingInstanceSource: seq_len must be positive");
  }
}

bool PackingInstanceSource::has_next() const {
  return !buffer_.empty() || source_->has_next();
}

Instance PackingInstanceSource::next() {
  // Pack documents into buffer until we have at least seq_len + 1 tokens
  while (static_cast<int64_t>(buffer_.size()) < seq_len_ + 1 &&
         source_->has_next()) {
    Document doc = source_->next();
    if (doc.tokens.empty()) continue;

    // If buffer is non-empty and we have an eos_token_id, add separator
    if (!buffer_.empty() && eos_token_id_ >= 0) {
      buffer_.push_back(eos_token_id_);
    }

    buffer_.insert(buffer_.end(), doc.tokens.begin(), doc.tokens.end());
  }

  if (buffer_.empty()) {
    throw std::runtime_error("PackingInstanceSource: no more instances");
  }

  Instance inst;

  if (static_cast<int64_t>(buffer_.size()) >= seq_len_ + 1) {
    // Enough tokens: take seq_len + 1 tokens
    inst.input_ids.assign(buffer_.begin(), buffer_.begin() + seq_len_);
    inst.labels.assign(buffer_.begin() + 1, buffer_.begin() + seq_len_ + 1);
    buffer_.erase(buffer_.begin(), buffer_.begin() + seq_len_);
  } else {
    // Not enough tokens: pad the remainder
    int64_t available = static_cast<int64_t>(buffer_.size());
    // input_ids: available - 1 real tokens + padding
    // labels: available - 1 real tokens (shifted) + -100 for padding
    int64_t real_len = available - 1;
    if (real_len <= 0) {
      // Only 1 or 0 tokens, not enough for input/label pair; pad fully
      inst.input_ids.resize(seq_len_, pad_token_id_);
      inst.labels.resize(seq_len_, -100);
      buffer_.clear();
      return inst;
    }

    inst.input_ids.assign(buffer_.begin(), buffer_.begin() + real_len);
    inst.labels.assign(buffer_.begin() + 1, buffer_.begin() + real_len + 1);

    // Pad to seq_len
    while (static_cast<int64_t>(inst.input_ids.size()) < seq_len_) {
      inst.input_ids.push_back(pad_token_id_);
      inst.labels.push_back(-100);  // ignore index
    }

    buffer_.clear();
  }

  return inst;
}

void PackingInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
}

// ---------------------------------------------------------------------------
// RandomInstanceSource
// ---------------------------------------------------------------------------

RandomInstanceSource::RandomInstanceSource(int64_t seq_len, int64_t vocab_size,
                                           int64_t num_instances, int64_t seed)
    : seq_len_(seq_len),
      vocab_size_(vocab_size),
      num_instances_(num_instances),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {}

bool RandomInstanceSource::has_next() const {
  return cursor_ < num_instances_;
}

Instance RandomInstanceSource::next() {
  if (!has_next()) {
    throw std::runtime_error("RandomInstanceSource: no more instances");
  }

  std::uniform_int_distribution<int64_t> dist(0, vocab_size_ - 1);

  Instance inst;
  // Generate seq_len + 1 tokens, then split into input/labels
  std::vector<int64_t> tokens(seq_len_ + 1);
  for (auto& t : tokens) {
    t = dist(rng_);
  }
  inst.input_ids.assign(tokens.begin(), tokens.begin() + seq_len_);
  inst.labels.assign(tokens.begin() + 1, tokens.begin() + seq_len_ + 1);

  ++cursor_;
  return inst;
}

void RandomInstanceSource::reset() {
  cursor_ = 0;
  rng_.seed(static_cast<unsigned>(seed_));
}

// ---------------------------------------------------------------------------
// MixingInstanceSource
// ---------------------------------------------------------------------------

MixingInstanceSource::MixingInstanceSource(
    std::vector<std::unique_ptr<InstanceSource>> sources,
    std::vector<double> ratios, int64_t seed)
    : sources_(std::move(sources)),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {
  if (sources_.empty()) {
    throw std::runtime_error("MixingInstanceSource: no sources provided");
  }
  if (sources_.size() != ratios.size()) {
    throw std::runtime_error(
        "MixingInstanceSource: sources and ratios size mismatch");
  }

  double sum = std::accumulate(ratios.begin(), ratios.end(), 0.0);
  if (sum <= 0.0) {
    throw std::runtime_error(
        "MixingInstanceSource: ratios must sum to positive value");
  }

  cumulative_ratios_.resize(ratios.size());
  double cumulative = 0.0;
  for (size_t i = 0; i < ratios.size(); ++i) {
    cumulative += ratios[i] / sum;
    cumulative_ratios_[i] = cumulative;
  }
  cumulative_ratios_.back() = 1.0;
}

bool MixingInstanceSource::has_next() const {
  for (const auto& s : sources_) {
    if (s->has_next()) {
      return true;
    }
  }
  return false;
}

Instance MixingInstanceSource::next() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);

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

  // Fallback
  for (auto& s : sources_) {
    if (s->has_next()) {
      return s->next();
    }
  }

  throw std::runtime_error("MixingInstanceSource: all sources exhausted");
}

void MixingInstanceSource::reset() {
  rng_.seed(static_cast<unsigned>(seed_));
  for (auto& s : sources_) {
    s->reset();
  }
}

// ---------------------------------------------------------------------------
// ShuffledInstanceSource
// ---------------------------------------------------------------------------

ShuffledInstanceSource::ShuffledInstanceSource(
    std::unique_ptr<InstanceSource> source, int64_t buffer_size, int64_t seed)
    : source_(std::move(source)),
      buffer_size_(buffer_size),
      rng_(static_cast<unsigned>(seed)),
      seed_(seed) {}

bool ShuffledInstanceSource::has_next() const {
  if (filled_ && cursor_ < buffer_.size()) {
    return true;
  }
  return source_->has_next();
}

Instance ShuffledInstanceSource::next() {
  // If buffer is consumed or not yet filled, refill it
  if (!filled_ || cursor_ >= buffer_.size()) {
    buffer_.clear();
    cursor_ = 0;

    // Fill buffer from source
    while (static_cast<int64_t>(buffer_.size()) < buffer_size_ &&
           source_->has_next()) {
      buffer_.push_back(source_->next());
    }

    if (buffer_.empty()) {
      throw std::runtime_error("ShuffledInstanceSource: no more instances");
    }

    // Shuffle the buffer
    std::shuffle(buffer_.begin(), buffer_.end(), rng_);
    filled_ = true;
  }

  return buffer_[cursor_++];
}

void ShuffledInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
  cursor_ = 0;
  filled_ = false;
  rng_.seed(static_cast<unsigned>(seed_));
}

}  // namespace olmo_cpp
