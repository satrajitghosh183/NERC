/**
 * src/data/vsl_dataset.cpp
 *
 * ─── What "VSL" stands for ──────────────────────────────────────────
 *
 * VSL = **V**ariable-**S**equence-**L**ength.  Plain TokenDataset
 * always returns sequences of exactly `seq_len` tokens by chopping
 * the corpus into uniform chunks.  That wastes capacity when natural
 * documents are shorter than seq_len (you pad) or longer (you split
 * across artificial boundaries that destroy mid-document context).
 *
 * VslDataset reads document boundaries from the corpus and produces
 * variable-length sequences that respect document structure, then
 * packs them into mini-batches via a length-bucketing scheme.
 * Sequences of similar length are batched together so padding is
 * minimal.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/vsl_dataset.hpp : VslDataset declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: alternative to TokenDataset for runs that need
 *     document-aware sequence packing. Off by default.
 *
 * --- Role in training pipeline ---
 *   Used only by configurations that want document-respecting
 *   training. The quickstart's TinyStories run uses TokenDataset.
 */
#include "olmo_cpp/data/vsl_dataset.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>

namespace olmo_cpp {

VSLInstanceSource::VSLInstanceSource(std::unique_ptr<DocumentSource> source,
                                     int64_t min_seq_len, int64_t max_seq_len,
                                     int64_t warmup_steps,
                                     VSLCurriculum curriculum,
                                     int64_t pad_token_id)
    : source_(std::move(source)),
      min_seq_len_(min_seq_len),
      max_seq_len_(max_seq_len),
      warmup_steps_(warmup_steps),
      curriculum_(curriculum),
      pad_token_id_(pad_token_id) {
  if (min_seq_len <= 0 || max_seq_len <= 0 || min_seq_len > max_seq_len) {
    throw std::runtime_error(
        "VSLInstanceSource: invalid seq_len range [" +
        std::to_string(min_seq_len) + ", " + std::to_string(max_seq_len) + "]");
  }
  if (warmup_steps <= 0) {
    throw std::runtime_error(
        "VSLInstanceSource: warmup_steps must be positive");
  }
  current_seq_len_ = compute_seq_len(0);
}

int64_t VSLInstanceSource::compute_seq_len(int64_t step) const {
  if (step >= warmup_steps_) {
    return max_seq_len_;
  }

  double progress = static_cast<double>(step) / static_cast<double>(warmup_steps_);
  // Clamp progress to [0, 1]
  progress = std::max(0.0, std::min(1.0, progress));

  double range = static_cast<double>(max_seq_len_ - min_seq_len_);

  switch (curriculum_) {
    case VSLCurriculum::Natural: {
      // Natural: random uniform in [min_seq_len, current_max]
      // where current_max = min + range * progress (linear growth of upper bound)
      int64_t current_max = min_seq_len_ +
          static_cast<int64_t>(range * progress);
      current_max = std::max(current_max, min_seq_len_);
      // For deterministic behavior based on step, use step as seed
      std::mt19937 step_rng(static_cast<unsigned>(step * 7919 + 31));
      std::uniform_int_distribution<int64_t> dist(min_seq_len_, current_max);
      return dist(step_rng);
    }
    case VSLCurriculum::Linear: {
      // Linear: min + (max - min) * step / warmup
      int64_t len = min_seq_len_ + static_cast<int64_t>(range * progress);
      return std::max(min_seq_len_, std::min(len, max_seq_len_));
    }
    case VSLCurriculum::Quadratic: {
      // Quadratic: min + (max - min) * (step / warmup)^2
      int64_t len = min_seq_len_ +
          static_cast<int64_t>(range * progress * progress);
      return std::max(min_seq_len_, std::min(len, max_seq_len_));
    }
    default:
      return max_seq_len_;
  }
}

void VSLInstanceSource::set_step(int64_t step) {
  current_step_ = step;
  current_seq_len_ = compute_seq_len(step);
}

bool VSLInstanceSource::has_next() const {
  return (buffer_.size() > head_) || source_->has_next();
}

Instance VSLInstanceSource::next() {
  // Recompute sequence length for current step
  current_seq_len_ = compute_seq_len(current_step_);
  ++current_step_;

  // Fill buffer from documents until we have enough unconsumed tokens.
  // Compaction step: when head_ has eaten more than half the buffer and
  // we're about to grow further, slide the live window to the front so the
  // buffer doesn't grow unbounded.
  auto live = [this]() -> int64_t {
    return static_cast<int64_t>(buffer_.size() - head_);
  };
  while (live() < current_seq_len_ + 1 && source_->has_next()) {
    if (head_ > 0 && head_ * 2 >= buffer_.size()) {
      // Compact: move live window to the front and drop the consumed prefix.
      buffer_.erase(buffer_.begin(),
                    buffer_.begin() + static_cast<std::ptrdiff_t>(head_));
      head_ = 0;
    }
    Document doc = source_->next();
    buffer_.insert(buffer_.end(), doc.tokens.begin(), doc.tokens.end());
  }

  Instance inst;

  if (live() >= current_seq_len_ + 1) {
    // Enough tokens: create input/labels pair from [head_, head_ + seq_len].
    inst.input_ids.assign(buffer_.begin() + static_cast<std::ptrdiff_t>(head_),
                          buffer_.begin() + static_cast<std::ptrdiff_t>(head_) + current_seq_len_);
    inst.labels.assign(buffer_.begin() + static_cast<std::ptrdiff_t>(head_) + 1,
                       buffer_.begin() + static_cast<std::ptrdiff_t>(head_) + current_seq_len_ + 1);
    head_ += static_cast<size_t>(current_seq_len_);
  } else if (live() > 0) {
    // Not enough tokens: use what we have and pad. Read from the live
    // window starting at head_, then drain the buffer.
    const int64_t available = live();
    const int64_t real_len  = available - 1;

    if (real_len <= 0) {
      // Not enough for even one input/label pair.
      inst.input_ids.resize(current_seq_len_, pad_token_id_);
      inst.labels.resize(current_seq_len_, -100);
      buffer_.clear();
      head_ = 0;
      return inst;
    }

    const auto begin = buffer_.begin() + static_cast<std::ptrdiff_t>(head_);
    inst.input_ids.assign(begin, begin + real_len);
    inst.labels.assign(begin + 1, begin + real_len + 1);

    // Pad to current_seq_len_
    while (static_cast<int64_t>(inst.input_ids.size()) < current_seq_len_) {
      inst.input_ids.push_back(pad_token_id_);
      inst.labels.push_back(-100);
    }

    buffer_.clear();
    head_ = 0;
  } else {
    throw std::runtime_error("VSLInstanceSource: no more tokens available");
  }

  return inst;
}

void VSLInstanceSource::reset() {
  source_->reset();
  buffer_.clear();
  current_step_ = 0;
  current_seq_len_ = compute_seq_len(0);
}

}  // namespace olmo_cpp
