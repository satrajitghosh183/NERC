#pragma once

/**
 * include/olmo_cpp/data/vsl_dataset.hpp
 *
 * Variable-Sequence-Length (VSL) instance source: a curriculum that grows
 * the per-batch sequence length from min_seq_len up to max_seq_len over
 * warmup_steps training steps. Short sequences early in training cost
 * less FLOPs per step and have been reported to speed up convergence
 * without harming final loss.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/instance_source.hpp: derives from
 *     InstanceSource so it slots into ComposableDataLoader.
 *   - olmo_cpp/data/composable/document_source.hpp: feeds raw documents.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. Wired in optionally by
 *   training scripts that opt into VSL curriculum via config.
 *
 * --- Role in training pipeline ---
 *   Replaces the fixed-length InstanceSource when VSL curriculum is
 *   enabled, controlling effective seq_len per step.
 */

#include "olmo_cpp/data/composable/instance_source.hpp"
#include "olmo_cpp/data/composable/document_source.hpp"
#include <memory>

namespace olmo_cpp {

/// Variable Sequence Length curriculum strategies
///   Natural:   uniform random in [min, current_max] (current_max grows
///              linearly with step). Mixed lengths within an epoch.
///   Linear:    deterministic linear ramp min -> max over warmup.
///   Quadratic: deterministic quadratic ramp (slower start, faster end).
enum class VSLCurriculum { Natural, Linear, Quadratic };

/// VSL instance source: dynamically adjusts sequence length during training
/// by setting a new current_seq_len_ each time set_step() is called.
class VSLInstanceSource : public InstanceSource {
 public:
  /// source: upstream document producer (concatenated tokens get chunked).
  /// min_seq_len / max_seq_len: curriculum bounds (max == final seq_len).
  /// warmup_steps: number of steps over which the curriculum ramps.
  /// curriculum: shape of the schedule (see enum).
  /// pad_token_id: filler when an instance falls short of current_seq_len_.
  VSLInstanceSource(std::unique_ptr<DocumentSource> source,
                    int64_t min_seq_len, int64_t max_seq_len,
                    int64_t warmup_steps, VSLCurriculum curriculum = VSLCurriculum::Linear,
                    int64_t pad_token_id = 0);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
  /// Inform the source which optimization step we're on; recomputes the
  /// scheduled current_seq_len_ from the curriculum.
  void set_step(int64_t step);
  /// Sequence length the next instance will be padded/truncated to.
  int64_t current_seq_len() const { return current_seq_len_; }
 private:
  /// Pure function of `step` and curriculum: returns the seq_len to use.
  int64_t compute_seq_len(int64_t step) const;
  std::unique_ptr<DocumentSource> source_;       // upstream document feed
  int64_t min_seq_len_, max_seq_len_, warmup_steps_;
  VSLCurriculum curriculum_;
  int64_t pad_token_id_;
  int64_t current_step_ = 0;        // monotonically incremented inside next()
  int64_t current_seq_len_;         // length of the next emitted instance
  // Token ring: buffer_ holds raw tokens, head_ points to the next
  // unconsumed token. Consuming advances head_ in O(1). Compaction (slide
  // [head_, end()) to the front) runs only when head_ exceeds half the
  // buffer's size, amortizing the shift to O(1) per consumed token vs the
  // O(n) shift the previous erase()-based code paid every step.
  std::vector<int64_t> buffer_;
  size_t               head_ = 0;
};

}  // namespace olmo_cpp
