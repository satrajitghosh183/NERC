#pragma once

#include <algorithm>
#include <cmath>

namespace zwt::optim {

// Cosine learning-rate schedule with linear warmup.
//
// Phase 1 (step < warmup_steps): linear warmup from 0 -> peak_lr.
// Phase 2 (warmup <= step < max_steps): cosine decay peak_lr -> min_lr.
// Phase 3 (step >= max_steps): held at min_lr.
struct CosineSchedule {
  float peak_lr      = 3e-4f;
  float min_lr       = 3e-5f;
  int64_t warmup_steps = 2000;
  int64_t max_steps    = 100000;

  float lr_at(int64_t step) const {
    if (step < warmup_steps) {
      return peak_lr * (float(step + 1) / float(warmup_steps));
    }
    if (step >= max_steps) return min_lr;
    double progress =
        double(step - warmup_steps) / double(max_steps - warmup_steps);
    progress = std::clamp(progress, 0.0, 1.0);
    double cos_val = 0.5 * (1.0 + std::cos(progress * 3.14159265358979323846));
    return float(min_lr + (peak_lr - min_lr) * cos_val);
  }
};

}  // namespace zwt::optim
