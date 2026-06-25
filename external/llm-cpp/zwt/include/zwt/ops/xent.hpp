#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// Fused softmax + negative-log-likelihood + mean reduction, in one kernel.
// Forward:
//   logits:  [N, V]   (bf16 or f32)
//   targets: [N]      (i64, -100 = ignore)
//   loss:    [1]      (f32)     — mean over non-ignored positions
//   grad_logits_out: [N, V]     — optionally produced here to fuse backward
//                                  into the same pass. Pass nullptr if you
//                                  want to compute backward later.
//
// This eliminates the "compute softmax → write V×N tensor → read again for
// backward" traffic pattern that dominates lm_head memory bandwidth.
void cross_entropy(const Tensor& logits, const Tensor& targets,
                   Tensor& loss,
                   Tensor* grad_logits_out = nullptr,
                   int64_t ignore_index = -100);

}  // namespace zwt::ops
