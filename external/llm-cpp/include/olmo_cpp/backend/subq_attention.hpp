#pragma once

/**
 * include/olmo_cpp/backend/subq_attention.hpp
 *
 * Content-selected (SubQ) attention with autograd-aware backward
 * (item HH / 7). See src/backend/subq_backward.cpp for the full
 * STE-based gradient derivation.
 *
 * Forward selects top-k cache positions per query, runs softmax
 * attention over only those, and returns the standard output shape.
 * Backward: gradient flows through the selected positions normally;
 * the discrete top-k itself is treated as identity (STE) so the
 * selection-head's parameters get a sensible signal.
 */

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor subq_attention_autograd(torch::Tensor q,
                                        torch::Tensor k,
                                        torch::Tensor v,
                                        torch::Tensor selection_scores,
                                        int64_t top_k);

}  // namespace olmo_cpp
