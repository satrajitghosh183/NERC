/**
 * include/olmo_cpp/backend/topp_radix.hpp
 *
 * Bucket-radix top-p (nucleus) sampling — fast-inference roadmap [7].
 *
 * Replaces the O(V log V) sort + cumsum that the CPU sample_logits path
 * uses for top-p with an O(V) histogram-based kernel:
 *
 *   1. Bin probabilities by log-magnitude into 16 fixed buckets:
 *        bucket(p) = clamp(floor(-log2(p)), 0, 15)
 *      bucket 0 = [0.5, 1.0], bucket 15 = [2^-16, 2^-15], tail dropped.
 *   2. Per-block partial histogram in shared memory; reduce to a 16-entry
 *      global histogram of (bucket_count, bucket_mass).
 *   3. Host scans the 16 entries to find cutoff bucket (cumulative mass
 *      first to reach top_p). One trip over a 16-int array — basically free.
 *   4. Mask kernel: zero out tokens with bucket_idx > cutoff. Tokens INSIDE
 *      the cutoff bucket are kept (approximation — within 2x prob range
 *      of the true cutoff). Optionally refine by sorting just that bucket
 *      if exactness matters.
 *
 * Skipping the long tail: anything with prob < 2^-16 ≈ 1.5e-5 contributes
 * essentially zero mass and is dropped without binning. Typically removes
 * 70-90% of the vocab outright.
 *
 * On a CPU: a reference implementation matching the same algorithm so
 * unit tests can run anywhere.
 *
 * NOT YET CALLED FROM chat.cpp. Once validated by unit tests, replaces
 * the sort+cumsum block in sample_logits.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Apply bucket-radix top-p filtering to a probability vector.
/// Modifies `probs` in place: tokens outside the nucleus are zeroed,
/// remaining mass is renormalized to sum to 1.
///
/// Inputs:
///   - probs: [V] tensor of non-negative probabilities (post-softmax).
///            float32. Modified in place.
///   - top_p: cutoff in (0, 1]. top_p >= 1.0 is a no-op.
///   - min_p: probabilities below this are zeroed without bucketing
///            (long-tail cutoff). Default 1.0e-5 (~2^-16). Pass 0.0 to
///            disable.
///
/// Dispatches to CUDA kernel if probs.is_cuda(), otherwise CPU reference.
/// CPU reference matches the kernel's bucketing semantics exactly so
/// outputs are bit-identical (modulo float-add ordering in the histogram).
void topp_radix_filter(torch::Tensor probs, float top_p, float min_p = 1.0e-5f);

/// CPU reference implementation. Exposed for testing / fallback.
/// Matches the CUDA kernel's behavior bit-for-bit.
void topp_radix_filter_cpu(torch::Tensor probs, float top_p, float min_p);

#ifdef OLMO_HAS_CUDA_KERNELS
/// CUDA kernel launcher. Caller must guarantee probs.is_cuda().
/// Defined in kernels/topp_radix.cu when CUDA is available.
void topp_radix_filter_cuda(torch::Tensor probs, float top_p, float min_p);
#endif

}  // namespace olmo_cpp
