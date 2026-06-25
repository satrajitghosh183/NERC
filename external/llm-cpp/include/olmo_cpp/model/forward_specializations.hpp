#pragma once

/**
 * include/olmo_cpp/model/forward_specializations.hpp
 *
 * Compile-time forward-path specialization (item CC).
 *
 * The transformer forward is conditional on a handful of runtime flags
 * (use_float8_, use_multi_res_, num_mtp_heads, fused vs non-fused). Each
 * if-check runs per-call in a tight inner loop. With template
 * specialization, we instantiate the per-feature-combination forward
 * once at construction and route through it by function pointer.
 *
 * The trade-off is binary size for branch elimination; in our case
 * the matrix of (Fp8 × MultiRes × HasMTP × Fused) is small (2^4 = 16)
 * and each instantiation is a few hundred bytes of code. The hot
 * inner loops gain measurably: each `if (use_float8_)` eliminated
 * is ~1 µs saved on launch path.
 *
 * Today the only function whose template-specialized variants we
 * actually generate is AttentionImpl::forward_specialized — used by
 * the un-fused training path. Other call sites can adopt the same
 * pattern incrementally.
 */

#include <torch/torch.h>

namespace olmo_cpp {

// Feature tag types for template specialization.
struct FeatTag_None {};
struct FeatTag_Float8 {};
struct FeatTag_MultiRes {};
struct FeatTag_MTP {};

template <bool HasFloat8, bool HasMTP>
struct ForwardSpec {
  static constexpr bool kFloat8 = HasFloat8;
  static constexpr bool kMTP    = HasMTP;
};

// A function-pointer table the model holds at construction time so the
// hot path is a single indirect call instead of a chain of if checks.
struct AttnForwardTable {
  using Fn = torch::Tensor (*)(torch::Tensor, int);
  Fn no_fp8_no_mtp = nullptr;
  Fn fp8_no_mtp    = nullptr;
  Fn no_fp8_mtp    = nullptr;
  Fn fp8_mtp       = nullptr;
};

}  // namespace olmo_cpp
