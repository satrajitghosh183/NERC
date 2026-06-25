#pragma once

/**
 * include/olmo_cpp/amp.hpp
 *
 * Lightweight Automatic Mixed Precision (AMP) helper plus a placeholder for
 * activation checkpointing. AMP wraps PyTorch's `at::autocast::*` API in a
 * scoped helper so call sites can run a forward in BF16 without manually
 * managing the global enable/disable state. The activation-checkpoint
 * helper currently just runs the function inline; a real implementation
 * would re-run forward inside a custom autograd backward to trade compute
 * for memory.
 *
 * Note: in practice the full training loop in `src/train.cpp` uses the
 * dedicated `AutocastGuard` from `train/autocast_guard.hpp` (which handles
 * CUDA-vs-CPU device dispatch correctly across PyTorch 2.x point
 * releases). `AMPContext` here is the simpler/older API, retained for
 * potential reuse and as a lower-friction primitive for unit tests.
 *
 * --- Includes from this project ---
 *   (none — pure torch/stdlib glue; the `at::autocast` header is pulled in
 *    by `<ATen/autocast_mode.h>`)
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - Direct callers not located via quick grep. Production training in
 *     src/train.cpp uses train/autocast_guard.hpp instead.
 *
 * --- Role in training pipeline ---
 *   Header-only utility. The optional `AMPContext::run(fn)` wraps a forward
 *   pass so individual ops execute in BF16 (or FP16) while master weights
 *   stay FP32 — the standard "autocast" recipe. Side effects are limited to
 *   toggling the global AMP enable flag for the duration of `fn`.
 */

#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <functional>

namespace olmo_cpp {

/// Mixed precision (BF16) training with autocast.
/// Uses at::autocast::set_autocast_enabled for CPU; CUDA uses device-specific API.
///
/// Pattern: construct once with the desired on/off flag, then call
/// `ctx.run([&]{ return model->forward(...); })`. Inside `fn`, eligible
/// ops (matmul, conv, attention) run in BF16; ops sensitive to dynamic
/// range (softmax, norms, loss) stay FP32. Master weights are unchanged.
class AMPContext {
 public:
  /// `enabled=false` makes `run()` a pass-through; useful so call sites
  /// can pipe `cfg.use_amp` through unconditionally.
  explicit AMPContext(bool enabled = true) : enabled_(enabled) {}

  /// Run fn under autocast if enabled. Enables autocast, runs fn, restores.
  template <typename F>
  auto run(F&& fn) -> decltype(fn()) {
    // Fast path: no autocast wanted, run directly with no flag toggling.
    if (!enabled_) return fn();
    // NB: `device` here is hard-coded to CPU; the real production guard in
    // train/autocast_guard.hpp dispatches on torch::Device for CUDA/MPS.
    auto device = at::DeviceType::CPU;  // Use CUDA when available
    // Save the previous AMP state so we can restore it (RAII-by-hand).
    bool prev = at::autocast::is_autocast_enabled(device);
    // Enter autocast: subsequent ops on `device` run in their AMP dtype.
    at::autocast::set_autocast_enabled(device, true);
    // Run user code (typically model->forward(...) returning a Tensor).
    auto result = fn();
    // Restore prior state — important if `fn` was nested under another AMP scope.
    at::autocast::set_autocast_enabled(device, prev);
    return result;
  }

  /// Whether this context will actually enable AMP when `run()` is called.
  bool enabled() const { return enabled_; }

 private:
  /// Master flag controlling whether `run()` toggles AMP at all.
  bool enabled_;
};

/// Activation checkpointing: recompute forward in backward to save memory.
/// Placeholder - when enabled, just runs fn (full implementation needs custom autograd).
///
/// Real activation checkpointing would: (a) discard intermediate
/// activations of `fn` during forward, (b) register a custom autograd
/// node, (c) re-execute `fn` during backward to materialise the
/// activations needed by gradient computation. Saves ~O(L) activation
/// memory at the cost of one extra forward per checkpointed segment.
inline torch::Tensor checkpoint(
    const std::function<torch::Tensor()>& fn,
    const std::vector<torch::Tensor>& /*inputs*/) {
  // Placeholder: just runs `fn` eagerly. The `inputs` argument is unused
  // here but is part of the intended signature for the real implementation
  // (it would tie those tensors into the recomputed graph).
  return fn();
}

}  // namespace olmo_cpp
