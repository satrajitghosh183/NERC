#pragma once
/**
 * include/olmo_cpp/train/autocast_guard.hpp
 *
 * RAII guard around PyTorch's "autocast" mode. Autocast tells ATen
 * to automatically run amp-friendly ops (matmul, conv) in BF16/FP16
 * while keeping the rest of the graph in FP32. Inside the guard's
 * scope, ops dispatch through the autocast key; on scope exit the
 * dispatch key is restored.
 *
 * Header-only because we need the same RAII behaviour from a few
 * different .cpp files (the train loop and activation
 * checkpointing's recompute path), and the API is portable across
 * PyTorch 2.x versions only via this header's compile-time branches.
 *
 * --- Includes from this project ---
 *   (none — torch headers only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp                         : guards each
 *     forward+backward when train_cfg.use_amp is set.
 *   - src/train/activation_checkpoint.cpp   : guards the recompute
 *     pass to keep its precision identical to the original forward.
 *
 * --- Role in training pipeline ---
 *   Mixed-precision plumbing. The quickstart's conf has amp=0 so the
 *   guard is a no-op there.
 */

#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#ifndef __APPLE__
#include <c10/core/impl/LocalDispatchKeySet.h>
#endif
#include <optional>

namespace olmo_cpp {

/// RAII guard that enables BF16 autocast on CUDA.
/// Portable across PyTorch 2.0 through 2.6+.
///
/// API variants detected at CMake configure time:
///   OLMO_AUTOCAST_DEVICE_API  — is_autocast_enabled(at::kCUDA)    (PyTorch >= 2.4)
///   OLMO_AUTOCAST_GPU_API     — is_autocast_gpu_enabled()          (some 2.x builds)
///   OLMO_AUTOCAST_LEGACY_API  — is_enabled() / set_enabled(bool)   (PyTorch 2.0-2.1)
///   (none)                    — dispatch-key fallback
struct AutocastGuard {
  explicit AutocastGuard(bool enabled, torch::Device device)
      : enabled_(enabled && device.is_cuda()) {
    if (enabled_) {
#if defined(OLMO_AUTOCAST_DEVICE_API)
      prev_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
      prev_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::increment_nesting();
#elif defined(OLMO_AUTOCAST_GPU_API)
      prev_enabled_ = at::autocast::is_autocast_gpu_enabled();
      prev_dtype_ = at::autocast::get_autocast_gpu_dtype();
      at::autocast::set_autocast_gpu_enabled(true);
      at::autocast::set_autocast_gpu_dtype(at::kBFloat16);
      at::autocast::increment_nesting();
#elif defined(OLMO_AUTOCAST_LEGACY_API)
      prev_enabled_ = at::autocast::is_enabled();
      prev_dtype_ = at::autocast::get_autocast_gpu_dtype();
      at::autocast::set_enabled(true);
      at::autocast::set_autocast_gpu_dtype(at::kBFloat16);
      at::autocast::increment_nesting();
#else
      dk_guard_.emplace(c10::DispatchKey::AutocastCUDA);
#endif
    }
  }

  ~AutocastGuard() {
    if (enabled_) {
#if defined(OLMO_AUTOCAST_DEVICE_API)
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_autocast_enabled(at::kCUDA, prev_enabled_);
      at::autocast::set_autocast_dtype(at::kCUDA, prev_dtype_);
#elif defined(OLMO_AUTOCAST_GPU_API)
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_autocast_gpu_enabled(prev_enabled_);
      at::autocast::set_autocast_gpu_dtype(prev_dtype_);
#elif defined(OLMO_AUTOCAST_LEGACY_API)
      at::autocast::decrement_nesting();
      at::autocast::clear_cache();
      at::autocast::set_enabled(prev_enabled_);
      at::autocast::set_autocast_gpu_dtype(prev_dtype_);
#else
      // dk_guard_ destructor removes the dispatch key automatically
#endif
    }
  }

  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;

 private:
  bool enabled_;
#if defined(OLMO_AUTOCAST_DEVICE_API) || defined(OLMO_AUTOCAST_GPU_API) || defined(OLMO_AUTOCAST_LEGACY_API)
  bool prev_enabled_{false};
  at::ScalarType prev_dtype_{at::kFloat};
#else
  std::optional<c10::impl::IncludeDispatchKeyGuard> dk_guard_;
#endif
};

}  // namespace olmo_cpp
