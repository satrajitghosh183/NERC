#pragma once

/**
 * include/olmo_cpp/types.hpp
 *
 * Tiny project-wide type alias header. Right now it just exposes
 * `olmo_cpp::Tensor` as a synonym for `torch::Tensor` so model and op code
 * can be written in a slightly less verbose namespace without dragging
 * `using namespace torch;` into every translation unit. Centralising the
 * alias here also gives us a single place to swap in a custom tensor
 * wrapper if we ever want to (e.g. a typed-shape sleeve, a half-precision
 * marker, or a backend-tagged tensor) without sweeping the codebase.
 *
 * --- Includes from this project ---
 *   (none — pure stdlib/torch glue)
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - Direct callers not located via quick grep. Many model headers use
 *     `torch::Tensor` directly rather than this alias; the alias remains
 *     for forward-compatibility and for tooling that prefers the project
 *     namespace.
 *
 * --- Role in training pipeline ---
 *   Pure compile-time alias — no runtime side effects. Included by header
 *   files that prefer `olmo_cpp::Tensor` for self-documenting signatures.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Project-local synonym for `torch::Tensor`. Behaves identically to
/// `torch::Tensor` (same shape, dtype, device, autograd semantics); the
/// alias merely keeps signatures inside `olmo_cpp::` self-contained.
using Tensor = torch::Tensor;

}  // namespace olmo_cpp
