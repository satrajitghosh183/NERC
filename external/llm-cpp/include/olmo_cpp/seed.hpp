#pragma once

/**
 * include/olmo_cpp/seed.hpp
 *
 * Reproducibility entry point. Declares `seed_all()` and friends to
 * deterministically seed every RNG the framework touches: torch CPU,
 * torch CUDA (per-device), `std::mt19937_64` for stdlib randomness, and a
 * dedicated `torch::Generator` used by weight init. Mirrors OLMo-core's
 * `utils.seed_all()` so a C++ run with the same seed reproduces the same
 * weight init, data shuffling, and dropout pattern.
 *
 * Also declares `RNGCheckpoint` for checkpoint-resume: serializes the
 * current PRNG state so resuming from step N picks up the same random
 * stream the un-interrupted run would have seen.
 *
 * --- Includes from this project ---
 *   (none — pure stdlib + torch)
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - src/main.cpp: calls `olmo_cpp::seed_all(seed_val)` once at startup,
 *     before device selection and model construction; the returned
 *     SeedState is passed to weight-init helpers.
 *   - src/main.cpp: after training, calls `print_rng_state_summary()`
 *     when `[training] profile=1` is set in the .conf file.
 *
 * --- Role in training pipeline ---
 *   First non-config call in `main()`. Establishes a single source of
 *   truth for all randomness in the run, prints the seed banner, and
 *   stashes the SeedState in a process-wide singleton retrievable via
 *   `global_seed_state()`. Without this, runs are non-reproducible.
 */

#include <cstdint>
#include <string>
#include <random>
#include <optional>
#include <torch/torch.h>

namespace olmo_cpp {

/// Global seed state — mirrors OLMo-core's seed_all() from utils.py.
///
/// Seeds:
///   - torch CPU RNG         (torch::manual_seed)
///   - torch CUDA RNG        (torch::cuda::manual_seed_all, if CUDA available)
///   - std::mt19937           (C++ stdlib, used for data shuffling)
///   - std::random_device     (not seedable, but we record it)
///
/// Prints the active seed so training runs are always reproducible.
/// If no seed is provided, generates one from std::random_device and prints it.

/// Bundle of seed metadata + RNGs returned from `seed_all()`. Stored as a
/// process-wide singleton via `global_seed_state()` and also returned by
/// value for direct use by callers (e.g. weight init).
struct SeedState {
  /// The 64-bit seed actually used (either user-supplied or generated).
  uint64_t seed;
  /// True if user provided --seed, false if auto-generated.
  bool was_explicit;  // true if user provided --seed, false if auto-generated

  /// The global C++ PRNG, seeded deterministically
  std::mt19937_64 rng;

  /// A torch::Generator seeded with the same seed, for weight init etc.
  torch::Generator torch_gen;
};

/// Seed all random number generators. Call this once at startup.
/// If seed == std::nullopt, a random seed is generated and printed.
/// Returns the SeedState for further use (e.g., passing to init_weights).
///
/// Side effects: torch::manual_seed, torch::cuda::manual_seed_all (if
/// CUDA available), populates a process-wide `g_seed_state`, prints a
/// banner to stdout. NOT thread-safe — must be called once before any
/// thread that uses an RNG starts.
SeedState seed_all(std::optional<uint64_t> seed = std::nullopt);

/// Get the global seed state (set by seed_all). Throws if seed_all not called.
SeedState& global_seed_state();

/// Print a summary of all RNG states (for debugging reproducibility).
/// Called from main when `[training] profile=1`.
void print_rng_state_summary();

/// Save RNG states to a checkpoint-compatible format.
/// Captures: torch CPU state, torch CUDA state (if available), mt19937 state.
///
/// Serialised inside training checkpoints so resuming from step N
/// continues the exact random stream the un-interrupted run would have.
struct RNGCheckpoint {
  /// Echo of `SeedState::seed` so debug tooling can identify the run.
  uint64_t original_seed;
  /// torch CPU generator state; in current impl just stores the seed as a
  /// 1-element int64 tensor (full mt19937 internal state isn't trivially
  /// reachable through the public C++ API).
  torch::Tensor torch_cpu_state;
  /// torch CUDA generator state (one per device), if CUDA is available.
  std::optional<torch::Tensor> torch_cuda_state;
  /// `std::mt19937_64` state serialised via `operator<<` (textual form).
  std::string mt19937_state;  // serialized as string
};

/// Snapshot all RNGs into an `RNGCheckpoint` for inclusion in a training
/// checkpoint. Inverse of `restore_rng_state`.
RNGCheckpoint capture_rng_state();
/// Restore all RNGs from a previously captured checkpoint. Used on resume.
void restore_rng_state(const RNGCheckpoint& checkpoint);

}  // namespace olmo_cpp
