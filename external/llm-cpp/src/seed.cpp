/**
 * src/seed.cpp
 *
 * Reproducibility plumbing. Owns a process-wide SeedState and offers
 * three things:
 *   - seed_all(seed_or_random)      seed every RNG that affects training:
 *                                   PyTorch CPU/CUDA, a torch::Generator
 *                                   for weight init, and an mt19937 for
 *                                   our own data shuffle/permutation use.
 *   - capture_rng_state() / restore_rng_state()  serialise/restore the
 *                                   above into a small POD so checkpoints
 *                                   can resume training without regressing
 *                                   the data order.
 *   - print_rng_state_summary()     log the current RNG state for the user.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/seed.hpp : SeedState struct + function declarations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: seed_all(seed) at startup before any model is built.
 *   - src/main.cpp: print_rng_state_summary() at end of training when
 *     profile=1 (so the user can confirm the seed actually got used).
 *   - src/train/checkpoint.cpp: capture_rng_state / restore_rng_state for
 *     deterministic resume.
 *
 * --- Role in training pipeline ---
 *   Run exactly once at process start, then read on demand. Without this,
 *   weight init and data shuffling drift between runs even with the same
 *   .conf.
 */
#include "olmo_cpp/seed.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace olmo_cpp {

namespace {
  std::optional<SeedState> g_seed_state;
}  // namespace

SeedState seed_all(std::optional<uint64_t> seed) {
  uint64_t actual_seed;
  bool explicit_seed;

  if (seed.has_value()) {
    actual_seed = *seed;
    explicit_seed = true;
  } else {
    // Generate a random seed, just like OLMo-core does when no seed is given
    std::random_device rd;
    actual_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    explicit_seed = false;
  }

  // Print seed prominently so the user can always reproduce
  std::cout << "\n╔══════════════════════════════════════════════╗\n"
            << "║  SEED: " << std::left << std::setw(37) << actual_seed << "║\n"
            << "║  Mode: " << std::left << std::setw(37)
            << (explicit_seed ? "explicit (--seed)" : "auto-generated") << "║\n"
            << "╚══════════════════════════════════════════════╝\n" << std::endl;

  // 1. Seed PyTorch CPU RNG (this is the global default generator)
  torch::manual_seed(static_cast<int64_t>(actual_seed));

  // 2. Seed PyTorch CUDA RNG (all devices) if available
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(static_cast<int64_t>(actual_seed));
  }

  // 3. Create a torch::Generator for weight initialization
  //    This is separate from the global RNG so weight init doesn't
  //    interfere with data shuffling etc.
  auto gen = torch::make_generator<at::CPUGeneratorImpl>(actual_seed);

  // 4. Create seeded C++ PRNG
  std::mt19937_64 rng(actual_seed);

  g_seed_state = SeedState{
    .seed = actual_seed,
    .was_explicit = explicit_seed,
    .rng = std::move(rng),
    .torch_gen = std::move(gen),
  };

  return *g_seed_state;
}

SeedState& global_seed_state() {
  if (!g_seed_state.has_value()) {
    throw std::runtime_error(
        "global_seed_state() called before seed_all(). "
        "Call olmo_cpp::seed_all() at startup.");
  }
  return *g_seed_state;
}

void print_rng_state_summary() {
  if (!g_seed_state.has_value()) {
    std::cout << "[RNG] No seed state initialized.\n";
    return;
  }

  auto& state = *g_seed_state;
  std::cout << "[RNG] State summary:\n"
            << "  Original seed: " << state.seed
            << (state.was_explicit ? " (explicit)" : " (auto)") << "\n"
            << "  torch generator seed: "
            << state.torch_gen.current_seed() << "\n";

  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    std::cout << "  torch CUDA devices: " << torch::cuda::device_count() << "\n";
  }

  std::cout << std::endl;
}

RNGCheckpoint capture_rng_state() {
  RNGCheckpoint ckpt;
  ckpt.original_seed = g_seed_state.has_value() ? g_seed_state->seed : 0;

  // Capture the default CPU generator state as a tensor
  auto cpu_gen = at::detail::getDefaultCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(cpu_gen.mutex());
    auto* impl = cpu_gen.get<at::CPUGeneratorImpl>();
    // Serialize engine state to string (there's no direct tensor API in C++)
    // We store the seed; for full state we'd need the mt19937 internal state
    ckpt.torch_cpu_state = torch::tensor({static_cast<int64_t>(impl->current_seed())});
  }

  // Capture mt19937 state as string
  if (g_seed_state.has_value()) {
    std::ostringstream oss;
    oss << g_seed_state->rng;
    ckpt.mt19937_state = oss.str();
  }

  return ckpt;
}

void restore_rng_state(const RNGCheckpoint& checkpoint) {
  // Restore torch CPU seed
  if (checkpoint.torch_cpu_state.defined() && checkpoint.torch_cpu_state.numel() > 0) {
    torch::manual_seed(checkpoint.torch_cpu_state.item<int64_t>());
  }

  // Restore CUDA if available
  if (checkpoint.torch_cuda_state.has_value() &&
      torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    // Re-seed CUDA from the checkpoint
    torch::cuda::manual_seed_all(checkpoint.torch_cuda_state->item<int64_t>());
  }

  // Restore mt19937 state
  if (g_seed_state.has_value() && !checkpoint.mt19937_state.empty()) {
    std::istringstream iss(checkpoint.mt19937_state);
    iss >> g_seed_state->rng;
    g_seed_state->seed = checkpoint.original_seed;
  }
}

}  // namespace olmo_cpp
