#include "zwt/core/determinism.hpp"

#include <atomic>
#include <cstdlib>

namespace zwt {

namespace {
std::atomic<bool> g_det{false};
}

bool is_deterministic() { return g_det.load(std::memory_order_relaxed); }

void set_deterministic(bool enabled) {
  g_det.store(enabled, std::memory_order_relaxed);
}

void init_determinism_env() {
  // cuBLAS consults CUBLAS_WORKSPACE_CONFIG at handle-creation time to
  // decide which algorithms are eligible for determinism. :4096:8 is the
  // documented setting that keeps reduction-order deterministic. Only set
  // it if the user hasn't already; never overwrite an explicit choice.
  if (std::getenv("CUBLAS_WORKSPACE_CONFIG") == nullptr) {
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/0);
  }
}

}  // namespace zwt
