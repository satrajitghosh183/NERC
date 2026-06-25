/**
 * src/distributed/ddp_stub.cpp
 *
 * No-op fallback for DDP. Compiled when OLMO_USE_DDP is OFF (or when
 * Gloo headers are not on the system; pip LibTorch ships without Gloo).
 *
 * Every public DDP entry point returns "not available" (std::nullopt
 * for factories, no-op bodies for member functions) so the train loop
 * happily falls through to single-process execution without any
 * conditional compilation in the call sites.
 *
 * The quickstart's 3060 path uses THIS file, not the real ddp.cpp.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/ddp.hpp : same header as the real impl.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: DDPContext::init_from_env() is called
 *     unconditionally; here it returns std::nullopt so the loop stays
 *     single-rank.
 *
 * --- Role in training pipeline ---
 *   Lets the project build out-of-the-box on a vanilla pip-installed
 *   LibTorch without breaking the CMake target list.
 */
// Stub implementation when Gloo is not available (pip LibTorch)
#include "olmo_cpp/distributed/ddp.hpp"
#include <vector>

namespace olmo_cpp {

std::optional<DDPContext> DDPContext::init_from_env() {
  return std::nullopt;
}

void DDPContext::broadcast_parameters(std::vector<torch::Tensor>& /*parameters*/) {}

void DDPContext::allreduce_gradients(const std::vector<torch::Tensor>& /*parameters*/) {}

void DDPContext::register_grad_hooks(std::vector<torch::Tensor>& /*parameters*/,
                                      int64_t /*bucket_bytes*/) {}

}  // namespace olmo_cpp
