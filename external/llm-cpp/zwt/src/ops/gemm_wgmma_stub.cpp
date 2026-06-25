// Always-compiled stub for ops::gemm_wgmma / ops::wgmma_available.
//
// When CMake is configured with -DZWT_USE_WGMMA=ON, the real
// implementation in zwt/src/ops/gemm_wgmma.cu provides these symbols
// (compiled into the dedicated zwt_wgmma static lib built for sm_90a).
// This file supplies the same symbols for builds *without* the flag so
// callers that link against zwt don't need to #ifdef.

#ifndef ZWT_USE_WGMMA

#include "zwt/ops/gemm_wgmma.hpp"

#include <stdexcept>

namespace zwt::ops {

bool wgmma_available() { return false; }

void gemm_wgmma(const Tensor&, bool,
                const Tensor&, bool,
                Tensor&, float, float) {
  throw std::runtime_error(
      "gemm_wgmma: built without ZWT_USE_WGMMA — rebuild with "
      "-DZWT_USE_WGMMA=ON on an sm_90 toolchain");
}

}  // namespace zwt::ops

#endif  // !ZWT_USE_WGMMA
