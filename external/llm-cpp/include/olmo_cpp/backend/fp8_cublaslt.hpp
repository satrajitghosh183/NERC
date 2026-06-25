#pragma once

/**
 * include/olmo_cpp/backend/fp8_cublaslt.hpp
 *
 * Real FP8 matmul via cuBLASLt FP8 GEMM — item S.
 *
 * Replaces the Float8LinearImpl STE quant/dequant emulation with a
 * direct cublasLtMatmul call configured for E4M3 inputs + fp32
 * accumulator + fp16/bf16 output. Drives the per-tensor scale via the
 * existing Float8ScaleState (amax history maintained on the host or
 * via a tiny device kernel each step).
 *
 * Hopper (sm_90) and Blackwell (sm_120) only. Non-FP8 hardware falls
 * back to the STE path automatically via the runtime device check.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// y = x_fp8 * weight_fp8.T  (per-tensor scaled)
/// scale_x, scale_w: scalar fp32 tensors; output dtype = out_dtype.
torch::Tensor fp8_linear_cublaslt(torch::Tensor x_bf16,
                                    torch::Tensor weight_bf16,
                                    torch::Tensor scale_x,
                                    torch::Tensor scale_w,
                                    torch::Dtype out_dtype = torch::kBFloat16);

bool device_supports_fp8(torch::Device device);

}  // namespace olmo_cpp
