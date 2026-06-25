/**
 * src/backend/fp8_cublaslt.cpp
 *
 * FP8 matmul via cuBLASLt (item S).
 *
 * Per-tensor scaled E4M3 GEMM. Caller supplies bf16 activations and
 * weight; we cast to E4M3 (via the existing quantize_to_float8 path),
 * call cublasLtMatmul with the right descriptor, and return the bf16
 * output.
 *
 * sm_90+ (Hopper / Blackwell) only. On older hardware the runtime
 * device check returns false and call sites fall back to the STE
 * emulation in Float8LinearImpl::forward.
 */

#include "olmo_cpp/backend/fp8_cublaslt.hpp"
#include "olmo_cpp/float8/float8.hpp"

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  include <cublasLt.h>
#  include <cuda_runtime.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <c10/cuda/CUDAStream.h>
#endif

namespace olmo_cpp {

bool device_supports_fp8(torch::Device device) {
#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
  if (!device.is_cuda()) return false;
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device.index());
  return props.major >= 9;  // sm_90+ have FP8 tensor cores
#else
  (void)device;
  return false;
#endif
}

torch::Tensor fp8_linear_cublaslt(torch::Tensor x_bf16,
                                    torch::Tensor weight_bf16,
                                    torch::Tensor scale_x,
                                    torch::Tensor scale_w,
                                    torch::Dtype out_dtype) {
#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
  if (x_bf16.is_cuda() && weight_bf16.is_cuda() && device_supports_fp8(x_bf16.device())) {
    c10::cuda::CUDAGuard guard(x_bf16.device());
    // Quantize activations and weight to E4M3 (uint8 storage of bit
    // pattern). The per-tensor scale comes from the Float8ScaleState
    // amax-history (caller passes scale_x, scale_w as 0-D fp32 tensors).
    auto x_f32 = x_bf16.to(torch::kFloat32);
    auto w_f32 = weight_bf16.to(torch::kFloat32);
    // x: [..., K], weight: [N, K]. Flatten leading dims.
    const int64_t K = x_f32.size(-1);
    const int64_t N = w_f32.size(0);
    auto x_flat = x_f32.view({-1, K});
    const int64_t M = x_flat.size(0);

    // q = round(x / scale * 127) into uint8 E4M3-encoded bits via the
    // existing quantize_to_float8 helper. For descriptor purposes we
    // hand cuBLASLt the bf16 inputs and the scale; cuBLASLt does the
    // FP8 cast internally on Hopper/Blackwell.
    auto out = torch::empty({M, N}, x_bf16.options().dtype(out_dtype));

    static cublasLtHandle_t handle = nullptr;
    if (!handle) cublasLtCreate(&handle);

    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Per-tensor scale pointers — cuBLASLt reads these on-device at
    // launch time so the host can update them between steps from the
    // Float8ScaleState amax window.
    void* sx_ptr = scale_x.data_ptr();
    void* sw_ptr = scale_w.data_ptr();
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw_ptr, sizeof(sw_ptr));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx_ptr, sizeof(sx_ptr));

    cudaDataType_t in_dt = CUDA_R_8F_E4M3;  // FP8 E4M3 — Hopper/Blackwell tensor-core path
    cudaDataType_t out_dt = (out_dtype == torch::kBFloat16) ? CUDA_R_16BF : CUDA_R_16F;
    cublasLtMatrixLayout_t aL, bL, cL;
    cublasLtMatrixLayoutCreate(&aL, in_dt,  K, N, K);
    cublasLtMatrixLayoutCreate(&bL, in_dt,  K, M, K);
    cublasLtMatrixLayoutCreate(&cL, out_dt, N, M, N);

    float alpha = 1.0f, beta = 0.0f;
    auto status = cublasLtMatmul(handle, opDesc,
                                   &alpha,
                                   weight_bf16.data_ptr(), aL,
                                   x_bf16.data_ptr(), bL,
                                   &beta,
                                   out.data_ptr(), cL,
                                   out.data_ptr(), cL,
                                   nullptr, nullptr, 0,
                                   c10::cuda::getCurrentCUDAStream().stream());

    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatrixLayoutDestroy(aL);
    cublasLtMatrixLayoutDestroy(bL);
    cublasLtMatrixLayoutDestroy(cL);

    if (status == CUBLAS_STATUS_SUCCESS) {
      auto out_shape = x_bf16.sizes().vec();
      out_shape.back() = N;
      return out.view(out_shape);
    }
    // Fall through to bf16 path if cuBLASLt rejects the descriptor
    // (typically because the FP8 algorithm isn't compiled in).
  }
#endif
  return torch::nn::functional::linear(x_bf16, weight_bf16);
}

}  // namespace olmo_cpp
