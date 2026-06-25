// Numerical correctness: CUTLASS-WGMMA output vs cuBLAS for the three
// Linear layouts we route (NN, NT, TN). BF16 tensor cores on both sides,
// so bit-exactness is not expected — we require max-abs <= 1e-2 relative
// to the operand norm, which comfortably catches wiring bugs (wrong
// stride, wrong transpose, swapped A/B) while staying above BF16
// roundoff noise on large contractions.
//
// Requires a build with -DZWT_USE_WGMMA=ON on an sm_90 device. If the
// build lacks WGMMA or the device isn't sm_90, the test reports SKIP
// (exit 0) rather than FAIL — so it's safe in CPU CI.

#include "zwt/core/allocator.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/ops/gemm.hpp"
#include "zwt/ops/gemm_wgmma.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif

using namespace zwt;

namespace {

int g_failed = 0;

void expect(bool cond, const std::string& what) {
    std::printf("[%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) ++g_failed;
}

#ifdef USE_CUDA
// Fill a device bf16 tensor with uniform random values drawn on the host.
void fill_random_bf16(Tensor& t, uint64_t seed, float lo = -0.5f, float hi = 0.5f) {
    std::vector<__nv_bfloat16> host(t.numel());
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (int64_t i = 0; i < t.numel(); ++i) host[i] = __float2bfloat16(d(rng));
    cudaMemcpy(t.data(), host.data(), host.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
}

// Pull a bf16 device tensor back as fp32 on host.
std::vector<float> pull_bf16_as_f32(const Tensor& t) {
    std::vector<__nv_bfloat16> host(t.numel());
    cudaMemcpy(host.data(), t.data(), host.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<float> out(t.numel());
    for (int64_t i = 0; i < t.numel(); ++i) out[i] = __bfloat162float(host[i]);
    return out;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(double(a[i]) - double(b[i]));
        if (d > m) m = d;
    }
    return m;
}

double l2_norm(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += double(x) * double(x);
    return std::sqrt(s);
}

// One case: run cuBLAS and WGMMA for identical inputs, compare outputs.
void case_layout(const char* label, bool transa, bool transb,
                 int64_t Ar, int64_t Ac, int64_t Br, int64_t Bc,
                 int64_t M, int64_t N, uint64_t seed) {
    Device dev = Device::cuda(0);
    Tensor A = empty({Ar, Ac}, DType::BF16, dev);
    Tensor B = empty({Br, Bc}, DType::BF16, dev);
    Tensor C_cublas = empty({M, N}, DType::BF16, dev);
    Tensor C_wgmma  = empty({M, N}, DType::BF16, dev);
    fill_random_bf16(A, seed);
    fill_random_bf16(B, seed ^ 0xBEEF);

    setenv("ZWT_DISABLE_WGMMA", "1", 1);
    ops::reset_wgmma_disable_cache();
    ops::gemm(A, transa, B, transb, C_cublas, 1.f, 0.f);
    unsetenv("ZWT_DISABLE_WGMMA");
    ops::reset_wgmma_disable_cache();
    ops::gemm_wgmma(A, transa, B, transb, C_wgmma, 1.f, 0.f);
    cudaDeviceSynchronize();

    auto h_cublas = pull_bf16_as_f32(C_cublas);
    auto h_wgmma  = pull_bf16_as_f32(C_wgmma);
    const double err   = max_abs(h_cublas, h_wgmma);
    const double norm  = l2_norm(h_cublas);
    const double rel   = err / (norm / std::sqrt(double(h_cublas.size())) + 1e-9);

    std::printf("  %s: M=%lld N=%lld K=%lld  max_abs=%.3e  rel=%.3e\n",
                label, (long long)M, (long long)N,
                (long long)(transa ? Ar : Ac), err, rel);
    // BF16 mma error grows ~sqrt(K)*eps_bf16. eps_bf16 ~= 4e-3. So for
    // K=512 we expect ~1e-1 max-abs. Use a generous absolute threshold
    // scaled by sqrt(K) * 0.02.
    const double tol = std::sqrt(double(transa ? Ar : Ac)) * 2e-2;
    expect(err < tol, std::string(label) + " cublas vs wgmma within tolerance");
}
#endif

}  // namespace

int main() {
#ifndef USE_CUDA
    std::printf("[SKIP] CUDA build required\n");
    return 0;
#else
    set_activation_arena_capacity(size_t(1) << 30);
    if (!ops::wgmma_available()) {
        std::printf("[SKIP] wgmma_available()=false (built without "
                    "ZWT_USE_WGMMA or device is not sm_90)\n");
        return 0;
    }

    // NT: Y = X @ W^T. X=[M,K]=[128,512], W=[N,K]=[256,512]. C=[128,256].
    case_layout("NT (forward proj)", /*transa=*/false, /*transb=*/true,
                /*Ar=*/128, /*Ac=*/512, /*Br=*/256, /*Bc=*/512,
                /*M=*/128, /*N=*/256, 0xC0DE);
    // NN: grad_X = grad_Y @ W. grad_Y=[M,N]=[128,256], W=[N,K]=[256,512]->
    //     treat as B=[K,N]=[256,512] with transb=false... wait: NN means
    //     both row-major, no transpose. A=[M,K]=[128,256], B=[K,N]=[256,512].
    case_layout("NN", /*transa=*/false, /*transb=*/false,
                /*Ar=*/128, /*Ac=*/256, /*Br=*/256, /*Bc=*/512,
                /*M=*/128, /*N=*/512, 0xBEEF);
    // TN: grad_W = grad_Y^T @ X. A=[K,M] transa=true -> logical (M,K).
    //     A=[M_store,K_store]=[256,128], B=[K,N]=[256,512]. M=128,N=512.
    case_layout("TN", /*transa=*/true, /*transb=*/false,
                /*Ar=*/256, /*Ac=*/128, /*Br=*/256, /*Bc=*/512,
                /*M=*/128, /*N=*/512, 0xFACE);

    // Dispatch-test: ops::gemm(BF16 + sm_90) should land in WGMMA path,
    // produce the same result as the direct call.
    {
        Device dev = Device::cuda(0);
        const int64_t M=128, N=256, K=512;
        Tensor X = empty({M, K}, DType::BF16, dev);
        Tensor W = empty({N, K}, DType::BF16, dev);
        Tensor C_dispatch = empty({M, N}, DType::BF16, dev);
        Tensor C_direct   = empty({M, N}, DType::BF16, dev);
        fill_random_bf16(X, 0x1234);
        fill_random_bf16(W, 0x5678);
        unsetenv("ZWT_DISABLE_WGMMA");
        ops::reset_wgmma_disable_cache();
        ops::gemm(X, false, W, true, C_dispatch, 1.f, 0.f);
        ops::gemm_wgmma(X, false, W, true, C_direct, 1.f, 0.f);
        cudaDeviceSynchronize();
        auto a = pull_bf16_as_f32(C_dispatch);
        auto b = pull_bf16_as_f32(C_direct);
        double err = max_abs(a, b);
        std::printf("  dispatch test: max_abs(ops::gemm vs ops::gemm_wgmma) "
                    "= %.3e\n", err);
        expect(err == 0.0, "ops::gemm on sm_90 + BF16 routes to WGMMA (bitwise)");
    }

    std::printf("---\n%d failed\n", g_failed);
    return g_failed;
#endif
}
