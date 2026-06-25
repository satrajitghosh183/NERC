// zwt_wgmma_bench — cuBLAS vs CUTLASS-WGMMA at typical Linear shapes.
//
// Both paths go through ops::gemm. The dispatch caches ZWT_DISABLE_WGMMA on
// first use; we A/B in-process by toggling the env var between measurement
// phases AND calling ops::reset_wgmma_disable_cache(). No dependence on
// cuBLAS internals.
//
// Output CSV to stdout:
//   shape,m,n,k,cublas_ms,wgmma_ms,cublas_tflops,wgmma_tflops,speedup
//
// Usage:
//   ./zwt_wgmma_bench                             # default shapes
//   ./zwt_wgmma_bench 4096 4096 4096              # single custom shape
//   ./zwt_wgmma_bench --iters 50 --warmup 10

#include "zwt/core/allocator.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/ops/gemm.hpp"
#include "zwt/ops/gemm_wgmma.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

namespace {

struct Shape2 { int64_t m, n, k; const char* label; };

// Representative projection shapes. Sequence length 2048, batch 8 -> M = 16384.
// Dims taken from OLMo-2 1B (d=2048), Llama-7B (d=4096), Llama-70B (d=8192).
const Shape2 kDefaultShapes[] = {
    { 16384,  2048,  2048, "1B-attn-proj"  },
    { 16384,  5504,  2048, "1B-ffn-up"     },
    { 16384,  2048,  5504, "1B-ffn-down"   },
    { 16384,  4096,  4096, "7B-attn-proj"  },
    { 16384, 11008,  4096, "7B-ffn-up"     },
    { 16384,  4096, 11008, "7B-ffn-down"   },
    { 16384,  8192,  8192, "70B-attn-proj" },
};

struct Args {
    int iters  = 20;
    int warmup = 5;
    std::vector<Shape2> shapes;
};

Args parse_args(int argc, char** argv) {
    Args a;
    int64_t m = 0, n = 0, k = 0;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--iters" && i + 1 < argc)        a.iters  = std::atoi(argv[++i]);
        else if (s == "--warmup" && i + 1 < argc)  a.warmup = std::atoi(argv[++i]);
        else {
            int64_t v = std::atoll(argv[i]);
            if (pos == 0) m = v;
            else if (pos == 1) n = v;
            else if (pos == 2) k = v;
            ++pos;
        }
    }
    if (pos == 3) a.shapes.push_back({m, n, k, "custom"});
    else for (const auto& s : kDefaultShapes) a.shapes.push_back(s);
    return a;
}

#ifdef USE_CUDA
double time_path(Tensor& X, Tensor& W, Tensor& Y, int iters, int warmup) {
    // Forward projection shape: Y = X @ W^T.
    for (int i = 0; i < warmup; ++i) {
        ops::gemm(X, /*transa=*/false, W, /*transb=*/true, Y, 1.f, 0.f);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        ops::gemm(X, /*transa=*/false, W, /*transb=*/true, Y, 1.f, 0.f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return double(ms) / double(iters);
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifndef USE_CUDA
    std::fprintf(stderr, "zwt_wgmma_bench: CUDA build required\n");
    return 2;
#else
    Args args = parse_args(argc, argv);
    set_activation_arena_capacity(size_t(4) << 30);  // 4 GB

    const bool have_wgmma = ops::wgmma_available();
    if (!have_wgmma) {
        std::fprintf(stderr,
            "NOTE: wgmma_available() = false. This binary was built without "
            "-DZWT_USE_WGMMA=ON or the current device is not sm_90. cuBLAS "
            "numbers still valid; WGMMA column will be 0.\n");
    }

    std::printf("shape,m,n,k,cublas_ms,wgmma_ms,cublas_tflops,"
                "wgmma_tflops,speedup\n");

    Device dev = Device::cuda(0);
    for (const auto& s : args.shapes) {
        Tensor X = empty({s.m, s.k}, DType::BF16, dev);
        Tensor W = empty({s.n, s.k}, DType::BF16, dev);
        Tensor Y = empty({s.m, s.n}, DType::BF16, dev);

        setenv("ZWT_DISABLE_WGMMA", "1", 1);
        ops::reset_wgmma_disable_cache();
        double cublas_ms = time_path(X, W, Y, args.iters, args.warmup);

        double wgmma_ms = 0.0;
        if (have_wgmma) {
            unsetenv("ZWT_DISABLE_WGMMA");
            ops::reset_wgmma_disable_cache();
            wgmma_ms = time_path(X, W, Y, args.iters, args.warmup);
        }

        const double flops = 2.0 * double(s.m) * double(s.n) * double(s.k);
        const double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        const double wgmma_tflops  =
            (wgmma_ms > 0) ? flops / (wgmma_ms * 1e-3) / 1e12 : 0.0;
        const double speedup = (wgmma_ms > 0) ? cublas_ms / wgmma_ms : 0.0;

        std::printf("%s,%lld,%lld,%lld,%.4f,%.4f,%.2f,%.2f,%.3f\n",
                    s.label, (long long)s.m, (long long)s.n, (long long)s.k,
                    cublas_ms, wgmma_ms, cublas_tflops, wgmma_tflops, speedup);
        std::fflush(stdout);
    }
    return 0;
#endif
}
