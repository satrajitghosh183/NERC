/**
 * src/backend/cublas_direct.cpp
 *
 * Direct cuBLAS / cuBLASLt entry points (item L). Caches the cuBLASLt
 * handle (one per CUDA device, thread-local) and a per-shape matmul
 * plan so the hot-path Linear / batched-matmul calls skip the ATen
 * dispatcher and go straight into the cuBLAS launcher.
 *
 * For the first cut: the cache is keyed on (dtype, m, n, k); plan
 * creation happens once per unique shape. Production-grade plan
 * selection would use cublasLtMatmulAlgoGetHeuristic; for now we let
 * cuBLASLt pick the first viable algorithm and stash it.
 *
 * Non-CUDA paths and unsupported dtypes fall through to ATen so
 * call sites can drop these helpers in unconditionally.
 */

#include "olmo_cpp/backend/cublas_direct.hpp"

#include <torch/torch.h>
#include <memory>
#include <mutex>
#include <unordered_map>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  define OLMO_HAS_CUBLASLT 1
#  include <cuda_runtime.h>
#  include <cublasLt.h>
#  include <c10/cuda/CUDAStream.h>
#  include <c10/cuda/CUDAGuard.h>
#endif

namespace olmo_cpp {

namespace {

#ifdef OLMO_HAS_CUBLASLT

struct LtHandleCache {
  cublasLtHandle_t handle = nullptr;
  std::mutex mu;
};
static LtHandleCache& lt_cache() {
  static LtHandleCache c;
  return c;
}

cublasLtHandle_t get_handle() {
  auto& c = lt_cache();
  std::lock_guard<std::mutex> lock(c.mu);
  if (!c.handle) cublasLtCreate(&c.handle);
  return c.handle;
}

cudaDataType_t to_cuda_dtype(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat16:   return CUDA_R_16F;
    case torch::kBFloat16:  return CUDA_R_16BF;
    case torch::kFloat32:   return CUDA_R_32F;
    default:                return CUDA_R_32F;
  }
}

bool supported_lt_dtype(torch::ScalarType t) {
  return t == torch::kFloat16 || t == torch::kBFloat16 || t == torch::kFloat32;
}

// D3 — descriptor cache. cublasLtMatmulDescCreate + 3 layout creates
// per call cost a few microseconds; for fixed-shape training loops
// the same descriptors get re-created every step. Cache them keyed
// by (dtype, M, N, K, opA, opB). One std::shared_ptr per plan keeps
// the cuBLASLt resources alive for the program's lifetime.
struct PlanKey {
  cudaDataType_t dtype;
  int64_t M, N, K;
  cublasOperation_t opA, opB;
  bool operator==(const PlanKey& o) const {
    return dtype == o.dtype && M == o.M && N == o.N && K == o.K
        && opA == o.opA && opB == o.opB;
  }
};
struct PlanKeyHash {
  size_t operator()(const PlanKey& k) const noexcept {
    size_t h = std::hash<int>{}(static_cast<int>(k.dtype));
    auto mix = [&](size_t x) {
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<int64_t>{}(k.M));
    mix(std::hash<int64_t>{}(k.N));
    mix(std::hash<int64_t>{}(k.K));
    mix(std::hash<int>{}(static_cast<int>(k.opA)));
    mix(std::hash<int>{}(static_cast<int>(k.opB)));
    return h;
  }
};
struct CachedPlan {
  cublasLtMatmulDesc_t   desc    = nullptr;
  cublasLtMatrixLayout_t aLayout = nullptr;
  cublasLtMatrixLayout_t bLayout = nullptr;
  cublasLtMatrixLayout_t cLayout = nullptr;
  cublasLtMatmulAlgo_t   algo{};
  bool                   has_algo = false;
  size_t                 ws_bytes = 0;
  ~CachedPlan() {
    if (desc)    cublasLtMatmulDescDestroy(desc);
    if (aLayout) cublasLtMatrixLayoutDestroy(aLayout);
    if (bLayout) cublasLtMatrixLayoutDestroy(bLayout);
    if (cLayout) cublasLtMatrixLayoutDestroy(cLayout);
  }
};

// D5 — workspace for cuBLASLt algos that need scratch. One buffer per
// device, allocated lazily on first use; the same buffer is passed to
// every matmul, which is safe because the GPU serialises kernel
// launches on a stream.
struct WorkspaceCache {
  void*  ptr = nullptr;
  size_t size = 0;
  std::mutex mu;
};
static WorkspaceCache& workspace_cache() {
  static WorkspaceCache w;
  return w;
}
static void* ensure_workspace(size_t bytes) {
  auto& w = workspace_cache();
  std::lock_guard<std::mutex> lock(w.mu);
  if (w.size >= bytes) return w.ptr;
  if (w.ptr) cudaFree(w.ptr);
  cudaMalloc(&w.ptr, bytes);
  w.size = bytes;
  return w.ptr;
}

// D5 — heuristic preference object. Built once per process; the
// max-workspace cap caps the algos cuBLAS may pick.
constexpr size_t kMaxWorkspaceBytes = 4 * 1024 * 1024;   // 4 MB
struct PrefHolder {
  cublasLtMatmulPreference_t pref = nullptr;
  PrefHolder() {
    cublasLtMatmulPreferenceCreate(&pref);
    size_t cap = kMaxWorkspaceBytes;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cap, sizeof(cap));
  }
  ~PrefHolder() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
  }
};
static PrefHolder& heuristic_pref() {
  static PrefHolder p;
  return p;
}

struct PlanCache {
  std::unordered_map<PlanKey, std::shared_ptr<CachedPlan>, PlanKeyHash> map;
  std::mutex mu;
};
static PlanCache& plan_cache() {
  static PlanCache c;
  return c;
}

// Returns a (cached) plan for the given matmul shape. The plan stores
// the descriptor + three layouts. ldA/ldB/ldC are the per-layout
// leading dimensions in cuBLAS column-major view.
std::shared_ptr<CachedPlan> get_or_create_plan(
    cudaDataType_t dtype,
    int64_t M, int64_t N, int64_t K,
    cublasOperation_t opA, cublasOperation_t opB,
    int64_t ldA_rows, int64_t ldA_cols, int64_t ldA,
    int64_t ldB_rows, int64_t ldB_cols, int64_t ldB,
    int64_t ldC_rows, int64_t ldC_cols, int64_t ldC) {
  PlanKey key{dtype, M, N, K, opA, opB};
  auto& cache = plan_cache();
  std::lock_guard<std::mutex> lock(cache.mu);
  auto it = cache.map.find(key);
  if (it != cache.map.end()) return it->second;

  auto plan = std::make_shared<CachedPlan>();
  cublasLtMatmulDescCreate(&plan->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                  &opA, sizeof(opA));
  cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                  &opB, sizeof(opB));
  cublasLtMatrixLayoutCreate(&plan->aLayout, dtype, ldA_rows, ldA_cols, ldA);
  cublasLtMatrixLayoutCreate(&plan->bLayout, dtype, ldB_rows, ldB_cols, ldB);
  cublasLtMatrixLayoutCreate(&plan->cLayout, dtype, ldC_rows, ldC_cols, ldC);

  // D5 — ask cuBLAS for the best algorithm for this shape. Store it
  // on the plan; subsequent matmuls pass it directly to cublasLtMatmul.
  cublasLtMatmulHeuristicResult_t result = {};
  int returned_count = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(
      get_handle(),
      plan->desc,
      plan->aLayout, plan->bLayout, plan->cLayout, plan->cLayout,
      heuristic_pref().pref,
      /*requestedAlgoCount=*/1,
      &result, &returned_count);
  if (s == CUBLAS_STATUS_SUCCESS && returned_count > 0) {
    plan->algo = result.algo;
    plan->has_algo = true;
    plan->ws_bytes = result.workspaceSize;
  }

  cache.map.emplace(key, plan);
  return plan;
}

#endif  // OLMO_HAS_CUBLASLT

}  // namespace

void cublas_direct_reset_cache() {
#ifdef OLMO_HAS_CUBLASLT
  {
    auto& c = lt_cache();
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.handle) {
      cublasLtDestroy(c.handle);
      c.handle = nullptr;
    }
  }
  {
    auto& pc = plan_cache();
    std::lock_guard<std::mutex> lock(pc.mu);
    pc.map.clear();  // CachedPlan dtors release descriptors
  }
#endif
}

torch::Tensor fast_linear(torch::Tensor x,
                           torch::Tensor weight,
                           torch::Tensor bias) {
#ifdef OLMO_HAS_CUBLASLT
  // x: [..., K], weight: [N, K]. Output: [..., N].
  if (x.is_cuda() && weight.is_cuda() && supported_lt_dtype(x.scalar_type())
      && x.scalar_type() == weight.scalar_type()) {
    c10::cuda::CUDAGuard guard(x.device());
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    const int64_t K = x_c.size(-1);
    const int64_t N = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == K, "fast_linear: weight inner dim mismatch");
    // Flatten leading dims so the matmul sees a 2-D problem.
    auto x2 = x_c.view({-1, K});
    const int64_t M = x2.size(0);
    auto opts = x_c.options();
    auto out2 = torch::empty({M, N}, opts);

    auto handle = get_handle();
    auto dtype = to_cuda_dtype(x_c.scalar_type());
    cublasOperation_t opA = CUBLAS_OP_T;  // weight (NxK) used as A^T -> KxN
    cublasOperation_t opN = CUBLAS_OP_N;

    // D3 — cached plan keyed by (dtype, M, N, K, opA, opB).
    auto plan = get_or_create_plan(
        dtype, M, N, K, opA, opN,
        K, N, K,
        K, M, K,
        N, M, N);

    float alpha = 1.0f, beta = 0.0f;
    void*  ws_ptr = plan->ws_bytes > 0 ? ensure_workspace(plan->ws_bytes) : nullptr;
    cublasLtMatmul(handle, plan->desc,
                    &alpha,
                    w_c.data_ptr(), plan->aLayout,
                    x2.data_ptr(),  plan->bLayout,
                    &beta,
                    out2.data_ptr(), plan->cLayout,
                    out2.data_ptr(), plan->cLayout,
                    plan->has_algo ? &plan->algo : nullptr,
                    ws_ptr, plan->ws_bytes,
                    c10::cuda::getCurrentCUDAStream().stream());

    auto out_shape = x_c.sizes().vec();
    out_shape.back() = N;
    auto out = out2.view(out_shape);
    if (bias.defined()) out = out + bias;
    return out;
  }
#endif
  // Fallback: regular ATen linear.
  return torch::nn::functional::linear(
      x, weight, bias.defined() ? bias : torch::Tensor());
}

torch::Tensor fast_bmm(torch::Tensor a, torch::Tensor b) {
  // For now, defer to torch::matmul. Direct cublasLtStridedBatched lands as
  // a follow-on once the shape/stride conversion is wired in.
  return torch::matmul(a, b);
}

torch::Tensor fast_matmul(torch::Tensor a,
                           torch::Tensor b,
                           bool transa,
                           bool transb) {
#ifdef OLMO_HAS_CUBLASLT
  if (a.is_cuda() && b.is_cuda() && supported_lt_dtype(a.scalar_type())
      && a.scalar_type() == b.scalar_type() && a.dim() == 2 && b.dim() == 2) {
    c10::cuda::CUDAGuard guard(a.device());
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    // Effective dims after op:
    //   op(a): [M, K]   op(b): [K, N]
    const int64_t M = transa ? a_c.size(1) : a_c.size(0);
    const int64_t Ka = transa ? a_c.size(0) : a_c.size(1);
    const int64_t Kb = transb ? b_c.size(1) : b_c.size(0);
    const int64_t N = transb ? b_c.size(0) : b_c.size(1);
    TORCH_CHECK(Ka == Kb,
                "fast_matmul: inner-dim mismatch ", Ka, " vs ", Kb);
    const int64_t K = Ka;

    auto opts = a_c.options();
    auto out = torch::empty({M, N}, opts);

    auto handle = get_handle();
    auto dtype = to_cuda_dtype(a_c.scalar_type());

    // cuBLAS sees column-major matrices. Each row-major torch tensor with
    // shape [r, c] (stride c) maps to a col-major matrix of shape [c, r]
    // (leading dim c). Two transposes (the row→col reinterp and the user's
    // transpose flag) collapse to: flip the cublasOperation when the user
    // does NOT want a transpose (because the row→col view already is one).
    cublasOperation_t opA_lt = transb ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB_lt = transa ? CUBLAS_OP_N : CUBLAS_OP_T;

    // D3 — cached plan.
    auto plan = get_or_create_plan(
        dtype, M, N, K, opA_lt, opB_lt,
        b_c.size(1), b_c.size(0), b_c.size(1),
        a_c.size(1), a_c.size(0), a_c.size(1),
        N, M, N);

    float alpha = 1.0f, beta = 0.0f;
    void*  ws_ptr = plan->ws_bytes > 0 ? ensure_workspace(plan->ws_bytes) : nullptr;
    cublasLtMatmul(handle, plan->desc,
                    &alpha,
                    b_c.data_ptr(), plan->aLayout,
                    a_c.data_ptr(), plan->bLayout,
                    &beta,
                    out.data_ptr(), plan->cLayout,
                    out.data_ptr(), plan->cLayout,
                    plan->has_algo ? &plan->algo : nullptr,
                    ws_ptr, plan->ws_bytes,
                    c10::cuda::getCurrentCUDAStream().stream());
    return out;
  }
#endif
  auto av = transa ? a.transpose(0, 1) : a;
  auto bv = transb ? b.transpose(0, 1) : b;
  return torch::matmul(av, bv);
}

}  // namespace olmo_cpp
