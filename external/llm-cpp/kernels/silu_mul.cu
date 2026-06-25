/**
 * kernels/silu_mul.cu
 *
 * ─── What this kernel computes ───────────────────────────────────────
 *
 * Inside every transformer "feed-forward" sub-layer (the second half of
 * a Block, after attention) there is a non-linearity called **SwiGLU**.
 * SwiGLU was popularised by PaLM and LLaMA and replaced the older
 * GELU+Linear pattern from GPT-2.
 *
 * Concretely, the FFN takes a hidden vector `x` of size D and computes:
 *
 *     gate = W1 x        (linear projection D -> H)
 *     up   = W3 x        (a SECOND independent linear projection D -> H)
 *     y    = silu(gate) ⊙ up        <-- this kernel does this elementwise step
 *     out  = W2 y        (linear projection H -> D)
 *
 * "⊙" is elementwise multiply.  silu (a.k.a. "swish") is the smooth-ReLU
 *
 *     silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
 *
 * The intuition: `gate` decides "how much information to let through" via
 * the silu activation, and `up` is the actual content.  Multiplying them
 * gives you a learned gating mechanism, which empirically works better
 * than a single non-linear projection.
 *
 * ─── Why we need a custom kernel ─────────────────────────────────────
 *
 * Without fusion, the elementwise step is THREE separate kernel launches
 * in PyTorch:
 *      tmp1 = silu(gate)    (read gate, write tmp1)
 *      tmp2 = tmp1 * up     (read tmp1+up, write tmp2)
 *      ...                   (each launch reads/writes the full tensor)
 *
 * With fusion this becomes ONE kernel launch and ONE pass through global
 * memory: read gate, read up, write out. For a 7B-class model the FFN
 * tensor is on the order of hundreds of MB, so cutting reads/writes by
 * 3x measurably moves training throughput.
 *
 * ─── Launch geometry ─────────────────────────────────────────────────
 *
 * The kernel is "embarrassingly parallel" — every output element is
 * independent of every other.  We launch a 1-D grid of 1-D blocks:
 *
 *      threads/block = 256       (a standard sweet-spot for compute-bound
 *                                  kernels on Ampere/Hopper)
 *      blocks        = ceil(N/256), capped at 65535
 *
 * Each thread loops `idx, idx+stride, idx+2*stride, ...` over the flat
 * tensor (called a "grid-stride loop") so we don't have to spawn a
 * gigantic grid for huge tensors.
 *
 * The FP32 path uses `float4` vector loads — one global-memory
 * transaction returns 4 floats, which roughly quadruples bandwidth.
 * The BF16 path is plain element-wise because BF16 vector intrinsics
 * complicate the silu computation (we still upcast to FP32 for the
 * sigmoid to keep accuracy).
 *
 * ─── Precision ───────────────────────────────────────────────────────
 *
 * BF16 path: read 16-bit, convert to FP32, do silu+multiply in FP32,
 * convert back to 16-bit. The sigmoid in BF16 is too lossy at the tails
 * — keeping the inner math in FP32 costs basically nothing because the
 * arithmetic is bandwidth-bound, not compute-bound.
 *
 * --- Includes from this project ---
 *   (none — kernel is standalone CUDA, packaged in olmo_kernels.so)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/cuda_backend.cpp: CUDABackend::silu_mul() resolves to
 *     this implementation through the torch::Library dispatch table.
 *   - src/model/feed_forward.cpp: get_backend().silu_mul(gate, up) on
 *     the SwiGLU forward path of every transformer block, every microbatch.
 *
 * --- Role in training pipeline ---
 *   On the hot path of every FFN forward. For typical 30M-7B configs the
 *   silu*mul time is ~3-5% of FFN time; fusing it shaves a few percent
 *   off step time vs. the unfused ATen reference.
 */
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

/// Single-element silu in FP32. `__forceinline__` because we want this
/// compiled into the calling kernel's body — a real function call would
/// trash performance for a 1-FLOP routine.
///   silu(x) = x * sigmoid(x) = x / (1 + e^{-x})
/// We use the second form because it avoids two separate ops (sigmoid +
/// multiply) — a single fused divide on the GPU's transcendental unit.
__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// ---- FP32 vectorized path ----------------------------------------------
//
// Loads gate and up as float4 (4 floats per memory transaction). On Ampere/
// Hopper this is the difference between bandwidth-saturating the L2 cache
// and bandwidth-starving the SM. The tail loop handles N % 4 leftovers.

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,   // [N] gate projection result (W1 x)
    const float* __restrict__ up,     // [N] up   projection result (W3 x)
    float* __restrict__ out,          // [N] output buffer (= silu(gate) * up)
    int64_t n) {                      // total element count
  // "Grid-stride loop" preamble: compute this thread's starting index
  // and how far it should hop on each iteration.
  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Reinterpret the float* as float4* so each memory access is 16 bytes
  // wide. Requires the input to be 16-byte aligned, which it is — torch
  // empty()/contiguous() always returns 256-byte aligned storage.
  int64_t vec_n = n / 4;
  const float4* gate4 = reinterpret_cast<const float4*>(gate);
  const float4* up4   = reinterpret_cast<const float4*>(up);
  float4* out4        = reinterpret_cast<float4*>(out);

  // Main vectorised loop: each iteration processes 4 elements.
  for (int64_t i = idx; i < vec_n; i += stride) {
    float4 g = gate4[i];   // one 16-byte global load
    float4 u = up4[i];     // one 16-byte global load
    float4 r;
    r.x = silu(g.x) * u.x; // four independent SwiGLU evaluations,
    r.y = silu(g.y) * u.y; // which the compiler can issue in parallel
    r.z = silu(g.z) * u.z; // because there's no inter-lane dependency.
    r.w = silu(g.w) * u.w;
    out4[i] = r;           // one 16-byte global store
  }

  // Tail loop: any leftover (n % 4) elements after the vectorised body.
  // Without this we'd silently truncate tensors whose size isn't a
  // multiple of 4.
  for (int64_t i = vec_n * 4 + idx; i < n; i += stride) {
    out[i] = silu(gate[i]) * up[i];
  }
}

// ---- BF16 with FP32 compute -------------------------------------------
//
// BF16 (Brain-Float-16): 1 sign bit, 8 exponent bits, 7 mantissa bits.
// Same dynamic range as FP32, less precision. On sm_80+ (A100/3060/H100)
// most matmuls already produce BF16 outputs, so accepting BF16 in/out
// here avoids a wasteful upcast.  We still do the silu in FP32 so the
// sigmoid keeps its accuracy at the tails of the distribution.

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t n) {
  // B4 — vectorise via uint4 (16 bytes = 8 bf16 elements per memory
  // transaction). Mirrors the fp32 path's float4 strategy. The math
  // stays in fp32 through __bfloat1622float2 pairs to keep the sigmoid
  // accurate at the tails of the distribution.
  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  const int64_t n_vec = n / 8;
  const uint4* gate4 = reinterpret_cast<const uint4*>(gate);
  const uint4* up4   = reinterpret_cast<const uint4*>(up);
  uint4* out4        = reinterpret_cast<uint4*>(out);

  for (int64_t i = idx; i < n_vec; i += stride) {
    uint4 g_raw = gate4[i];
    uint4 u_raw = up4[i];
    __nv_bfloat162* gb = reinterpret_cast<__nv_bfloat162*>(&g_raw);
    __nv_bfloat162* ub = reinterpret_cast<__nv_bfloat162*>(&u_raw);
    __nv_bfloat162 rb[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 gf = __bfloat1622float2(gb[j]);
      float2 uf = __bfloat1622float2(ub[j]);
      float2 rf = make_float2(silu(gf.x) * uf.x, silu(gf.y) * uf.y);
      rb[j] = __float22bfloat162_rn(rf);
    }
    out4[i] = *reinterpret_cast<uint4*>(rb);
  }

  // Tail: any leftover n % 8 elements after the vectorised body.
  for (int64_t i = n_vec * 8 + idx; i < n; i += stride) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i]  = __float2bfloat16(silu(g) * u);
  }
}

}  // namespace

/// Host-side launcher. Validates the inputs, makes them contiguous (the
/// kernels assume a flat-stride layout), allocates the output, picks the
/// right kernel based on dtype, and launches.
torch::Tensor silu_mul_cuda_impl(
    const torch::Tensor& gate,
    const torch::Tensor& up) {
  // Sanity checks — fail loudly if the caller mis-routed a CPU tensor or
  // mismatched shapes. TORCH_CHECK is the standard Torch assertion macro.
  TORCH_CHECK(gate.is_cuda() && up.is_cuda());
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");

  // Force a packed/contiguous memory layout. If the caller already passed
  // contiguous tensors this is a cheap no-op; otherwise it triggers a copy.
  auto gate_c = gate.contiguous();
  auto up_c   = up.contiguous();

  // Allocate the output with the same shape / dtype / device as gate.
  auto out = torch::empty_like(gate_c);
  auto n   = gate_c.numel();

  // Make sure subsequent CUDA calls run on the right device. CUDAGuard
  // restores the previous device on scope exit (RAII).
  c10::cuda::CUDAGuard device_guard(gate.device());

  // Launch geometry: 256 threads per block is a generic good choice on
  // Ampere/Hopper. Cap blocks at 65535 (the historical CUDA grid-X limit
  // on older arches; modern arches allow 2^31-1 but capping makes the
  // grid-stride loop do useful work and keeps kernel launch overhead low).
  int threads = 256;
  int blocks  = std::min(static_cast<int64_t>((n + threads - 1) / threads),
                         static_cast<int64_t>(65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch on dtype. Note: scalar_type() returns torch's dtype enum.
  if (gate.scalar_type() == torch::kBFloat16) {
    silu_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n);
  } else {
    silu_mul_f32_kernel<<<blocks, threads, 0, stream>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
  }
  return out;
}

// Plug ourselves into the torch::Library dispatch system. When code calls
// torch::ops::olmo_ops::silu_mul(...) with CUDA tensors, this implementation
// is selected by the dispatcher. CPU/MPS callers fall through to the
// reference path defined elsewhere.
TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("silu_mul", silu_mul_cuda_impl);
}
