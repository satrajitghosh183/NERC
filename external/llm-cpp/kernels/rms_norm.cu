/**
 * kernels/rms_norm.cu
 *
 * ─── What RMSNorm is ─────────────────────────────────────────────────
 *
 * "RMS" stands for **Root Mean Square**. RMSNorm is a normalisation
 * layer used inside every transformer block: it rescales each token's
 * D-dim hidden vector so its overall "magnitude" is roughly 1, then
 * lets a learned weight vector restore per-channel scale.
 *
 *     rms(x)   = sqrt( mean_i( x_i^2 ) + eps )           // scalar
 *     y_i      = (x_i / rms(x)) * weight_i               // per-element
 *
 * `eps` is a tiny constant (1e-5..1e-6) that prevents division-by-zero
 * when a row is all zeros. `weight` is the learnable gain — same shape
 * as one row of x.
 *
 * Why not LayerNorm (which subtracts the mean too)? RMSNorm omits the
 * mean-centering step from LayerNorm. It turns out (Zhang & Sennrich
 * 2019) that the mean term contributes very little to model quality
 * but takes a non-trivial chunk of the FLOPs and memory traffic.
 * LLaMA, Gemma and OLMo-2 all use RMSNorm.
 *
 * ─── What "fused" means here ─────────────────────────────────────────
 *
 * A naive implementation in PyTorch would be five separate kernels
 * (square, mean, add eps, sqrt+reciprocal, multiply by weight). Each
 * launch reads/writes the full tensor. This file fuses everything into
 * one kernel: the row is loaded once, the reduction happens in
 * registers + shared memory, and the result is written back once.
 *
 * There's also a "residual_rms_norm" variant that absorbs the
 * preceding `x = x + residual` add into the same kernel, saving yet
 * another full-tensor read/write. That add is ubiquitous in
 * transformers (every block has at least two of them).
 *
 * ─── Launch geometry ─────────────────────────────────────────────────
 *
 *   blocks  = number of rows in the [N, D] input
 *   threads = 128 if D <= 256, else 256
 *
 * One block handles ONE row of length D. Threads inside the block
 * cooperate to compute sum(x^2) by:
 *   1. Each thread reads a stride-`blockDim.x` slice of the row,
 *      accumulating x^2 into a private FP32 register.
 *   2. warpReduceSum() collapses the per-lane sums via __shfl_down_sync
 *      (intra-warp register exchange — no shared memory).
 *   3. blockReduceSum() collects warp results in a 32-element shared-
 *      memory array, then a final warpReduceSum on the first warp gets
 *      the per-block answer.
 * After this, thread 0 computes the scaling factor (1 / rms) and
 * stashes it in shared memory. Every thread then re-reads the row,
 * multiplies by the scale and weight, and writes the output.
 *
 * ─── Precision ───────────────────────────────────────────────────────
 *
 * Both FP32 and BF16 entry points exist. The BF16 path always
 * accumulates the sum-of-squares in FP32 — accumulating thousands of
 * BF16 squared values into a BF16 accumulator silently loses the LSBs
 * and the resulting `rms` is biased. FP32 accumulation costs nothing
 * because the kernel is bandwidth-bound, not compute-bound.
 *
 * --- Includes from this project ---
 *   (none — pure CUDA kernel, packaged in libolmo_kernels.so)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/cuda_backend.cpp: CUDABackend::rms_norm() and
 *     ::residual_rms_norm() resolve here through the dispatcher.
 *   - src/model/layer_norm.cpp / layer_norm_variants.cpp:
 *     get_backend().rms_norm(...) is the entry point used by every
 *     block in the forward pass.
 *
 * --- Role in training pipeline ---
 *   RMSNorm is invoked at least 2N times per forward (pre-attn and
 *   pre-ffn for each of the N transformer layers). With QK-norm
 *   enabled (use_qk_norm=1) it's another 2N invocations. Every one
 *   of those goes through this kernel on CUDA hosts.
 */
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

/// Sum the `val` register across all 32 lanes of a warp using
/// butterfly-style shuffles. This is a tree-reduction:
///   step 1: lanes (0,16),(1,17),... swap and add  -> 16 unique sums
///   step 2: lanes (0,8) (1,9) ...                  -> 8
///   step 3: ...                                    -> 4
///   step 4:                                        -> 2
///   step 5:                                        -> 1   (in lane 0)
/// `__shfl_down_sync(mask, val, offset)` returns the value of `val`
/// from lane (current_lane + offset). The 0xffffffff mask says "all
/// 32 lanes participate" — same behaviour as the deprecated __shfl_down.
__device__ __forceinline__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/// Reduce a value across the entire CUDA block.
///   warps in a 256-thread block = 8
///   shared[8] holds one float per warp (warp 0 writes shared[0], etc.)
/// After warp-reduce, the lane-0 of each warp publishes its sum into
/// shared[wid]. Then we __syncthreads() so every thread can see the
/// publication, and warp-0 does a final warpReduceSum across the first
/// `(blockDim.x / 32)` lanes — i.e. across the per-warp sums — to get
/// the block-wide total. Only thread 0 ends up with the correct value;
/// callers either use `__syncthreads()` + a shared scalar to broadcast
/// it (as we do below) or only consume it from thread 0.
__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float shared[32];      // up to 32 warps per block (1024 threads)
  int lane = threadIdx.x % 32;       // lane index inside the warp
  int wid  = threadIdx.x / 32;       // warp index inside the block

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;  // each warp publishes its partial sum
  __syncthreads();

  // Read back the per-warp sums into the first warp; then reduce again.
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// ---- FP32 vectorized kernel ----------------------------------------
//
// Each block normalises one row.  The work splits into three stages:
//   1. parallel sum-of-squares across the row, accumulating in FP32
//      registers, then a block-wide reduction.
//   2. thread 0 turns the sum into the scaling factor `1 / sqrt(mean+eps)`
//      and broadcasts it via shared memory.
//   3. every thread writes its share of the output:  out_i = x_i * scale * w_i

__global__ void rms_norm_f32_kernel(
    const float* __restrict__ x,        // [N, dim] input rows, contiguous
    const float* __restrict__ weight,   // [dim] learnable gain (or nullptr)
    float* __restrict__ out,            // [N, dim] output, same layout as x
    int64_t dim,
    float eps) {
  // One CUDA block == one row. Compute pointers to the start of that row.
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  float* row_out = out + row * dim;

  // ----- Stage 1: per-thread partial sum of x^2 -----
  // We use float4 vector loads so each memory transaction is 16 bytes
  // wide (saturates HBM bandwidth on Ampere/Hopper).
  float sum_sq = 0.0f;
  int64_t vec_dim = dim / 4;
  const float4* x4 = reinterpret_cast<const float4*>(row_x);

  // Each thread strides through the row; for d_model=4096 with 256
  // threads, each thread sees 4 float4s.
  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 v = x4[i];
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  // Tail loop for the (dim % 4) leftover scalars — without it, models
  // whose hidden size isn't a multiple of 4 silently get truncated.
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i];
    sum_sq += v * v;
  }

  // Block-wide reduction. Only thread 0 holds the correct answer after
  // this returns — that's why the next step reads it through shared mem.
  sum_sq = blockReduceSum(sum_sq);

  // ----- Stage 2: compute the scale once, broadcast to all threads -----
  // rsqrtf(z) = 1/sqrt(z), executed in a single hardware instruction on
  // the SFU. We add eps inside the rsqrt to avoid /0 on degenerate rows.
  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();           // make the publication visible to all threads

  float scale = s_rms_scale;  // private copy in each thread's register

  // ----- Stage 3: write the normalised row out -----
  float4* out4 = reinterpret_cast<float4*>(row_out);
  const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 v = x4[i];
    float4 r;
    if (w4) {
      // Apply the learned per-channel gain along with the global scale.
      float4 w = w4[i];
      r.x = v.x * scale * w.x;
      r.y = v.y * scale * w.y;
      r.z = v.z * scale * w.z;
      r.w = v.w * scale * w.w;
    } else {
      // weight==nullptr is the "no learnable gain" case (rare; some
      // QK-norm variants pass nullptr because the gain would be redundant).
      r.x = v.x * scale;
      r.y = v.y * scale;
      r.z = v.z * scale;
      r.w = v.w * scale;
    }
    out4[i] = r;
  }
  // Tail loop matching the load tail above.
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i] * scale;
    if (weight) v *= weight[i];
    row_out[i] = v;
  }
}

// ---- BF16 kernel with FP32 accumulation ----------------------------
//
// Same algorithm as the FP32 version, but we read/write 16-bit values.
// Vectorised loads aren't worth the complexity for BF16 because the
// type is already half the width of FP32; the simpler scalar path is
// already very close to bandwidth-limit on this kernel.

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const __nv_bfloat16* row_x = x + row * dim;
  __nv_bfloat16* row_out = out + row * dim;

  // Stage 1: sum of squares.  CRITICAL: we accumulate in FP32 even
  // though the inputs are BF16.  BF16 has only 7 mantissa bits, so
  // adding thousands of small squared values into a BF16 register
  // would silently quantise them to zero.
  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __bfloat162float(row_x[i]);   // intrinsic = ~1 PTX op
    sum_sq += v * v;
  }

  // Stage 1b: block-wide reduction (same primitive as the FP32 path).
  sum_sq = blockReduceSum(sum_sq);

  // Stage 2: compute & broadcast the scale.
  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  // Stage 3: write normalised+gained row.  Upcast for the multiply,
  // downcast just before the store so the on-disk format stays BF16.
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __bfloat162float(row_x[i]) * scale;
    if (weight) v *= __bfloat162float(weight[i]);
    row_out[i] = __float2bfloat16(v);
  }
}

// ---- FP32 residual + RMSNorm --------------------------------------
//
// "Residual + RMSNorm" is the fused version of the canonical pre-norm
// transformer pattern:
//
//     x = x + sublayer_output(prev_x)        // the residual add
//     h = rms_norm(x)                        // the norm
//
// Doing both in one kernel saves the round-trip read/write of the
// post-add tensor.  We need TWO outputs: the post-norm `h` (consumed
// by the next sublayer) and the post-add `x` itself (kept around for
// the NEXT residual add at the end of the next sublayer).

__global__ void residual_rms_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float* __restrict__ residual_out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x = x + row * dim;
  const float* row_res = residual + row * dim;
  float* row_out = out + row * dim;
  float* row_res_out = residual_out + row * dim;

  float sum_sq = 0.0f;
  int64_t vec_dim = dim / 4;
  const float4* x4 = reinterpret_cast<const float4*>(row_x);
  const float4* res4 = reinterpret_cast<const float4*>(row_res);
  float4* res_out4 = reinterpret_cast<float4*>(row_res_out);

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 xv = x4[i];
    float4 rv = res4[i];
    float4 h;
    h.x = xv.x + rv.x;
    h.y = xv.y + rv.y;
    h.z = xv.z + rv.z;
    h.w = xv.w + rv.w;
    res_out4[i] = h;
    sum_sq += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
  }
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float h = row_x[i] + row_res[i];
    row_res_out[i] = h;
    sum_sq += h * h;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  float4* out4 = reinterpret_cast<float4*>(row_out);
  const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;

  for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    float4 h = res_out4[i];
    float4 r;
    if (w4) {
      float4 w = w4[i];
      r.x = h.x * scale * w.x;
      r.y = h.y * scale * w.y;
      r.z = h.z * scale * w.z;
      r.w = h.w * scale * w.w;
    } else {
      r.x = h.x * scale;
      r.y = h.y * scale;
      r.z = h.z * scale;
      r.w = h.w * scale;
    }
    out4[i] = r;
  }
  for (int64_t i = vec_dim * 4 + threadIdx.x; i < dim; i += blockDim.x) {
    float h = row_res_out[i] * scale;
    if (weight) h *= weight[i];
    row_out[i] = h;
  }
}

// ---- FP32 rms_norm_add: out = residual + rms_norm(x) * weight ----
//
// Semantics differ from residual_rms_norm above:
//   residual_rms_norm : out = rms_norm(x + residual)             ← add-then-norm
//   rms_norm_add      : out = residual + rms_norm(x) * weight    ← norm-then-add
//
// rms_norm_add matches reordered-norm transformer blocks (OLMo-2, LLaMA-2)
// where the sublayer output is normalized BEFORE being merged back into the
// residual stream. Two kernel-launches collapse to one; the bf16 variant
// keeps the I/O dtype while doing the reduction in fp32 (essential because
// summing thousands of squared values in bf16 silently underflows). This is
// item H from the optimization roadmap.

__global__ void rms_norm_add_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const float* row_x   = x + row * dim;
  const float* row_res = residual + row * dim;
  float* row_out       = out + row * dim;

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_x[i];
    sum_sq += v * v;
  }
  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_scale;
  if (threadIdx.x == 0) {
    s_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  float scale = s_scale;

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float normed = row_x[i] * scale;
    if (weight) normed *= weight[i];
    row_out[i] = row_res[i] + normed;
  }
}

__global__ void rms_norm_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const __nv_bfloat16* row_x   = x + row * dim;
  const __nv_bfloat16* row_res = residual + row * dim;
  __nv_bfloat16* row_out       = out + row * dim;

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __bfloat162float(row_x[i]);
    sum_sq += v * v;
  }
  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_scale;
  if (threadIdx.x == 0) {
    s_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  float scale = s_scale;

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float normed = __bfloat162float(row_x[i]) * scale;
    if (weight) normed *= __bfloat162float(weight[i]);
    float ri = __bfloat162float(row_res[i]);
    row_out[i] = __float2bfloat16(ri + normed);
  }
}

// ---- BF16 residual + RMSNorm ----

__global__ void residual_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ residual_out,
    int64_t dim,
    float eps) {
  int64_t row = blockIdx.x;
  const __nv_bfloat16* row_x = x + row * dim;
  const __nv_bfloat16* row_res = residual + row * dim;
  __nv_bfloat16* row_out = out + row * dim;
  __nv_bfloat16* row_res_out = residual_out + row * dim;

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float h = __bfloat162float(row_x[i]) + __bfloat162float(row_res[i]);
    row_res_out[i] = __float2bfloat16(h);
    sum_sq += h * h;
  }

  sum_sq = blockReduceSum(sum_sq);

  __shared__ float s_rms_scale;
  if (threadIdx.x == 0) {
    s_rms_scale = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  float scale = s_rms_scale;

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float h = __bfloat162float(row_res_out[i]) * scale;
    if (weight) h *= __bfloat162float(weight[i]);
    row_out[i] = __float2bfloat16(h);
  }
}

}  // namespace

// ---- C++ dispatch entry points -------------------------------------
//
// Each *_cuda_impl function:
//   1. validates inputs,
//   2. forces a contiguous memory layout (the kernels assume row-major
//      pack with stride 1 on the last dim),
//   3. allocates outputs,
//   4. picks block size (smaller blocks for short rows so we don't
//      waste threads),
//   5. launches the FP32 or BF16 kernel based on input dtype.

torch::Tensor rms_norm_cuda_impl(
    const torch::Tensor& x,                              // [..., dim]
    const c10::optional<torch::Tensor>& weight,          // [dim] or none
    double eps) {
  // Refuse non-CUDA tensors and 0-D / 1-D inputs (last-axis is the norm axis).
  TORCH_CHECK(x.dim() >= 2 && x.is_cuda());
  auto x_contig = x.contiguous();
  auto dim = x_contig.size(-1);                          // length of one row
  auto rows = x_contig.numel() / dim;                    // total row count
  auto out = torch::empty_like(x_contig);
  c10::cuda::CUDAGuard device_guard(x.device());

  // Heuristic: short rows don't benefit from a 256-thread block.
  // 128 threads keeps occupancy reasonable while reducing wasted lanes.
  int threads = (dim <= 256) ? 128 : 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kBFloat16) {
    rms_norm_bf16_kernel<<<rows, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        weight.has_value() ? reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr<at::BFloat16>()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        dim, static_cast<float>(eps));
  } else {
    rms_norm_f32_kernel<<<rows, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        weight.has_value() ? weight->data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        dim, static_cast<float>(eps));
  }
  return out;
}

std::vector<torch::Tensor> residual_rms_norm_cuda_impl(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda());
  auto x_contig = x.contiguous();
  auto res_contig = residual.contiguous();
  auto dim = x_contig.size(-1);
  auto rows = x_contig.numel() / dim;
  auto out = torch::empty_like(x_contig);
  auto residual_out = torch::empty_like(x_contig);
  c10::cuda::CUDAGuard device_guard(x.device());

  int threads = (dim <= 256) ? 128 : 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kBFloat16) {
    residual_rms_norm_bf16_kernel<<<rows, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(res_contig.data_ptr<at::BFloat16>()),
        weight.has_value() ? reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr<at::BFloat16>()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        dim, static_cast<float>(eps));
  } else {
    residual_rms_norm_f32_kernel<<<rows, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        res_contig.data_ptr<float>(),
        weight.has_value() ? weight->data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        residual_out.data_ptr<float>(),
        dim, static_cast<float>(eps));
  }
  return {out, residual_out};
}

// rms_norm_add dispatch (item H): one-pass `out = residual + rms_norm(x) * weight`.
torch::Tensor rms_norm_add_cuda_impl(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda(),
              "rms_norm_add_cuda_impl: inputs must be CUDA");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "rms_norm_add_cuda_impl: x and residual must have the same shape");
  auto x_contig   = x.contiguous();
  auto res_contig = residual.contiguous();
  const int64_t dim  = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / dim;
  auto out = torch::empty_like(x_contig);
  c10::cuda::CUDAGuard guard(x.device());
  int threads = (dim <= 256) ? 128 : 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kBFloat16) {
    TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
                "rms_norm_add_cuda_impl: x is bf16 but residual is not");
    rms_norm_add_bf16_kernel<<<rows, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(res_contig.data_ptr<at::BFloat16>()),
        weight.has_value() ? reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr<at::BFloat16>()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        dim, static_cast<float>(eps));
  } else {
    rms_norm_add_f32_kernel<<<rows, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        res_contig.data_ptr<float>(),
        weight.has_value() ? weight->data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        dim, static_cast<float>(eps));
  }
  return out;
}

// Library registration
TORCH_LIBRARY(olmo_ops, m) {
  m.def("rms_norm(Tensor x, Tensor? weight, float eps=1e-6) -> Tensor");
  m.def("residual_rms_norm(Tensor x, Tensor residual, Tensor? weight, float eps=1e-6) -> Tensor[]");
  m.def("rms_norm_add(Tensor x, Tensor residual, Tensor? weight, float eps=1e-6) -> Tensor");
  m.def("apply_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor");
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
  m.def("apply_rope_qk(Tensor q, Tensor k, Tensor cos_q, Tensor sin_q, Tensor cos_k, Tensor sin_k) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("rms_norm", rms_norm_cuda_impl);
  m.impl("residual_rms_norm", residual_rms_norm_cuda_impl);
  m.impl("rms_norm_add", rms_norm_add_cuda_impl);
}
