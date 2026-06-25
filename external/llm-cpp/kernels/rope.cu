/**
 * kernels/rope.cu
 *
 * ─── What RoPE is ────────────────────────────────────────────────────
 *
 * "RoPE" = **Ro**tary **P**osition **E**mbedding (Su et al. 2021).
 * It's how this transformer tells the attention mechanism *where* in
 * the sequence each token sits.
 *
 * Older models used additive sinusoidal position embeddings: you
 * compute a fixed vector for each position and add it to the token
 * embedding before feeding it into the network.  RoPE takes a very
 * different approach: it leaves the token embeddings alone and instead
 * **rotates** the query and key vectors in attention by an angle that
 * depends on the token's position.
 *
 * Concretely: pair up the head_dim D into D/2 disjoint 2-D pairs
 * (x_0, x_{D/2}), (x_1, x_{D/2+1}), ... and rotate each pair by an
 * angle theta_p = position * base^{-2j/D}, where j indexes the pair.
 * In matrix form, for one pair:
 *
 *     [ x'_a ]   [  cos θ   -sin θ ] [ x_a ]
 *     [ x'_b ] = [  sin θ    cos θ ] [ x_b ]
 *
 * Each pair is rotated by a different frequency — high-frequency pairs
 * carry fine positional information, low-frequency pairs carry coarse
 * sequence-position information.
 *
 * Why this is nice:
 *   - The dot product Q·K depends only on the *relative* offset
 *     between the two tokens, not their absolute positions. That's
 *     exactly what attention should care about.
 *   - It extrapolates to longer sequences than were seen at training
 *     better than additive position embeddings, especially with the
 *     scaling tricks in src/model/rope_scaling.cpp (YaRN, ABF, etc.).
 *
 * The cos/sin tables are precomputed once in src/model/rope.cpp and
 * passed in as `cos_buf` / `sin_buf`.  This kernel just consumes them.
 *
 * ─── How the in-place rotation is implemented here ───────────────────
 *
 * Reading the kernel body it looks weird: we do
 *
 *     if col < D/2:  x_rot = -x[row, col + D/2]
 *     else:          x_rot =  x[row, col - D/2]
 *     out[col] = x[col]*cos[col] + x_rot * sin[col]
 *
 * This is the standard "interleaved-pair" formulation.  Splitting the
 * D-dim vector into a "first half" and "second half", you can show
 * that (x_a, x_b) -> (x_a cos - x_b sin, x_a sin + x_b cos) is
 * equivalent to writing each output element as
 *     out_i = x_i * cos_i + rot_i * sin_i
 * where rot_i is "the partner element with a sign depending on which
 * half of the vector you're in".  This avoids the conditional pair
 * indexing you'd otherwise need.
 *
 * ─── Launch geometry ─────────────────────────────────────────────────
 *
 * One thread per element (Q is shape [B*S*H, D]).  Grid-stride loop so
 * the same kernel handles small sequences and 32k-context sequences.
 * The "qk" variant fuses Q and K into the same launch — saves the
 * second kernel launch's overhead, which can dominate for short seqs.
 *
 * ─── Precision ───────────────────────────────────────────────────────
 *
 * BF16 inputs are upcast to FP32 for the trig multiplies because
 * sin/cos of tiny angles in BF16 carry ~7 mantissa bits — enough to
 * hurt long-context attention accuracy.  Outputs are written back as
 * BF16 so the dtype contract with downstream attention is preserved.
 *
 * --- Includes from this project ---
 *   (none — pure CUDA kernel.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/cuda_backend.cpp: CUDABackend::apply_rope() and
 *     ::apply_rope_qk() forward to these implementations.
 *   - src/model/attention.cpp / fused_attention.cpp:
 *     get_backend().apply_rope(q, k, ...) is called right before the
 *     SDPA call inside every attention block.
 *
 * --- Role in training pipeline ---
 *   Q and K each get one RoPE invocation per attention block per
 *   microbatch. Fusing the rotation into a single kernel halves the
 *   memory traffic compared to ATen's "two multiplies + an add" recipe.
 */
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

// ---- FP32 single-tensor RoPE -----------------------------------------
//
// Treat the input as a flat array of shape [..., D] where D is head_dim
// (always even). For each element at column `col`, find its pair
// partner at col±D/2, and apply the 2-D rotation matrix entry-by-entry.

__global__ void apply_rope_f32_kernel(
    const float* __restrict__ x,         // [..., D] queries or keys, contiguous
    const float* __restrict__ cos_buf,   // [D]   precomputed cosines
    const float* __restrict__ sin_buf,   // [D]   precomputed sines
    float* __restrict__ out,             // [..., D] rotated output
    int64_t total_elements,              // numel(x)
    int64_t dim) {                       // D (head_dim)
  int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half   = dim / 2;              // boundary between "first half" and "second half"

  // Grid-stride loop: every thread processes one element per iteration.
  for (int64_t i = idx; i < total_elements; i += stride) {
    int64_t col = i % dim;     // position within the head-dim
    int64_t row = i / dim;     // which (batch, seq, head) row we're in
    float x_val = x[i];
    float x_rot;
    // Find the rotation partner. The sign flip in the "first half"
    // branch is exactly the -sin term of the 2x2 rotation matrix.
    if (col < half) {
      x_rot = -x[row * dim + col + half];
    } else {
      x_rot = x[row * dim + col - half];
    }
    // Apply the rotation.  cos_buf[col] and sin_buf[col] together
    // encode the position-dependent angle for this dim slot.
    out[i] = x_val * cos_buf[col] + x_rot * sin_buf[col];
  }
}

// ---- BF16 single-tensor RoPE with FP32 compute -----------------------
//
// Same algorithm as the FP32 path, but inputs/outputs are 16-bit and
// the trig multiplies happen in FP32 to keep the cos/sin precision.
// __bfloat162float / __float2bfloat16 are single-PTX-op intrinsics.

__global__ void apply_rope_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    __nv_bfloat16* __restrict__ out,
    int64_t total_elements,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;

  for (int64_t i = idx; i < total_elements; i += stride) {
    int64_t col = i % dim;
    int64_t row = i / dim;
    float x_val = __bfloat162float(x[i]);
    float x_rot;
    if (col < half) {
      x_rot = -__bfloat162float(x[row * dim + col + half]);
    } else {
      x_rot = __bfloat162float(x[row * dim + col - half]);
    }
    float c = __bfloat162float(cos_buf[col]);
    float s = __bfloat162float(sin_buf[col]);
    out[i] = __float2bfloat16(x_val * c + x_rot * s);
  }
}

// ---- FP32 fused Q+K RoPE -------------------------------------------
//
// Q and K both need RoPE before being fed to attention.  Instead of
// launching two kernels we issue one whose total iteration count is
// q_total + k_total. The first q_total iterations process Q; the rest
// process K. Halves the launch overhead — especially noticeable for
// short sequences where each kernel is microseconds.
//
// Q and K may have DIFFERENT cos/sin buffers when GQA (grouped-query
// attention) is in use with different scaling factors per stream.

__global__ void apply_rope_qk_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_q,
    const float* __restrict__ sin_q,
    const float* __restrict__ cos_k,
    const float* __restrict__ sin_k,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int64_t q_total,
    int64_t k_total,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;
  int64_t combined_total = q_total + k_total;

  for (int64_t i = idx; i < combined_total; i += stride) {
    bool is_q = (i < q_total);
    int64_t local_i = is_q ? i : (i - q_total);
    const float* src = is_q ? q : k;
    float* dst = is_q ? q_out : k_out;
    const float* cos_ptr = is_q ? cos_q : cos_k;
    const float* sin_ptr = is_q ? sin_q : sin_k;

    int64_t col = local_i % dim;
    int64_t row = local_i / dim;

    float x_val = src[local_i];
    float x_rot;
    if (col < half) {
      x_rot = -src[row * dim + col + half];
    } else {
      x_rot = src[row * dim + col - half];
    }
    dst[local_i] = x_val * cos_ptr[col] + x_rot * sin_ptr[col];
  }
}

// ---- BF16 fused Q+K RoPE ----

__global__ void apply_rope_qk_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos_q,
    const __nv_bfloat16* __restrict__ sin_q,
    const __nv_bfloat16* __restrict__ cos_k,
    const __nv_bfloat16* __restrict__ sin_k,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int64_t q_total,
    int64_t k_total,
    int64_t dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t half = dim / 2;
  int64_t combined_total = q_total + k_total;

  for (int64_t i = idx; i < combined_total; i += stride) {
    bool is_q = (i < q_total);
    int64_t local_i = is_q ? i : (i - q_total);
    const __nv_bfloat16* src = is_q ? q : k;
    __nv_bfloat16* dst = is_q ? q_out : k_out;
    const __nv_bfloat16* cos_ptr = is_q ? cos_q : cos_k;
    const __nv_bfloat16* sin_ptr = is_q ? sin_q : sin_k;

    int64_t col = local_i % dim;
    int64_t row = local_i / dim;

    float x_val = __bfloat162float(src[local_i]);
    float x_rot;
    if (col < half) {
      x_rot = -__bfloat162float(src[row * dim + col + half]);
    } else {
      x_rot = __bfloat162float(src[row * dim + col - half]);
    }
    float c = __bfloat162float(cos_ptr[col]);
    float s = __bfloat162float(sin_ptr[col]);
    dst[local_i] = __float2bfloat16(x_val * c + x_rot * s);
  }
}

}  // namespace

// ---- Dispatch ----

torch::Tensor apply_rope_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda());
  auto x_c = x.contiguous();
  auto cos_c = cos.contiguous();
  auto sin_c = sin.contiguous();
  auto out = torch::empty_like(x_c);
  auto numel = x_c.numel();
  auto dim = x_c.size(-1);
  c10::cuda::CUDAGuard device_guard(x.device());

  int threads = 256;
  int blocks = std::min(static_cast<int64_t>((numel + threads - 1) / threads),
                        static_cast<int64_t>(65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kBFloat16) {
    apply_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        numel, dim);
  } else {
    apply_rope_f32_kernel<<<blocks, threads, 0, stream>>>(
        x_c.data_ptr<float>(),
        cos_c.data_ptr<float>(),
        sin_c.data_ptr<float>(),
        out.data_ptr<float>(),
        numel, dim);
  }
  return out;
}

std::vector<torch::Tensor> apply_rope_qk_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_q,
    const torch::Tensor& sin_q,
    const torch::Tensor& cos_k,
    const torch::Tensor& sin_k) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda());
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto cos_q_c = cos_q.contiguous();
  auto sin_q_c = sin_q.contiguous();
  auto cos_k_c = cos_k.contiguous();
  auto sin_k_c = sin_k.contiguous();

  auto q_out = torch::empty_like(q_c);
  auto k_out = torch::empty_like(k_c);
  auto q_total = q_c.numel();
  auto k_total = k_c.numel();
  auto dim = q_c.size(-1);
  c10::cuda::CUDAGuard device_guard(q.device());

  int threads = 256;
  int64_t combined = q_total + k_total;
  int blocks = std::min((combined + threads - 1) / threads,
                        static_cast<int64_t>(65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == torch::kBFloat16) {
    apply_rope_qk_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_q_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_k_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        q_total, k_total, dim);
  } else {
    apply_rope_qk_f32_kernel<<<blocks, threads, 0, stream>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        cos_q_c.data_ptr<float>(),
        sin_q_c.data_ptr<float>(),
        cos_k_c.data_ptr<float>(),
        sin_k_c.data_ptr<float>(),
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        q_total, k_total, dim);
  }

  return {q_out, k_out};
}

TORCH_LIBRARY_IMPL(olmo_ops, CUDA, m) {
  m.impl("apply_rope", apply_rope_cuda);
  m.impl("apply_rope_qk", apply_rope_qk_cuda);
}
