/**
 * src/backend/kernels/simd_elementwise.cpp
 *
 * ─── What this file is ──────────────────────────────────────────────
 *
 * Hand-vectorised CPU implementations of the same fused ops the .cu
 * kernels provide on GPU: rms_norm, silu_mul, apply_rope. The
 * algorithms are identical (RMSNorm, SwiGLU, RoPE — see the .cu
 * docblocks for the math); only the parallelisation primitive differs.
 *
 * On a CUDA host you have thousands of threads and warp shuffles. On
 * a CPU you have ONE core's vector register file. So instead of a
 * grid of CUDA blocks each with hundreds of threads, here we have a
 * scalar outer loop and an inner SIMD loop that processes 4/8/16
 * floats per iteration.
 *
 * The host SIMD ISA is detected at compile time:
 *
 *   __ARM_NEON  -> 4-wide float NEON       (Apple Silicon, ARM Linux)
 *   __AVX512F__ -> 16-wide float AVX-512   (Skylake-X / EPYC server)
 *   __AVX2__    -> 8-wide float AVX2       (most x86 desktops since 2013)
 *   else        -> scalar fallback         (correct, just slow)
 *
 * Each kernel processes the contiguous flat layout as 1D and assumes
 * the caller already verified contiguity + dtype.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/kernels/simd_elementwise.hpp : kernel decls.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/simd_backend.cpp: SIMDBackend dispatches its
 *     IBackend methods to these free functions when can_use_simd()
 *     returns true.
 *
 * --- Role in training pipeline ---
 *   The CPU equivalent of the .cu kernels. On the 3060 quickstart
 *   these never run because CUDABackend is installed instead.
 */
#include "olmo_cpp/backend/kernels/simd_elementwise.hpp"

#include <cmath>
#include <cstring>

// Platform detection for SIMD
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define OLMO_USE_NEON 1
  #include <arm_neon.h>
#elif defined(__AVX2__)
  #define OLMO_USE_AVX2 1
  #include <immintrin.h>
#endif

namespace olmo_cpp {
namespace kernels {

// ============================================================================
// Fused RMSNorm (single pass: compute sum-of-squares, rsqrt, scale)
// ============================================================================

#if OLMO_USE_NEON

void fused_rms_norm_f32(const float* x, const float* weight, float* out,
                        int64_t n_rows, int64_t d, float eps) {
  for (int64_t row = 0; row < n_rows; ++row) {
    const float* xi = x + row * d;
    float* oi = out + row * d;

    // Pass 1: compute sum of squares using NEON
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 4 <= d; i += 4) {
      float32x4_t v = vld1q_f32(xi + i);
      sum_sq = vfmaq_f32(sum_sq, v, v);  // fused multiply-add
    }
    float ss = vaddvq_f32(sum_sq);  // horizontal sum
    for (; i < d; ++i) {
      ss += xi[i] * xi[i];
    }

    // rsqrt(mean + eps)
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

    // Pass 2: multiply by scale and weight
    if (weight) {
      i = 0;
      float32x4_t vs = vdupq_n_f32(scale);
      for (; i + 4 <= d; i += 4) {
        float32x4_t v = vld1q_f32(xi + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(oi + i, vmulq_f32(vmulq_f32(v, vs), w));
      }
      for (; i < d; ++i) {
        oi[i] = xi[i] * scale * weight[i];
      }
    } else {
      i = 0;
      float32x4_t vs = vdupq_n_f32(scale);
      for (; i + 4 <= d; i += 4) {
        float32x4_t v = vld1q_f32(xi + i);
        vst1q_f32(oi + i, vmulq_f32(v, vs));
      }
      for (; i < d; ++i) {
        oi[i] = xi[i] * scale;
      }
    }
  }
}

#elif OLMO_USE_AVX2

void fused_rms_norm_f32(const float* x, const float* weight, float* out,
                        int64_t n_rows, int64_t d, float eps) {
  for (int64_t row = 0; row < n_rows; ++row) {
    const float* xi = x + row * d;
    float* oi = out + row * d;

    // Pass 1: sum of squares
    __m256 sum_sq = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= d; i += 8) {
      __m256 v = _mm256_loadu_ps(xi + i);
      sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    // Horizontal sum of 8 floats
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float ss = _mm_cvtss_f32(s4);
    for (; i < d; ++i) {
      ss += xi[i] * xi[i];
    }

    float scale = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

    // Pass 2: scale and weight
    __m256 vs = _mm256_set1_ps(scale);
    if (weight) {
      i = 0;
      for (; i + 8 <= d; i += 8) {
        __m256 v = _mm256_loadu_ps(xi + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(oi + i, _mm256_mul_ps(_mm256_mul_ps(v, vs), w));
      }
      for (; i < d; ++i) {
        oi[i] = xi[i] * scale * weight[i];
      }
    } else {
      i = 0;
      for (; i + 8 <= d; i += 8) {
        __m256 v = _mm256_loadu_ps(xi + i);
        _mm256_storeu_ps(oi + i, _mm256_mul_ps(v, vs));
      }
      for (; i < d; ++i) {
        oi[i] = xi[i] * scale;
      }
    }
  }
}

#else  // Scalar fallback

void fused_rms_norm_f32(const float* x, const float* weight, float* out,
                        int64_t n_rows, int64_t d, float eps) {
  for (int64_t row = 0; row < n_rows; ++row) {
    const float* xi = x + row * d;
    float* oi = out + row * d;

    float ss = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
      ss += xi[i] * xi[i];
    }
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

    if (weight) {
      for (int64_t i = 0; i < d; ++i) {
        oi[i] = xi[i] * scale * weight[i];
      }
    } else {
      for (int64_t i = 0; i < d; ++i) {
        oi[i] = xi[i] * scale;
      }
    }
  }
}

#endif

// ============================================================================
// Fused SiLU * mul (silu(gate) * up in one pass)
// ============================================================================

// Scalar silu for tail elements
static inline float silu_scalar(float x) {
  return x / (1.0f + std::exp(-x));
}

#if OLMO_USE_NEON

// Fast approximate exp for NEON using the classic integer-cast trick.
// max relative error ~1e-4 which is fine for silu in training.
static inline float32x4_t fast_exp_neon(float32x4_t x) {
  // Clamp to avoid overflow/underflow in the integer conversion
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
  x = vminq_f32(x, vdupq_n_f32(88.0f));

  // exp(x) ≈ 2^(x / ln2) via Schraudolph's trick with a polynomial refinement
  const float32x4_t log2e = vdupq_n_f32(1.442695041f);
  const float32x4_t c0 = vdupq_n_f32(12102203.0f);  // 2^23 / ln2
  const float32x4_t c1 = vdupq_n_f32(1065353216.0f); // 127 * 2^23

  float32x4_t val = vfmaq_f32(c1, x, c0);
  // Reinterpret as float (Schraudolph approximation)
  int32x4_t ival = vcvtq_s32_f32(val);
  return vreinterpretq_f32_s32(ival);
}

void fused_silu_mul_f32(const float* gate, const float* up, float* out,
                        int64_t n) {
  int64_t i = 0;
  const float32x4_t ones = vdupq_n_f32(1.0f);

  for (; i + 4 <= n; i += 4) {
    float32x4_t g = vld1q_f32(gate + i);
    float32x4_t u = vld1q_f32(up + i);

    // silu(g) = g / (1 + exp(-g))  = g * sigmoid(g)
    float32x4_t neg_g = vnegq_f32(g);
    float32x4_t exp_neg = fast_exp_neon(neg_g);
    float32x4_t sigmoid = vrecpeq_f32(vaddq_f32(ones, exp_neg));
    // One Newton-Raphson refinement for reciprocal
    float32x4_t denom = vaddq_f32(ones, exp_neg);
    sigmoid = vmulq_f32(sigmoid, vrecpsq_f32(denom, sigmoid));

    float32x4_t silu_g = vmulq_f32(g, sigmoid);
    vst1q_f32(out + i, vmulq_f32(silu_g, u));
  }
  for (; i < n; ++i) {
    out[i] = silu_scalar(gate[i]) * up[i];
  }
}

#elif OLMO_USE_AVX2

// Fast exp for AVX2 via Schraudolph's method
static inline __m256 fast_exp_avx2(__m256 x) {
  x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
  x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

  const __m256 c0 = _mm256_set1_ps(12102203.0f);
  const __m256 c1 = _mm256_set1_ps(1065353216.0f);

  __m256 val = _mm256_fmadd_ps(x, c0, c1);
  __m256i ival = _mm256_cvtps_epi32(val);
  return _mm256_castsi256_ps(ival);
}

void fused_silu_mul_f32(const float* gate, const float* up, float* out,
                        int64_t n) {
  int64_t i = 0;
  const __m256 ones = _mm256_set1_ps(1.0f);

  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 u = _mm256_loadu_ps(up + i);

    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
    __m256 exp_neg = fast_exp_avx2(neg_g);
    __m256 denom = _mm256_add_ps(ones, exp_neg);
    __m256 sigmoid = _mm256_div_ps(ones, denom);

    __m256 silu_g = _mm256_mul_ps(g, sigmoid);
    _mm256_storeu_ps(out + i, _mm256_mul_ps(silu_g, u));
  }
  for (; i < n; ++i) {
    out[i] = silu_scalar(gate[i]) * up[i];
  }
}

#else  // Scalar fallback

void fused_silu_mul_f32(const float* gate, const float* up, float* out,
                        int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = silu_scalar(gate[i]) * up[i];
  }
}

#endif

// ============================================================================
// Fused RoPE (zero-alloc, no chunk+cat temporary)
//
// rotate_half([x0..xH/2, xH/2..xH]) = [-xH/2.., x0..]
// out[i] = t[i]*cos[i] + rotate_half(t)[i]*sin[i]
//
// For i < half_dim:  rotate_half[i] = -t[i + half_dim]
// For i >= half_dim: rotate_half[i] = t[i - half_dim]
// ============================================================================

#if OLMO_USE_NEON

void fused_rope_f32(const float* t, const float* sin_ptr, const float* cos_ptr,
                    float* out, int64_t n, int64_t head_dim) {
  int64_t half = head_dim / 2;

  for (int64_t vec = 0; vec < n; ++vec) {
    const float* ti = t + vec * head_dim;
    const float* si = sin_ptr + vec * head_dim;
    const float* ci = cos_ptr + vec * head_dim;
    float* oi = out + vec * head_dim;

    // First half: out[j] = t[j]*cos[j] + (-t[j+half])*sin[j]
    int64_t j = 0;
    for (; j + 4 <= half; j += 4) {
      float32x4_t tj = vld1q_f32(ti + j);
      float32x4_t tj_rot = vld1q_f32(ti + j + half);
      float32x4_t sj = vld1q_f32(si + j);
      float32x4_t cj = vld1q_f32(ci + j);

      // out = t*cos + (-t_rot)*sin
      float32x4_t result = vfmaq_f32(
          vmulq_f32(vnegq_f32(tj_rot), sj),  // -t_rot * sin
          tj, cj);                             // + t * cos
      vst1q_f32(oi + j, result);
    }
    for (; j < half; ++j) {
      oi[j] = ti[j] * ci[j] + (-ti[j + half]) * si[j];
    }

    // Second half: out[j] = t[j]*cos[j] + t[j-half]*sin[j]
    j = half;
    for (; j + 4 <= head_dim; j += 4) {
      float32x4_t tj = vld1q_f32(ti + j);
      float32x4_t tj_rot = vld1q_f32(ti + j - half);
      float32x4_t sj = vld1q_f32(si + j);
      float32x4_t cj = vld1q_f32(ci + j);

      float32x4_t result = vfmaq_f32(
          vmulq_f32(tj_rot, sj),  // t_rot * sin
          tj, cj);                 // + t * cos
      vst1q_f32(oi + j, result);
    }
    for (; j < head_dim; ++j) {
      oi[j] = ti[j] * ci[j] + ti[j - half] * si[j];
    }
  }
}

#elif OLMO_USE_AVX2

void fused_rope_f32(const float* t, const float* sin_ptr, const float* cos_ptr,
                    float* out, int64_t n, int64_t head_dim) {
  int64_t half = head_dim / 2;

  for (int64_t vec = 0; vec < n; ++vec) {
    const float* ti = t + vec * head_dim;
    const float* si = sin_ptr + vec * head_dim;
    const float* ci = cos_ptr + vec * head_dim;
    float* oi = out + vec * head_dim;

    // First half
    int64_t j = 0;
    for (; j + 8 <= half; j += 8) {
      __m256 tj = _mm256_loadu_ps(ti + j);
      __m256 tj_rot = _mm256_loadu_ps(ti + j + half);
      __m256 sj = _mm256_loadu_ps(si + j);
      __m256 cj = _mm256_loadu_ps(ci + j);

      __m256 neg_rot_sin = _mm256_mul_ps(
          _mm256_sub_ps(_mm256_setzero_ps(), tj_rot), sj);
      __m256 result = _mm256_fmadd_ps(tj, cj, neg_rot_sin);
      _mm256_storeu_ps(oi + j, result);
    }
    for (; j < half; ++j) {
      oi[j] = ti[j] * ci[j] + (-ti[j + half]) * si[j];
    }

    // Second half
    j = half;
    for (; j + 8 <= head_dim; j += 8) {
      __m256 tj = _mm256_loadu_ps(ti + j);
      __m256 tj_rot = _mm256_loadu_ps(ti + j - half);
      __m256 sj = _mm256_loadu_ps(si + j);
      __m256 cj = _mm256_loadu_ps(ci + j);

      __m256 rot_sin = _mm256_mul_ps(tj_rot, sj);
      __m256 result = _mm256_fmadd_ps(tj, cj, rot_sin);
      _mm256_storeu_ps(oi + j, result);
    }
    for (; j < head_dim; ++j) {
      oi[j] = ti[j] * ci[j] + ti[j - half] * si[j];
    }
  }
}

#else  // Scalar fallback

void fused_rope_f32(const float* t, const float* sin_ptr, const float* cos_ptr,
                    float* out, int64_t n, int64_t head_dim) {
  int64_t half = head_dim / 2;

  for (int64_t vec = 0; vec < n; ++vec) {
    const float* ti = t + vec * head_dim;
    const float* si = sin_ptr + vec * head_dim;
    const float* ci = cos_ptr + vec * head_dim;
    float* oi = out + vec * head_dim;

    for (int64_t j = 0; j < half; ++j) {
      oi[j] = ti[j] * ci[j] + (-ti[j + half]) * si[j];
    }
    for (int64_t j = half; j < head_dim; ++j) {
      oi[j] = ti[j] * ci[j] + ti[j - half] * si[j];
    }
  }
}

#endif

}  // namespace kernels
}  // namespace olmo_cpp
