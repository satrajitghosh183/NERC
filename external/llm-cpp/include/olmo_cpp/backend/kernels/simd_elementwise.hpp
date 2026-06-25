#pragma once

/**
 * include/olmo_cpp/backend/kernels/simd_elementwise.hpp
 *
 * Pure-C-style entry points for the CPU SIMD elementwise kernels. The
 * three free functions all take raw float* pointers and dimensions —
 * no std::vector, no torch::Tensor — so the SIMD code stays simple,
 * inlinable, and free of LibTorch include dependencies. The .cpp file
 * picks NEON, AVX2 or scalar fallback at compile time via macros.
 *
 * --- Includes from this project ---
 *   - (none — leaf header)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/simd_backend.cpp: SIMDBackend::rms_norm,
 *     SIMDBackend::silu_mul, SIMDBackend::apply_rope each unwrap their
 *     torch::Tensor arguments and call the matching f32 kernel here.
 *
 * --- Role in training pipeline ---
 *   The "fast inner loop" for the CPU path. Everything above this
 *   layer is bookkeeping (shape checks, contiguity, fallback); the
 *   actual FLOPs happen here.
 */

#include <cstdint>

namespace olmo_cpp {
namespace kernels {

/// Fused RMSNorm: single pass computing variance, rsqrt, scale by weight.
/// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i % d]
/// @param x      input data  (n_rows * d contiguous floats)
/// @param weight affine weight (d floats, nullptr to skip)
/// @param out    output buffer (n_rows * d floats)
/// @param n_rows number of rows (batch * seq)
/// @param d      hidden dimension
/// @param eps    epsilon for numerical stability
void fused_rms_norm_f32(const float* x, const float* weight, float* out,
                        int64_t n_rows, int64_t d, float eps);

/// Fused SwiGLU elementwise: out[i] = silu(gate[i]) * up[i]
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Single pass, no temporaries.
void fused_silu_mul_f32(const float* gate, const float* up, float* out,
                        int64_t n);

/// Fused RoPE apply: out = t * cos + rotate_half(t) * sin
/// Zero-allocation: computes rotate_half inline without chunk+cat.
/// @param t    input (n * head_dim contiguous floats)
/// @param sin  sin values (n * head_dim floats, broadcast ok)
/// @param cos  cos values (n * head_dim floats, broadcast ok)
/// @param out  output (n * head_dim floats)
/// @param n    number of vectors (batch * heads * seq_len)
/// @param head_dim  must be even
void fused_rope_f32(const float* t, const float* sin, const float* cos,
                    float* out, int64_t n, int64_t head_dim);

}  // namespace kernels
}  // namespace olmo_cpp
