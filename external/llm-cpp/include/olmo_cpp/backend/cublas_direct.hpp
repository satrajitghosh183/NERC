#pragma once

/**
 * include/olmo_cpp/backend/cublas_direct.hpp
 *
 * Direct cuBLAS / cuBLASLt entry points that bypass the ATen dispatcher
 * (item L). At ~26k launches/step on the speed-xp-sg profile, the
 * dispatcher itself cost ~150 ms/step (4-7 µs/call); the fast path
 * keeps a single cuBLASLt handle + a per-shape descriptor cache so
 * the inner hot ops jump straight into the cuBLAS launcher.
 *
 * Scope right now:
 *   - fast_linear(x, weight, bias): equivalent to torch::nn::functional::linear
 *     for 2-D weight, single launch.
 *   - fast_bmm(a, b): batched matmul for attention's Q·K and ·V.
 *
 * Non-CUDA / non-supported-dtype paths fall back to the regular ATen
 * implementation; numerics are identical.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// y = x @ weight.T + bias. weight: [out_features, in_features], x: [..., in].
/// Output: [..., out]. Bias is optional (pass undefined tensor to skip).
torch::Tensor fast_linear(torch::Tensor x,
                           torch::Tensor weight,
                           torch::Tensor bias = torch::Tensor());

/// Batched matmul over leading dims of a, b. Equivalent to torch::matmul
/// for 3-D / 4-D inputs but avoids the dispatcher.
torch::Tensor fast_bmm(torch::Tensor a, torch::Tensor b);

/// General 2-D matmul: y = op(a) @ op(b), where op(t) is t.T when the
/// corresponding flag is true. Inputs must already be 2-D (callers
/// flatten higher-rank tensors). Used by the FFN backward and other
/// hot paths that need a transpose flag torch::matmul can't express
/// without an explicit .t().contiguous() copy.
torch::Tensor fast_matmul(torch::Tensor a,
                           torch::Tensor b,
                           bool transa = false,
                           bool transb = false);

/// Force-reset the cached cuBLAS plans (e.g. after dtype changes).
void cublas_direct_reset_cache();

}  // namespace olmo_cpp
