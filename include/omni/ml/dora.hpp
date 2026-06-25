// omni/ml/dora.hpp — Weight-Decomposed Low-Rank Adaptation (DoRA), from scratch.
//
// DoRA decomposes a frozen pretrained weight W0 (d x k) into a per-column magnitude
// vector m (length k) and a direction, adapting only the direction with a low-rank
// term BA (B: d x r, A: r x k):
//
//     V       = W0 + B*A
//     W'[:,j] = m[j] * V[:,j] / ||V[:,j]||_2          (per output column j)
//
// Trainable: m, A, B  (k + d*r + r*k params)   <<   full fine-tune (d*k).
// Initialised with m[j] = ||W0[:,j]|| and A = B = 0  =>  W' == W0 (exact identity).
// This is the CPU reference for the DoRALinear module to be ported into vendored
// llm-cpp (PLAN.md §12). Beats LoRA because magnitude and direction adapt separately.
#pragma once
#include <cstdint>
#include <vector>

namespace omni::ml {

struct DoRALinear {
    int d = 0;   // output dim (rows of W0)
    int k = 0;   // input dim  (cols of W0)
    int r = 0;   // low-rank
    std::vector<float> W0;  // d*k, row-major: W0[i*k + j]   (frozen base)
    std::vector<float> A;   // r*k, row-major: A[l*k + j]    (trainable)
    std::vector<float> B;   // d*r, row-major: B[i*r + l]    (trainable)
    std::vector<float> m;   // k                              (trainable magnitudes)

    DoRALinear(int d_, int k_, int r_) : d(d_), k(k_), r(r_),
        W0(d_ * k_, 0), A(r_ * k_, 0), B(d_ * r_, 0), m(k_, 0) {}

    // Set m[j] = ||W0[:,j]|| and A=B=0 so W' reproduces W0 exactly.
    void init_from_base();

    // Effective adapted weight W' (d*k, row-major).
    std::vector<float> effective_weight() const;

    // y = W' * x   (x: length k, y: length d).
    std::vector<float> forward(const std::vector<float>& x) const;

    // Trainable parameter count vs a full fine-tune.
    uint64_t trainable_params() const { return (uint64_t)k + (uint64_t)d * r + (uint64_t)r * k; }
    uint64_t full_params() const { return (uint64_t)d * k; }
};

} // namespace omni::ml
