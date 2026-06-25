/**
 * src/optim/muon.cpp
 *
 * ─── What Muon is ────────────────────────────────────────────────────
 *
 * Muon (Bernstein 2024) is a relatively new optimizer that has been
 * reporting AdamW-beating results on transformer pre-training. The
 * idea, in plain English:
 *
 *   - Take the usual momentum buffer m_t = β m_{t-1} + g_t.
 *   - Treat m_t as a 2-D matrix (it usually IS — Linear weight grads
 *     are 2-D). Compute the matrix's "polar factor": the orthogonal
 *     part U V^T from its SVD U Σ V^T. This bounds every singular
 *     value to 1 — every direction the optimizer steps in has equal
 *     scale.
 *   - Step: w_t ← w_{t-1} − lr · (polar(m_t) + λ w_{t-1}).
 *
 * Computing the polar factor exactly via SVD is too slow. Muon
 * approximates it using a **Newton-Schulz** matrix iteration: a
 * 5-step polynomial that converges to U V^T from a normalised input.
 * Five matmuls per parameter per step — much cheaper than an SVD,
 * still much more expensive than AdamW's elementwise update. The
 * tradeoff is favourable when the optimiser is the bottleneck (large
 * batch / small model) but not always otherwise.
 *
 * Muon is NOT used for 1-D tensors (norm scales, biases), embeddings,
 * or the LM head — those are routed to AdamW elsewhere because the
 * Newton-Schulz iteration is undefined for non-2D parameters.
 *
 * The "async_muon" flag (handled here on CUDA) overlaps the
 * Newton-Schulz iteration with the next forward pass to hide its
 * latency.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/muon.hpp : MuonOptions + MuonOptimizer declarations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when optimizer="muon" in the .conf, the train
 *     loop constructs MuonOptimizer and calls .step() each microbatch.
 *
 * --- Role in training pipeline ---
 *   Replaces AdamW for 2-D weights when selected.
 */
#include "olmo_cpp/optim/muon.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace olmo_cpp {

namespace {

struct MuonParamState : public torch::optim::OptimizerCloneableParamState<MuonParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);
  TORCH_ARG(int64_t, step) = 0;

  void serialize(torch::serialize::OutputArchive& archive) const override {
    archive.write("step", torch::scalar_tensor(step(), torch::kInt64));
    if (momentum_buffer().defined()) {
      archive.write("momentum_buffer", momentum_buffer());
    }
  }

  void serialize(torch::serialize::InputArchive& archive) override {
    torch::Tensor t;
    archive.read("step", t);
    step(t.item<int64_t>());
    torch::Tensor buf;
    if (archive.try_read("momentum_buffer", buf)) {
      momentum_buffer(buf);
    }
  }
};

}  // namespace

Muon::Muon(std::vector<torch::Tensor> params, MuonOptions defaults)
    : Optimizer(
          {torch::optim::OptimizerParamGroup(std::move(params))},
          std::make_unique<MuonOptions>(defaults)) {}

Muon::Muon(std::vector<torch::optim::OptimizerParamGroup> param_groups, MuonOptions defaults)
    : Optimizer(
          std::move(param_groups),
          std::make_unique<MuonOptions>(defaults)) {}

torch::Tensor Muon::newton_schulz_orthogonalize(torch::Tensor G, int64_t steps) {
  // Device-side only — called on the async Muon side stream for 2D params.
  // Any host sync here (e.g. .item()) serializes the side stream against the
  // main stream and destroys the async overlap.
  bool transposed = false;
  if (G.size(0) < G.size(1)) {
    G = G.t();
    transposed = true;
  }

  // clamp_min keeps the division safe without a host-side branch. For a
  // genuinely zero-norm grad the result is a vanishingly small X that the
  // NS iteration leaves near-zero; the caller's learning rate scales it
  // down further, so the numerical effect is indistinguishable from the
  // old early-return.
  auto norm = G.norm().clamp_min(1e-12);
  auto X = G / norm;

  // X @ (3I - A)/2  ≡  1.5*X - 0.5*X@A.
  // Equivalent matmul count, but avoids allocating and materializing the
  // (X.size(1) × X.size(1)) identity matrix every NS iteration.
  for (int64_t i = 0; i < steps; ++i) {
    auto A = torch::mm(X.t(), X);
    auto XA = torch::mm(X, A);
    X = X * 1.5 - XA * 0.5;
  }

  X = X * norm;

  if (transposed) {
    X = X.t();
  }
  return X;
}

torch::Tensor Muon::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  for (auto& group : param_groups_) {
    auto& options = static_cast<MuonOptions&>(group.options());
    const double lr = options.lr();
    const double mom = options.momentum();
    const double weight_decay = options.weight_decay();
    const int64_t ns_steps = options.ns_steps();
    const bool async_ns = options.async_ns();
    const int64_t async_min = options.async_min_numel();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }

      auto grad = p.grad();
      auto key = p.unsafeGetTensorImpl();

      // Single hash lookup per param per step. The old code ran find() then
      // an unconditional operator[] — two lookups every step on the hot path.
      auto it = state_.find(key);
      if (it == state_.end()) {
        auto s = std::make_unique<MuonParamState>();
        s->momentum_buffer(torch::zeros_like(p.data()));
        s->step(0);
        it = state_.emplace(key, std::move(s)).first;
      }

      auto& state = static_cast<MuonParamState&>(*it->second);
      auto& buf = state.momentum_buffer();
      state.step(state.step() + 1);

      if (weight_decay != 0.0) {
        p.data().add_(p.data(), -lr * weight_decay);
      }

      buf.mul_(mom).add_(grad);

      torch::Tensor update;

      if (p.dim() == 2) {
#ifdef USE_CUDA
        if (async_ns && p.is_cuda() && p.numel() >= async_min) {
          // ── Async path: apply previous step's ortho, launch current on side stream ──
          auto it = async_states_.find(key);
          if (it != async_states_.end() && it->second.has_pending) {
            // Wait for the side stream to finish previous NS
            it->second.ready_event.block(at::cuda::getCurrentCUDAStream(p.device().index()));
            // Apply the stale-by-1-step orthogonalized update
            p.data().add_(it->second.pending_update, -lr);
          }

          // Launch current NS on side stream
          if (!ns_stream_.has_value()) {
            ns_stream_ = at::cuda::getStreamFromPool(false, p.device().index());
          }

          auto& as = async_states_[key];
          {
            // Record an event on the current stream so the NS stream waits
            // until buf is fully computed before reading it.
            at::cuda::CUDAEvent buf_ready;
            buf_ready.record(at::cuda::getCurrentCUDAStream(p.device().index()));
            buf_ready.block(*ns_stream_);

            c10::cuda::CUDAStreamGuard guard(*ns_stream_);
            as.pending_update = newton_schulz_orthogonalize(buf, ns_steps);
            as.ready_event.record(*ns_stream_);
            as.has_pending = true;
          }
          continue;  // skip synchronous apply — will be applied next step
        }
#endif
        update = newton_schulz_orthogonalize(buf, ns_steps);
      } else {
        update = buf;
      }

      p.data().add_(update, -lr);
    }
  }

  return loss;
}

}  // namespace olmo_cpp
