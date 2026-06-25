/**
 * src/optim/eight_bit_adamw.cpp
 *
 * EightBitAdamW (item O) implementation.
 *
 * State per parameter:
 *   q_m       : int8 tensor, same numel as param
 *   q_v       : int8 tensor, same numel as param
 *   scales_m  : fp32 tensor, length = ceil(numel / block_size)
 *   scales_v  : fp32 tensor, length = ceil(numel / block_size)
 *
 * Each step():
 *   1. Dequantize q_m, q_v to fp32 by block scale.
 *   2. Apply AdamW update (param decay, moment update, denom, step).
 *   3. Requantize m, v into q_m, q_v with fresh block scales.
 *
 * Step 2 reuses fused vector ops; steps 1 and 3 are done per-block
 * via ATen reshape + abs.max + clamp/round. The kernel-level fast path
 * (bitsandbytes-style 8-bit adam CUDA kernel) is a follow-on; this
 * commit ships the algorithm so the storage savings land first.
 */

#include "olmo_cpp/optim/eight_bit_adamw.hpp"

#include <cmath>

namespace olmo_cpp {

namespace {

struct State : public torch::optim::OptimizerCloneableParamState<State> {
  TORCH_ARG(torch::Tensor, q_m);
  TORCH_ARG(torch::Tensor, q_v);
  TORCH_ARG(torch::Tensor, scales_m);
  TORCH_ARG(torch::Tensor, scales_v);

  void serialize(torch::serialize::OutputArchive& a) const override {
    if (q_m().defined())      a.write("q_m", q_m());
    if (q_v().defined())      a.write("q_v", q_v());
    if (scales_m().defined()) a.write("scales_m", scales_m());
    if (scales_v().defined()) a.write("scales_v", scales_v());
  }
  void serialize(torch::serialize::InputArchive& a) override {
    torch::Tensor t;
    if (a.try_read("q_m", t))      q_m(t);
    if (a.try_read("q_v", t))      q_v(t);
    if (a.try_read("scales_m", t)) scales_m(t);
    if (a.try_read("scales_v", t)) scales_v(t);
  }
};

// Quantize a 1-D fp32 buffer in `block_size`-element blocks. Returns
// (q_int8, scales_fp32) where scales[b] = max_abs(block_b) / 127.
std::pair<torch::Tensor, torch::Tensor>
quantize_blockwise(torch::Tensor x_flat, int64_t block_size) {
  TORCH_CHECK(x_flat.dim() == 1, "quantize_blockwise: expect 1-D input");
  const int64_t N = x_flat.size(0);
  const int64_t n_blocks = (N + block_size - 1) / block_size;
  // Pad to a multiple of block_size for easy reshape.
  if (N % block_size != 0) {
    auto pad = torch::zeros({n_blocks * block_size - N}, x_flat.options());
    x_flat = torch::cat({x_flat, pad}, 0);
  }
  auto blocks = x_flat.view({n_blocks, block_size});
  auto max_abs = std::get<0>(blocks.abs().max(/*dim=*/1));        // [n_blocks]
  auto scales  = (max_abs / 127.0f).clamp_min(1e-12f);            // [n_blocks]
  auto q = (blocks / scales.unsqueeze(-1)).round().clamp(-127.0f, 127.0f)
              .to(torch::kInt8).view({n_blocks * block_size}).narrow(0, 0, N);
  return {q, scales};
}

// Dequantize back to fp32 of length `target_numel`.
torch::Tensor dequantize_blockwise(torch::Tensor q, torch::Tensor scales,
                                     int64_t block_size, int64_t target_numel) {
  const int64_t n_blocks = scales.size(0);
  const int64_t padded   = n_blocks * block_size;
  auto qf = torch::zeros({padded}, torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(q.device()));
  qf.narrow(0, 0, q.size(0)).copy_(q.to(torch::kFloat32));
  auto blocks = qf.view({n_blocks, block_size});
  auto out = (blocks * scales.unsqueeze(-1)).view({padded}).narrow(0, 0, target_numel);
  return out;
}

}  // namespace

EightBitAdamW::EightBitAdamW(std::vector<torch::Tensor> params,
                               EightBitAdamWOptions defaults)
    : Optimizer({torch::optim::OptimizerParamGroup(std::move(params))},
                std::make_unique<EightBitAdamWOptions>(defaults)) {}

EightBitAdamW::EightBitAdamW(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                               EightBitAdamWOptions defaults)
    : Optimizer(std::move(param_groups),
                std::make_unique<EightBitAdamWOptions>(defaults)) {}

torch::Tensor EightBitAdamW::step(LossClosure closure) {
  torch::NoGradGuard no_grad;
  torch::Tensor loss = {};
  if (closure) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  step_count_++;

  for (auto& group : param_groups_) {
    auto& opts = static_cast<EightBitAdamWOptions&>(group.options());
    const double lr  = opts.lr();
    const double b1  = opts.beta1();
    const double b2  = opts.beta2();
    const double eps = opts.eps();
    const double wd  = opts.weight_decay();
    const int64_t block_size = opts.block_size();

    const double bc1 = 1.0 - std::pow(b1, static_cast<double>(step_count_));
    const double bc2 = 1.0 - std::pow(b2, static_cast<double>(step_count_));
    const double step_size = lr / bc1;
    const double bc2_sqrt  = std::sqrt(bc2);

    for (auto& p : group.params()) {
      if (!p.grad().defined()) continue;
      auto& slot = state_[p.unsafeGetTensorImpl()];
      if (slot == nullptr) {
        slot = std::make_unique<State>();
      }
      auto& st = static_cast<State&>(*slot);
      const auto N = p.numel();
      const auto n_blocks = (N + block_size - 1) / block_size;

      if (!st.q_m().defined()) {
        // Init zero-state.
        st.q_m(torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt8).device(p.device())));
        st.q_v(torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt8).device(p.device())));
        st.scales_m(torch::zeros({n_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(p.device())));
        st.scales_v(torch::zeros({n_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(p.device())));
      }

      // Dequantize state.
      auto m = dequantize_blockwise(st.q_m(), st.scales_m(), block_size, N);
      auto v = dequantize_blockwise(st.q_v(), st.scales_v(), block_size, N);

      // Decoupled weight decay on the parameter.
      auto p_flat = p.data().view({N});
      auto g_flat = p.grad().view({N});
      if (wd != 0.0) p_flat.mul_(1.0 - lr * wd);

      // Moment updates.
      m = m * b1 + g_flat.to(torch::kFloat32) * (1.0 - b1);
      v = v * b2 + g_flat.to(torch::kFloat32).pow(2) * (1.0 - b2);

      // Parameter update.
      auto denom = v.sqrt() / bc2_sqrt + eps;
      p_flat.add_( - step_size * (m / denom).to(p_flat.dtype()) );

      // Requantize state.
      auto [new_qm, new_sm] = quantize_blockwise(m, block_size);
      auto [new_qv, new_sv] = quantize_blockwise(v, block_size);
      st.q_m(new_qm); st.scales_m(new_sm);
      st.q_v(new_qv); st.scales_v(new_sv);
    }
  }
  return loss;
}

void EightBitAdamW::reset_state() {
  state_.clear();
  step_count_ = 0;
}

int64_t EightBitAdamW::state_bytes() const {
  int64_t total = 0;
  for (const auto& kv : state_) {
    const auto& s = static_cast<const State&>(*kv.second);
    if (s.q_m().defined())      total += s.q_m().numel() * s.q_m().element_size();
    if (s.q_v().defined())      total += s.q_v().numel() * s.q_v().element_size();
    if (s.scales_m().defined()) total += s.scales_m().numel() * s.scales_m().element_size();
    if (s.scales_v().defined()) total += s.scales_v().numel() * s.scales_v().element_size();
  }
  return total;
}

void EightBitAdamW::save(torch::serialize::OutputArchive& archive) const {
  Optimizer::save(archive);
  archive.write("step_count_", torch::tensor(step_count_));
}

void EightBitAdamW::load(torch::serialize::InputArchive& archive) {
  Optimizer::load(archive);
  torch::Tensor t;
  if (archive.try_read("step_count_", t)) step_count_ = t.item<int64_t>();
}

}  // namespace olmo_cpp
