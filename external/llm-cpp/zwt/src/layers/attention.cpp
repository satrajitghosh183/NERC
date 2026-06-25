#include "zwt/layers/attention.hpp"
#include "zwt/core/allocator.hpp"
#include "zwt/core/profiler.hpp"
#include "zwt/ops/attn.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/rope.hpp"

#include <stdexcept>
#include <utility>

namespace zwt {

namespace {

// Disjoint seed salts for Q / K / V / O so their weights differ at init.
constexpr uint64_t kSeedSaltQ = 0x17'00'00'00ULL;
constexpr uint64_t kSeedSaltK = 0x23'00'00'00ULL;
constexpr uint64_t kSeedSaltV = 0x31'00'00'00ULL;
constexpr uint64_t kSeedSaltO = 0x47'00'00'00ULL;

int64_t kv_heads(const Attention::Config& c) {
  return (c.n_kv_heads > 0) ? c.n_kv_heads : c.n_heads;
}

}  // namespace

Attention::Attention(const Config& cfg, DType dtype, Device device,
                     uint64_t init_seed)
    : cfg_(cfg),
      q_proj_(cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltQ),
      k_proj_(cfg.d_model, kv_heads(cfg) * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltK),
      v_proj_(cfg.d_model, kv_heads(cfg) * cfg.head_dim, cfg.bias, dtype, device,
              init_seed ^ kSeedSaltV),
      out_proj_(cfg.n_heads * cfg.head_dim, cfg.d_model, cfg.bias, dtype, device,
                init_seed ^ kSeedSaltO) {
  if (cfg.d_model <= 0 || cfg.n_heads <= 0 || cfg.head_dim <= 0 || cfg.max_seq <= 0)
    throw std::runtime_error("Attention: invalid config");
  if (cfg.d_model != cfg.n_heads * cfg.head_dim)
    throw std::runtime_error("Attention: d_model must equal n_heads * head_dim");
  const int64_t Hkv = kv_heads(cfg);
  if (cfg.n_heads % Hkv != 0)
    throw std::runtime_error("Attention: n_heads must be a multiple of n_kv_heads");
  rope_table_ = ops::rope_build_table(cfg.max_seq, cfg.head_dim, cfg.rope_base, device);
}

Tensor Attention::forward(const Tensor& x) {
  if (x.rank() != 3) throw std::runtime_error("Attention::forward: x must be [B,S,d]");
  const int64_t B   = x.dim(0);
  const int64_t S   = x.dim(1);
  const int64_t H   = cfg_.n_heads;
  const int64_t Hkv = kv_heads(cfg_);
  const int64_t D   = cfg_.head_dim;
  const int64_t group = H / Hkv;
  saved_input_ = x.view(x.shape());
  Device dev = x.device();

  Tensor q, k, v;
  {
    ZWT_PROFILE_GPU("attn.qkv_proj.fwd", dev);
    q = q_proj_.forward(x);  // [B, S, H  *D]
    k = k_proj_.forward(x);  // [B, S, Hkv*D]
    v = v_proj_.forward(x);  // [B, S, Hkv*D]
  }

  Tensor q4 = q.view({B, S, H,   D});
  Tensor k4 = k.view({B, S, Hkv, D});
  Tensor v4 = v.view({B, S, Hkv, D});

  // RoPE on Q (full width) and K (compact Hkv width). RoPE is shape-agnostic
  // on the H axis, so it applies directly at compact width — cheaper than
  // rotating after expansion.
  {
    ZWT_PROFILE_GPU("attn.rope.fwd", dev);
    ops::rope_apply(q4, rope_table_);
    ops::rope_apply(k4, rope_table_);
  }

  // Head-major transpose at compact widths.
  saved_q_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  {
    ZWT_PROFILE_GPU("attn.transpose.fwd", dev);
    ops::transpose_bshd_to_bhsd(q4, saved_q_bhsd_);

    if (group == 1) {
      saved_k_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
      saved_v_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
      ops::transpose_bshd_to_bhsd(k4, saved_k_bhsd_);
      ops::transpose_bshd_to_bhsd(v4, saved_v_bhsd_);
    } else {
      // GQA path: transpose into compact head-major buffers, then replicate
      // across groups into the saved_*_bhsd_ (full H) buffers fed to sdpa().
      // This is the work native-GQA would eliminate; the profiler tells us
      // exactly how much it costs.
      Tensor k_bhsd_kv = empty_scratch({B, Hkv, S, D}, x.dtype(), x.device());
      Tensor v_bhsd_kv = empty_scratch({B, Hkv, S, D}, x.dtype(), x.device());
      ops::transpose_bshd_to_bhsd(k4, k_bhsd_kv);
      ops::transpose_bshd_to_bhsd(v4, v_bhsd_kv);
      saved_k_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
      saved_v_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
      {
        ZWT_PROFILE_GPU("attn.gqa_kv_expand.fwd", dev);
        ops::repeat_kv_heads(k_bhsd_kv, saved_k_bhsd_);
        ops::repeat_kv_heads(v_bhsd_kv, saved_v_bhsd_);
      }
    }
  }

  saved_out_bhsd_ = empty_scratch({B, H, S, D}, x.dtype(), x.device());
  {
    ZWT_PROFILE_GPU("attn.sdpa.fwd", dev);
    ops::sdpa(saved_q_bhsd_, saved_k_bhsd_, saved_v_bhsd_,
              saved_out_bhsd_, /*is_causal=*/true);
  }

  Tensor out_bshd = empty_scratch({B, S, H, D}, x.dtype(), x.device());
  ops::transpose_bhsd_to_bshd(saved_out_bhsd_, out_bshd);
  Tensor out_flat = out_bshd.view({B, S, H * D});

  Tensor out;
  {
    ZWT_PROFILE_GPU("attn.oproj.fwd", dev);
    out = out_proj_.forward(out_flat);
  }
  return out;
}

Tensor Attention::backward(const Tensor& grad_y) {
  const int64_t B   = saved_input_.dim(0);
  const int64_t S   = saved_input_.dim(1);
  const int64_t H   = cfg_.n_heads;
  const int64_t Hkv = kv_heads(cfg_);
  const int64_t D   = cfg_.head_dim;
  const int64_t group = H / Hkv;
  Device dev = grad_y.device();

  Tensor grad_out_flat;
  {
    ZWT_PROFILE_GPU("attn.oproj.bwd", dev);
    grad_out_flat = out_proj_.backward(grad_y);                   // [B,S,H*D]
  }
  Tensor grad_out_bshd = grad_out_flat.view({B, S, H, D});

  Tensor grad_out_bhsd = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  ops::transpose_bshd_to_bhsd(grad_out_bshd, grad_out_bhsd);

  Tensor grad_q_bhsd      = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_k_bhsd_full = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_v_bhsd_full = empty_scratch({B, H, S, D}, grad_y.dtype(), grad_y.device());
  {
    ZWT_PROFILE_GPU("attn.sdpa.bwd", dev);
    ops::sdpa_backward(grad_out_bhsd,
                       saved_q_bhsd_, saved_k_bhsd_, saved_v_bhsd_,
                       saved_out_bhsd_,
                       grad_q_bhsd, grad_k_bhsd_full, grad_v_bhsd_full,
                       /*is_causal=*/true);
  }

  Tensor grad_k_bhsd = (group == 1)
      ? std::move(grad_k_bhsd_full)
      : empty_scratch({B, Hkv, S, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_v_bhsd = (group == 1)
      ? std::move(grad_v_bhsd_full)
      : empty_scratch({B, Hkv, S, D}, grad_y.dtype(), grad_y.device());
  if (group != 1) {
    ZWT_PROFILE_GPU("attn.gqa_kv_reduce.bwd", dev);
    ops::reduce_kv_heads_sum(grad_k_bhsd_full, grad_k_bhsd);
    ops::reduce_kv_heads_sum(grad_v_bhsd_full, grad_v_bhsd);
  }

  Tensor grad_q_bshd = empty_scratch({B, S, H,   D}, grad_y.dtype(), grad_y.device());
  Tensor grad_k_bshd = empty_scratch({B, S, Hkv, D}, grad_y.dtype(), grad_y.device());
  Tensor grad_v_bshd = empty_scratch({B, S, Hkv, D}, grad_y.dtype(), grad_y.device());
  {
    ZWT_PROFILE_GPU("attn.transpose.bwd", dev);
    ops::transpose_bhsd_to_bshd(grad_q_bhsd, grad_q_bshd);
    ops::transpose_bhsd_to_bshd(grad_k_bhsd, grad_k_bshd);
    ops::transpose_bhsd_to_bshd(grad_v_bhsd, grad_v_bshd);
  }

  {
    ZWT_PROFILE_GPU("attn.rope.bwd", dev);
    ops::rope_apply_backward(grad_q_bshd, rope_table_);
    ops::rope_apply_backward(grad_k_bshd, rope_table_);
  }

  Tensor gq_flat = grad_q_bshd.view({B, S, H   * D});
  Tensor gk_flat = grad_k_bshd.view({B, S, Hkv * D});
  Tensor gv_flat = grad_v_bshd.view({B, S, Hkv * D});

  Tensor grad_x_q, grad_x_k, grad_x_v;
  {
    ZWT_PROFILE_GPU("attn.qkv_proj.bwd", dev);
    grad_x_q = q_proj_.backward(gq_flat);
    grad_x_k = k_proj_.backward(gk_flat);
    grad_x_v = v_proj_.backward(gv_flat);
  }

  Tensor grad_x = empty_scratch(saved_input_.shape(), grad_y.dtype(), grad_y.device());
  {
    ZWT_PROFILE_GPU("attn.qkv_grad_sum.bwd", dev);
    ops::add(grad_x, grad_x_q, grad_x_k);
    ops::axpy(grad_x, grad_x_v, 1.0f);
  }
  return grad_x;
}

void Attention::collect_params(std::vector<Parameter*>& out) {
  q_proj_.collect_params(out);
  k_proj_.collect_params(out);
  v_proj_.collect_params(out);
  out_proj_.collect_params(out);
}

}  // namespace zwt
