#include "zwt/layers/transformer_block.hpp"
#include "zwt/core/allocator.hpp"
#include "zwt/ops/elementwise.hpp"

namespace zwt {

namespace {

Attention::Config make_attn_cfg(const TransformerBlock::Config& c) {
  Attention::Config a;
  a.d_model    = c.d_model;
  a.n_heads    = c.n_heads;
  a.n_kv_heads = (c.n_kv_heads > 0) ? c.n_kv_heads : c.n_heads;
  a.head_dim   = c.head_dim;
  a.max_seq    = c.max_seq;
  a.rope_base  = c.rope_base;
  a.bias       = c.bias;
  return a;
}

}  // namespace

TransformerBlock::TransformerBlock(const Config& cfg, DType dtype, Device device,
                                   uint64_t init_seed)
    : cfg_(cfg),
      norm1_(cfg.d_model, cfg.norm_eps, dtype, device),
      attn_(make_attn_cfg(cfg), dtype, device, init_seed ^ 0x1'00'00'00ULL),
      norm2_(cfg.d_model, cfg.norm_eps, dtype, device),
      ffn_(cfg.d_model, cfg.d_ffn, dtype, device, init_seed ^ 0x2'00'00'00ULL) {}

Tensor TransformerBlock::forward(const Tensor& x) {
  saved_input_ = x.view(x.shape());

  // y = x + attn(norm1(x))
  Tensor n1 = norm1_.forward(x);
  Tensor a  = attn_.forward(n1);
  saved_after_attn_ = empty_scratch(x.shape(), x.dtype(), x.device());
  ops::add(saved_after_attn_, x, a);

  // z = y + ffn(norm2(y))
  Tensor n2 = norm2_.forward(saved_after_attn_);
  Tensor f  = ffn_.forward(n2);
  Tensor z  = empty_scratch(x.shape(), x.dtype(), x.device());
  ops::add(z, saved_after_attn_, f);
  return z;
}

Tensor TransformerBlock::backward(const Tensor& grad_z) {
  // dL/dy = dL/dz + d(ffn(norm2(y)))/dy
  // Residual add: grad_y_via_residual = grad_z (identity); grad_y_via_ffn gets
  // multiplied by the Jacobian of ffn ∘ norm2.
  Tensor grad_f = ffn_.backward(grad_z);                        // dL/d(norm2(y))
  Tensor grad_y_via_norm = norm2_.backward(grad_f);             // dL/dy (from ffn branch)
  // grad_y = grad_z + grad_y_via_norm
  Tensor grad_y = empty_scratch(grad_z.shape(), grad_z.dtype(), grad_z.device());
  ops::add(grad_y, grad_z, grad_y_via_norm);

  // dL/dx = dL/dy + d(attn(norm1(x)))/dx
  Tensor grad_a = attn_.backward(grad_y);                       // dL/d(norm1(x))
  Tensor grad_x_via_norm = norm1_.backward(grad_a);             // dL/dx from attn branch
  Tensor grad_x = empty_scratch(grad_z.shape(), grad_z.dtype(), grad_z.device());
  ops::add(grad_x, grad_y, grad_x_via_norm);
  return grad_x;
}

void TransformerBlock::collect_params(std::vector<Parameter*>& out) {
  norm1_.collect_params(out);
  attn_.collect_params(out);
  norm2_.collect_params(out);
  ffn_.collect_params(out);
}

}  // namespace zwt
