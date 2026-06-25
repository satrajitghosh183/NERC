/**
 * src/backend/int4_kv.cpp
 *
 * INT4 KV cache codec (item U). Per-vector dynamic max-abs scaling.
 * Pack 2 nibbles per byte along head_dim.
 */

#include "olmo_cpp/backend/int4_kv.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

Int4KVBlock quantize_kv_int4(torch::Tensor kv) {
  TORCH_CHECK(kv.dim() >= 2, "quantize_kv_int4: last two dims are [..., head_dim]");
  auto kvc = kv.contiguous().to(torch::kFloat32);
  const int64_t head_dim = kvc.size(-1);
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  auto lead_shape = kvc.sizes().vec();
  lead_shape.pop_back();
  auto flat = kvc.reshape({-1, head_dim});                                 // [N, head_dim]
  auto max_abs = std::get<0>(flat.abs().max(/*dim=*/1));                   // [N]
  auto scales = (max_abs / 7.0f).clamp_min(1e-8f);
  auto q = (flat / scales.unsqueeze(-1)).round().clamp(-8.0f, 7.0f).to(torch::kInt8) + 8;
  auto packed = torch::empty({flat.size(0), head_dim / 2},
                              torch::TensorOptions().dtype(torch::kUInt8).device(kv.device()));
  auto src = q.flatten().contiguous();
  auto* sp = src.data_ptr<int8_t>();
  auto* dp = packed.data_ptr<uint8_t>();
  const int64_t N = src.numel();
  for (int64_t i = 0; i < N; i += 2) {
    uint8_t lo = static_cast<uint8_t>(sp[i]) & 0xF;
    uint8_t hi = static_cast<uint8_t>(sp[i + 1]) & 0xF;
    dp[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
  }
  Int4KVBlock out;
  out.packed = packed.reshape([&] {
    auto s = lead_shape; s.push_back(head_dim / 2); return s;
  }());
  out.scales = scales.to(torch::kFloat16).reshape(lead_shape);
  return out;
}

torch::Tensor dequantize_kv_int4(const Int4KVBlock& q, int64_t head_dim) {
  auto p = q.packed.contiguous();
  auto s = q.scales.to(torch::kFloat32).contiguous();
  const int64_t N = p.numel() / (head_dim / 2);
  auto pf = p.reshape({N, head_dim / 2});
  auto out = torch::empty({N, head_dim},
                          torch::TensorOptions().dtype(torch::kFloat32).device(p.device()));
  auto* pp = pf.data_ptr<uint8_t>();
  auto* op = out.data_ptr<float>();
  auto* sp = s.reshape({N}).data_ptr<float>();
  for (int64_t i = 0; i < N; ++i) {
    const float scale = sp[i];
    for (int64_t j = 0; j < head_dim / 2; ++j) {
      uint8_t byte = pp[i * (head_dim / 2) + j];
      int lo = (int)(byte & 0xF) - 8;
      int hi = (int)((byte >> 4) & 0xF) - 8;
      op[i * head_dim + 2 * j]     = (float)lo * scale;
      op[i * head_dim + 2 * j + 1] = (float)hi * scale;
    }
  }
  auto lead = q.scales.sizes().vec();
  lead.push_back(head_dim);
  return out.reshape(lead).to(q.packed.options().dtype(torch::kFloat32));
}

}  // namespace olmo_cpp
