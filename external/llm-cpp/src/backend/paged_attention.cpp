/**
 * src/backend/paged_attention.cpp
 *
 * CPU reference + dispatch for the paged attention decode kernel
 * (fast-inference [9]).
 *
 * The CPU path gathers K/V via the page table into contiguous buffers,
 * then runs reference attention. Slow but correct — useful for
 * validating the CUDA kernel against ground truth on small inputs.
 *
 * DRAFT.
 */

#include "olmo_cpp/backend/paged_attention.hpp"

#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

torch::Tensor paged_attention_decode_cpu(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cpu() && k_pool.is_cpu() && v_pool.is_cpu() && page_table.is_cpu(),
              "paged_attention_decode_cpu: all tensors must be CPU");

  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k_pool.contiguous().to(torch::kFloat32);
  auto v_c = v_pool.contiguous().to(torch::kFloat32);
  auto pt  = page_table.contiguous().to(torch::kInt32);

  const int64_t n_q_heads  = q_c.size(0);
  const int64_t head_dim   = q_c.size(1);
  const int64_t max_pages  = k_c.size(0);
  const int64_t page_size  = k_c.size(1);
  const int64_t n_kv_heads = k_c.size(2);

  // Gather K and V into [n_tokens, n_kv_heads, head_dim].
  auto k_gathered = torch::empty({n_tokens, n_kv_heads, head_dim}, k_c.options());
  auto v_gathered = torch::empty({n_tokens, n_kv_heads, head_dim}, v_c.options());

  auto pt_a = pt.accessor<int32_t, 1>();
  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t bi  = t / page_size;
    int64_t off = t % page_size;
    int32_t pg  = pt_a[bi];
    TORCH_CHECK(pg >= 0 && pg < max_pages, "page_table out of range");
    k_gathered[t].copy_(k_c[pg][off]);
    v_gathered[t].copy_(v_c[pg][off]);
  }

  // Standard attention: scores = (Q @ K^T) * sm_scale, softmax, @ V.
  // GQA: expand kv heads to n_q_heads if needed.
  if (n_kv_heads != n_q_heads) {
    int64_t group = n_q_heads / n_kv_heads;
    k_gathered = k_gathered.repeat_interleave(group, /*dim=*/1);
    v_gathered = v_gathered.repeat_interleave(group, /*dim=*/1);
  }

  // q: [Hq, D] → [Hq, 1, D]; k_gathered: [T, Hq, D] → [Hq, T, D]; same for v.
  auto q_b = q_c.unsqueeze(1);                              // [Hq, 1, D]
  auto k_b = k_gathered.transpose(0, 1);                    // [Hq, T, D]
  auto v_b = v_gathered.transpose(0, 1);                    // [Hq, T, D]

  auto scores = torch::matmul(q_b, k_b.transpose(-1, -2)) * sm_scale;  // [Hq, 1, T]
  auto attn   = torch::softmax(scores, -1);                            // [Hq, 1, T]
  auto out    = torch::matmul(attn, v_b).squeeze(1);                    // [Hq, D]

  return out.to(q.dtype());
}

torch::Tensor paged_attention_decode(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.is_cuda()) {
    return paged_attention_decode_cuda(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
  }
#endif
  return paged_attention_decode_cpu(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
}

// ── Graph-capture-friendly variant ────────────────────────────────────────
//
// CPU reference reads the scalar tensor value once and delegates to the
// existing static-n_tokens CPU path. The "dyn" suffix is meaningful on the
// CUDA path, where the kernel reads from a device pointer to keep the
// captured graph correct as n_tokens advances.

torch::Tensor paged_attention_decode_dyn_cpu(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
  TORCH_CHECK(n_tokens.is_cpu(), "paged_attention_decode_dyn_cpu: n_tokens must be CPU");
  TORCH_CHECK(n_tokens.numel() == 1, "paged_attention_decode_dyn_cpu: n_tokens must be a scalar");
  const int64_t n = static_cast<int64_t>(n_tokens.to(torch::kInt64).item<int64_t>());
  return paged_attention_decode_cpu(q, k_pool, v_pool, page_table, n, sm_scale);
}

torch::Tensor paged_attention_decode_dyn(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.is_cuda()) {
    return paged_attention_decode_dyn_cuda(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
  }
#endif
  return paged_attention_decode_dyn_cpu(q, k_pool, v_pool, page_table, n_tokens, sm_scale);
}

// ── Graph-safe paged K/V write ────────────────────────────────────────────

void paged_kv_write_dyn_cpu(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens) {
  TORCH_CHECK(k_src.is_cpu() && v_src.is_cpu() && k_pool.is_cpu() && v_pool.is_cpu()
              && page_table.is_cpu() && n_tokens.is_cpu(),
              "paged_kv_write_dyn_cpu: all tensors must be CPU");
  TORCH_CHECK(k_src.dim() == 3 && v_src.dim() == 3,
              "k_src / v_src must be [S, n_kv_heads, head_dim]");
  TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
              "k_pool / v_pool must be [max_pages, page_size, n_kv_heads, head_dim]");
  TORCH_CHECK(n_tokens.numel() == 1, "n_tokens must be a scalar");

  const int64_t S          = k_src.size(0);
  const int64_t n_kv_heads = k_src.size(1);
  const int64_t head_dim   = k_src.size(2);
  const int64_t max_pages  = k_pool.size(0);
  const int64_t page_size  = k_pool.size(1);
  TORCH_CHECK(k_pool.size(2) == n_kv_heads && k_pool.size(3) == head_dim,
              "k_pool dims must match k_src");

  const int32_t n_tok = n_tokens.to(torch::kInt32).item<int32_t>();
  const int32_t start = n_tok - static_cast<int32_t>(S);
  TORCH_CHECK(start >= 0, "paged_kv_write_dyn_cpu: n_tokens < S");

  auto pt = page_table.contiguous().to(torch::kInt32);
  auto pt_a = pt.accessor<int32_t, 1>();

  for (int64_t i = 0; i < S; ++i) {
    const int32_t global_pos = start + static_cast<int32_t>(i);
    const int32_t blk        = global_pos / static_cast<int32_t>(page_size);
    const int32_t slot       = global_pos % static_cast<int32_t>(page_size);
    const int32_t pg         = pt_a[blk];
    TORCH_CHECK(pg >= 0 && pg < max_pages, "paged_kv_write_dyn_cpu: page out of range");
    // [n_kv_heads, head_dim] slice each side. Cast handles dtype mismatch
    // (e.g. bf16 pool + fp32 src) without an explicit pre-copy.
    k_pool[pg][slot].copy_(k_src[i].to(k_pool.dtype()));
    v_pool[pg][slot].copy_(v_src[i].to(v_pool.dtype()));
  }
}

void paged_kv_write_dyn(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (k_pool.is_cuda()) {
    paged_kv_write_dyn_cuda(k_src, v_src, k_pool, v_pool, page_table, n_tokens);
    return;
  }
#endif
  paged_kv_write_dyn_cpu(k_src, v_src, k_pool, v_pool, page_table, n_tokens);
}

// ── INT4-KV paged attention (item U) ───────────────────────────────────

torch::Tensor paged_attention_decode_int4_cpu(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cpu(), "int4 cpu path: tensors must be CPU");
  const int n_tok = n_tokens.to(torch::kInt32).item<int32_t>();
  // Dequantize K / V pools to fp32 on the fly via the existing nibble
  // unpack from int4_kv.cpp's logic. For the CPU path we just call
  // dequantize_kv_int4 once per page reference — slow but correct.
  // The CUDA kernel does inline dequant for real speed.
  auto q_c = q.contiguous().to(torch::kFloat32);
  const int64_t n_q_heads = q_c.size(0);
  const int64_t head_dim  = q_c.size(1);
  const int64_t max_pages = k_pool.size(0);
  const int64_t page_size = k_pool.size(1);
  const int64_t n_kv_heads = k_pool.size(2);
  TORCH_CHECK(k_pool.size(3) == head_dim / 2,
              "int4 pool last dim must be head_dim/2");

  auto pt = page_table.contiguous().to(torch::kInt32);
  auto pt_a = pt.accessor<int32_t, 1>();

  // Gather K, V into [n_tok, n_kv_heads, head_dim] fp32.
  auto k = torch::zeros({n_tok, n_kv_heads, head_dim}, torch::kFloat32);
  auto v = torch::zeros({n_tok, n_kv_heads, head_dim}, torch::kFloat32);
  for (int t = 0; t < n_tok; ++t) {
    int blk = t / page_size;
    int off = t % page_size;
    int pg  = pt_a[blk];
    for (int h = 0; h < n_kv_heads; ++h) {
      float ks = k_scales.index({pg, off, h}).item<float>();
      float vs = v_scales.index({pg, off, h}).item<float>();
      auto kbyte = k_pool.index({pg, off, h}).to(torch::kCPU).to(torch::kInt32);
      auto vbyte = v_pool.index({pg, off, h}).to(torch::kCPU).to(torch::kInt32);
      auto kbyte_a = kbyte.accessor<int32_t, 1>();
      auto vbyte_a = vbyte.accessor<int32_t, 1>();
      for (int j = 0; j < head_dim / 2; ++j) {
        int lo_k = (kbyte_a[j] & 0xF) - 8;
        int hi_k = ((kbyte_a[j] >> 4) & 0xF) - 8;
        int lo_v = (vbyte_a[j] & 0xF) - 8;
        int hi_v = ((vbyte_a[j] >> 4) & 0xF) - 8;
        k.index_put_({t, h, 2*j},     static_cast<float>(lo_k) * ks);
        k.index_put_({t, h, 2*j + 1}, static_cast<float>(hi_k) * ks);
        v.index_put_({t, h, 2*j},     static_cast<float>(lo_v) * vs);
        v.index_put_({t, h, 2*j + 1}, static_cast<float>(hi_v) * vs);
      }
    }
  }
  // GQA expand if needed.
  if (n_kv_heads != n_q_heads) {
    const int64_t group = n_q_heads / n_kv_heads;
    k = k.repeat_interleave(group, /*dim=*/1);
    v = v.repeat_interleave(group, /*dim=*/1);
  }
  auto q_b = q_c.unsqueeze(1);
  auto k_b = k.transpose(0, 1);
  auto v_b = v.transpose(0, 1);
  auto scores = torch::matmul(q_b, k_b.transpose(-1, -2)) * sm_scale;
  auto attn   = torch::softmax(scores, -1);
  auto out    = torch::matmul(attn, v_b).squeeze(1);
  return out.to(q.dtype());
}

torch::Tensor paged_attention_decode_int4(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.is_cuda()) {
    return paged_attention_decode_int4_cuda(q, k_pool, k_scales, v_pool, v_scales,
                                              page_table, n_tokens, sm_scale);
  }
#endif
  return paged_attention_decode_int4_cpu(q, k_pool, k_scales, v_pool, v_scales,
                                           page_table, n_tokens, sm_scale);
}

}  // namespace olmo_cpp
