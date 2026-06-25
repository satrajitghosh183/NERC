// zwt_train — end-to-end training demo for the Zero-Wait Trainer.
//
// What it does: builds a two-layer MLP language-model head on top of an
// embedding, trains it with fused AdamW on token data, and reports tokens/s.
// This is deliberately a simple model — its purpose is to exercise every
// path of the framework (data loader, ops, layers, optimizer, activation
// arena reset) so you can confirm the plumbing works end-to-end before we
// wire the full transformer block.
//
// Build (CUDA):   cmake --build build --target zwt_train
// Run:            ./build/zwt_train <tokens.bin> [steps]

#include "zwt/core/allocator.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/data/token_loader.hpp"
#include "zwt/layers/embedding.hpp"
#include "zwt/layers/ffn.hpp"
#include "zwt/layers/linear.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/rmsnorm.hpp"
#include "zwt/ops/xent.hpp"
#include "zwt/optim/adamw.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s <tokens.bin> [steps=200] [batch=4] [seq=512] [vocab=50257]\n",
        argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  int    steps  = (argc >= 3) ? std::atoi(argv[2]) : 200;
  int    batch  = (argc >= 4) ? std::atoi(argv[3]) : 4;
  int    seq    = (argc >= 5) ? std::atoi(argv[4]) : 512;
  int    vocab  = (argc >= 6) ? std::atoi(argv[5]) : 50257;
  int    d_model = 256;
  int    hidden  = 1024;

#ifdef USE_CUDA
  Device dev = Device::cuda(0);
  DType  param_dtype = DType::BF16;
#else
  Device dev = Device::cpu();
  DType  param_dtype = DType::F32;
#endif

  std::fprintf(stderr,
      "zwt_train: dev=%s dtype=%s steps=%d batch=%d seq=%d vocab=%d d_model=%d\n",
      dev.is_cuda() ? "cuda:0" : "cpu",
      dtype_name(param_dtype), steps, batch, seq, vocab, d_model);

  // Enlarge the activation arena — 512 MiB should handle a toy model easily.
  set_activation_arena_capacity(512ULL << 20);

  // Model: token emb -> RMSNorm -> Linear -> SwiGLU FFN -> RMSNorm -> LM head.
  Embedding tok_emb(vocab,   d_model,  param_dtype, dev);
  RMSNorm   norm_in(d_model, 1e-5f,    param_dtype, dev);
  Linear    proj(d_model, d_model, false, param_dtype, dev);
  FFN       ffn(d_model, hidden,   param_dtype, dev);
  RMSNorm   norm_out(d_model, 1e-5f, param_dtype, dev);
  Linear    lm_head(d_model, vocab, false, param_dtype, dev);

  std::vector<Parameter*> params;
  tok_emb.collect_params(params);
  norm_in.collect_params(params);
  proj.collect_params(params);
  ffn.collect_params(params);
  norm_out.collect_params(params);
  lm_head.collect_params(params);

  int64_t total_params = 0;
  for (auto* p : params) total_params += p->numel();
  std::fprintf(stderr, "zwt_train: params = %ld M\n", (long)(total_params / 1'000'000));

  optim::AdamWConfig cfg;
  cfg.lr = 3e-4f;
  cfg.weight_decay = 0.0f;  // demo
  optim::AdamW opt(params, cfg);

  data::TokenLoader::Options dopts;
  dopts.path = path;
  dopts.seq_len = seq;
  dopts.batch_size = batch;
  dopts.device = dev;
  data::TokenLoader loader(dopts);
  loader.start();

  auto t_start = std::chrono::steady_clock::now();
  int64_t tokens_seen = 0;

  for (int step = 1; step <= steps; ++step) {
    step_begin();                   // reset activation arenas
    auto batch_data = loader.next();

    Tensor h0 = tok_emb.forward(batch_data.input);       // [B, S, D]
    Tensor h1 = norm_in.forward(h0);
    Tensor h2 = proj.forward(h1);
    Tensor h3 = ffn.forward(h2);
    Tensor h4 = norm_out.forward(h3);
    Tensor logits = lm_head.forward(h4);                 // [B, S, V]

    // Flatten to [B*S, V] for cross entropy.
    Shape logits2d{batch * seq, vocab};
    Tensor logits_flat = logits.view(logits2d);
    Tensor tgt_flat = batch_data.target.view({batch * seq});

    Tensor loss = empty_scratch({1}, DType::F32, dev);
    Tensor grad_logits = empty_scratch(logits2d, param_dtype, dev);
    ops::cross_entropy(logits_flat, tgt_flat, loss, &grad_logits, /*ignore=*/-100);

    // Backward
    opt.zero_grad();
    Tensor g4 = lm_head.backward(grad_logits.view(logits.shape()));
    Tensor g3 = norm_out.backward(g4);
    Tensor g2 = ffn.backward(g3);
    Tensor g1 = proj.backward(g2);
    Tensor g0 = norm_in.backward(g1);
    tok_emb.backward(g0);

    opt.step();

    tokens_seen += (int64_t)batch * seq;

    if (step % 10 == 0 || step == 1) {
      // Pull the loss to host for logging. This is the only sync per N steps.
      float loss_h = 0.f;
#ifdef USE_CUDA
      cudaMemcpy(&loss_h, loss.data(), sizeof(float), cudaMemcpyDeviceToHost);
#else
      loss_h = *loss.as<float>();
#endif
      auto t_now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t_now - t_start).count();
      double tps = tokens_seen / std::max(secs, 1e-9);
      std::fprintf(stderr,
          "step %4d  loss %.4f  %.1f tok/s\n", step, loss_h, tps);
    }
  }

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return 0;
}
