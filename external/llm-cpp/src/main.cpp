/**
 * OLMo C++ Training
 *
 * Usage: ./build/olmo_train conf/olmo.conf
 */

#include "olmo_cpp/common/config_ini.hpp"
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include "olmo_cpp/model/mup_init.hpp"
#include "olmo_cpp/train.hpp"
#include "olmo_cpp/seed.hpp"
#include "olmo_cpp/profiler.hpp"
#include "olmo_cpp/backend/simd_backend.hpp"
#include "olmo_cpp/backend/cuda_backend.hpp"
#include "olmo_cpp/train/callbacks/all_callbacks.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

torch::Device select_device(const std::string& preferred) {
  if (preferred == "mps" || preferred == "metal") {
#ifdef __APPLE__
    if (torch::mps::is_available()) {
      std::cout << "Device: Metal (MPS) on Apple Silicon\n";
      return torch::Device(torch::kMPS);
    }
#endif
    std::cerr << "MPS not available, falling back to CPU.\n";
  }
  if (preferred == "cuda") {
    if (torch::cuda::is_available()) {
      std::cout << "Device: CUDA GPU\n";
      return torch::Device(torch::kCUDA);
    }
    std::cerr << "CUDA not available, falling back to CPU.\n";
  }
  std::cout << "Device: CPU\n";
  return torch::Device(torch::kCPU);
}

int64_t count_parameters(const torch::nn::Module& model) {
  int64_t total = 0;
  for (const auto& p : model.parameters()) total += p.numel();
  return total;
}

template<typename Model>
void run(Model& model, const olmo_cpp::TransformerConfig& cfg,
         olmo_cpp::TrainConfig train_cfg, torch::Device device,
         bool use_mup, bool use_fused, bool use_multi_res,
         bool enable_profile, const std::string& save_path,
         const olmo_cpp::SeedState& seed_state) {
  // Init weights
  if (use_mup) {
    olmo_cpp::MuPConfig mup_cfg;
    mup_cfg.base_width = 256.0;
    mup_cfg.target_width = static_cast<double>(cfg.d_model);
    olmo_cpp::apply_mup_init(*model, cfg, mup_cfg, seed_state.torch_gen);
  } else {
    model->init_weights(seed_state.torch_gen);
  }

  int64_t n_params = count_parameters(*model);
  std::cout << "Model: " << (n_params / 1000000) << "M params"
            << " (d=" << cfg.d_model
            << ", layers=" << cfg.n_layers
            << ", heads=" << cfg.n_heads;
  if (use_fused) std::cout << ", FUSED";
  if (use_mup) std::cout << ", µP";
  if (use_multi_res) std::cout << ", DC-MRE";
  if (cfg.num_mtp_heads > 0) std::cout << ", mtp_heads=" << cfg.num_mtp_heads;
  std::cout << ")\n";

  model->to(device);

  // Pure BF16 mode: convert every floating-point parameter AND buffer to BF16
  // after init. Halves memory for weights + optimizer state + gradients
  // (~3x savings). Integer buffers (embedding indices, trigram maps) MUST be
  // skipped — torch::nn::Module::to(kBFloat16) would clobber them and break
  // embedding lookups. Incompatible with autocast (which expects FP32 master
  // weights), so we auto-disable it.
  //
  // Why buffers matter: DC-MRE registers float buffers (char_trigram_count)
  // that participate in forward arithmetic. If they stay FP32 while weights
  // are BF16, the first BF16/FP32 op promotes activations to FP32 and the
  // next Linear call dies with "mat1 and mat2 have different dtypes".
  if (train_cfg.use_bf16 && device.is_cuda()) {
    if (train_cfg.use_amp) {
      std::cerr << "WARNING: bf16=1 and amp=1 are mutually exclusive; disabling amp.\n";
      train_cfg.use_amp = false;
    }
    torch::NoGradGuard no_grad;
    for (auto& p : model->parameters()) {
      if (p.is_floating_point()) {
        p.set_data(p.data().to(torch::kBFloat16));
      }
    }
    for (auto& b : model->buffers()) {
      if (b.is_floating_point()) {
        b.set_data(b.to(torch::kBFloat16));
      }
    }
    std::cout << "Weights: BF16 (saves ~50% GPU memory)\n";
  }

  // ── Build callbacks ──
  std::vector<std::shared_ptr<olmo_cpp::Callback>> callbacks;
  std::shared_ptr<olmo_cpp::GradientStatsCallback> grad_stats_cb;
  if (!train_cfg.grad_stats_path.empty()) {
    grad_stats_cb = std::make_shared<olmo_cpp::GradientStatsCallback>(
        train_cfg.grad_stats_path, "sample", train_cfg.grad_stats_interval);
    grad_stats_cb->set_model(model.ptr());
    callbacks.push_back(grad_stats_cb);
  }

  // When profiling on CUDA, switch the ProfileScope timers to CUDA events so
  // the per-stage report reflects true GPU execution time (forward/backward/
  // optimizer), not just kernel-launch wall time. Adds a per-stage sync.
  if (enable_profile && device.is_cuda()) {
    olmo_cpp::profiler().set_cuda_timing(true);
  }

  olmo_cpp::train(model, cfg, train_cfg, device, std::move(callbacks));

  if (enable_profile) {
    olmo_cpp::profiler().report("Training Profile (GPU time)");
    olmo_cpp::print_memory_summary(device);
    olmo_cpp::print_rng_state_summary();
  }

  if (!save_path.empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(save_path).parent_path());
    torch::save(model, save_path);
    std::cout << "Checkpoint saved: " << save_path << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <conf_file>\n"
              << "  e.g.: " << argv[0] << " conf/olmo.conf\n";
    return (argc == 1) ? 0 : 1;
  }

  const std::string conf_path = argv[1];

  try {
    // ── Load all sections from single .conf file ──
    ConfigINI model_ini(conf_path, "model");
    ConfigINI train_ini(conf_path, "training");
    ConfigINI data_ini(conf_path, "data");
    ConfigINI opt_ini(conf_path, "optimization");
    ConfigINI dev_ini(conf_path, "device");

    // ── Model config ──
    olmo_cpp::TransformerConfig cfg;
    model_ini.get("d_model", cfg.d_model);
    model_ini.get("vocab_size", cfg.vocab_size);
    model_ini.get("n_layers", cfg.n_layers);
    model_ini.get("n_heads", cfg.n_heads);
    cfg.n_kv_heads          = model_ini.get_or<int64_t>("n_kv_heads", -1);
    cfg.head_dim            = model_ini.get_or<int64_t>("head_dim", -1);
    cfg.rope_theta          = model_ini.get_or<int64_t>("rope_theta", 500000);
    cfg.layer_norm_eps      = model_ini.get_or<double>("layer_norm_eps", 1e-6);
    cfg.init_std            = model_ini.get_or<double>("init_std", 0.02);
    cfg.use_qk_norm         = model_ini.get_or<bool>("use_qk_norm", true);
    cfg.use_head_qk_norm    = model_ini.get_or<bool>("use_head_qk_norm", false);
    cfg.hidden_size_multiple_of = model_ini.get_or<int64_t>("hidden_size_multiple_of", 256);
    cfg.hidden_size_multiplier  = model_ini.get_or<double>("hidden_size_multiplier", 1.5);
    cfg.num_mtp_heads       = model_ini.get_or<int64_t>("num_mtp_heads", 0);
    cfg.mtp_loss_weight     = model_ini.get_or<double>("mtp_loss_weight", 0.1);

    // Sliding window attention
    cfg.sliding_window_size = model_ini.get_or<int64_t>("sliding_window_size", -1);

    // Convolution
    cfg.use_conv            = model_ini.get_or<bool>("use_conv", false);
    cfg.conv_kernel_size    = model_ini.get_or<int64_t>("conv_kernel_size", 4);

    // Float8
    cfg.use_float8          = model_ini.get_or<bool>("use_float8", false);

    // Block type: reordered_norm, peri_norm, normalized_ngpt, layer_norm_scaled,
    //             moe_reordered_norm, moe_hybrid_reordered_norm
    {
      auto s = model_ini.get_or<std::string>("block_type", "reordered_norm");
      if (s == "peri_norm")                    cfg.block_type = olmo_cpp::TransformerConfig::BlockType::PeriNorm;
      else if (s == "normalized_ngpt")         cfg.block_type = olmo_cpp::TransformerConfig::BlockType::NormalizedNGPT;
      else if (s == "layer_norm_scaled")       cfg.block_type = olmo_cpp::TransformerConfig::BlockType::LayerNormScaled;
      else if (s == "moe_reordered_norm")      cfg.block_type = olmo_cpp::TransformerConfig::BlockType::MoEReorderedNorm;
      else if (s == "moe_hybrid_reordered_norm") cfg.block_type = olmo_cpp::TransformerConfig::BlockType::MoEHybridReorderedNorm;
      else                                     cfg.block_type = olmo_cpp::TransformerConfig::BlockType::ReorderedNorm;
    }

    // Attention backend: sdpa, flash2, flash3, transformer_engine
    {
      auto s = model_ini.get_or<std::string>("attention_backend", "sdpa");
      if (s == "flash2")                cfg.attention_backend = olmo_cpp::TransformerConfig::AttentionBackend::FlashAttention2;
      else if (s == "flash3")           cfg.attention_backend = olmo_cpp::TransformerConfig::AttentionBackend::FlashAttention3;
      else if (s == "transformer_engine") cfg.attention_backend = olmo_cpp::TransformerConfig::AttentionBackend::TransformerEngine;
      else                              cfg.attention_backend = olmo_cpp::TransformerConfig::AttentionBackend::SDPA;
    }

    // Gated attention: none, headwise, elementwise
    {
      auto s = model_ini.get_or<std::string>("gated_attention", "none");
      if (s == "headwise")         cfg.gated_attention = olmo_cpp::TransformerConfig::GatedAttentionType::Headwise;
      else if (s == "elementwise") cfg.gated_attention = olmo_cpp::TransformerConfig::GatedAttentionType::Elementwise;
      else                         cfg.gated_attention = olmo_cpp::TransformerConfig::GatedAttentionType::None;
    }

    // RoPE scaling: none, abf, position_interpolation, stepwise, yarn
    {
      auto s = model_ini.get_or<std::string>("rope_scaling_type", "none");
      if (s == "abf")                        cfg.rope_scaling_type = olmo_cpp::TransformerConfig::RoPEScalingType::ABF;
      else if (s == "position_interpolation") cfg.rope_scaling_type = olmo_cpp::TransformerConfig::RoPEScalingType::PositionInterpolation;
      else if (s == "stepwise")              cfg.rope_scaling_type = olmo_cpp::TransformerConfig::RoPEScalingType::Stepwise;
      else if (s == "yarn")                  cfg.rope_scaling_type = olmo_cpp::TransformerConfig::RoPEScalingType::YaRN;
      else                                   cfg.rope_scaling_type = olmo_cpp::TransformerConfig::RoPEScalingType::None;
    }
    cfg.rope_scaling_factor   = model_ini.get_or<double>("rope_scaling_factor", 1.0);
    cfg.rope_yarn_beta_fast   = model_ini.get_or<double>("rope_yarn_beta_fast", 32.0);
    cfg.rope_yarn_beta_slow   = model_ini.get_or<double>("rope_yarn_beta_slow", 1.0);

    // Layer norm type: rms_norm, layer_norm, l2_norm, fused_rms_norm
    {
      auto s = model_ini.get_or<std::string>("layer_norm_type", "rms_norm");
      if (s == "layer_norm")          cfg.layer_norm_type = olmo_cpp::TransformerConfig::LayerNormType::LayerNorm;
      else if (s == "l2_norm")        cfg.layer_norm_type = olmo_cpp::TransformerConfig::LayerNormType::L2Norm;
      else if (s == "fused_rms_norm") cfg.layer_norm_type = olmo_cpp::TransformerConfig::LayerNormType::FusedRMSNorm;
      else                            cfg.layer_norm_type = olmo_cpp::TransformerConfig::LayerNormType::RMSNorm;
    }

    // Activation checkpointing mode: none, full, selected_blocks
    {
      auto s = model_ini.get_or<std::string>("activation_checkpoint_mode", "none");
      if (s == "full")                   cfg.activation_checkpoint_mode = olmo_cpp::TransformerConfig::ActivationCheckpointMode::Full;
      else if (s == "selected_blocks")   cfg.activation_checkpoint_mode = olmo_cpp::TransformerConfig::ActivationCheckpointMode::SelectedBlocks;
      else                               cfg.activation_checkpoint_mode = olmo_cpp::TransformerConfig::ActivationCheckpointMode::None;
    }
    cfg.activation_checkpoint_interval = model_ini.get_or<int64_t>("activation_checkpoint_interval", 1);

    // MoE config
    cfg.use_moe              = model_ini.get_or<bool>("use_moe", false);
    cfg.moe_num_experts      = model_ini.get_or<int64_t>("moe_num_experts", 8);
    cfg.moe_top_k            = model_ini.get_or<int64_t>("moe_top_k", 2);
    cfg.moe_hidden_size      = model_ini.get_or<int64_t>("moe_hidden_size", -1);
    cfg.moe_capacity_factor  = model_ini.get_or<double>("moe_capacity_factor", 1.25);
    cfg.moe_dropless         = model_ini.get_or<bool>("moe_dropless", true);
    cfg.moe_zloss_weight     = model_ini.get_or<double>("moe_zloss_weight", 1e-3);
    cfg.moe_lb_loss_weight   = model_ini.get_or<double>("moe_lb_loss_weight", 1e-2);
    cfg.moe_hybrid           = model_ini.get_or<bool>("moe_hybrid", false);
    cfg.moe_hybrid_interval  = model_ini.get_or<int64_t>("moe_hybrid_interval", 2);

    // Multi-res DC-MRE
    cfg.multi_res_char_buckets   = model_ini.get_or<int64_t>("multi_res_char_buckets", 4096);
    cfg.multi_res_phrase_buckets = model_ini.get_or<int64_t>("multi_res_phrase_buckets", 8192);
    cfg.multi_res_inner_dim      = model_ini.get_or<int64_t>("multi_res_inner_dim", 64);

    // ── Training config ──
    olmo_cpp::TrainConfig train_cfg;
    train_ini.get("steps", train_cfg.num_steps);
    train_ini.get("batch_size", train_cfg.batch_size);
    train_ini.get("seq_len", train_cfg.seq_len);
    train_ini.get("lr", train_cfg.lr);
    train_cfg.warmup_steps     = train_ini.get_or<int64_t>("warmup_steps", 100);
    train_cfg.grad_accum_steps = train_ini.get_or<int64_t>("grad_accum", 1);
    train_cfg.optimizer        = train_ini.get_or<std::string>("optimizer", "adamw");
    train_cfg.use_amp          = train_ini.get_or<bool>("amp", false);
    train_cfg.use_bf16         = train_ini.get_or<bool>("bf16", false);
    train_cfg.use_grad_scaler  = train_ini.get_or<bool>("grad_scaler", false);
    train_cfg.max_grad_norm    = train_ini.get_or<double>("max_grad_norm", 1.0);
    train_cfg.weight_decay     = train_ini.get_or<double>("weight_decay", 0.01);
    train_cfg.scheduler        = train_ini.get_or<std::string>("scheduler", "cosine");
    train_cfg.activation_checkpoint_interval = train_ini.get_or<int64_t>("activation_checkpoint_interval", 0);

    // Evaluation
    train_cfg.eval_data_path   = train_ini.get_or<std::string>("eval_data_path", "");
    if (train_cfg.eval_data_path && train_cfg.eval_data_path->empty())
      train_cfg.eval_data_path = std::nullopt;
    train_cfg.eval_interval    = train_ini.get_or<int64_t>("eval_interval", 500);

    // Checkpointing
    train_cfg.checkpoint_dir      = train_ini.get_or<std::string>("checkpoint_dir", "");
    train_cfg.checkpoint_interval = train_ini.get_or<int64_t>("checkpoint_interval", 1000);
    train_cfg.keep_checkpoints    = train_ini.get_or<int>("keep_checkpoints", 3);
    train_cfg.resume              = train_ini.get_or<bool>("resume", false);

    // Sequence/batch curriculum scheduling
    train_cfg.target_seq_len       = train_ini.get_or<int64_t>("target_seq_len", -1);
    train_cfg.seq_len_warmup_steps = train_ini.get_or<int64_t>("seq_len_warmup_steps", 0);
    train_cfg.target_batch_size    = train_ini.get_or<int64_t>("target_batch_size", -1);
    train_cfg.batch_size_ramp_steps = train_ini.get_or<int64_t>("batch_size_ramp_steps", 0);

    int64_t seed_val       = train_ini.get_or<int64_t>("seed", 42);
    bool enable_profile    = train_ini.get_or<bool>("profile", false);
    std::string save_path  = train_ini.get_or<std::string>("save", "");

    // ── Data config ──
    std::string data_path;
    data_ini.get("data_path", data_path);
    train_cfg.data_path = data_path;

    std::string bpe_vocab = data_ini.get_or<std::string>("bpe_vocab", "");

    // ── Optimization flags ──
    bool use_fused     = opt_ini.get_or<bool>("fused", false);
    bool use_mup       = opt_ini.get_or<bool>("mup", false);
    bool use_multi_res = opt_ini.get_or<bool>("multi_res", false);

    // ── Performance tuning ──
    train_cfg.use_foreach_optimizer = opt_ini.get_or<bool>("foreach_optimizer", true);
    train_cfg.use_zero1             = opt_ini.get_or<bool>("zero1", false);
    train_cfg.use_fsdp              = opt_ini.get_or<bool>("fsdp", false);
    train_cfg.gpu_resident_data     = opt_ini.get_or<bool>("gpu_data", true);
    train_cfg.max_gpu_data_tokens   = opt_ini.get_or<int64_t>("gpu_data_max_tokens", 0);
    train_cfg.log_interval          = train_ini.get_or<int64_t>("log_interval", 10);
    train_cfg.use_cuda_graph        = opt_ini.get_or<bool>("cuda_graph", false);
    train_cfg.async_muon            = opt_ini.get_or<bool>("async_muon", false);

    // ── Heartbeat monitoring ──
    train_cfg.report_every          = train_ini.get_or<double>("report_every", 300.0);
    train_cfg.heartbeat_path        = train_ini.get_or<std::string>("heartbeat_path", "");

    // ── Speculative Gradient Prediction ──
    train_cfg.sgp_enabled           = opt_ini.get_or<bool>("sgp", false);
    train_cfg.sgp_version           = opt_ini.get_or<int64_t>("sgp_version", 1);
    train_cfg.sgp_initial_k         = opt_ini.get_or<int64_t>("sgp_initial_k", 2);
    train_cfg.sgp_max_k             = opt_ini.get_or<int64_t>("sgp_max_k", 8);
    train_cfg.sgp_warmup_steps      = opt_ini.get_or<int64_t>("sgp_warmup_steps", 100);
    train_cfg.sgp_rank              = opt_ini.get_or<int64_t>("sgp_rank", 4);

    // ── Gradient statistics (SGP Phase 0) ──
    train_cfg.grad_stats_path       = train_ini.get_or<std::string>("grad_stats_path", "");
    train_cfg.grad_stats_interval   = train_ini.get_or<int64_t>("grad_stats_interval", 10);

    // ── Device ──
    std::string device_pref = dev_ini.get_or<std::string>("device", "auto");

    // ── Apply multi-res DC-MRE settings ──
    if (use_multi_res) {
      cfg.use_multi_res = true;
      cfg.bpe_vocab_path = bpe_vocab;
    }

    cfg.validate();

    // Resolve "auto" device
    if (device_pref == "auto") {
#ifdef __APPLE__
      device_pref = torch::mps::is_available() ? "mps" : "cpu";
#else
      device_pref = torch::cuda::is_available() ? "cuda" : "cpu";
#endif
    }

    // Seed all RNGs
    auto seed_state = olmo_cpp::seed_all(static_cast<uint64_t>(seed_val));

    auto device = select_device(device_pref);

    // Select backend
    if (device.is_cuda()) {
      olmo_cpp::use_cuda_backend();
      // cuDNN benchmark: auto-select fastest convolution/GEMM algorithm
      at::globalContext().setBenchmarkCuDNN(true);
      // TF32 for any FP32 matmuls — 2x faster, negligible accuracy loss
      at::globalContext().setAllowTF32CuBLAS(true);
      at::globalContext().setAllowTF32CuDNN(true);
      // Disable deterministic mode for best performance
      at::globalContext().setDeterministicAlgorithms(false, false);
      std::cout << "Backend: CUDA (fused kernels, cuDNN benchmark, TF32)\n";
    } else if (device.is_cpu()) {
      olmo_cpp::use_simd_backend();
      std::cout << "Backend: SIMD (fused kernels + arena allocator)\n";
    }

    std::cout << "Config: " << conf_path << "\n";

    // Create and train model
    if (use_fused) {
      auto model = olmo_cpp::FusedTransformer(cfg);
      run(model, cfg, train_cfg, device, use_mup, true,
          use_multi_res, enable_profile, save_path, seed_state);
    } else {
      auto model = olmo_cpp::Transformer(cfg);
      run(model, cfg, train_cfg, device, use_mup, false,
          use_multi_res, enable_profile, save_path, seed_state);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
