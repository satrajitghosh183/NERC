#pragma once

#include "zwt/layers/transformer.hpp"
#include "zwt/optim/adamw.hpp"
#include "zwt/optim/lr_schedule.hpp"

#include <cstdint>
#include <string>

namespace zwt::train {

// All knobs for an end-to-end pretraining run, loaded from a simple INI file.
// The layout mirrors what a human would edit: sections for model, data,
// optimization, runtime — no nested dicts, no YAML, no surprises.
struct TrainConfig {
  // [model]
  Transformer::Config model;

  // [data]
  std::string data_path;             // token file (.npy or raw i64)
  int64_t     seq_len        = 2048;
  int64_t     batch_size     = 4;    // per-step micro batch
  int64_t     grad_accum     = 1;    // number of micro-batches per optimizer step
  uint64_t    data_seed      = 0xC0FFEE;
  bool        shuffle        = true;

  // [optim]
  optim::AdamWConfig  adamw;
  optim::CosineSchedule schedule;
  float       grad_clip      = 1.0f;

  // [runtime]
  int64_t     max_steps      = 100000;
  int64_t     log_interval   = 10;
  int64_t     ckpt_interval  = 2000;
  std::string ckpt_path      = "zwt_ckpt.bin";
  std::string resume_from;           // if non-empty, load before training
  uint64_t    init_seed      = 0xC0DEBA5EULL;
  int64_t     arena_mb       = 2048; // activation arena per step
  bool        deterministic  = false; // reproducible kernels + cuBLAS; slower

  // [device]
  // device is selected automatically — CUDA if compiled with USE_CUDA,
  // else CPU. No knob needed.

  // [dist] — multi-GPU data-parallel knobs.
  // world_size == 1 disables every DDP code path; the trainer behaves
  // bit-identically to the single-GPU baseline.
  int         rank          = 0;
  int         local_rank    = 0;
  int         world_size    = 1;
  std::string master_addr   = "127.0.0.1";
  int         master_port   = 29500;
  // Per-bucket byte budget for gradient allreduce. 256 MiB is right at
  // 2-GPU NVLink scale: bandwidth-bound, so smaller buckets just add ring
  // trips. Lower for unit tests / single-host smoke.
  int64_t     bucket_mb     = 256;
};

// Load a TrainConfig from an INI-style file. Unknown keys are errors; missing
// keys fall back to the struct defaults above. Throws on any parse failure.
TrainConfig load_train_config(const std::string& path);

// Dump a config back to a string in the same INI format. Round-trippable.
std::string dump_train_config(const TrainConfig& c);

}  // namespace zwt::train
