#pragma once

#include "zwt/layers/parameter.hpp"
#include "zwt/optim/adamw.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zwt::train {

// Binary checkpoint format. Single file containing:
//   * header with magic / version / step / seed / data_cursor
//   * for each parameter, three tensor records:
//       "<name>"     -> parameter value  (BF16 or F32)
//       "<name>.m"   -> AdamW 1st moment (F32)
//       "<name>.v"   -> AdamW 2nd moment (F32)
//
// Tensor records carry shape + dtype + bytes, so the loader can sanity-check
// against the live model before touching any buffers. Host<->device transfer
// is handled internally — caller passes the live params and optimizer and the
// checkpoint code deals with all the staging.
//
// The format is intentionally flat and uncompressed. Time to save a 3B model
// at 6 GB on an NVMe SSD is ~3 s; compression would cost more CPU than it's
// worth at training cadence.
struct CheckpointMeta {
  int64_t  step        = 0;
  uint64_t seed        = 0;
  int64_t  data_cursor = 0;   // TokenLoader chunk cursor for resume
  float    lr          = 0.f;
  float    loss        = 0.f;
};

// Write a full checkpoint to `path`. Overwrites any existing file atomically
// by writing to `path + ".tmp"` and renaming on success.
void save_checkpoint(const std::string& path,
                     const std::vector<Parameter*>& params,
                     optim::AdamW& opt,
                     const CheckpointMeta& meta);

// Restore parameters + optimizer state + meta from `path`. Throws if shapes
// or dtypes don't line up with the live model.
CheckpointMeta load_checkpoint(const std::string& path,
                               const std::vector<Parameter*>& params,
                               optim::AdamW& opt);

// Read just the metadata header from a checkpoint without touching tensors.
// Useful for tooling / quick inspection.
CheckpointMeta read_checkpoint_meta(const std::string& path);

}  // namespace zwt::train
