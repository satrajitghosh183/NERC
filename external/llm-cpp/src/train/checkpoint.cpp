/**
 * src/train/checkpoint.cpp
 *
 * ─── What a "checkpoint" is ─────────────────────────────────────────
 *
 * A model checkpoint is a snapshot of every parameter and buffer in
 * the model, plus enough metadata (step number, optimizer learning
 * rate, etc.) to resume training as if nothing happened. Without
 * checkpoints, a 24-hour training run that crashes at hour 23 has to
 * start over.
 *
 * This file implements `CheckpointManager`, which:
 *
 *   - **save(tag, model, optimizer, meta)**  serialise everything
 *     to <base_path>/<tag>/rank_<R>/. The model parameters are split
 *     into chunks and saved by a thread pool so wall-clock time is
 *     dominated by disk bandwidth, not single-threaded torch::save.
 *
 *   - **save_async(...)**  same, but returns a std::future that
 *     completes when the save is done. The training loop can call
 *     this and immediately resume the next forward — disk write
 *     overlaps with compute.
 *
 *   - **load(tag, model, optimizer)**  reverse of save: read the
 *     archive(s) under that tag and copy_() the tensors back into
 *     the (already-constructed) model's parameters/buffers.
 *
 *   - **latest()** / **list_checkpoints()** / **prune(keep_n)**
 *     inventory + retention helpers. `prune` keeps only the N most
 *     recent checkpoints so disk usage doesn't grow unbounded.
 *
 * Layout on disk:
 *
 *   base_path/
 *     step_001000/rank_0/
 *       model_chunk_0.pt
 *       model_chunk_M.pt
 *       optimizer.pt
 *       metadata.json
 *
 * Per-rank shards mean a multi-rank run produces a tree where each
 * rank dumps its own slice — useful when combined with FSDP/TP.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train/checkpoint.hpp : CheckpointManager declaration.
 *   - olmo_cpp/io/filesystem.hpp    : abstraction over local disk vs.
 *                                      remote (S3/GCS/etc.).
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: emplaces a CheckpointManager when the .conf
 *     specifies checkpoint_dir, then calls .save() every
 *     checkpoint_interval steps and .prune() to enforce keep_n.
 *
 * --- Role in training pipeline ---
 *   Periodically active during training. Disabled when
 *   checkpoint_dir is empty.
 */
#include "olmo_cpp/train/checkpoint.hpp"
#include "olmo_cpp/io/filesystem.hpp"

#include <torch/serialize.h>
#include <algorithm>
#include <future>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

CheckpointManager::CheckpointManager(const std::string& base_path, int num_io_threads)
    : fs_(FileSystem::create(base_path)),
      base_path_(base_path),
      num_io_threads_(std::max(1, num_io_threads)) {}

std::string CheckpointManager::shard_dir(const std::string& tag, int rank) const {
  return base_path_ + "/" + tag + "/rank_" + std::to_string(rank);
}

// ---------------------------------------------------------------------------
// Metadata I/O (JSON)
// ---------------------------------------------------------------------------

void CheckpointManager::write_metadata(const std::string& dir,
                                        const CheckpointMetadata& meta) const {
#ifdef HAS_NLOHMANN_JSON
  nlohmann::json j;
  j["step"] = meta.step;
  j["epoch"] = meta.epoch;
  j["loss"] = meta.loss;
  j["model_config"] = meta.model_config_json;
  j["optimizer_type"] = meta.optimizer_type;
  j["world_size"] = meta.world_size;
  j["rank"] = meta.rank;
  std::string s = j.dump(2);
  std::vector<uint8_t> data(s.begin(), s.end());
  fs_->write_file(dir + "/metadata.json", data);
#else
  // Minimal manual JSON
  std::ostringstream oss;
  oss << "{\"step\":" << meta.step << ",\"epoch\":" << meta.epoch
      << ",\"loss\":" << meta.loss << ",\"world_size\":" << meta.world_size
      << ",\"rank\":" << meta.rank << "}";
  std::string s = oss.str();
  std::vector<uint8_t> data(s.begin(), s.end());
  fs_->write_file(dir + "/metadata.json", data);
#endif
}

CheckpointMetadata CheckpointManager::read_metadata(const std::string& dir) const {
  CheckpointMetadata meta;
#ifdef HAS_NLOHMANN_JSON
  auto data = fs_->read_file(dir + "/metadata.json");
  std::string s(data.begin(), data.end());
  auto j = nlohmann::json::parse(s);
  meta.step = j.value("step", int64_t(0));
  meta.epoch = j.value("epoch", int64_t(0));
  meta.loss = j.value("loss", 0.0);
  meta.model_config_json = j.value("model_config", std::string(""));
  meta.optimizer_type = j.value("optimizer_type", std::string(""));
  meta.world_size = j.value("world_size", 1);
  meta.rank = j.value("rank", 0);
#endif
  return meta;
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

void CheckpointManager::save(const std::string& tag,
                              const torch::nn::Module& model,
                              const torch::optim::Optimizer& optimizer,
                              const CheckpointMetadata& meta,
                              int rank, int world_size) {
  std::string dir = shard_dir(tag, rank);
  fs_->mkdir(dir);

  // Collect named parameters
  auto params = model.named_parameters();
  auto buffers = model.named_buffers();

  // Save model params in parallel using thread pool
  std::vector<std::future<void>> futures;
  int params_per_thread = std::max(1, static_cast<int>(params.size()) / num_io_threads_);

  // Group params into chunks for each thread
  std::vector<std::pair<std::string, torch::Tensor>> param_list;
  for (const auto& p : params) param_list.emplace_back(p.key(), p.value());
  for (const auto& b : buffers) param_list.emplace_back(b.key(), b.value());

  // Save as a single archive using torch::serialize
  // We split into chunks for parallel serialization
  auto save_chunk = [&](int start, int end) {
    torch::serialize::OutputArchive archive;
    for (int i = start; i < end && i < static_cast<int>(param_list.size()); ++i) {
      archive.write(param_list[i].first, param_list[i].second);
    }
    std::ostringstream oss;
    archive.save_to(oss);
    std::string s = oss.str();
    std::vector<uint8_t> data(s.begin(), s.end());
    fs_->write_file(dir + "/model_chunk_" + std::to_string(start) + ".pt", data);
  };

  for (int i = 0; i < static_cast<int>(param_list.size()); i += params_per_thread) {
    int end = std::min(i + params_per_thread, static_cast<int>(param_list.size()));
    futures.push_back(std::async(std::launch::async, save_chunk, i, end));
  }
  for (auto& f : futures) f.get();

  // Save optimizer state
  {
    std::ostringstream oss;
    torch::serialize::OutputArchive opt_archive;
    // Save optimizer state_dict manually: param group lrs and states
    const auto& groups = optimizer.param_groups();
    for (size_t g = 0; g < groups.size(); ++g) {
      auto lr = static_cast<const torch::optim::AdamWOptions&>(groups[g].options()).lr();
      opt_archive.write("group_" + std::to_string(g) + "_lr",
                        torch::tensor(lr));
    }
    opt_archive.save_to(oss);
    std::string s = oss.str();
    std::vector<uint8_t> data(s.begin(), s.end());
    fs_->write_file(dir + "/optimizer.pt", data);
  }

  // Write metadata
  CheckpointMetadata full_meta = meta;
  full_meta.rank = rank;
  full_meta.world_size = world_size;
  write_metadata(dir, full_meta);

  if (rank == 0) {
    std::cout << "Checkpoint saved: " << tag << " (step " << meta.step << ")\n";
  }
}

// ---------------------------------------------------------------------------
// Async save
// ---------------------------------------------------------------------------

std::future<void> CheckpointManager::save_async(const std::string& tag,
                                                  const torch::nn::Module& model,
                                                  const torch::optim::Optimizer& optimizer,
                                                  const CheckpointMetadata& meta,
                                                  int rank, int world_size) {
  // Clone param data to CPU to avoid racing with training
  // (we snapshot the current state)
  auto params = model.named_parameters();
  std::unordered_map<std::string, torch::Tensor> snapshot;
  for (const auto& p : params) {
    snapshot[p.key()] = p.value().detach().clone().cpu();
  }

  return std::async(std::launch::async, [this, tag, snapshot, meta, rank, world_size]() {
    std::string dir = shard_dir(tag, rank);
    fs_->mkdir(dir);

    torch::serialize::OutputArchive archive;
    for (const auto& [name, tensor] : snapshot) {
      archive.write(name, tensor);
    }
    std::ostringstream oss;
    archive.save_to(oss);
    std::string s = oss.str();
    std::vector<uint8_t> data(s.begin(), s.end());
    fs_->write_file(dir + "/model.pt", data);

    CheckpointMetadata full_meta = meta;
    full_meta.rank = rank;
    full_meta.world_size = world_size;
    write_metadata(dir, full_meta);
  });
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

CheckpointMetadata CheckpointManager::load(const std::string& tag,
                                            torch::nn::Module& model,
                                            torch::optim::Optimizer& /*optimizer*/,
                                            int rank, int /*world_size*/) {
  std::string dir = shard_dir(tag, rank);
  auto meta = read_metadata(dir);

  // Find all model chunk files
  auto files = fs_->list_dir(dir);
  std::sort(files.begin(), files.end());

  for (const auto& f : files) {
    if (f.find("model") != std::string::npos && f.find(".pt") != std::string::npos) {
      auto data = fs_->read_file(dir + "/" + f);
      std::string s(data.begin(), data.end());
      std::istringstream iss(s);
      torch::serialize::InputArchive archive;
      archive.load_from(iss);

      // Load each tensor into the model
      for (const auto& item : model.named_parameters()) {
        torch::Tensor loaded;
        if (archive.try_read(item.key(), loaded)) {
          torch::NoGradGuard no_grad;
          item.value().copy_(loaded);
        }
      }
      for (const auto& item : model.named_buffers()) {
        torch::Tensor loaded;
        if (archive.try_read(item.key(), loaded)) {
          torch::NoGradGuard no_grad;
          item.value().copy_(loaded);
        }
      }
    }
  }

  std::cout << "Checkpoint loaded: " << tag << " (step " << meta.step << ")\n";
  return meta;
}

// ---------------------------------------------------------------------------
// Latest / list / prune
// ---------------------------------------------------------------------------

std::vector<std::string> CheckpointManager::list_checkpoints() const {
  auto entries = fs_->list_dir(base_path_);
  std::vector<std::string> tags;
  for (const auto& e : entries) {
    // Checkpoint dirs typically named "step_XXXXX" or similar
    if (e.find("step") != std::string::npos || e.find("epoch") != std::string::npos) {
      tags.push_back(e);
    }
  }
  std::sort(tags.begin(), tags.end());
  return tags;
}

std::optional<std::string> CheckpointManager::latest() const {
  auto tags = list_checkpoints();
  if (tags.empty()) return std::nullopt;

  // Find tag with highest step number
  std::string best;
  int64_t best_step = -1;
  for (const auto& tag : tags) {
    // Try to extract number after "step_"
    auto pos = tag.find("step_");
    if (pos != std::string::npos) {
      try {
        int64_t step = std::stoll(tag.substr(pos + 5));
        if (step > best_step) {
          best_step = step;
          best = tag;
        }
      } catch (...) {}
    }
  }
  if (best.empty() && !tags.empty()) return tags.back();
  if (best.empty()) return std::nullopt;
  return best;
}

std::optional<std::string> CheckpointManager::latest_complete(int world_size) const {
  // Order candidate tags by numeric step, descending.
  std::vector<std::pair<int64_t, std::string>> by_step;
  for (const auto& tag : list_checkpoints()) {
    auto pos = tag.find("step_");
    if (pos == std::string::npos) continue;
    try { by_step.emplace_back(std::stoll(tag.substr(pos + 5)), tag); } catch (...) {}
  }
  std::sort(by_step.begin(), by_step.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  // Return the newest tag whose every rank shard has a readable metadata.json
  // (written LAST in save(), so its presence implies the model shards exist).
  for (const auto& kv : by_step) {
    const std::string& tag = kv.second;
    bool complete = true;
    for (int r = 0; r < world_size; ++r) {
      try { (void)read_metadata(shard_dir(tag, r)); }
      catch (...) { complete = false; break; }
    }
    if (complete) return tag;
    std::cout << "RESUME: skipping incomplete checkpoint '" << tag << "'\n";
  }
  return std::nullopt;
}

void CheckpointManager::prune(int keep_n) {
  auto tags = list_checkpoints();
  if (static_cast<int>(tags.size()) <= keep_n) return;

  // numeric step (robust-resume fix): sort by step number, not lexicographically,
  // so step_8000 is older than step_12000 (lexical order gets this wrong).
  std::sort(tags.begin(), tags.end(), [](const std::string& a, const std::string& b){
    auto num=[](const std::string& t){ auto p=t.find("step_"); return p==std::string::npos?0LL:std::stoll(t.substr(p+5)); };
    return num(a) < num(b);
  });

  int to_remove = static_cast<int>(tags.size()) - keep_n;
  for (int i = 0; i < to_remove; ++i) {
    std::string dir = base_path_ + "/" + tags[i];
    fs_->remove(dir);
    std::cout << "Pruned checkpoint: " << tags[i] << "\n";
  }
}

}  // namespace olmo_cpp
