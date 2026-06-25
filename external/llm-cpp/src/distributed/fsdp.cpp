/**
 * src/distributed/fsdp.cpp
 *
 * ─── What FSDP is ────────────────────────────────────────────────────
 *
 * FSDP = **F**ully **S**harded **D**ata **P**arallel (Meta, 2022).
 *
 * Plain DDP keeps a full copy of the model on every GPU. For a 70B
 * parameter model in BF16 that's 140 GB — too big for a single GPU.
 *
 * FSDP shards each parameter across all ranks, so each GPU only
 * stores 1/world_size of every weight, gradient, and optimizer-state
 * tensor. When a layer's forward needs the full parameter, an
 * **all_gather** reconstructs it on the fly into a temporary buffer;
 * after the layer is done the buffer is freed. After backward, the
 * gradients are **reduce_scattered** so each rank only ends up
 * holding the gradient slice corresponding to its parameter shard.
 *
 *   forward:  for each layer:  all_gather(params) → compute → free
 *   backward: for each layer:  all_gather(params) → backward → free
 *                              reduce_scatter(grads)
 *
 * The cost is the extra communication. The benefit is that the
 * memory footprint per GPU shrinks linearly with world_size, so you
 * can train models that would not otherwise fit anywhere.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/fsdp.hpp : FSDPContext + sharding helpers.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when FSDP is configured, the train loop uses
 *     FSDPContext::pre_forward / post_backward hooks.
 *
 * --- Role in training pipeline ---
 *   Memory-saving alternative to DDP for very large models. Like the
 *   other DDP-family files, the implementation is gated by OLMO_HAS_DDP.
 *   Single-GPU runs (the 3060 quickstart) don't touch this file.
 */
#include "olmo_cpp/distributed/fsdp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace olmo_cpp {

#if defined(OLMO_HAS_DDP) || defined(OLMO_HAS_NCCL)

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FSDPContext::FSDPContext(c10::intrusive_ptr<c10d::Backend> backend,
                        c10::intrusive_ptr<c10d::Backend> inter_backend,
                        int rank, int world_size, ShardingStrategy strategy)
    : backend_(std::move(backend)),
      inter_backend_(std::move(inter_backend)),
      rank_(rank),
      world_size_(world_size),
      strategy_(strategy) {}

std::optional<FSDPContext> FSDPContext::create(
    c10::intrusive_ptr<c10d::Backend> backend,
    ShardingStrategy strategy) {
  if (!backend) return std::nullopt;
  int ws = backend->getSize();
  if (ws < 2) return std::nullopt;
  return FSDPContext(backend, nullptr, backend->getRank(), ws, strategy);
}

std::optional<FSDPContext> FSDPContext::create_hsdp(
    c10::intrusive_ptr<c10d::Backend> intra_backend,
    c10::intrusive_ptr<c10d::Backend> inter_backend,
    ShardingStrategy strategy) {
  if (!intra_backend) return std::nullopt;
  int ws = intra_backend->getSize();
  if (ws < 2) return std::nullopt;
  return FSDPContext(intra_backend, inter_backend,
                     intra_backend->getRank(), ws, strategy);
}

// ---------------------------------------------------------------------------
// Shard params: flatten each param, keep 1/N slice
// ---------------------------------------------------------------------------

void FSDPContext::shard_params(std::vector<torch::Tensor>& params) {
  if (strategy_ == ShardingStrategy::NO_SHARD) return;
  if (strategy_ == ShardingStrategy::SHARD_GRAD_OP) return;  // ZeRO-2 keeps full params
  if (is_sharded_) return;

  sharded_params_.clear();
  original_shapes_.clear();
  original_numels_.clear();

  for (auto& p : params) {
    if (!p.defined()) {
      sharded_params_.push_back(torch::Tensor());
      original_shapes_.push_back({});
      original_numels_.push_back(0);
      continue;
    }

    original_shapes_.push_back(p.sizes().vec());
    int64_t numel = p.numel();
    original_numels_.push_back(numel);

    // Pad to be divisible by world_size
    int64_t padded = ((numel + world_size_ - 1) / world_size_) * world_size_;
    auto flat = p.detach().reshape({-1});
    if (padded > numel) {
      flat = torch::cat({flat, torch::zeros({padded - numel}, flat.options())});
    }

    int64_t shard_size = padded / world_size_;
    auto shard = flat.narrow(0, rank_ * shard_size, shard_size).clone();
    sharded_params_.push_back(shard);

    // Replace parameter data with the shard (saves memory)
    {
      torch::NoGradGuard no_grad;
      p.set_data(shard);
    }
  }
  is_sharded_ = true;
}

// ---------------------------------------------------------------------------
// Unshard (allgather): reconstruct full params from shards
// ---------------------------------------------------------------------------

void FSDPContext::unshard_params(std::vector<torch::Tensor>& params) {
  if (strategy_ == ShardingStrategy::NO_SHARD) return;
  if (strategy_ == ShardingStrategy::SHARD_GRAD_OP) return;
  if (!is_sharded_) return;

  for (size_t i = 0; i < params.size(); ++i) {
    if (!params[i].defined() || !sharded_params_[i].defined()) continue;

    auto shard = sharded_params_[i];
    int64_t shard_size = shard.numel();

    // Allgather: each rank contributes its shard
    std::vector<at::Tensor> gathered(world_size_);
    for (int r = 0; r < world_size_; ++r) {
      gathered[r] = torch::empty({shard_size}, shard.options());
    }
    std::vector<std::vector<at::Tensor>> output_tensors = {gathered};
    std::vector<at::Tensor> input_tensors = {shard};
    backend_->allgather(output_tensors, input_tensors)->wait();

    // Concatenate and reshape to original
    auto full = torch::cat(gathered, 0);
    int64_t orig_numel = original_numels_[i];
    full = full.narrow(0, 0, orig_numel).view(original_shapes_[i]);

    {
      torch::NoGradGuard no_grad;
      params[i].set_data(full);
    }
  }
}

// ---------------------------------------------------------------------------
// Reduce-scatter gradients
// ---------------------------------------------------------------------------

void FSDPContext::reduce_scatter_grads(std::vector<torch::Tensor>& grads) {
  if (strategy_ == ShardingStrategy::NO_SHARD) {
    // Simple allreduce for NO_SHARD
    for (auto& g : grads) {
      if (g.defined()) {
        std::vector<at::Tensor> tensors = {g};
        backend_->allreduce(tensors)->wait();
        g.div_(world_size_);
      }
    }
    return;
  }

  for (size_t i = 0; i < grads.size(); ++i) {
    if (!grads[i].defined()) continue;

    auto grad = grads[i].reshape({-1});
    int64_t numel = grad.numel();
    int64_t padded = ((numel + world_size_ - 1) / world_size_) * world_size_;

    if (padded > numel) {
      grad = torch::cat({grad, torch::zeros({padded - numel}, grad.options())});
    }

    int64_t shard_size = padded / world_size_;

    // Split into chunks for each rank
    std::vector<at::Tensor> input_chunks;
    for (int r = 0; r < world_size_; ++r) {
      input_chunks.push_back(grad.narrow(0, r * shard_size, shard_size).contiguous());
    }

    auto output = torch::empty({shard_size}, grad.options());

    // reduce_scatter: sum across ranks, scatter result
    std::vector<std::vector<at::Tensor>> input_per_rank = {input_chunks};
    std::vector<at::Tensor> outputs = {output};
    backend_->reduce_scatter(outputs, input_per_rank)->wait();

    output.div_(world_size_);
    grads[i] = output;
  }

  // For HSDP: also allreduce across inter-node group
  if (inter_backend_) {
    for (auto& g : grads) {
      if (g.defined()) {
        std::vector<at::Tensor> tensors = {g};
        inter_backend_->allreduce(tensors)->wait();
        g.div_(inter_backend_->getSize());
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Re-shard: free full param memory, go back to local shards
// ---------------------------------------------------------------------------

void FSDPContext::reshard_params(std::vector<torch::Tensor>& params) {
  if (strategy_ == ShardingStrategy::NO_SHARD) return;
  if (strategy_ == ShardingStrategy::SHARD_GRAD_OP) return;
  if (!is_sharded_) return;

  for (size_t i = 0; i < params.size(); ++i) {
    if (!params[i].defined() || !sharded_params_[i].defined()) continue;
    torch::NoGradGuard no_grad;
    params[i].set_data(sharded_params_[i]);
  }
}

#else  // !OLMO_HAS_DDP

FSDPContext::FSDPContext(int rank, int world_size, ShardingStrategy strategy)
    : rank_(rank), world_size_(world_size), strategy_(strategy) {}

void FSDPContext::shard_params(std::vector<torch::Tensor>&) {}
void FSDPContext::unshard_params(std::vector<torch::Tensor>&) {}
void FSDPContext::reduce_scatter_grads(std::vector<torch::Tensor>&) {}
void FSDPContext::reshard_params(std::vector<torch::Tensor>&) {}

#endif

}  // namespace olmo_cpp
