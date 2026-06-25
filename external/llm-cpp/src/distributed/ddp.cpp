/**
 * src/distributed/ddp.cpp
 *
 * ─── What DDP is ────────────────────────────────────────────────────
 *
 * DDP = **D**istributed **D**ata **P**arallel — the simplest way to
 * scale training across GPUs.
 *
 * Each GPU (called a "rank") holds an IDENTICAL copy of the model.
 * They process DIFFERENT microbatches of data in parallel. After
 * backward, each rank has its own gradient that reflects only its
 * microbatch. To stay synchronised, an **all_reduce(SUM)** sums the
 * gradients across all ranks, then each rank divides by world_size to
 * get the average gradient. Since the model and optimizer were
 * identical going in, applying the averaged gradient keeps them
 * identical going out.
 *
 *   per step: forward -> backward -> all_reduce(grads) -> optim.step()
 *
 * One allreduce per step, regardless of how many parameters — the
 * communication is overlap-friendly with backward (you can start
 * reducing the gradient of layer N while still backpropping through
 * layer N-1).
 *
 * ─── How init works here ────────────────────────────────────────────
 *
 * DDPContext::init_from_env() reads the standard env vars:
 *   MASTER_ADDR / MASTER_PORT   where rank 0 listens
 *   RANK         my rank in [0, WORLD_SIZE)
 *   WORLD_SIZE   total number of ranks
 * It opens a c10d::TCPStore on rank 0 (the rendezvous service), then
 * constructs a c10d::ProcessGroupGloo. Returns std::nullopt if the
 * env vars are absent so single-process runs are unaffected.
 *
 * Why **Gloo** and not NCCL? Gloo is CPU-friendly and ships with
 * pip-installed LibTorch. NCCL is faster on GPU clusters but adds a
 * dep that not every developer has. This codebase uses Gloo for
 * portability; rebuild against NCCL for production multi-node runs.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/ddp.hpp : DDPContext declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: DDPContext::init_from_env() at startup; allreduce
 *     is called inside the "allreduce" ProfileScope after backward.
 *
 * --- Role in training pipeline ---
 *   Compiled when OLMO_USE_DDP=ON. Without it ddp_stub.cpp takes its
 *   place. The quickstart's 3060 single-GPU flow uses the stub.
 */
#include "olmo_cpp/distributed/ddp.hpp"
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Backend selection. NCCL (GPU collectives, ships in pip torch via
// nvidia-nccl-cu12) is preferred for H100; Gloo (CPU, needs separate headers)
// is the portability fallback. The build picks exactly one via -DOLMO_USE_NCCL
// or -DOLMO_USE_DDP; this file is only compiled when one of them is set
// (otherwise ddp_stub.cpp is used).
#if defined(OLMO_HAS_NCCL)
#  include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#  include <c10/cuda/CUDAFunctions.h>
#elif defined(OLMO_HAS_DDP)
#  include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif

namespace olmo_cpp {

std::optional<DDPContext> DDPContext::init_from_env() {
  const char* rank_str = std::getenv("RANK");
  const char* world_size_str = std::getenv("WORLD_SIZE");
  const char* master_addr = std::getenv("MASTER_ADDR");
  const char* master_port = std::getenv("MASTER_PORT");

  if (!rank_str || !world_size_str || !master_addr || !master_port) {
    return std::nullopt;
  }

  int rank = std::stoi(rank_str);
  int world_size = std::stoi(world_size_str);
  uint16_t port = static_cast<uint16_t>(std::stoi(master_port));

  c10d::TCPStoreOptions store_opts;
  store_opts.port = port;
  store_opts.isServer = (rank == 0);
  store_opts.numWorkers = world_size;
  store_opts.waitWorkers = true;

  auto store = c10::make_intrusive<c10d::TCPStore>(std::string(master_addr), store_opts);

  c10::intrusive_ptr<c10d::Backend> backend;
#if defined(OLMO_HAS_NCCL)
  // Pin this rank to its GPU BEFORE creating the NCCL process group — NCCL
  // binds the communicator to the current device. LOCAL_RANK is the per-node
  // GPU index (set by the launcher); fall back to global rank for 1 node.
  const char* local_rank_str = std::getenv("LOCAL_RANK");
  int local_rank = local_rank_str ? std::stoi(local_rank_str) : rank;
  c10::cuda::set_device(static_cast<c10::DeviceIndex>(local_rank));
  auto nccl_opts = c10d::ProcessGroupNCCL::Options::create();
  backend = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world_size, nccl_opts);
#elif defined(OLMO_HAS_DDP)
  auto gloo_opts = c10d::ProcessGroupGloo::Options::create();
  backend = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, world_size, gloo_opts);
#else
  return std::nullopt;  // no backend compiled — stub should have been used
#endif

  return DDPContext(backend, rank, world_size);
}

DDPContext::DDPContext(c10::intrusive_ptr<c10d::Backend> backend, int rank, int world_size)
    : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {}

void DDPContext::broadcast_parameters(std::vector<torch::Tensor>& parameters) {
  if (!backend_) return;
  for (auto& p : parameters) {
    if (p.defined()) {
      std::vector<at::Tensor> tensors = {p};
      c10d::BroadcastOptions opts;
      opts.rootRank = 0;
      backend_->broadcast(tensors, opts)->wait();
    }
  }
}

void DDPContext::register_grad_hooks(std::vector<torch::Tensor>& parameters,
                                      int64_t bucket_bytes) {
  if (!backend_) return;
  if (!hook_state_.buckets.empty()) return;  // already registered

  // Build buckets in REVERSE parameter order (deeper layers first, matching
  // backward order). Each bucket holds N consecutive params; when the last
  // param's hook fires for a bucket, the bucket's grads are dispatched as
  // one allreduce.
  std::vector<std::pair<size_t, size_t>> bucket_ranges;  // (start, count) in reverse order
  int64_t cur_bytes = 0;
  size_t bucket_start = 0;
  size_t reverse_idx = 0;  // count from the back

  // First pass: figure out which params go into which bucket. We walk
  // parameters in reverse so reverse_idx == 0 is the deepest layer.
  std::vector<size_t> param_to_bucket(parameters.size(),
                                      std::numeric_limits<size_t>::max());
  size_t bucket_id = 0;
  for (auto it = parameters.rbegin(); it != parameters.rend(); ++it, ++reverse_idx) {
    const auto& p = *it;
    if (!p.defined() || !p.requires_grad()) continue;
    const int64_t pbytes = p.numel() * p.element_size();
    if (cur_bytes + pbytes > bucket_bytes && cur_bytes > 0) {
      bucket_id++;
      cur_bytes = 0;
    }
    const size_t fwd_idx = parameters.size() - 1 - reverse_idx;
    param_to_bucket[fwd_idx] = bucket_id;
    cur_bytes += pbytes;
  }
  const size_t n_buckets = bucket_id + 1;

  hook_state_.buckets.assign(n_buckets, Bucket{});
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (param_to_bucket[i] == std::numeric_limits<size_t>::max()) continue;
    const size_t bidx = param_to_bucket[i];
    hook_state_.buckets[bidx].total_count++;
  }

  // Register one hook per trainable parameter. The hook stores the
  // gradient passed by autograd into the bucket and increments the
  // bucket's ready counter. When the bucket fills, the hook dispatches
  // an allreduce (unless sync_required_ is false — that's the no_sync
  // case for non-final accumulation steps).
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (param_to_bucket[i] == std::numeric_limits<size_t>::max()) continue;
    const size_t bidx = param_to_bucket[i];
    auto& p = parameters[i];
    // Capture bidx + this by value; `this` is the DDPContext.
    p.register_hook([this, bidx](const at::Tensor& grad) -> at::Tensor {
      if (!this->sync_required_) return grad;
      std::lock_guard<std::mutex> lock(*this->hook_state_.mu);
      Bucket& b = this->hook_state_.buckets[bidx];
      b.grads.push_back(grad);
      b.ready_count++;
      if (b.ready_count == b.total_count) {
        // Bucket complete — dispatch the collective. ProcessGroupNCCL.allreduce
        // takes ONE tensor per call (the multi-tensor form is deprecated and
        // throws "Expecting one tensor only"), so reduce each grad individually;
        // each returns a c10::Work we store and wait on at finalize time.
        for (auto& g : b.grads) {
          std::vector<at::Tensor> single{g};
          this->hook_state_.pending_works.push_back(this->backend_->allreduce(single));
        }
      }
      return grad;
    });
  }
}

void DDPContext::allreduce_gradients(const std::vector<torch::Tensor>& parameters) {
  if (!backend_) return;

  // Hook mode: most of the work has already been kicked off during
  // backward. Wait on all the in-flight bucket Works, reset bucket
  // state for the next iteration, then divide by world_size.
  if (has_hooks()) {
    std::vector<c10::intrusive_ptr<c10d::Work>> works_to_wait;
    {
      std::lock_guard<std::mutex> lock(*hook_state_.mu);
      works_to_wait = std::move(hook_state_.pending_works);
      hook_state_.pending_works.clear();
      // Reset bucket fill state. The grad tensor references are owned by
      // the model's parameter .grad attributes, which are still valid;
      // we just drop our pointers so the next backward starts fresh.
      for (auto& b : hook_state_.buckets) {
        b.grads.clear();
        b.ready_count = 0;
      }
    }
    for (auto& w : works_to_wait) {
      w->wait();
    }

    // Divide all gradients by world_size in one fused kernel.
    std::vector<at::Tensor> grads;
    grads.reserve(parameters.size());
    for (const auto& p : parameters) {
      if (p.defined() && p.requires_grad() && p.grad().defined()) {
        grads.push_back(p.grad());
      }
    }
    if (grads.empty()) return;
    const auto inv_ws = 1.0 / static_cast<double>(world_size_);
#if defined(__cpp_lib_ranges) || (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 14000)
    at::_foreach_mul_(grads, c10::Scalar(inv_ws));
#else
    for (auto& g : grads) g.mul_(inv_ws);
#endif
    return;
  }

  // Legacy path (no hooks registered): bucketed end-of-step allreduce.

  // Collect every defined gradient. Walk parameters in REVERSE order so the
  // first bucket flushed contains the deepest-layer gradients — those are
  // ready first during backward, so this ordering pairs cleanly with the
  // backward-hook variant (TODO below) where bucket-N completes before
  // bucket-N+1 even starts.
  std::vector<at::Tensor> grads;
  grads.reserve(parameters.size());
  for (auto it = parameters.rbegin(); it != parameters.rend(); ++it) {
    const auto& p = *it;
    if (p.defined() && p.requires_grad() && p.grad().defined()) {
      grads.push_back(p.grad());
    }
  }
  if (grads.empty()) return;

  // Bucket gradients into ~25MB groups and issue an allreduce per bucket
  // concurrently. c10d::Work objects are independent — by dispatching
  // multiple buckets before waiting, we pipeline collective dispatch and
  // (on NCCL builds) get free overlap between bucket transfers. With the
  // single-fused allreduce the old code did, the dispatcher had to wait
  // for the entire bucket before any bytes went over the wire.
  //
  // NOTE: this still does NOT overlap with backward — that requires
  // per-parameter hooks (`register_hook` / `register_post_accumulate_grad_hook`)
  // that fire DURING backward and dispatch buckets as their last param
  // completes, plus a no_sync()-equivalent flag honoring gradient
  // accumulation. The hook variant is the right long-term fix; this
  // bucketed end-of-step pass is the minimum-risk improvement we can
  // make without validating multi-rank locally. See README / fast-inference [T-1].
  constexpr int64_t kBucketBytes = 25 * 1024 * 1024;
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  works.reserve(8);
  std::vector<at::Tensor> bucket;
  int64_t bucket_bytes = 0;
  auto flush = [&]() {
    if (bucket.empty()) return;
    // ProcessGroupNCCL.allreduce is one-tensor-per-call (multi-tensor
    // deprecated); reduce each grad in the bucket individually.
    for (auto& g : bucket) {
      std::vector<at::Tensor> single{g};
      works.push_back(backend_->allreduce(single));
    }
    bucket.clear();
    bucket_bytes = 0;
  };
  for (auto& g : grads) {
    const int64_t n = g.numel() * g.element_size();
    if (bucket_bytes + n > kBucketBytes && !bucket.empty()) {
      flush();
    }
    bucket.push_back(g);
    bucket_bytes += n;
  }
  flush();

  // Wait for all in-flight bucket allreduces before the divide step.
  for (auto& w : works) {
    w->wait();
  }

  // Single fused divide over the full gradient list — runs on the compute
  // stream after every bucket's reduction has landed.
  const auto inv_ws = 1.0 / static_cast<double>(world_size_);
#if defined(__cpp_lib_ranges) || (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 14000)
  at::_foreach_mul_(grads, c10::Scalar(inv_ws));
#else
  for (auto& g : grads) g.mul_(inv_ws);
#endif
}

}  // namespace olmo_cpp
