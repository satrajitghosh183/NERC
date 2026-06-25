/**
 * src/distributed/pipeline_parallel.cpp
 *
 * ─── What "Pipeline Parallelism" is ─────────────────────────────────
 *
 * Suppose you have 64 transformer layers and 4 GPUs. Pipeline
 * Parallelism (PP) assigns layers 0-15 to GPU 0, 16-31 to GPU 1, etc.
 * A microbatch goes through GPU 0 (layers 0-15), then its output
 * activations are sent to GPU 1 (layers 16-31), and so on, like an
 * assembly line.
 *
 * The naive version wastes 75% of GPU time (only one stage works at
 * any moment). The classic fix is to split the global batch into
 * many small microbatches and pipeline them — while GPU 1 is working
 * on microbatch i, GPU 0 already started microbatch i+1. The
 * standard schedule is "1F1B" (one-forward-one-backward).
 *
 * Communication is point-to-point send/recv (NOT a collective).
 * That's why PP often pairs well with TP/DP: it doesn't tax the
 * cross-GPU all-reduce bandwidth.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/pipeline_parallel.hpp : the PP context type.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when a PP context is constructed, the train loop
 *     interleaves microbatch fwd/bwd via PipelineParallelContext.
 *
 * --- Role in training pipeline ---
 *   Used only for very large models where activation memory pressure
 *   forces splitting the layer stack across devices. Off by default.
 */
#include "olmo_cpp/distributed/pipeline_parallel.hpp"

namespace olmo_cpp {

std::optional<PipelineParallelContext> PipelineParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend) {
  if (!backend) return std::nullopt;
  int world_size = backend->getSize();
  if (world_size < 2) return std::nullopt;
  int rank = backend->getRank();
  // Placeholder: assume equal layer split. Actual layer count comes from config.
  // For now we just store rank/world_size; layer ranges set by caller.
  return PipelineParallelContext(backend, rank, world_size, rank, rank + 1);
}

PipelineParallelContext::PipelineParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend,
    int rank, int world_size,
    int64_t start_layer, int64_t end_layer)
    : backend_(std::move(backend)),
      rank_(rank),
      world_size_(world_size),
      start_layer_(start_layer),
      end_layer_(end_layer) {}

void PipelineParallelContext::send_to_next_stage(const torch::Tensor& t) {
  if (rank_ + 1 >= world_size_) return;  // last stage; no downstream
  std::vector<at::Tensor> tensors = {t.contiguous()};
  // tag=0 reserved for forward activations; gradients use tag=1.
  backend_->send(tensors, /*dstRank=*/rank_ + 1, /*tag=*/0)->wait();
}

torch::Tensor PipelineParallelContext::recv_from_prev_stage(const torch::TensorOptions& opts) {
  TORCH_CHECK(act_meta_set_,
              "PipelineParallelContext::recv_from_prev_stage requires set_activation_meta()");
  TORCH_CHECK(rank_ > 0,
              "first stage cannot recv_from_prev_stage");
  auto buf = torch::empty(act_shape_,
                          opts.dtype().has_value() ? opts : opts.dtype(act_dtype_));
  std::vector<at::Tensor> tensors = {buf};
  backend_->recv(tensors, /*srcRank=*/rank_ - 1, /*tag=*/0)->wait();
  return buf;
}

void PipelineParallelContext::send_grad_to_prev_stage(const torch::Tensor& grad) {
  if (rank_ == 0) return;  // first stage; nothing upstream
  std::vector<at::Tensor> tensors = {grad.contiguous()};
  backend_->send(tensors, /*dstRank=*/rank_ - 1, /*tag=*/1)->wait();
}

torch::Tensor PipelineParallelContext::recv_grad_from_next_stage(const torch::TensorOptions& opts) {
  TORCH_CHECK(act_meta_set_,
              "PipelineParallelContext::recv_grad_from_next_stage requires set_activation_meta()");
  TORCH_CHECK(rank_ + 1 < world_size_,
              "last stage cannot recv_grad_from_next_stage");
  auto buf = torch::empty(act_shape_,
                          opts.dtype().has_value() ? opts : opts.dtype(act_dtype_));
  std::vector<at::Tensor> tensors = {buf};
  backend_->recv(tensors, /*srcRank=*/rank_ + 1, /*tag=*/1)->wait();
  return buf;
}

void PipelineParallelContext::run_1f1b(
    int num_microbatches,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& targets,
    StageForwardFn fwd, StageBackwardFn bwd) {
  TORCH_CHECK(act_meta_set_, "run_1f1b: call set_activation_meta() first");
  const bool first = is_first_stage();
  const bool last  = is_last_stage();
  const int S = world_size_;
  const int r = rank_;
  // Number of warmup (and trailing cooldown) microbatches at this stage.
  // Stage 0 does S-1 warmups before any backwards; the deepest stage
  // does 0. Symmetric for cooldown.
  const int warmup = std::min(S - 1 - r, num_microbatches);
  const int steady = num_microbatches - warmup;

  auto opts = torch::TensorOptions().dtype(act_dtype_).device(act_device_);

  // For each microbatch, we remember the stage's forward output so the
  // backward callback can find what gradient it needs to flow through.
  // The callback itself is responsible for any tape stashing it needs.
  std::vector<torch::Tensor> stash_out(num_microbatches);

  // ── Warmup: warmup forwards, no backwards yet. ─────────────────────
  for (int i = 0; i < warmup; ++i) {
    torch::Tensor in;
    if (first) {
      TORCH_CHECK(static_cast<int>(inputs.size()) > i,
                  "run_1f1b: stage 0 needs `inputs` per microbatch");
      in = inputs[i];
    } else {
      in = recv_from_prev_stage(opts);
    }
    auto out = fwd(in, i);
    stash_out[i] = out;
    if (!last) send_to_next_stage(out);
  }

  // ── Steady: alternate forward and backward. ───────────────────────
  for (int i = 0; i < steady; ++i) {
    const int fwd_idx = warmup + i;
    const int bwd_idx = i;

    // Forward of microbatch fwd_idx
    torch::Tensor in;
    if (first) {
      TORCH_CHECK(static_cast<int>(inputs.size()) > fwd_idx,
                  "run_1f1b: stage 0 needs `inputs` per microbatch");
      in = inputs[fwd_idx];
    } else {
      in = recv_from_prev_stage(opts);
    }
    auto out = fwd(in, fwd_idx);
    stash_out[fwd_idx] = out;
    if (!last) send_to_next_stage(out);

    // Backward of microbatch bwd_idx
    torch::Tensor upstream_grad;
    if (last) {
      // On the last stage, the seed grad is implicit (the loss).
      // We pass an empty tensor; the user's bwd callback handles
      // `out.backward()` semantics with the cached output and target.
      TORCH_CHECK(static_cast<int>(targets.size()) > bwd_idx,
                  "run_1f1b: last stage needs `targets` per microbatch");
      upstream_grad = targets[bwd_idx];
    } else {
      upstream_grad = recv_grad_from_next_stage(opts);
    }
    auto grad_in = bwd(upstream_grad, bwd_idx);
    if (!first) send_grad_to_prev_stage(grad_in);
  }

  // ── Cooldown: trailing backwards. ─────────────────────────────────
  for (int i = 0; i < warmup; ++i) {
    const int bwd_idx = steady + i;
    torch::Tensor upstream_grad;
    if (last) {
      TORCH_CHECK(static_cast<int>(targets.size()) > bwd_idx,
                  "run_1f1b: last stage needs `targets` per microbatch");
      upstream_grad = targets[bwd_idx];
    } else {
      upstream_grad = recv_grad_from_next_stage(opts);
    }
    auto grad_in = bwd(upstream_grad, bwd_idx);
    if (!first) send_grad_to_prev_stage(grad_in);
  }
}

}  // namespace olmo_cpp
