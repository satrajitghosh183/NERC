/**
 * src/train/activation_checkpoint.cpp
 *
 * ─── What "activation checkpointing" is ─────────────────────────────
 *
 * To compute gradients via backprop, the framework normally keeps in
 * memory every intermediate activation produced during forward — for
 * a 32-layer model that's ~32x the activation memory of a single
 * layer. On a 12 GB 3060 this can blow the VRAM budget for any
 * serious model size.
 *
 * Activation checkpointing trades **compute** for **memory**: drop
 * intermediate activations during forward, and **recompute** them
 * during backward right before they're needed. You pay one extra
 * forward pass per checkpointed segment but save the matching memory.
 *
 * This file wires that into LibTorch's autograd. We use a
 * `torch::autograd::Function` — the standard primitive for inserting
 * a custom forward/backward pair — whose forward saves only the
 * input, and whose backward re-runs the segment under
 * `torch::enable_grad()` to materialise gradients.
 *
 * Two policies (selected by cfg.activation_checkpoint_mode):
 *   - "full"             : checkpoint every block.
 *   - "selected_blocks"  : checkpoint every Nth block (controlled by
 *                          activation_checkpoint_interval). Halving
 *                          the count of checkpointed blocks roughly
 *                          halves the memory savings AND halves the
 *                          recompute cost.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train/activation_checkpoint.hpp : entry-point decl.
 *   - olmo_cpp/train/autocast_guard.hpp        : keeps the recompute
 *     pass under the same autocast precision as the original forward.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/transformer.cpp / fused_transformer.cpp: forward()
 *     wraps each block's call in checkpoint(...) when the .conf
 *     enables it.
 *
 * --- Role in training pipeline ---
 *   Memory-saving feature, opt-in. Off in the 30M quickstart conf
 *   (it fits trivially in 12 GB) but ON in the 125M conf.
 */
#include "olmo_cpp/train/activation_checkpoint.hpp"
#include "olmo_cpp/train/autocast_guard.hpp"

namespace olmo_cpp {

// Thread-local storage for the checkpoint function and device.
// Only tensors can safely go through autograd Function::apply.
// These are set before apply() and read inside forward/backward.
static thread_local std::function<torch::Tensor(torch::Tensor)>* tl_ckpt_fn = nullptr;

// Custom autograd function — only takes a single Tensor through apply()
class CheckpointFunction : public torch::autograd::Function<CheckpointFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input) {
    // Save input for recomputation during backward
    ctx->save_for_backward({input});

    // Heap-allocate a copy of the function for backward (the thread-local
    // and the caller's stack frame won't exist when backward runs later)
    auto* fn_copy = new std::function<torch::Tensor(torch::Tensor)>(*tl_ckpt_fn);
    ctx->saved_data["fn_ptr"] = reinterpret_cast<int64_t>(fn_copy);

    // Save whether autocast is active so backward can re-enable it.
    // Without this, recomputation runs without autocast → dtype mismatch
    // when weights are BF16 but intermediate tensors are FP32.
    bool has_autocast = false;
#if defined(OLMO_AUTOCAST_DEVICE_API)
    has_autocast = at::autocast::is_autocast_enabled(at::kCUDA);
#elif defined(OLMO_AUTOCAST_GPU_API)
    has_autocast = at::autocast::is_autocast_gpu_enabled();
#endif
    ctx->saved_data["autocast"] = has_autocast;
    ctx->saved_data["is_cuda"] = input.is_cuda();

    // Run forward WITHOUT gradient tracking — this is the memory saving:
    // intermediate activations are NOT stored in the autograd graph
    torch::Tensor output;
    {
      torch::NoGradGuard no_grad;
      output = (*tl_ckpt_fn)(input);
    }
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    // Retrieve function pointer and autocast state
    auto fn = reinterpret_cast<std::function<torch::Tensor(torch::Tensor)>*>(
        ctx->saved_data["fn_ptr"].toInt());
    bool had_autocast = ctx->saved_data["autocast"].toBool();
    bool is_cuda = ctx->saved_data["is_cuda"].toBool();

    // Recompute forward pass WITH gradients enabled AND autocast restored
    torch::Tensor input_detached = input.detach().requires_grad_(true);
    torch::Tensor output;
    {
      torch::AutoGradMode enable_grad(true);
      AutocastGuard ac(had_autocast,
          is_cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU));
      output = (*fn)(input_detached);
    }

    // Backpropagate through recomputed graph
    output.backward(grad_outputs[0]);

    // Free the heap-allocated function copy
    delete fn;

    return {input_detached.grad()};
  }
};

torch::Tensor ActivationCheckpoint::checkpoint(
    std::function<torch::Tensor(torch::Tensor)> fn,
    torch::Tensor input) {
  // Store function in thread-local so forward() can access it
  // without passing a non-tensor through apply()
  tl_ckpt_fn = &fn;
  auto result = CheckpointFunction::apply(input);
  tl_ckpt_fn = nullptr;
  return result;
}

bool ActivationCheckpoint::should_checkpoint(int64_t layer_idx, int64_t interval) {
  if (interval <= 0) return false;
  return (layer_idx % interval) == 0;
}

}  // namespace olmo_cpp
