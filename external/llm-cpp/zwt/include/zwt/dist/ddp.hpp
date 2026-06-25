#pragma once

#include "zwt/core/device.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace zwt::dist {

// Data-parallel gradient synchronization with bucketing.
//
// Each rank owns a full model copy and computes gradients on its own shard
// of the global batch. After backward, every gradient tensor must be
// all-reduced across ranks before the optimizer step sees it.
//
// Doing one all-reduce per parameter is the wrong shape: a 1B model has
// hundreds of parameters, each all-reduce incurs a launch + ring-trip cost,
// and small allreduces are latency-bound on NCCL. Instead we:
//   1. Bucket parameters by reverse-collect order (approximates the order
//      gradients finish in the backward pass — output layer first).
//   2. Size each bucket at a fixed byte budget (default 256 MiB at 2-GPU
//      NVLink scale; small buckets are pure latency on NVLink).
//   3. As soon as all params in a bucket have their grad ready, gather
//      (D2D copy) each grad into a contiguous fp32 device buffer at its
//      precomputed offset, then fire one ncclAllReduce(Avg) on that buffer
//      on a side stream.
//   4. At step end, the compute stream waits on every bucket_done event,
//      then scatter (D2D copy) the reduced buffer back into each
//      Parameter::grad. Allreduce-Avg already divided by world_size, so the
//      scatter is a pure copy.
//
// This class owns the bucketing logic and the device staging buffers. The
// actual NCCL call is parameterized via an AllReduceFn callback so that
//   (a) the NCCL dep stays at the boundary, and
//   (b) CPU builds and unit tests can exercise bucketing without linking
//       NCCL — the host path uses a std::vector<float> staging buffer and
//       a no-op or memcpy-style callback.
class BucketManager {
 public:
  // params: in the order the optimizer sees them (collect_params traversal).
  //         Bucketing is assigned in reverse so the lm_head's params (which
  //         finish first in backward) end up in bucket 0.
  // bucket_bytes: target bucket size. Actual buckets may exceed this by one
  //         parameter — we never split a single parameter across buckets.
  // world_size: kept for diagnostics; the divide-by-N is done by ncclAvg
  //         inside the AllReduceFn (the host fallback is responsible for
  //         doing the same if a CPU path ever runs in DDP mode).
  BucketManager(const std::vector<Parameter*>& params,
                size_t bucket_bytes,
                int world_size);
  ~BucketManager();

  BucketManager(const BucketManager&) = delete;
  BucketManager& operator=(const BucketManager&) = delete;

  // Callback signature: (fp32 buffer, element count, stream).
  // The buffer is the bucket's contiguous staging buffer (device on CUDA,
  // host on CPU). The callback is responsible for performing an
  // average-allreduce in place on this buffer; the caller does NOT divide
  // by world_size afterwards — ncclAvg does that for us.
  using AllReduceFn = std::function<void(float* buf, size_t n, StreamHandle s)>;

  void set_allreduce(AllReduceFn fn) { allreduce_ = std::move(fn); }

  // Indicate that grad for parameter pid (or *p) is ready. When a bucket
  // becomes complete, gather all of its parameters' fp32 grads into the
  // bucket's staging buffer (D2D async on CUDA), then invoke `allreduce_`.
  //
  // Both overloads exist: the integer-index path is preserved for the unit
  // tests. The Parameter* path is what the model's backward calls.
  void mark_ready(int param_index, StreamHandle s);
  void mark_ready(Parameter* p, StreamHandle s);

  // Wait for all in-flight all-reduces, scatter results back into each
  // parameter's grad, and reset the bucket state for the next step. The
  // caller must ensure the compute stream `s` has already waited on every
  // bucket's done event (OverlapHookup::sync_and_finalize handles this).
  void finalize(StreamHandle s = nullptr);

  // Reset ready-state. Call at the start of each step (or the end of
  // finalize — same effect).
  void begin_step();

  int    num_buckets() const { return static_cast<int>(buckets_.size()); }
  size_t bucket_bytes() const { return bucket_bytes_; }
  int    world_size()   const { return world_size_; }

  // Visible for the loopback path: which device do staging buffers live on?
  Device device() const { return device_; }

 private:
  struct Bucket {
    std::vector<int>    param_ids;   // indexes into params_
    std::vector<size_t> offsets;     // float-offset of each param within buf
    size_t              total_floats = 0;
    void*               dev_buf      = nullptr;  // cudaMalloc'd; null on CPU
    std::vector<float>  host_buf;                // used on CPU only
    int                 remaining = 0;
    bool                fired     = false;
  };

  // Snapshots taken once at construction so we don't iterate params_ on the
  // hot path. Each Bucket owns its piece.
  void build_buckets_(const std::vector<Parameter*>& params);
  void allocate_buffers_();
  void free_buffers_();

  // Issue a gather (per-param D2D/H2H copy into bucket buf). Runs on `s`.
  void gather_bucket_(Bucket& bk, StreamHandle s);
  // Issue a scatter (per-param copy from bucket buf back into Parameter::grad).
  void scatter_bucket_(const Bucket& bk, StreamHandle s);

  std::vector<Parameter*> params_;        // owned by the caller, copied here
  size_t                  bucket_bytes_;
  int                     world_size_;
  Device                  device_;        // CPU vs CUDA, snapshot at ctor
  std::vector<int>        param_to_bucket_;  // pid → bucket index
  std::unordered_map<Parameter*, int> ptr_to_pid_;
  std::vector<Bucket>     buckets_;
  AllReduceFn             allreduce_;
};

// Default bucket size. At 2-GPU NVLink an allreduce on a 1B fp32 grad payload
// (~4 GiB) is bandwidth-bound, not latency-bound, so small buckets just add
// ring-trip cost. The unit tests pass an explicit small value. Real training
// should pass 256 MiB or larger.
inline constexpr size_t kDefaultBucketBytes = size_t(256) << 20;

// Walk a list of parameters and call mark_ready on each. Used by
// Transformer::backward to signal "this layer's grads are done" without
// coupling layer code to BucketManager internals.
void signal_params_ready(const std::vector<Parameter*>& params,
                         BucketManager& mgr,
                         StreamHandle s);

}  // namespace zwt::dist
