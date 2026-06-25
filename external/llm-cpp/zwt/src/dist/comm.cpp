#include "zwt/dist/comm.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

// C ABI exposed by zwt/src/dist/nccl_backend.cu. Declared inline here so
// comm.cpp can stay a plain .cpp without pulling cuda_runtime.h or nccl.h.
// When ZWT_HAVE_NCCL is undefined the symbols don't exist; we guard with
// the macro and throw at runtime so callers see a clean failure.
#if ZWT_HAVE_NCCL
extern "C" {
  void* zwt_nccl_init(int rank, int world_size,
                      const char* master_addr, int master_port,
                      int device_index,
                      char* errbuf, size_t errlen);
  // Average-allreduce: ncclAvg sums across ranks and divides by world_size
  // in one pass, so BucketManager's scatter doesn't need a separate divide.
  int   zwt_nccl_allreduce_avg_f32(void* handle, void* buf, size_t count,
                                   void* stream,
                                   char* errbuf, size_t errlen);
  void  zwt_nccl_destroy(void* handle);
  void  zwt_nccl_abort(void* handle);
}
#endif

namespace zwt::dist {

OverlapHookup::OverlapHookup(BucketManager& mgr, CommContext ctx)
    : mgr_(mgr), ctx_(ctx) {
  bucket_done_.reserve(mgr_.num_buckets());
  for (int b = 0; b < mgr_.num_buckets(); ++b) {
    bucket_done_.push_back(Event::create(ctx_.device));
  }

  // Wire a callback that: records grad-ready on compute, makes comm_stream
  // wait on it, performs the reduction on comm_stream, and records
  // bucket_done_[b] on comm_stream. The mark_ready() call site already runs
  // on the compute stream, so we record the ready event on the
  // currently-active stream for the bucket device — passed in via
  // StreamHandle. The "which bucket just fired" is not surfaced by the
  // AllReduceFn signature yet (it only gets buf+n+stream). We chain the
  // ordering implicitly: all-reduces are issued sequentially on the same
  // comm_stream in bucket order, so the final bucket's done event
  // transitively ordered-after every earlier reduction.
  int* fire_idx = new int(0);
  void* backend = ctx_.backend;
  Stream comm_stream = ctx_.comm_stream;
  Device dev = ctx_.device;
  mgr_.set_allreduce([this, fire_idx, backend, comm_stream, dev](
                         float* buf, size_t n, StreamHandle compute_s) {
    Stream compute{dev, compute_s};
    Event  grad_ready = Event::create(dev);
    grad_ready.record(compute);
    grad_ready.wait(comm_stream);

#if ZWT_HAVE_NCCL
    if (backend) {
      char err[256] = {0};
      int  rc = zwt_nccl_allreduce_avg_f32(backend, buf, n,
                                            comm_stream.handle, err, sizeof(err));
      if (rc != 0) {
        std::fprintf(stderr, "zwt::dist::OverlapHookup allreduce failed: %s\n", err);
        std::abort();
      }
    }
#else
    (void)backend; (void)buf; (void)n;
#endif

    int b = (*fire_idx)++;
    if (b < static_cast<int>(bucket_done_.size())) {
      bucket_done_[b].record(comm_stream);
    }
  });
  // Ownership leak of fire_idx is intentional for the duration of the
  // process — BucketManager's callback holds it. OverlapHookup lifetime is
  // tied to the training loop; cleaning it up at OverlapHookup dtor would
  // require tracking the std::function back, which is not worth the code.
}

void OverlapHookup::sync_and_finalize() {
  // Make the compute stream wait on every bucket_done event so the scatter
  // pass (and the subsequent optimizer step) read fully-reduced grads.
  Stream compute = compute_stream(ctx_.device);
  for (auto& ev : bucket_done_) ev.wait(compute);
  mgr_.finalize(compute.handle);
}

CommContext make_loopback_ctx(Device dev) {
  CommContext c;
  c.rank        = 0;
  c.world_size  = 1;
  c.device      = dev;
  c.comm_stream = side_stream(dev);
  c.backend     = nullptr;
  return c;
}

CommContext make_nccl_ctx(int rank, int world_size,
                          const std::string& master_addr, int master_port,
                          int device_index, Device dev) {
#if ZWT_HAVE_NCCL
  if (!dev.is_cuda()) {
    throw std::runtime_error("make_nccl_ctx: NCCL requires a CUDA device");
  }
  char err[256] = {0};
  void* handle = zwt_nccl_init(rank, world_size,
                                master_addr.c_str(), master_port,
                                device_index, err, sizeof(err));
  if (!handle) {
    throw std::runtime_error(std::string("make_nccl_ctx: ") + err);
  }
  CommContext c;
  c.rank        = rank;
  c.world_size  = world_size;
  c.device      = dev;
  c.comm_stream = side_stream(dev);
  c.backend     = handle;
  return c;
#else
  (void)rank; (void)world_size; (void)master_addr; (void)master_port;
  (void)device_index; (void)dev;
  throw std::runtime_error(
      "make_nccl_ctx: zwt was built without NCCL support. "
      "Rebuild with NCCL headers + libnccl on the include/library path.");
#endif
}

void nccl_destroy(CommContext& ctx) {
#if ZWT_HAVE_NCCL
  if (ctx.backend) {
    zwt_nccl_destroy(ctx.backend);
    ctx.backend = nullptr;
  }
#else
  (void)ctx;
#endif
}

void nccl_abort(CommContext& ctx) {
#if ZWT_HAVE_NCCL
  if (ctx.backend) {
    zwt_nccl_abort(ctx.backend);
  }
#else
  (void)ctx;
#endif
}

}  // namespace zwt::dist
