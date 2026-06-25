# DDP comm/compute overlap

At 1B params with 16 layers the backward pass spans ~500 ms on H100. A
single post-backward all-reduce of every gradient as one buffer would add
20–40 ms of latency hidden behind compute, which is fine — but it also means
the whole fp32 grad buffer is live at optimizer time, doubling grad memory.
Bucketing lets us fire reductions *during* backward: the output-layer
gradients finish first, so their all-reduce begins while the embedding-layer
backward is still running.

## Streams and events

```
compute_stream:  … bwd(block_N)  ── [grad_ready_N] ──  bwd(block_N-1)  …  opt.step()
                                     │                                    ▲
                                     ▼                                    │
comm_stream:         allreduce(bucket_k)  ── [bucket_done_k] ─── wait ────┘
```

- `compute_stream` is zwt's main training stream (`zwt::compute_stream(dev)`).
- `comm_stream` is a side stream (`zwt::side_stream(dev)`) dedicated to NCCL.
- `grad_ready` events flow compute → comm (comm_stream waits before reducing).
- `bucket_done` events flow comm → compute (compute_stream waits before
  reading grads in the optimizer step).

## API

```cpp
auto mgr = zwt::dist::make_ddp(params, /*bucket_bytes=*/25 << 20, world_size);
auto ctx = zwt::dist::make_loopback_ctx(dev);   // real NCCL context in prod
zwt::dist::OverlapHookup hook(mgr, ctx);

for (step in steps) {
  mgr.begin_step();
  model.forward(...);
  model.backward(grad_logits);
  // Linear::backward calls mgr.mark_ready(param_id, compute_stream.handle)
  // immediately after producing each param's grad — the bucket fires as
  // soon as its last param is ready.
  hook.sync_and_finalize();   // join bucket_done events → compute stream
  opt.step();
}
```

## Bucketing heuristic

- Packing in **reverse** registration order matches the order gradients
  emerge from backward — the output layer is last in forward and therefore
  first in backward. Bucket 0 always holds the last-forward / first-backward
  parameters so its reduction overlaps the most of backward.
- Default bucket size 25 MiB matches PyTorch DDP's default. At bf16 grads
  and fp32 staging, this is ~6.25M params per bucket → ~3 MB NCCL message.
  Below that, NCCL latency dominates; above it, overlap opportunity shrinks.

## NCCL hookup (pending)

Today `OverlapHookup`'s allreduce callback exercises event sequencing only;
`comm.cpp` contains the exact spot where `ncclAllReduce` on `comm_stream`
slots in. Landing NCCL proper requires:

1. A `dist/nccl_backend.cpp` TU that creates `ncclComm_t` via
   `ncclCommInitRank` + rendezvous (MPI, torchrun-style env vars, or file-
   based). Stored in `CommContext::backend`.
2. A staging buffer per bucket — allocated via `device_pool(dev)` as an
   fp32 tensor sized to `bucket.total_floats`. The per-step cost is one
   grad→staging copy per bucket, then one NCCL reduce over the staging, then
   a staging→grad scatter scaled by `1/world_size_`.
3. CMake guard on `-DZWT_USE_NCCL=ON` that links `libnccl`. CPU builds and
   single-rank runs continue to use the loopback context.
```
