// Exercises DDP bucket scheduling without NCCL: builds a synthetic parameter
// list, wires a counting callback as AllReduceFn, and asserts on:
//   * bucket count relative to byte budget
//   * allreduce is called exactly once per bucket per step
//   * ready-order → bucket-fire-order respects the scheduling contract
//
// Exit code 0 = all PASS. Non-zero = failure count.

#include "zwt/core/allocator.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/dist/ddp.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zwt;

namespace {

int g_failed = 0;

void expect(bool cond, const std::string& what) {
  std::printf("[%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
  if (!cond) ++g_failed;
}

// Build a Parameter owning a tiny fp32 tensor with the given name + numel.
Parameter make_param(const std::string& name, int64_t n) {
  Tensor t = empty({n}, DType::F32, Device::cpu());
  std::memset(t.data(), 0, static_cast<size_t>(n) * sizeof(float));
  Parameter p(name, std::move(t));
  p.ensure_grad();
  return p;
}

}  // namespace

int main() {
  set_activation_arena_capacity(size_t(16) << 20);

  // 6 params, numels [1000, 2000, 3000, 4000, 5000, 6000] floats (4 * n bytes).
  // Reversed order for bucketing: p5(6000), p4(5000), p3(4000), p2(3000),
  //                                p1(2000), p0(1000).
  // With bucket_bytes = 30_000 (= 7500 floats per bucket at 4 B/float):
  //   bucket 0: p5 (6000)          — adding p4 -> 11000 > 7500 ⇒ close
  //   bucket 1: p4 (5000)          — adding p3 -> 9000 > 7500  ⇒ close
  //   bucket 2: p3, p2 (4000+3000=7000)  — ok; p1 -> 9000 > 7500 ⇒ close
  //   bucket 3: p1, p0 (2000+1000=3000)
  // Total: 4 buckets.
  std::vector<Parameter> owners;
  owners.reserve(6);
  for (int i = 0; i < 6; ++i) {
    owners.push_back(make_param("p" + std::to_string(i), (i + 1) * 1000));
  }
  std::vector<Parameter*> params;
  for (auto& p : owners) params.push_back(&p);

  dist::BucketManager mgr(params, /*bucket_bytes=*/30'000, /*world_size=*/4);
  expect(mgr.num_buckets() == 4, "bucket count matches byte-budget packing");

  int fire_count = 0;
  std::vector<size_t> fired_sizes;
  mgr.set_allreduce([&](float* /*buf*/, size_t n, StreamHandle /*s*/) {
    ++fire_count;
    fired_sizes.push_back(n);
  });

  // Mark ready in reverse parameter order (like backward would).
  // bucket 0 fires after p5 ready; bucket 1 after p4; bucket 2 after p3 + p2;
  // bucket 3 after p1 + p0.
  for (int i = 5; i >= 0; --i) {
    mgr.mark_ready(i, nullptr);
  }
  expect(fire_count == 4, "one allreduce per bucket per step");
  expect(fired_sizes.size() == 4, "recorded 4 bucket sizes");

  // Each staging buffer size should equal its bucket's total_floats.
  // We reconstruct the expected sizes deterministically: reversed-order
  // walk with the same packing logic.
  // Expected: [6000, 5000, 7000, 3000]
  std::vector<size_t> want = {6000, 5000, 7000, 3000};
  for (size_t i = 0; i < want.size(); ++i) {
    expect(fired_sizes[i] == want[i],
           "bucket " + std::to_string(i) + " staging size == " + std::to_string(want[i]));
  }

  // finalize() must pass when all buckets fired.
  bool ok = true;
  try { mgr.finalize(); } catch (...) { ok = false; }
  expect(ok, "finalize() after all fired");

  // After finalize() begin_step() is implicit — marking ready again should
  // work without error.
  mgr.mark_ready(5, nullptr);
  mgr.mark_ready(4, nullptr);
  mgr.mark_ready(3, nullptr);
  mgr.mark_ready(2, nullptr);
  mgr.mark_ready(1, nullptr);
  mgr.mark_ready(0, nullptr);
  try { mgr.finalize(); } catch (...) { ok = false; }
  expect(ok, "second step fires correctly after finalize reset");

  // Duplicate mark_ready on same param within a step must throw.
  bool threw = false;
  mgr.mark_ready(5, nullptr);
  try { mgr.mark_ready(5, nullptr); } catch (...) { threw = true; }
  expect(threw, "mark_ready twice throws");

  std::printf("---\n%d failed\n", g_failed);
  return g_failed;
}
