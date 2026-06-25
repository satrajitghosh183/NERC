/**
 * tools/test_prefix_cache.cpp
 *
 * Unit test for SharedBlockPool — the foundation for prefix caching
 * (I-7) and continuous batching (I-2).
 *
 * Exercises:
 *   1. allocate() hands out distinct pages with refcount=1.
 *   2. find_or_lease() hits on identical content + prev_hash chains.
 *   3. find_or_lease() misses on changed content.
 *   4. inc_ref / dec_ref refcount accounting; dec_ref to 0 frees + drops
 *      hash entry, and the page comes back through the freelist.
 *   5. release_all() drops all listed pages.
 *   6. The same hash chain reproduces the same page sequence across
 *      requests — the actual prefix-caching property.
 *
 * Runs on CPU. No model needed.
 */

#include "olmo_cpp/model/shared_block_pool.hpp"

#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using olmo_cpp::SharedBlockPool;

namespace {

struct Failure { std::string what; };

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::ostringstream _os; _os << msg;                          \
      throw Failure{_os.str()};                                    \
    }                                                              \
  } while (0)

}  // namespace

int main() {
  const int64_t n_layers = 2;
  const int64_t n_kv_heads = 1;
  const int64_t head_dim = 4;
  const int64_t page_size = 4;
  const int64_t max_pages = 8;
  auto device = torch::kCPU;
  auto dtype  = torch::kFloat32;

  try {
    SharedBlockPool pool(n_layers, n_kv_heads, head_dim, page_size, max_pages,
                         device, dtype);
    EXPECT(pool.allocated_count() == 0, "fresh pool should have 0 allocated");

    int32_t a = pool.allocate();
    int32_t b = pool.allocate();
    EXPECT(a != b, "allocate must return distinct pages");
    EXPECT(pool.allocated_count() == 2, "two pages should be live");

    pool.dec_ref(a);
    pool.dec_ref(b);
    EXPECT(pool.allocated_count() == 0, "after dec_ref, pool empty");

    std::vector<int32_t> ids1 = {7, 13, 21, 99};
    auto r1 = pool.find_or_lease(0, ids1.data(), ids1.size());
    EXPECT(!r1.was_hit, "first lease should miss");
    EXPECT(pool.allocated_count() == 1, "one page leased");

    auto r2 = pool.find_or_lease(0, ids1.data(), ids1.size());
    EXPECT(r2.was_hit, "second lease with same content must hit");
    EXPECT(r2.page_idx == r1.page_idx, "hit must reuse the same physical page");
    EXPECT(pool.allocated_count() == 1, "no new allocation on hit");
    EXPECT(r1.hash == r2.hash, "matching content -> matching hash");

    std::vector<int32_t> ids2 = {7, 13, 21, 100};
    auto r3 = pool.find_or_lease(0, ids2.data(), ids2.size());
    EXPECT(!r3.was_hit, "different content should miss");
    EXPECT(r3.page_idx != r1.page_idx, "different content -> different page");
    EXPECT(pool.allocated_count() == 2, "two distinct pages now");

    auto r4 = pool.find_or_lease(r1.hash, ids1.data(), ids1.size());
    EXPECT(!r4.was_hit, "different prev_hash should miss");

    // State here: r1.page_idx has refcount = 2 (from r1 lease + r2 hit).
    pool.dec_ref(r1.page_idx);   // ref=1
    auto r5 = pool.find_or_lease(0, ids1.data(), ids1.size());
    EXPECT(r5.was_hit && r5.page_idx == r1.page_idx,
           "after one dec_ref, page still valid + hashable");
    // r5 bumped ref back to 2. Two more dec_refs frees it.
    pool.dec_ref(r1.page_idx);   // ref=1
    pool.dec_ref(r1.page_idx);   // ref=0 -> freed, hash dropped
    auto r6 = pool.find_or_lease(0, ids1.data(), ids1.size());
    EXPECT(!r6.was_hit, "freed page should no longer be hashed");

    pool.release_all({r3.page_idx, r6.page_idx, r4.page_idx});

    std::vector<std::vector<int32_t>> prompt_a_blocks = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}
    };
    std::vector<int32_t> req_a_pages;
    uint64_t hA = 0;
    for (const auto& blk : prompt_a_blocks) {
      auto r = pool.find_or_lease(hA, blk.data(), blk.size());
      req_a_pages.push_back(r.page_idx);
      hA = r.hash;
      EXPECT(!r.was_hit, "first request all misses");
    }

    std::vector<std::vector<int32_t>> prompt_b_blocks = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {99, 99, 99, 99}
    };
    std::vector<int32_t> req_b_pages;
    uint64_t hB = 0;
    for (size_t i = 0; i < prompt_b_blocks.size(); ++i) {
      const auto& blk = prompt_b_blocks[i];
      auto r = pool.find_or_lease(hB, blk.data(), blk.size());
      req_b_pages.push_back(r.page_idx);
      hB = r.hash;
      if (i < 2) {
        EXPECT(r.was_hit, "shared-prefix block should hit");
        EXPECT(r.page_idx == req_a_pages[i], "must reuse same physical page");
      } else {
        EXPECT(!r.was_hit, "divergent block should miss");
        EXPECT(r.page_idx != req_a_pages[2], "divergent block must be a new page");
      }
    }

    std::cout << "SharedBlockPool OK: "
              << "alloc/free, hash chain, refcount, prefix sharing all hold\n";
    return 0;
  } catch (const Failure& f) {
    std::cerr << "FAIL: " << f.what << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
  }
}
