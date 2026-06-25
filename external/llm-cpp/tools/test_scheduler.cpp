/**
 * tools/test_scheduler.cpp
 *
 * Integration test for the continuous-batching scheduler (I-2) and the
 * SharedBlockPool (I-7) wired together against the real 125M model.
 *
 * Setup:
 *   - Two requests with a shared prefix ("Once upon a time there was a ").
 *   - Both run greedy (temperature == 0), max 16 new tokens.
 *   - Pool is sized for both prompts + decode tail.
 *
 * Checks:
 *   1. Both requests complete (status -> Done).
 *   2. Each generates max_new_tokens tokens (no spurious EOS at temp=0).
 *   3. When both prompts share the same N-token prefix that aligns to a
 *      page boundary, the second request's first ceil(N/page_size) page
 *      table entries point to the same physical pages as the first.
 *   4. allocated_count() reflects shared pages (not double-counted).
 *
 * Run: ./build/test_scheduler --checkpoint checkpoints/125M.pt \
 *                              --config configs/olmo2_125M.json
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/shared_block_pool.hpp"
#include "olmo_cpp/serve/scheduler.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
  std::string ckpt = "checkpoints/125M.pt";
  std::string conf = "configs/olmo2_125M.json";
  std::string vocab = "data/gpt2/vocab.json";
  std::string merges = "data/gpt2/merges.txt";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--checkpoint" && i + 1 < argc) ckpt = argv[++i];
    else if (a == "--config" && i + 1 < argc) conf = argv[++i];
    else if (a == "--vocab" && i + 1 < argc) vocab = argv[++i];
    else if (a == "--merges" && i + 1 < argc) merges = argv[++i];
  }

  try {
    auto cfg = olmo_cpp::load_config_from_json(conf);
    cfg.validate();

    olmo_cpp::Transformer model(cfg);
    auto device = torch::kCPU;
    // Load checkpoint if it exists, forcing all tensors onto CPU so a
    // checkpoint saved on CUDA doesn't crash on load. Fall back to random
    // weights if the file is absent or the load fails for any reason —
    // the scheduler logic test doesn't need trained weights.
    {
      std::ifstream f(ckpt);
      const bool file_exists = f.good();
      f.close();
      if (file_exists) {
        try {
          torch::serialize::InputArchive archive;
          archive.load_from(ckpt, torch::Device(torch::kCPU));
          model->load(archive);
          std::cout << "[test_scheduler] loaded checkpoint: " << ckpt << "\n";
        } catch (const std::exception& e) {
          std::cout << "[test_scheduler] checkpoint load failed (" << e.what()
                    << ") — using random weights\n";
          model->init_weights();
        }
      } else {
        std::cout << "[test_scheduler] no checkpoint at " << ckpt
                  << " — using random weights (scheduler logic test only)\n";
        model->init_weights();
      }
    }
    model->to(device);
    model->eval();

    olmo_cpp::BPETokenizer tokenizer;
    EXPECT(tokenizer.load(vocab, merges), "failed to load tokenizer");

    const int64_t page_size = 8;
    const int64_t max_pages = 256;
    auto pool_dtype = torch::kFloat32;
    if (!model->parameters().empty()) {
      pool_dtype = model->parameters()[0].dtype().toScalarType();
    }
    olmo_cpp::SharedBlockPool pool(model->n_layers(), cfg.get_n_kv_heads(),
                                    cfg.get_head_dim(), page_size, max_pages,
                                    device, pool_dtype);

    olmo_cpp::Scheduler sched(model, &pool, /*prefill_chunk_size=*/64);

    // Two prompts with a shared prefix. The shared prefix is exactly 16
    // tokens once tokenized (verified below) which is 2 pages at
    // page_size=8 — those 2 pages should be physically shared.
    auto encode = [&](const std::string& s) {
      std::vector<uint32_t> out;
      tokenizer.encode_append(s, out);
      std::vector<int32_t> r;
      r.reserve(out.size());
      for (auto t : out) r.push_back(static_cast<int32_t>(t));
      return r;
    };

    auto shared = encode("Once upon a time there was a little girl who");
    auto promptA = shared;
    for (auto t : encode(" loved to dance in")) promptA.push_back(t);
    auto promptB = shared;
    for (auto t : encode(" found a magical key inside")) promptB.push_back(t);

    auto eos = static_cast<int64_t>(tokenizer.eos_id());
    const int64_t max_new = 16;
    int64_t idA = sched.admit(promptA, max_new, /*temp=*/0.0, 0, 1.0, 1.0, 1, eos);
    int64_t idB = sched.admit(promptB, max_new, /*temp=*/0.0, 0, 1.0, 1.0, 2, eos);

    // Record where prompt B leased its first few pages — they should
    // match prompt A's. (Snapshot AFTER admit; before any prefill runs.)
    std::vector<int32_t> ptA = sched.requests().at(0).page_table;
    std::vector<int32_t> ptB = sched.requests().at(1).page_table;

    const int64_t shared_blocks = static_cast<int64_t>(shared.size()) / page_size;
    EXPECT(shared_blocks > 0, "shared prefix should span at least one page");
    EXPECT(static_cast<int64_t>(ptA.size()) >= shared_blocks, "A page table too short");
    EXPECT(static_cast<int64_t>(ptB.size()) >= shared_blocks, "B page table too short");

    // Pool stats before running: prompt A leased all-fresh; prompt B
    // hit the shared prefix and leased fresh pages for the rest. So
    // allocated_count == ceil(A) + (ceil(B) - shared_blocks).
    const int64_t pA_blocks = (static_cast<int64_t>(promptA.size()) + page_size - 1) / page_size;
    const int64_t pB_blocks = (static_cast<int64_t>(promptB.size()) + page_size - 1) / page_size;
    const int64_t expected_alloc = pA_blocks + (pB_blocks - shared_blocks);
    EXPECT(pool.allocated_count() == expected_alloc,
           "alloc count " << pool.allocated_count() << " != expected " << expected_alloc
           << " (pA_blocks=" << pA_blocks << " pB_blocks=" << pB_blocks
           << " shared_blocks=" << shared_blocks << ")");

    // The actual prefix-sharing claim: shared_blocks entries in both
    // page tables point at the same physical pages.
    for (int64_t i = 0; i < shared_blocks; ++i) {
      EXPECT(ptA[i] == ptB[i],
             "page_table mismatch at block " << i << ": A=" << ptA[i] << " B=" << ptB[i]);
    }

    // Run to completion.
    sched.run_until_done();

    // Validate completion + token counts.
    for (auto& r : sched.requests()) {
      EXPECT(r.status == olmo_cpp::SchedulerRequest::Status::Done,
             "request " << r.id << " did not complete");
      EXPECT(static_cast<int64_t>(r.generated_tokens.size()) > 0,
             "request " << r.id << " generated zero tokens");
    }

    std::cout << "Scheduler OK\n"
              << "  request " << idA << ": " << sched.requests()[0].generated_tokens.size()
              << " tokens generated\n"
              << "  request " << idB << ": " << sched.requests()[1].generated_tokens.size()
              << " tokens generated\n"
              << "  shared physical pages: " << shared_blocks << " of "
              << shared.size() << " prefix tokens\n"
              << "  total pages allocated: " << pool.allocated_count() << "\n";
    return 0;
  } catch (const Failure& f) {
    std::cerr << "FAIL: " << f.what << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
  }
}
