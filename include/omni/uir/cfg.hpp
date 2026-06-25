// omni/uir/cfg.hpp — control-flow graph, dominators, post-dominators, reconvergence.
//
// Branch terminators carry target UIR BlockIds in imm0/imm1. We build succ/pred, then
// the dominator and post-dominator trees (Cooper-Harvey-Kennedy). The immediate
// post-dominator of a divergent branch is its PDOM reconvergence point; the thread
// frontier is the set of blocks on divergent paths before reconvergence (CMU 2011,
// PLAN.md §4.1 / §11 item 2).
#pragma once
#include "omni/uir/ir.hpp"
#include <vector>
#include <unordered_map>

namespace omni::uir {

struct CFG {
    FuncId fn = INVALID;
    std::vector<BlockId> blocks;                 // local index -> BlockId; blocks[0] == entry
    std::unordered_map<BlockId, int> local;      // BlockId -> local index
    std::vector<std::vector<int>> succ, pred;    // local adjacency
    std::vector<int> idom;                        // immediate dominator (local); entry -> entry
    std::vector<int> ipdom;                       // immediate post-dominator; exit -> -1

    int index_of(BlockId b) const { auto it = local.find(b); return it == local.end() ? -1 : it->second; }
    int entry() const { return 0; }
};

CFG build_cfg(const Module& m, FuncId f);

bool dominates(const CFG& g, int a, int b);        // does a dominate b?
bool post_dominates(const CFG& g, int a, int b);   // does a post-dominate b?
int  reconvergence(const CFG& g, int branch_block);// PDOM reconvergence point (local), or -1 (exit)
std::vector<int> thread_frontier(const CFG& g, int branch_block); // divergent blocks pre-reconvergence

} // namespace omni::uir
