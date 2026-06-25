#include "omni/uir/cfg.hpp"
#include "omni/uir/verify.hpp"
#include <algorithm>

namespace omni::uir {

// Cooper-Harvey-Kennedy iterative dominators over a graph given succ (for postorder
// from `entry`) and pred (for the intersect step). Returns idom (entry -> entry).
static std::vector<int> compute_idom(int n, int entry,
                                     const std::vector<std::vector<int>>& succ,
                                     const std::vector<std::vector<int>>& pred) {
    std::vector<int> post; post.reserve(n);
    std::vector<char> seen(n, 0);
    std::vector<std::pair<int, size_t>> stk;
    stk.push_back({entry, 0}); seen[entry] = 1;
    while (!stk.empty()) {                              // iterative postorder DFS
        auto& [node, ci] = stk.back();
        if (ci < succ[node].size()) {
            int s = succ[node][ci++];
            if (!seen[s]) { seen[s] = 1; stk.push_back({s, 0}); }
        } else { post.push_back(node); stk.pop_back(); }
    }
    std::vector<int> pono(n, -1);
    for (int i = 0; i < (int)post.size(); ++i) pono[post[i]] = i;
    std::vector<int> idom(n, -1); idom[entry] = entry;
    std::vector<int> rpo(post.rbegin(), post.rend());

    auto intersect = [&](int a, int b) {
        while (a != b) {
            while (pono[a] < pono[b]) a = idom[a];
            while (pono[b] < pono[a]) b = idom[b];
        }
        return a;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (int b : rpo) {
            if (b == entry) continue;
            int newidom = -1;
            for (int p : pred[b]) {
                if (pono[p] < 0 || idom[p] == -1) continue;
                newidom = (newidom == -1) ? p : intersect(p, newidom);
            }
            if (newidom != -1 && idom[b] != newidom) { idom[b] = newidom; changed = true; }
        }
    }
    return idom;
}

CFG build_cfg(const Module& m, FuncId f) {
    CFG g; g.fn = f;
    const Function& fn = m.function(f);
    for (BlockId b : fn.blocks) { g.local[b] = (int)g.blocks.size(); g.blocks.push_back(b); }
    int n = (int)g.blocks.size();
    g.succ.assign(n, {}); g.pred.assign(n, {});

    for (int i = 0; i < n; ++i) {
        const BasicBlock& b = m.block(g.blocks[i]);
        if (b.insts.empty()) continue;
        const Instruction& term = m.inst(b.insts.back());
        auto add_edge = [&](BlockId target) {
            int t = g.index_of(target);
            if (t >= 0) { g.succ[i].push_back(t); g.pred[t].push_back(i); }
        };
        switch (term.op) {
            case Op::Branch:            add_edge(term.imm0); break;
            case Op::BranchConditional: add_edge(term.imm0); add_edge(term.imm1); break;
            default: break;             // Return/Kill/Unreachable: no successors
        }
    }

    // Forward dominators.
    g.idom = compute_idom(n, g.entry(), g.succ, g.pred);

    // Post-dominators: reverse graph with a virtual exit (index n) linked to all exits.
    std::vector<std::vector<int>> rsucc(n + 1), rpred(n + 1);
    for (int i = 0; i < n; ++i)
        for (int s : g.succ[i]) { rsucc[s].push_back(i); rpred[i].push_back(s); }
    for (int i = 0; i < n; ++i)
        if (g.succ[i].empty()) { rsucc[n].push_back(i); rpred[i].push_back(n); }
    std::vector<int> ripdom = compute_idom(n + 1, n, rsucc, rpred);
    g.ipdom.assign(n, -1);
    for (int i = 0; i < n; ++i) g.ipdom[i] = (ripdom[i] == n) ? -1 : ripdom[i];
    return g;
}

bool dominates(const CFG& g, int a, int b) {
    if (a < 0 || b < 0) return false;
    for (int x = b; ; x = g.idom[x]) { if (x == a) return true; if (x == g.entry()) break; }
    return a == g.entry();
}

bool post_dominates(const CFG& g, int a, int b) {
    if (a < 0 || b < 0) return false;
    for (int x = b; x != -1; x = g.ipdom[x]) { if (x == a) return true; }
    return false;
}

int reconvergence(const CFG& g, int branch_block) {
    return (branch_block >= 0 && branch_block < (int)g.ipdom.size()) ? g.ipdom[branch_block] : -1;
}

std::vector<int> thread_frontier(const CFG& g, int branch_block) {
    int recon = reconvergence(g, branch_block);
    std::vector<int> frontier;
    std::vector<char> seen(g.blocks.size(), 0);
    std::vector<int> stk(g.succ[branch_block].begin(), g.succ[branch_block].end());
    while (!stk.empty()) {
        int b = stk.back(); stk.pop_back();
        if (b == recon || seen[b]) continue;
        seen[b] = 1; frontier.push_back(b);
        for (int s : g.succ[b]) if (!seen[s]) stk.push_back(s);
    }
    std::sort(frontier.begin(), frontier.end());
    return frontier;
}

} // namespace omni::uir
