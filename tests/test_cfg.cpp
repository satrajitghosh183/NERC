#include "omni/test.hpp"
#include "omni/uir/ir.hpp"
#include "omni/uir/cfg.hpp"
#include <vector>
#include <algorithm>

using namespace omni::uir;

// Diamond:  entry -> {then, else} -> merge -> return
TEST(cfg, diamond_dominators_and_reconvergence) {
    Module m;
    TypeId fn = m.function_type(m.void_type(), {});
    FuncId f = m.new_function(fn, "main");
    BlockId entry = m.new_block(f), bt = m.new_block(f), be = m.new_block(f), mg = m.new_block(f);
    ValueId cond = m.const_bool(true);
    m.emit(entry, Op::BranchConditional, INVALID, {cond}, bt, be);
    m.emit(bt, Op::Branch, INVALID, {}, mg);
    m.emit(be, Op::Branch, INVALID, {}, mg);
    m.emit(mg, Op::Return, INVALID, {});

    CFG g = build_cfg(m, f);
    int ie = g.index_of(entry), it = g.index_of(bt), iel = g.index_of(be), im = g.index_of(mg);

    // Dominance: entry dominates all; then/else dominate only themselves.
    CHECK(dominates(g, ie, im));
    CHECK(dominates(g, ie, it));
    CHECK(!dominates(g, it, im));     // merge is reachable without going through `then`
    CHECK_EQ(g.idom[it], ie);
    CHECK_EQ(g.idom[im], ie);         // merge's idom is entry (two incoming paths)

    // Post-dominance: merge post-dominates entry/then/else; entry does not post-dom merge.
    CHECK(post_dominates(g, im, ie));
    CHECK(post_dominates(g, im, it));
    CHECK(!post_dominates(g, ie, im));

    // Reconvergence point of the divergent branch is the merge block.
    CHECK_EQ(reconvergence(g, ie), im);

    // Thread frontier of the branch is exactly {then, else}.
    auto tf = thread_frontier(g, ie);
    std::vector<int> expect = {it, iel}; std::sort(expect.begin(), expect.end());
    CHECK(tf == expect);
}

// Loop:  entry -> header; header -> {body, exit}; body -> header (back edge); exit -> return
TEST(cfg, loop_back_edge) {
    Module m;
    TypeId fn = m.function_type(m.void_type(), {});
    FuncId f = m.new_function(fn, "loop");
    BlockId entry = m.new_block(f), header = m.new_block(f), body = m.new_block(f), exit = m.new_block(f);
    ValueId cond = m.const_bool(true);
    m.emit(entry, Op::Branch, INVALID, {}, header);
    m.emit(header, Op::BranchConditional, INVALID, {cond}, body, exit);
    m.emit(body, Op::Branch, INVALID, {}, header);   // back edge
    m.emit(exit, Op::Return, INVALID, {});

    CFG g = build_cfg(m, f);
    int ih = g.index_of(header), ib = g.index_of(body), ix = g.index_of(exit), ient = g.index_of(entry);

    CHECK(dominates(g, ient, ib));        // entry dominates body
    CHECK(dominates(g, ih, ib));          // header dominates body
    CHECK_EQ(g.idom[ib], ih);             // body's idom is header
    CHECK(post_dominates(g, ix, ih));     // exit post-dominates header
    CHECK_EQ(reconvergence(g, ih), ix);   // loop exit is the reconvergence point
}

// Single-block shader: no successors, no reconvergence.
TEST(cfg, single_block) {
    Module m;
    TypeId fn = m.function_type(m.void_type(), {});
    FuncId f = m.new_function(fn, "k");
    BlockId b = m.new_block(f);
    m.emit(b, Op::Return, INVALID, {});
    CFG g = build_cfg(m, f);
    CHECK_EQ(g.blocks.size(), 1u);
    CHECK_EQ(g.idom[0], 0);
    CHECK_EQ(reconvergence(g, 0), -1);    // it is itself the exit
}
