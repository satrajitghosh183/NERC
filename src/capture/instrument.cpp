#include "omni/capture/instrument.hpp"
#include "omni/uir/verify.hpp"
#include <unordered_set>

namespace omni::capture {
using namespace omni::uir;

bool is_capturable(const Module& m, TypeId t) {
    if (t == INVALID) return false;
    const Type& ty = m.type(t);
    if (ty.kind == TypeKind::Int || ty.kind == TypeKind::Float || ty.kind == TypeKind::Bool)
        return true;
    if (ty.kind == TypeKind::Vector) {
        const Type& e = m.type(ty.elem);
        return e.kind == TypeKind::Int || e.kind == TypeKind::Float;
    }
    return false; // pointers, structs, images, void
}

std::vector<TapSite> insert_taps(Module& m, FuncId f,
                                 const std::function<bool(const Instruction&)>& accept) {
    std::vector<TapSite> sites;
    uint32_t next_id = 0;

    // Idempotency: values that already have a TraceTap must not be tapped again.
    std::unordered_set<ValueId> already_tapped;
    for (BlockId bid : m.function(f).blocks)
        for (InstId iid : m.block(bid).insts) {
            const Instruction& in = m.inst(iid);
            if (in.op == Op::TraceTap && in.operand_count >= 1)
                already_tapped.insert(m.operands(in)[0]);
        }

    for (BlockId bid : m.function(f).blocks) {
        std::vector<InstId>& insts = m.block(bid).insts;
        std::vector<InstId> rebuilt;
        rebuilt.reserve(insts.size() * 2);

        for (InstId iid : insts) {
            rebuilt.push_back(iid);
            const Instruction& in = m.inst(iid);
            if (is_terminator(in.op)) continue;          // never tap a terminator
            if (in.op == Op::TraceTap) continue;         // idempotent: don't tap taps
            if (in.result == INVALID) continue;          // only value-producing insts
            if (!is_capturable(m, in.type)) continue;    // skip pointers/structs/void
            if (already_tapped.count(in.result)) continue; // idempotent
            if (accept && !accept(in)) continue;         // region-of-interest filter

            uint32_t site = next_id++;
            // NB: capture `in.result` before further mutation; `in` ref stays valid since
            // create_inst only appends to pools (no relocation of existing entries we read).
            ValueId captured = in.result;
            Op def_op = in.op;
            uint32_t line = in.line;
            InstId tap = m.create_inst(bid, Op::TraceTap, INVALID, {captured}, site, 0, line);
            rebuilt.push_back(tap);
            sites.push_back({site, captured, def_op, line});
        }
        m.block(bid).insts = std::move(rebuilt);
    }
    return sites;
}

} // namespace omni::capture
