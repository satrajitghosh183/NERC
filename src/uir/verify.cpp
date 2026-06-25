#include "omni/uir/verify.hpp"

namespace omni::uir {

bool is_terminator(Op op) {
    switch (op) {
        case Op::Branch: case Op::BranchConditional: case Op::Switch:
        case Op::Return: case Op::ReturnValue: case Op::Kill: case Op::Unreachable:
            return true;
        default: return false;
    }
}

bool verify(const Module& m, std::vector<std::string>& errors) {
    const size_t before = errors.size();
    auto err = [&](std::string s) { errors.push_back(std::move(s)); };

    for (size_t fi = 0; fi < m.num_functions(); ++fi) {
        const Function& f = m.function((FuncId)fi);
        if (f.blocks.empty()) { err("function '" + f.name + "' has no blocks"); continue; }

        for (BlockId bid : f.blocks) {
            const BasicBlock& b = m.block(bid);
            if (b.insts.empty()) { err("block " + std::to_string(bid) + " is empty"); continue; }

            // Exactly one terminator, and it must be last.
            for (size_t i = 0; i < b.insts.size(); ++i) {
                const Instruction& in = m.inst(b.insts[i]);
                bool last = (i + 1 == b.insts.size());
                if (is_terminator(in.op) && !last)
                    err("block " + std::to_string(bid) + ": terminator '" +
                        op_name(in.op) + "' not at end");
                if (!is_terminator(in.op) && last)
                    err("block " + std::to_string(bid) + ": last inst '" +
                        op_name(in.op) + "' is not a terminator");
            }
        }
    }

    // Per-instruction operand & result-type sanity.
    for (size_t i = 0; i < m.num_insts(); ++i) {
        const Instruction& in = m.inst((InstId)i);
        const ValueId* ops = m.operands(in);
        for (uint32_t k = 0; k < in.operand_count; ++k) {
            ValueId v = ops[k];
            if (v == INVALID || v >= m.num_values())
                err(std::string(op_name(in.op)) + ": operand " + std::to_string(k) +
                    " references invalid value " + std::to_string(v));
        }
        if (in.result != INVALID) {
            if (in.result >= m.num_values()) { err("bad result id"); continue; }
            const Value& rv = m.value(in.result);
            if (rv.kind != ValueKind::InstResult || rv.inst != (InstId)i)
                err(std::string(op_name(in.op)) + ": result value not linked back to instruction");
            if (rv.type != in.type)
                err(std::string(op_name(in.op)) + ": result value type mismatches instruction type");
        }
    }

    return errors.size() == before;
}

} // namespace omni::uir
