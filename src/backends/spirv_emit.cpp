#include "omni/backends/spirv_emit.hpp"
#include <unordered_map>
#include <cstring>

namespace omni::backends {
using namespace omni::uir;

namespace spv {
enum : uint16_t {
    OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
    OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypePointer = 32, OpTypeFunction = 33, OpTypeStruct = 30,
    OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
    OpFunction = 54, OpFunctionParameter = 55, OpFunctionEnd = 56,
    OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
    OpCompositeConstruct = 80, OpCompositeExtract = 81, OpUndef = 1,
    OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpReturn = 253,
    OpReturnValue = 254, OpKill = 252, OpUnreachable = 255,
    OpFNegate = 127, OpIAdd = 128, OpFAdd = 129, OpISub = 130, OpFSub = 131,
    OpIMul = 132, OpFMul = 133, OpFDiv = 136, OpVectorTimesScalar = 142, OpDot = 148,
};
constexpr uint32_t MAGIC = 0x07230203u, VERSION_1_0 = 0x00010000u;
} // namespace spv

// UIR op -> SPIR-V opcode for the "simple" [resultType,resultId,operands...] forms.
static uint16_t spv_simple(Op op) {
    switch (op) {
        case Op::FNegate: return spv::OpFNegate; case Op::IAdd: return spv::OpIAdd;
        case Op::FAdd: return spv::OpFAdd; case Op::ISub: return spv::OpISub;
        case Op::FSub: return spv::OpFSub; case Op::IMul: return spv::OpIMul;
        case Op::FMul: return spv::OpFMul; case Op::FDiv: return spv::OpFDiv;
        case Op::VectorTimesScalar: return spv::OpVectorTimesScalar; case Op::Dot: return spv::OpDot;
        default: return 0;
    }
}

namespace {
struct Emitter {
    const Module& m;
    const EmitOptions& opt;
    std::vector<uint32_t> head, types_consts, body; // module sections
    std::unordered_map<TypeId, uint32_t> tid;
    std::unordered_map<ValueId, uint32_t> vid;
    std::unordered_map<FuncId, uint32_t> fid;
    std::unordered_map<BlockId, uint32_t> bid;
    uint32_t next_id = 1;

    Emitter(const Module& m_, const EmitOptions& o) : m(m_), opt(o) {}
    uint32_t fresh() { return next_id++; }

    static void inst(std::vector<uint32_t>& out, uint16_t op, std::initializer_list<uint32_t> ops) {
        out.push_back(((uint32_t)(ops.size() + 1) << 16) | op);
        for (uint32_t o : ops) out.push_back(o);
    }
    static void inst_v(std::vector<uint32_t>& out, uint16_t op, const std::vector<uint32_t>& ops) {
        out.push_back(((uint32_t)(ops.size() + 1) << 16) | op);
        for (uint32_t o : ops) out.push_back(o);
    }
    static void push_string(std::vector<uint32_t>& ops, const std::string& s) {
        size_t words = s.size() / 4 + 1;
        for (size_t w = 0; w < words; ++w) {
            uint32_t word = 0;
            for (int b = 0; b < 4; ++b) { size_t i = w * 4 + b; if (i < s.size()) word |= (uint32_t)(uint8_t)s[i] << (8 * b); }
            ops.push_back(word);
        }
    }

    void alloc() {
        for (size_t i = 0; i < m.num_types(); ++i) tid[(TypeId)i] = fresh();
        for (size_t i = 0; i < m.num_values(); ++i) vid[(ValueId)i] = fresh();
        for (size_t i = 0; i < m.num_functions(); ++i) fid[(FuncId)i] = fresh();
        for (size_t i = 0; i < m.num_blocks(); ++i) bid[(BlockId)i] = fresh();
    }

    void emit_types() {
        for (size_t i = 0; i < m.num_types(); ++i) {
            const Type& t = m.type((TypeId)i); uint32_t id = tid[(TypeId)i];
            switch (t.kind) {
                case TypeKind::Void:  inst(types_consts, spv::OpTypeVoid, {id}); break;
                case TypeKind::Bool:  inst(types_consts, spv::OpTypeBool, {id}); break;
                case TypeKind::Int:   inst(types_consts, spv::OpTypeInt, {id, t.width, (uint32_t)(t.is_signed ? 1 : 0)}); break;
                case TypeKind::Float: inst(types_consts, spv::OpTypeFloat, {id, t.width}); break;
                case TypeKind::Vector:inst(types_consts, spv::OpTypeVector, {id, tid[t.elem], t.count}); break;
                case TypeKind::Pointer:inst(types_consts, spv::OpTypePointer, {id, t.storage_class, tid[t.elem]}); break;
                case TypeKind::Function: {
                    std::vector<uint32_t> ops = {id, tid[t.ret]};
                    for (TypeId p : t.members) ops.push_back(tid[p]);
                    inst_v(types_consts, spv::OpTypeFunction, ops);
                } break;
                case TypeKind::Struct: {
                    std::vector<uint32_t> ops = {id};
                    for (TypeId mm : t.members) ops.push_back(tid[mm]);
                    inst_v(types_consts, spv::OpTypeStruct, ops);
                } break;
                default: inst(types_consts, spv::OpTypeVoid, {id}); break; // unsupported -> placeholder
            }
        }
        // constants / globals / undef
        for (size_t i = 0; i < m.num_values(); ++i) {
            const Value& v = m.value((ValueId)i); uint32_t id = vid[(ValueId)i];
            switch (v.kind) {
                case ValueKind::ConstInt:  inst(types_consts, spv::OpConstant, {tid[v.type], id, (uint32_t)v.const_bits}); break;
                case ValueKind::ConstFloat:inst(types_consts, spv::OpConstant, {tid[v.type], id, (uint32_t)v.const_bits}); break;
                case ValueKind::ConstBool: inst(types_consts, v.const_bits ? spv::OpConstantTrue : spv::OpConstantFalse, {tid[v.type], id}); break;
                case ValueKind::ConstComposite: {
                    std::vector<uint32_t> ops = {tid[v.type], id};
                    for (ValueId e : v.elements) ops.push_back(vid[e]);
                    inst_v(types_consts, spv::OpConstantComposite, ops);
                } break;
                case ValueKind::Undef:    inst(types_consts, spv::OpUndef, {tid[v.type], id}); break;
                case ValueKind::GlobalVar:inst(types_consts, spv::OpVariable, {tid[v.type], id, v.storage_class}); break;
                default: break; // InstResult / Param emitted in function body
            }
        }
    }

    void emit_inst(const Instruction& in) {
        const ValueId* ops = m.operands(in);
        auto rid = [&](ValueId v){ return vid[v]; };
        if (uint16_t s = spv_simple(in.op)) {
            std::vector<uint32_t> o = {tid[in.type], vid[in.result]};
            for (uint32_t k = 0; k < in.operand_count; ++k) o.push_back(rid(ops[k]));
            inst_v(body, s, o); return;
        }
        switch (in.op) {
            case Op::Load:  inst(body, spv::OpLoad, {tid[in.type], vid[in.result], rid(ops[0])}); break;
            case Op::Store: inst(body, spv::OpStore, {rid(ops[0]), rid(ops[1])}); break;
            case Op::CompositeExtract: inst(body, spv::OpCompositeExtract, {tid[in.type], vid[in.result], rid(ops[0]), in.imm0}); break;
            case Op::CompositeConstruct: {
                std::vector<uint32_t> o = {tid[in.type], vid[in.result]};
                for (uint32_t k = 0; k < in.operand_count; ++k) o.push_back(rid(ops[k]));
                inst_v(body, spv::OpCompositeConstruct, o);
            } break;
            case Op::Return: inst(body, spv::OpReturn, {}); break;
            case Op::ReturnValue: inst(body, spv::OpReturnValue, {rid(ops[0])}); break;
            case Op::Kill: inst(body, spv::OpKill, {}); break;
            case Op::Unreachable: inst(body, spv::OpUnreachable, {}); break;
            case Op::Branch: inst(body, spv::OpBranch, {bid[(BlockId)in.imm0]}); break;
            case Op::BranchConditional: inst(body, spv::OpBranchConditional, {rid(ops[0]), bid[(BlockId)in.imm0], bid[(BlockId)in.imm1]}); break;
            case Op::TraceTap: break; // not lowered to SPIR-V here (handled by capture backend)
            default: break;
        }
    }

    void emit_functions() {
        for (size_t fi = 0; fi < m.num_functions(); ++fi) {
            const Function& f = m.function((FuncId)fi);
            const Type& ft = m.type(f.type);
            inst(body, spv::OpFunction, {tid[ft.ret], fid[(FuncId)fi], 0 /*FunctionControl None*/, tid[f.type]});
            for (ValueId p : f.params) inst(body, spv::OpFunctionParameter, {tid[m.value(p).type], vid[p]});
            for (BlockId b : f.blocks) {
                inst(body, spv::OpLabel, {bid[b]});
                for (InstId iid : m.block(b).insts) emit_inst(m.inst(iid));
            }
            inst(body, spv::OpFunctionEnd, {});
        }
    }

    std::vector<uint32_t> run() {
        alloc();
        // header section
        inst(head, spv::OpCapability, {1 /*Shader*/});
        inst(head, spv::OpMemoryModel, {0 /*Logical*/, 1 /*GLSL450*/});
        { // OpEntryPoint <model> %entry "name" <interface = global vars>
            std::vector<uint32_t> ops = {(uint32_t)opt.model, fid[opt.entry]};
            push_string(ops, opt.entry_name);
            for (size_t i = 0; i < m.num_values(); ++i)
                if (m.value((ValueId)i).kind == ValueKind::GlobalVar) ops.push_back(vid[(ValueId)i]);
            inst_v(head, spv::OpEntryPoint, ops);
        }
        if (opt.model == ExecModel::GLCompute)
            inst(head, spv::OpExecutionMode, {fid[opt.entry], 17 /*LocalSize*/, opt.local_size[0], opt.local_size[1], opt.local_size[2]});
        else
            inst(head, spv::OpExecutionMode, {fid[opt.entry], 7 /*OriginUpperLeft*/});

        emit_types();
        emit_functions();

        std::vector<uint32_t> out;
        out.push_back(spv::MAGIC);
        out.push_back(spv::VERSION_1_0);
        out.push_back(0);             // generator
        out.push_back(next_id);       // bound
        out.push_back(0);             // schema
        out.insert(out.end(), head.begin(), head.end());
        out.insert(out.end(), types_consts.begin(), types_consts.end());
        out.insert(out.end(), body.begin(), body.end());
        return out;
    }
};
} // namespace

std::vector<uint32_t> emit_spirv(const Module& m, const EmitOptions& opt) {
    Emitter e(m, opt);
    return e.run();
}

} // namespace omni::backends
