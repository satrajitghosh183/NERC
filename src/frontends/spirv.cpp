#include "omni/frontends/spirv.hpp"
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <optional>

namespace omni::frontends {
using namespace omni::uir;

namespace spv {
// SPIR-V opcode numbers (subset we handle); see the SPIR-V specification.
enum : uint16_t {
    OpSource = 3, OpName = 5, OpMemberName = 6, OpExtInstImport = 11, OpExtInst = 12,
    OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
    OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeMatrix = 24, OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27,
    OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
    OpConstantNull = 46,
    OpFunction = 54, OpFunctionParameter = 55, OpFunctionEnd = 56, OpFunctionCall = 57,
    OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
    OpDecorate = 71, OpMemberDecorate = 72,
    OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81, OpCompositeInsert = 82,
    OpConvertFToU = 109, OpConvertFToS = 110, OpConvertSToF = 111, OpConvertUToF = 112, OpBitcast = 124,
    OpSNegate = 126, OpFNegate = 127,
    OpIAdd = 128, OpFAdd = 129, OpISub = 130, OpFSub = 131, OpIMul = 132, OpFMul = 133,
    OpUDiv = 134, OpSDiv = 135, OpFDiv = 136, OpUMod = 137, OpSMod = 139, OpFRem = 140,
    OpVectorTimesScalar = 142, OpDot = 148,
    OpLogicalEqual = 164, OpLogicalOr = 166, OpLogicalAnd = 167, OpLogicalNot = 168,
    OpIEqual = 170, OpINotEqual = 171, OpUGreaterThan = 172, OpSGreaterThan = 173,
    OpUGreaterThanEqual = 174, OpSGreaterThanEqual = 175, OpULessThan = 176, OpSLessThan = 177,
    OpULessThanEqual = 178, OpSLessThanEqual = 179,
    OpFOrdEqual = 180, OpFOrdNotEqual = 182, OpFOrdLessThan = 184, OpFOrdGreaterThan = 186,
    OpFOrdLessThanEqual = 188, OpFOrdGreaterThanEqual = 190,
    OpShiftRightLogical = 194, OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196,
    OpBitwiseOr = 197, OpBitwiseXor = 198, OpBitwiseAnd = 199, OpNot = 200,
    OpDPdx = 207, OpDPdy = 208, OpFwidth = 209,
    OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249,
    OpBranchConditional = 250, OpSwitch = 251, OpKill = 252, OpReturn = 253, OpReturnValue = 254,
    OpUnreachable = 255,
    OpSelect = 169,
};
constexpr uint32_t MAGIC = 0x07230203u;
} // namespace spv

// SPIR-V storage classes (subset).
namespace sc { enum { UniformConstant=0, Input=1, Uniform=2, Output=3, Workgroup=4, Function=7, PushConstant=9, StorageBuffer=12, Private=6 }; }

// Map "simple" SPIR-V opcodes (layout: resultType, resultId, value-operands...) to UIR.
static std::optional<Op> simple_op(uint16_t o) {
    switch (o) {
        case spv::OpSNegate: return Op::SNegate; case spv::OpFNegate: return Op::FNegate;
        case spv::OpIAdd: return Op::IAdd; case spv::OpFAdd: return Op::FAdd;
        case spv::OpISub: return Op::ISub; case spv::OpFSub: return Op::FSub;
        case spv::OpIMul: return Op::IMul; case spv::OpFMul: return Op::FMul;
        case spv::OpUDiv: return Op::UDiv; case spv::OpSDiv: return Op::SDiv; case spv::OpFDiv: return Op::FDiv;
        case spv::OpUMod: return Op::UMod; case spv::OpSMod: return Op::SMod; case spv::OpFRem: return Op::FRem;
        case spv::OpDot: return Op::Dot;
        case spv::OpConvertFToU: return Op::ConvertFToU; case spv::OpConvertFToS: return Op::ConvertFToS;
        case spv::OpConvertSToF: return Op::ConvertSToF; case spv::OpConvertUToF: return Op::ConvertUToF;
        case spv::OpBitcast: return Op::Bitcast;
        case spv::OpLogicalEqual: return Op::LogicalEqual; case spv::OpLogicalOr: return Op::LogicalOr;
        case spv::OpLogicalAnd: return Op::LogicalAnd; case spv::OpLogicalNot: return Op::LogicalNot;
        case spv::OpIEqual: return Op::IEqual; case spv::OpINotEqual: return Op::INotEqual;
        case spv::OpUGreaterThan: return Op::UGreaterThan; case spv::OpSGreaterThan: return Op::SGreaterThan;
        case spv::OpUGreaterThanEqual: return Op::UGreaterThanEqual; case spv::OpSGreaterThanEqual: return Op::SGreaterThanEqual;
        case spv::OpULessThan: return Op::ULessThan; case spv::OpSLessThan: return Op::SLessThan;
        case spv::OpULessThanEqual: return Op::ULessThanEqual; case spv::OpSLessThanEqual: return Op::SLessThanEqual;
        case spv::OpFOrdEqual: return Op::FOrdEqual; case spv::OpFOrdNotEqual: return Op::FOrdNotEqual;
        case spv::OpFOrdLessThan: return Op::FOrdLessThan; case spv::OpFOrdGreaterThan: return Op::FOrdGreaterThan;
        case spv::OpFOrdLessThanEqual: return Op::FOrdLessThanEqual; case spv::OpFOrdGreaterThanEqual: return Op::FOrdGreaterThanEqual;
        case spv::OpShiftRightLogical: return Op::ShiftRightLogical; case spv::OpShiftRightArithmetic: return Op::ShiftRightArithmetic;
        case spv::OpShiftLeftLogical: return Op::ShiftLeftLogical;
        case spv::OpBitwiseOr: return Op::BitwiseOr; case spv::OpBitwiseXor: return Op::BitwiseXor;
        case spv::OpBitwiseAnd: return Op::BitwiseAnd; case spv::OpNot: return Op::Not;
        case spv::OpVectorTimesScalar: return Op::VectorTimesScalar;
        case spv::OpDPdx: return Op::DPdx; case spv::OpDPdy: return Op::DPdy; case spv::OpFwidth: return Op::Fwidth;
        default: return std::nullopt;
    }
}

namespace {
struct Lifter {
    const uint32_t* w;
    size_t n;
    Module& m;
    SpirvLiftResult res;

    std::unordered_map<uint32_t, TypeId> type_map;   // spv id -> UIR type
    std::unordered_map<uint32_t, ValueId> val_map;   // spv id -> UIR value
    std::unordered_map<uint32_t, BlockId> label_map; // spv id -> UIR block
    std::unordered_map<uint32_t, std::string> names; // spv id -> debug name
    std::unordered_map<uint32_t, uint32_t> result_type_of; // spv id -> spv result-type id (for forward fwd refs)

    FuncId cur_fn = INVALID;
    BlockId cur_blk = INVALID;

    Lifter(const uint32_t* w_, size_t n_, Module& m_) : w(w_), n(n_), m(m_) {}

    TypeId T(uint32_t spv_id) {
        auto it = type_map.find(spv_id);
        return it == type_map.end() ? INVALID : it->second;
    }
    // Resolve a value operand; if undefined (forward ref / unhandled def), synthesize an Undef.
    ValueId V(uint32_t spv_id) {
        auto it = val_map.find(spv_id);
        if (it != val_map.end()) return it->second;
        TypeId t = INVALID;
        auto rt = result_type_of.find(spv_id);
        if (rt != result_type_of.end()) t = T(rt->second);
        ValueId u = m.undef(t == INVALID ? m.void_type() : t);
        val_map[spv_id] = u;
        return u;
    }

    void prepass() {
        size_t i = 5;
        while (i < n) {
            uint32_t word0 = w[i];
            uint16_t len = word0 >> 16, op = word0 & 0xffff;
            if (len == 0) break;
            const uint32_t* a = &w[i + 1];
            uint16_t ac = len - 1;
            if (op == spv::OpName && ac >= 2) {
                names[a[0]] = std::string((const char*)&a[1]);
            }
            // Pre-record result types for ops with [resultType, resultId, ...] layout so
            // forward references (Phi/CFG) can synthesize correctly-typed placeholders.
            if (simple_op(op) || op == spv::OpLoad || op == spv::OpAccessChain ||
                op == spv::OpCompositeConstruct || op == spv::OpCompositeExtract ||
                op == spv::OpVectorShuffle || op == spv::OpPhi || op == spv::OpSelect ||
                op == spv::OpExtInst || op == spv::OpFunctionCall) {
                if (ac >= 2) result_type_of[a[1]] = a[0];
            }
            i += len;
        }
    }

    const char* literal_string(const uint32_t* a, uint16_t ac, uint16_t start) {
        (void)ac; return (const char*)&a[start];
    }

    void make_type(uint16_t op, uint32_t rid, const uint32_t* a, uint16_t ac) {
        switch (op) {
            case spv::OpTypeVoid:  type_map[rid] = m.void_type(); break;
            case spv::OpTypeBool:  type_map[rid] = m.bool_type(); break;
            case spv::OpTypeInt:   type_map[rid] = m.int_type((uint16_t)a[1], a[2] != 0); break;
            case spv::OpTypeFloat: type_map[rid] = m.float_type((uint16_t)a[1]); break;
            case spv::OpTypeVector: type_map[rid] = m.vector_type(T(a[1]), a[2]); break;
            case spv::OpTypeMatrix: type_map[rid] = m.add_type([&]{ Type t; t.kind=TypeKind::Matrix; t.elem=T(a[1]); t.count=a[2]; return t; }()); break;
            case spv::OpTypeArray: type_map[rid] = m.array_type(T(a[1]), /*len id*/ a[2]); break;
            case spv::OpTypeRuntimeArray: type_map[rid] = m.add_type([&]{ Type t; t.kind=TypeKind::RuntimeArray; t.elem=T(a[1]); return t; }()); break;
            case spv::OpTypePointer: type_map[rid] = m.pointer_type(T(a[2]), a[1]); break;
            case spv::OpTypeFunction: {
                std::vector<TypeId> params;
                for (uint16_t k = 2; k < ac; ++k) params.push_back(T(a[k]));
                type_map[rid] = m.function_type(T(a[1]), std::move(params));
            } break;
            case spv::OpTypeStruct: {
                std::vector<TypeId> mem;
                for (uint16_t k = 1; k < ac; ++k) mem.push_back(T(a[k]));
                type_map[rid] = m.struct_type(std::move(mem));
            } break;
            default: // image/sampler etc: model as opaque struct so ids resolve
                type_map[rid] = m.struct_type({});
                break;
        }
    }

    void dispatch(uint16_t op, const uint32_t* a, uint16_t ac) {
        // ---- module-level type/const/var ------------------------------------
        if (op >= spv::OpTypeVoid && op <= spv::OpTypeFunction && op != 31) {
            make_type(op, a[0], a, ac); ++res.handled_insts; return;
        }
        switch (op) {
        case spv::OpExtInstImport: case spv::OpCapability: case spv::OpMemoryModel:
        case spv::OpEntryPoint: case spv::OpExecutionMode: case spv::OpSource:
        case spv::OpName: case spv::OpMemberName: case spv::OpDecorate: case spv::OpMemberDecorate:
            ++res.skipped_insts; return; // metadata: intentionally ignored

        case spv::OpConstant: {
            TypeId t = T(a[0]); const Type& ty = m.type(t);
            ValueId v;
            if (ty.kind == TypeKind::Float) {
                if (ty.width == 64 && ac >= 4) { uint64_t bits = (uint64_t)a[2] | ((uint64_t)a[3] << 32); double d; std::memcpy(&d,&bits,8); v = m.const_float(t, d); }
                else { float f; uint32_t b = a[2]; std::memcpy(&f,&b,4); v = m.const_float(t, (double)f); }
            } else {
                uint64_t bits = a[2];
                if (ty.width == 64 && ac >= 4) bits = (uint64_t)a[2] | ((uint64_t)a[3] << 32);
                v = m.const_int(t, bits);
            }
            val_map[a[1]] = v; ++res.handled_insts; return;
        }
        case spv::OpConstantTrue:  val_map[a[1]] = m.const_bool(true);  ++res.handled_insts; return;
        case spv::OpConstantFalse: val_map[a[1]] = m.const_bool(false); ++res.handled_insts; return;
        case spv::OpConstantComposite: {
            std::vector<ValueId> elems; for (uint16_t k=2;k<ac;++k) elems.push_back(V(a[k]));
            val_map[a[1]] = m.const_composite(T(a[0]), std::move(elems)); ++res.handled_insts; return;
        }
        case spv::OpConstantNull: {
            Value dummy; // represent null as a typed Undef-like const
            val_map[a[1]] = m.undef(T(a[0])); ++res.handled_insts; return;
        }

        case spv::OpFunction: {
            // a = [resultType, resultId, functionControl, functionTypeId]
            cur_fn = m.new_function(T(a[3]), names.count(a[1]) ? names[a[1]] : "fn");
            ++res.handled_insts; return;
        }
        case spv::OpFunctionParameter: {
            ValueId p = m.param(T(a[0]), names.count(a[1]) ? names[a[1]] : "");
            val_map[a[1]] = p; if (cur_fn != INVALID) m.function(cur_fn).params.push_back(p);
            ++res.handled_insts; return;
        }
        case spv::OpFunctionEnd: cur_fn = INVALID; cur_blk = INVALID; ++res.handled_insts; return;

        case spv::OpLabel: {
            BlockId b = m.new_block(cur_fn);
            label_map[a[0]] = b; cur_blk = b; ++res.handled_insts; return;
        }
        case spv::OpVariable: {
            // a = [resultType(ptr), resultId, storageClass, (initializer?)]
            uint32_t storage = a[2];
            if (storage == sc::Function && cur_blk != INVALID) {
                InstId in = m.emit(cur_blk, Op::Variable, T(a[0]), {}, storage);
                val_map[a[1]] = m.result_of(in);
            } else {
                val_map[a[1]] = m.global_var(T(a[0]), storage, names.count(a[1]) ? names[a[1]] : "");
            }
            ++res.handled_insts; return;
        }
        case spv::OpLoad: {
            InstId in = m.emit(cur_blk, Op::Load, T(a[0]), {V(a[2])});
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpStore: {
            m.emit(cur_blk, Op::Store, INVALID, {V(a[0]), V(a[1])}); ++res.handled_insts; return;
        }
        case spv::OpAccessChain: {
            std::vector<ValueId> ops; ops.push_back(V(a[2]));
            for (uint16_t k=3;k<ac;++k) ops.push_back(V(a[k]));
            InstId in = m.emit(cur_blk, Op::AccessChain, T(a[0]), ops);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpCompositeConstruct: {
            std::vector<ValueId> ops; for (uint16_t k=2;k<ac;++k) ops.push_back(V(a[k]));
            InstId in = m.emit(cur_blk, Op::CompositeConstruct, T(a[0]), ops);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpCompositeExtract: {
            // a = [resultType, resultId, composite, literalIndex...]
            uint32_t i0 = ac > 3 ? a[3] : 0, i1 = ac > 4 ? a[4] : 0;
            InstId in = m.emit(cur_blk, Op::CompositeExtract, T(a[0]), {V(a[2])}, i0, i1);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpVectorShuffle: {
            // a = [resultType, resultId, vec1, vec2, comp...]
            std::vector<ValueId> ops = {V(a[2]), V(a[3])};
            uint32_t c0 = ac > 4 ? a[4] : 0, c1 = ac > 5 ? a[5] : 0;
            InstId in = m.emit(cur_blk, Op::VectorShuffle, T(a[0]), ops, c0, c1);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpSelect: {
            InstId in = m.emit(cur_blk, Op::Select, T(a[0]), {V(a[2]), V(a[3]), V(a[4])});
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpPhi: {
            std::vector<ValueId> ops; for (uint16_t k=2;k<ac;++k) ops.push_back(V(a[k]));
            InstId in = m.emit(cur_blk, Op::Phi, T(a[0]), ops);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }
        case spv::OpReturn: m.emit(cur_blk, Op::Return, INVALID, {}); ++res.handled_insts; return;
        case spv::OpReturnValue: m.emit(cur_blk, Op::ReturnValue, INVALID, {V(a[0])}); ++res.handled_insts; return;
        case spv::OpKill: m.emit(cur_blk, Op::Kill, INVALID, {}); ++res.handled_insts; return;
        case spv::OpUnreachable: m.emit(cur_blk, Op::Unreachable, INVALID, {}); ++res.handled_insts; return;
        case spv::OpBranch: m.emit(cur_blk, Op::Branch, INVALID, {}, a[0]); ++res.handled_insts; return; // target spv-label in imm0
        case spv::OpBranchConditional:
            m.emit(cur_blk, Op::BranchConditional, INVALID, {V(a[0])}, a[1], a[2]); ++res.handled_insts; return;
        case spv::OpSelectionMerge: m.emit(cur_blk, Op::SelectionMerge, INVALID, {}, a[0]); ++res.handled_insts; return;
        case spv::OpLoopMerge: m.emit(cur_blk, Op::LoopMerge, INVALID, {}, a[0], a[1]); ++res.handled_insts; return;

        default: break;
        }

        // ---- generic simple ops ---------------------------------------------
        if (auto uop = simple_op(op)) {
            std::vector<ValueId> ops; for (uint16_t k=2;k<ac;++k) ops.push_back(V(a[k]));
            InstId in = m.emit(cur_blk, *uop, T(a[0]), ops);
            val_map[a[1]] = m.result_of(in); ++res.handled_insts; return;
        }

        // ---- unhandled: record, synthesize result if it has one -------------
        char buf[32]; std::snprintf(buf, sizeof buf, "op#%u", op);
        res.unhandled_opcodes.push_back(buf);
        ++res.skipped_insts;
    }

    SpirvLiftResult run() {
        if (n < 5 || w[0] != spv::MAGIC) { res.error = "bad SPIR-V magic"; return res; }
        res.id_bound = w[3];
        prepass();
        size_t i = 5;
        while (i < n) {
            uint32_t word0 = w[i];
            uint16_t len = word0 >> 16, op = word0 & 0xffff;
            if (len == 0 || i + len > n) { res.error = "truncated instruction"; return res; }
            dispatch(op, &w[i + 1], len - 1);
            i += len;
        }
        // Resolve branch terminator targets from SPIR-V label ids to UIR BlockIds so the
        // CFG builder can read them directly.
        for (size_t bi = 0; bi < m.num_blocks(); ++bi) {
            BasicBlock& blk = m.block((BlockId)bi);
            if (blk.insts.empty()) continue;
            Instruction& term = m.inst(blk.insts.back());
            auto fix = [&](uint32_t& imm) { auto it = label_map.find(imm); if (it != label_map.end()) imm = it->second; };
            if (term.op == Op::Branch) fix(term.imm0);
            else if (term.op == Op::BranchConditional) { fix(term.imm0); fix(term.imm1); }
        }
        res.ok = res.error.empty();
        return res;
    }
};
} // namespace

SpirvLiftResult lift_spirv(const uint32_t* words, size_t word_count, Module& out) {
    Lifter L(words, word_count, out);
    return L.run();
}

SpirvLiftResult lift_spirv_file(const std::string& path, Module& out) {
    SpirvLiftResult r;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { r.error = "cannot open " + path; return r; }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) { std::fclose(f); r.error = "bad file size"; return r; }
    std::vector<uint32_t> words(sz / 4);
    size_t got = std::fread(words.data(), 4, words.size(), f);
    std::fclose(f);
    if (got != words.size()) { r.error = "short read"; return r; }
    return lift_spirv(words.data(), words.size(), out);
}

} // namespace omni::frontends
