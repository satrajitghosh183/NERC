#include "omni/test.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/uir/ir.hpp"
#include "omni/uir/verify.hpp"
#include <string>

using namespace omni;

static std::string spv_path() {
    return std::string(OMNI_SOURCE_DIR) + "/data/shaders/simple.frag.spv";
}

// Count UIR instructions of a given opcode across the whole module.
static int count_op(const uir::Module& m, uir::Op op) {
    int n = 0;
    for (size_t i = 0; i < m.num_insts(); ++i) if (m.inst((uir::InstId)i).op == op) ++n;
    return n;
}

TEST(spirv, lift_real_fragment_shader) {
    uir::Module m;
    auto r = frontends::lift_spirv_file(spv_path(), m);
    if (!r.ok) std::printf("    lift error: %s\n", r.error.c_str());
    REQUIRE(r.ok);
    std::printf("    id_bound=%u handled=%u skipped=%u unhandled_opcodes=%zu\n",
                r.id_bound, r.handled_insts, r.skipped_insts, r.unhandled_opcodes.size());
    for (auto& o : r.unhandled_opcodes) std::printf("    UNHANDLED %s\n", o.c_str());

    // Every SPIR-V opcode in simple.frag must be handled (no unknowns).
    CHECK(r.unhandled_opcodes.empty());

    // Structure recovered: a 'main' function with one block ending in Return.
    REQUIRE(m.num_functions() >= 1);
    bool found_main = false;
    for (size_t i = 0; i < m.num_functions(); ++i)
        if (m.function((uir::FuncId)i).name == "main") found_main = true;
    CHECK(found_main);

    // Instruction mix matches the disassembly we compiled from.
    CHECK_EQ(count_op(m, uir::Op::Load), 3);
    CHECK_EQ(count_op(m, uir::Op::Store), 2);
    CHECK_EQ(count_op(m, uir::Op::CompositeExtract), 5);
    CHECK_EQ(count_op(m, uir::Op::CompositeConstruct), 2);
    CHECK_EQ(count_op(m, uir::Op::VectorTimesScalar), 1);
    CHECK_EQ(count_op(m, uir::Op::AccessChain), 1);
    CHECK_EQ(count_op(m, uir::Op::Return), 1);

    // Types recovered: float, vec2/vec3/vec4 of float.
    bool f32 = false, v2 = false, v3 = false, v4 = false;
    for (size_t i = 0; i < m.num_types(); ++i) {
        const auto& t = m.type((uir::TypeId)i);
        if (t.kind == uir::TypeKind::Float && t.width == 32) f32 = true;
        if (t.kind == uir::TypeKind::Vector) {
            if (t.count == 2) v2 = true; if (t.count == 3) v3 = true; if (t.count == 4) v4 = true;
        }
    }
    CHECK(f32); CHECK(v2); CHECK(v3); CHECK(v4);

    // The lifted module is structurally valid.
    std::vector<std::string> errs;
    bool ok = verify(m, errs);
    for (auto& e : errs) std::printf("    verify: %s\n", e.c_str());
    CHECK(ok);
}
