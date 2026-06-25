#include "omni/test.hpp"
#include "omni/backends/spirv_emit.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/uir/ir.hpp"
#include "omni/uir/verify.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace omni;

static int count_op(const uir::Module& m, uir::Op op) {
    int n = 0; for (size_t i = 0; i < m.num_insts(); ++i) if (m.inst((uir::InstId)i).op == op) ++n; return n;
}
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(), &s) == 0; }
static std::string spirv_val() {
    if (const char* e = std::getenv("OMNI_SPIRV_VAL"); e && exists(e)) return e;
    std::string p = "/Users/satrajitghosh/Software/VulkanSDK/1.4.328.1/macOS/bin/spirv-val";
    return exists(p) ? p : "";
}

// Build a UIR compute module:  void main() { (2*3)+4; }  -> emit -> validate -> re-lift.
TEST(spirv_backend, emit_validate_relift_roundtrip) {
    uir::Module m;
    uir::TypeId tvoid = m.void_type(), f32 = m.float_type(32);
    uir::FuncId f = m.new_function(m.function_type(tvoid, {}), "main");
    uir::BlockId b = m.new_block(f);
    uir::ValueId c2 = m.const_float(f32, 2.0), c3 = m.const_float(f32, 3.0), c4 = m.const_float(f32, 4.0);
    uir::InstId mul = m.emit(b, uir::Op::FMul, f32, {c2, c3});
    m.emit(b, uir::Op::FAdd, f32, {m.result_of(mul), c4});
    m.emit(b, uir::Op::Return, uir::INVALID, {});

    backends::EmitOptions opt; opt.entry = f; opt.entry_name = "main";
    opt.model = backends::ExecModel::GLCompute; opt.local_size[0] = 1;
    auto words = backends::emit_spirv(m, opt);
    REQUIRE(words.size() > 5);
    CHECK_EQ(words[0], 0x07230203u);   // magic

    // Write out and validate with the real spirv-val.
    const char* tmpdir = std::getenv("TMPDIR"); std::string dir = tmpdir ? tmpdir : "/tmp";
    std::string path = dir + "/omni_emit_" + std::to_string(getpid()) + ".spv";
    FILE* fp = std::fopen(path.c_str(), "wb");
    REQUIRE(fp != nullptr);
    std::fwrite(words.data(), 4, words.size(), fp); std::fclose(fp);

    std::string val = spirv_val();
    if (!val.empty()) {
        std::string cmd = "'" + val + "' '" + path + "' 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        std::string out; char buf[1024]; size_t g;
        while (p && (g = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, g);
        int rc = p ? pclose(p) : -1;
        if (rc != 0) std::printf("    spirv-val FAILED:\n%s\n", out.c_str());
        CHECK_EQ(rc, 0);               // our emitted SPIR-V is valid per the official validator
        std::printf("    spirv-val: clean\n");
    } else {
        std::printf("    spirv-val unavailable; skipping external validation\n");
    }

    // Re-lift the emitted module and confirm the arithmetic survived the round-trip.
    uir::Module m2;
    auto lr = frontends::lift_spirv_file(path, m2);
    REQUIRE(lr.ok);
    CHECK_EQ(count_op(m2, uir::Op::FMul), 1);
    CHECK_EQ(count_op(m2, uir::Op::FAdd), 1);
    CHECK_EQ(count_op(m2, uir::Op::Return), 1);
    std::vector<std::string> errs;
    CHECK(verify(m2, errs));
    std::remove(path.c_str());
}
