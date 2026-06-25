// bench_trace_codec — measures the divergence-aware codec across SIMT coherence
// regimes and writes the results to data/trace_codec.{json,csv} for the paper.
#include "omni/bench.hpp"
#include "omni/trace/codec.hpp"
#include "omni/trace/synth_columns.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace omni::trace;
using omni::bench::Suite;

static void run(Suite& s, const std::string& name, const Column& c) {
    auto bytes = encode(c);
    Column d = decode(bytes.data(), bytes.size());

    // Lossless check (active lanes).
    bool lossless = (d.size() == c.size());
    for (size_t i = 0; lossless && i < c.size(); ++i)
        if (c.lane_active(i) && d.bits[i] != c.bits[i]) lossless = false;

    size_t active = c.active_count();
    double achieved_bpv = active ? (double)bytes.size() * 8.0 / (double)active : 0.0;
    double ratio = achieved_bpv > 0 ? 32.0 / achieved_bpv : 0.0;
    double h_res = warp_conditional_entropy_bits(c);
    double h0 = order0_entropy_bits(c.bits);

    // Throughput.
    auto enc = omni::bench::time_ns([&]{ volatile auto b = encode(c); (void)b; }, 7, 2);
    auto dec = omni::bench::time_ns([&]{ volatile auto x = decode(bytes.data(), bytes.size()); (void)x; }, 7, 2);
    double enc_ns_per = enc.median / (double)c.size();
    double dec_ns_per = dec.median / (double)c.size();

    std::printf("  %-18s lossless=%d  %.3f bpv  ratio=%.2fx  Hres=%.2f H0=%.2f  enc=%.2f dec=%.2f ns/val\n",
                name.c_str(), (int)lossless, achieved_bpv, ratio, h_res, h0, enc_ns_per, dec_ns_per);

    s.add(name, "bits_per_value", achieved_bpv, "bits");
    s.add(name, "compression_ratio_vs_raw32", ratio, "x");
    s.add(name, "residual_entropy_bound", h_res, "bits");
    s.add(name, "order0_entropy", h0, "bits");
    s.add(name, "encode_throughput", enc_ns_per, "ns/value");
    s.add(name, "decode_throughput", dec_ns_per, "ns/value");
    s.add(name, "lossless", lossless ? 1.0 : 0.0, "bool");
}

int main() {
    const size_t N = 1u << 20; // ~1M invocations per column
    std::printf("OmniTrace codec benchmark — %zu invocations/column, warp=32\n", N);
    Suite s("trace_codec");

    run(s, "constant",        synth::constant(N, 0xABCD));
    run(s, "quad_coherent",   synth::quad_coherent(N));
    run(s, "warp_gradient",   synth::warp_gradient(N));
    run(s, "float_smooth",    synth::float_smooth(N));
    run(s, "random32",        synth::random32(N, 12345));
    // Divergence regimes (control flow): half / 10% lanes active.
    run(s, "gradient_div50",  synth::with_divergence(synth::warp_gradient(N), 0.5, 2));
    run(s, "gradient_div10",  synth::with_divergence(synth::warp_gradient(N), 0.1, 2));

    s.write();
    return 0;
}
