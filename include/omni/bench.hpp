// omni/bench.hpp — from-scratch micro-benchmark harness with JSON/CSV output.
// Records land under data/ so paper figures are reproducible.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <filesystem>

namespace omni::bench {

struct Record {
    std::string name;     // benchmark / configuration name
    std::string metric;   // e.g. "ns_per_op", "bits_per_value", "compression_ratio"
    double value = 0.0;
    std::string unit;     // e.g. "ns", "bits", "x"
};

struct Stats { double min, median, mean, max; uint64_t n; };

// Time a callable `iters` times (after `warmup`), return ns/op stats.
template <class F>
inline Stats time_ns(F&& f, int iters = 9, int warmup = 2) {
    using clk = std::chrono::steady_clock;
    for (int i = 0; i < warmup; ++i) f();
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0; for (double s : samples) sum += s;
    return Stats{ samples.front(), samples[samples.size()/2], sum/samples.size(),
                  samples.back(), (uint64_t)samples.size() };
}

class Suite {
public:
    explicit Suite(std::string name) : name_(std::move(name)) {}

    void add(const std::string& n, const std::string& metric, double v, const std::string& unit) {
        records_.push_back({n, metric, v, unit});
        std::printf("  %-44s %14.4f %s\n", (n + " [" + metric + "]").c_str(), v, unit.c_str());
    }

    // Resolve data dir robustly: env OMNI_DATA_DIR, else ./data, creating it.
    static std::filesystem::path data_dir() {
        const char* env = std::getenv("OMNI_DATA_DIR");
        std::filesystem::path d = env ? env : "data";
        std::error_code ec; std::filesystem::create_directories(d, ec);
        return d;
    }

    void write() const {
        auto dir = data_dir();
        write_json(dir / (name_ + ".json"));
        write_csv(dir / (name_ + ".csv"));
        std::printf("  -> wrote %zu records to %s.{json,csv}\n",
                    records_.size(), (dir / name_).c_str());
    }

private:
    static std::string esc(const std::string& s) {
        std::string o; for (char c : s) { if (c=='"'||c=='\\') o.push_back('\\'); o.push_back(c); } return o;
    }
    void write_json(const std::filesystem::path& p) const {
        FILE* f = std::fopen(p.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "{\n  \"suite\": \"%s\",\n  \"records\": [\n", esc(name_).c_str());
        for (size_t i = 0; i < records_.size(); ++i) {
            const auto& r = records_[i];
            std::fprintf(f, "    {\"name\": \"%s\", \"metric\": \"%s\", \"value\": %.10g, \"unit\": \"%s\"}%s\n",
                         esc(r.name).c_str(), esc(r.metric).c_str(), r.value, esc(r.unit).c_str(),
                         i + 1 < records_.size() ? "," : "");
        }
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
    }
    void write_csv(const std::filesystem::path& p) const {
        FILE* f = std::fopen(p.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "suite,name,metric,value,unit\n");
        for (const auto& r : records_)
            std::fprintf(f, "%s,%s,%s,%.10g,%s\n", name_.c_str(), r.name.c_str(),
                         r.metric.c_str(), r.value, r.unit.c_str());
        std::fclose(f);
    }

    std::string name_;
    std::vector<Record> records_;
};

} // namespace omni::bench
