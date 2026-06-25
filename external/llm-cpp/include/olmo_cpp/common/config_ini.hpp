#pragma once
/**
 * include/olmo_cpp/common/config_ini.hpp
 *
 * Header-only INI-style key/value parser. The .conf files in conf/
 * (e.g. conf/quickstart_3060.conf) are read with this — it provides
 * `ConfigINI(path, section)` and a `get<T>(key)` /
 * `get_or<T>(key, default)` API.
 *
 * Format:
 *
 *   [section_name]
 *   key1   value1
 *   key2   value2     # comment
 *
 *   [other_section]
 *   ...
 *
 * Whitespace separates key from value (tabs preferred); '#' starts a
 * comment. Sections are mutually exclusive — you instantiate one
 * ConfigINI per [section] you want to read.
 *
 * Header-only because it's tiny and used by both the C++ training
 * binary and the standalone tools (which don't link to any common
 * lib that would otherwise own the implementation).
 *
 * --- Includes from this project ---
 *   (none — pure stdlib + templates.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: parses every .conf section into TrainingConfig +
 *     TransformerConfig.
 *   - tools/dump_embeddings.cpp: re-parses the same .conf to know
 *     what shape model to allocate before loading the checkpoint.
 *
 * --- Role in training pipeline ---
 *   The single source of truth for "what the user typed in their
 *   .conf". Anything not in the .conf falls back to its default.
 */

#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>

/**
 * Map-backed INI-style config reader.
 * Ported from kuiper-tts Config.hpp — same API, same file format.
 *
 * File format (conf/olmo.conf):
 *   [section_name]
 *   key<TAB>value      <- tab delimiter (preferred)
 *   key = value        <- equals delimiter (also accepted)
 *   # comment lines ignored
 *
 * Entire file parsed into memory at construction — O(1) key lookups.
 *
 * Usage:
 *   ConfigINI cfg("conf/olmo.conf", "training");
 *   int steps = cfg.get_or("steps", 1000);
 *   double lr; cfg.get("lr", lr);
 */
class ConfigINI {
public:
    explicit ConfigINI(const std::string& conf_path, const std::string& section = "");

    ConfigINI(const char env_varname[], const char conf_filename[],
              const std::string& section = "");

    template<typename T>
    void get(const std::string& key, T& out) const;

    template<typename T>
    bool get_optional(const std::string& key, T& out) const;

    template<typename T>
    T get_or(const std::string& key, T default_val) const;

    bool has(const std::string& key) const { return values_.count(key) != 0; }

    const std::string& path() const { return path_; }
    const std::string& section() const { return section_; }

private:
    using KVMap      = std::unordered_map<std::string, std::string>;
    using SectionMap = std::unordered_map<std::string, KVMap>;

    KVMap       values_;
    std::string section_;
    std::string path_;

    static void trim(std::string& s) {
        const char* ws = " \t\r\n";
        auto b = s.find_first_not_of(ws);
        if (b == std::string::npos) { s.clear(); return; }
        s = s.substr(b, s.find_last_not_of(ws) - b + 1);
    }

    static SectionMap parse_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Config: cannot open '" + path + "'");
        SectionMap all;
        std::string cur_section;
        std::string line;
        while (std::getline(f, line)) {
            trim(line);
            if (line.empty() || line[0] == '#') continue;
            if (line[0] == '[' && line.back() == ']') {
                cur_section = line.substr(1, line.size() - 2);
                trim(cur_section);
                continue;
            }
            size_t sep = line.find('\t');
            if (sep == std::string::npos) sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key   = line.substr(0, sep);
            std::string value = line.substr(sep + 1);
            while (!value.empty() && (value[0] == '\t' || value[0] == ' '))
                value.erase(0, 1);
            // Strip inline comments: anything after # preceded by whitespace
            auto hash = value.find('#');
            if (hash != std::string::npos && hash > 0 &&
                (value[hash - 1] == ' ' || value[hash - 1] == '\t')) {
                value = value.substr(0, hash);
            }
            trim(key); trim(value);
            if (!key.empty())
                all[cur_section][key] = value;
        }
        return all;
    }

    static std::string env_conf_path(const char varname[], const char filename[]) {
        const char* root = std::getenv(varname);
        if (!root || root[0] == '\0')
            throw std::runtime_error(
                std::string("Config: environment variable ") + varname + " is not set.\n"
                "Run: export " + varname + "=/path/to/llm-cpp");
        return std::string(root) + "/conf/" + filename;
    }

    void load(const std::string& path, const std::string& section) {
        path_    = path;
        section_ = section;
        auto all = parse_file(path);
        auto it  = all.find(section);
        if (!section.empty() && it == all.end())
            throw std::runtime_error(
                "Config: section [" + section + "] not found in '" + path + "'");
        values_ = (it != all.end()) ? it->second : KVMap{};
    }

    template<typename T>
    static T convert(const std::string& v);
};

// ── convert<T> specialisations ──────────────────────────────────────────────

template<> inline uint32_t ConfigINI::convert<uint32_t>(const std::string& v) {
    try { return static_cast<uint32_t>(std::stoul(v)); } catch (...) { return 0u; }
}
template<> inline int ConfigINI::convert<int>(const std::string& v) {
    try { return std::stoi(v); } catch (...) { return 0; }
}
template<> inline int64_t ConfigINI::convert<int64_t>(const std::string& v) {
    try { return std::stoll(v); } catch (...) { return 0; }
}
template<> inline float ConfigINI::convert<float>(const std::string& v) {
    try { return std::stof(v); } catch (...) { return 0.f; }
}
template<> inline double ConfigINI::convert<double>(const std::string& v) {
    try { return std::stod(v); } catch (...) { return 0.0; }
}
template<> inline std::string ConfigINI::convert<std::string>(const std::string& v) {
    return v;
}
template<> inline bool ConfigINI::convert<bool>(const std::string& v) {
    std::string lv = v;
    std::transform(lv.begin(), lv.end(), lv.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return lv == "1" || lv == "true" || lv == "yes" || lv == "on";
}

// ── Constructors ────────────────────────────────────────────────────────────

inline ConfigINI::ConfigINI(const std::string& conf_path, const std::string& section) {
    load(conf_path, section);
}
inline ConfigINI::ConfigINI(const char env_varname[], const char conf_filename[],
                            const std::string& section) {
    load(env_conf_path(env_varname, conf_filename), section);
}

// ── get() / get_optional() / get_or() ───────────────────────────────────────

template<typename T>
void ConfigINI::get(const std::string& key, T& out) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
        std::string ctx = section_.empty() ? path_
                                           : path_ + " [" + section_ + "]";
        throw std::runtime_error("Config: key '" + key + "' not found in " + ctx);
    }
    out = convert<T>(it->second);
}

template<typename T>
bool ConfigINI::get_optional(const std::string& key, T& out) const {
    auto it = values_.find(key);
    if (it == values_.end()) return false;
    out = convert<T>(it->second);
    return true;
}

template<typename T>
T ConfigINI::get_or(const std::string& key, T default_val) const {
    auto it = values_.find(key);
    if (it == values_.end()) return default_val;
    return convert<T>(it->second);
}
