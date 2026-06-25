#include "omni/synth/corpus.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace omni::synth {

std::vector<uint16_t> tokenize(const std::string& text) {
    std::vector<uint16_t> t; t.reserve(text.size());
    for (unsigned char c : text) t.push_back((uint16_t)c);   // byte-level
    return t;
}

std::string detokenize(const std::vector<uint16_t>& toks) {
    std::string s; s.reserve(toks.size());
    for (uint16_t t : toks) if (t < 256) s.push_back((char)(unsigned char)t);  // skip specials
    return s;
}

bool npy_write_u16(const std::string& path, const std::vector<uint16_t>& data) {
    std::ostringstream hs;
    hs << "{'descr': '<u2', 'fortran_order': False, 'shape': (" << data.size() << ",), }";
    std::string header = hs.str();
    // Pad so that (10 + header.size() + 1) is a multiple of 64; header ends with '\n'.
    size_t base = 10 + header.size() + 1;
    size_t pad = (64 - (base % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const unsigned char magic[8] = {0x93, 'N','U','M','P','Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    uint16_t hlen = (uint16_t)header.size();
    std::fwrite(&hlen, 2, 1, f);                 // little-endian (host is LE)
    std::fwrite(header.data(), 1, header.size(), f);
    std::fwrite(data.data(), 2, data.size(), f);
    std::fclose(f);
    return true;
}

bool npy_read_u16(const std::string& path, std::vector<uint16_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || magic[0] != 0x93 || std::memcmp(magic + 1, "NUMPY", 5) != 0) { std::fclose(f); return false; }
    uint16_t hlen; std::fread(&hlen, 2, 1, f);
    std::string header(hlen, '\0'); std::fread(header.data(), 1, hlen, f);
    if (header.find("'<u2'") == std::string::npos) { std::fclose(f); return false; }
    // shape (N,)
    size_t sp = header.find("'shape': (");
    uint64_t n = 0;
    if (sp != std::string::npos) n = std::strtoull(header.c_str() + sp + 10, nullptr, 10);
    out.resize(n);
    size_t got = std::fread(out.data(), 2, n, f);
    std::fclose(f);
    return got == n;
}

std::vector<std::string> ingest_dir(const std::string& dir, const std::vector<std::string>& exts) {
    namespace fs = std::filesystem;
    std::vector<std::string> shaders;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return shaders;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        std::string ext = it->path().extension().string();
        bool match = false; for (const auto& e : exts) if (ext == e) match = true;
        if (!match) continue;
        std::ifstream in(it->path(), std::ios::binary);
        std::ostringstream ss; ss << in.rdbuf();
        shaders.push_back(ss.str());
    }
    return shaders;
}

CorpusStats build_token_npy(const std::vector<std::string>& shaders, const std::string& npy_path) {
    CorpusStats st;
    std::vector<uint16_t> stream;
    for (const auto& s : shaders) {
        stream.push_back(TOK_BOS);
        auto t = tokenize(s);
        stream.insert(stream.end(), t.begin(), t.end());
        stream.push_back(TOK_SEP);
        ++st.num_shaders;
    }
    st.num_tokens = stream.size();
    npy_write_u16(npy_path, stream);
    return st;
}

} // namespace omni::synth
