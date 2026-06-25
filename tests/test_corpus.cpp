#include "omni/test.hpp"
#include "omni/synth/corpus.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace omni::synth;

static std::string tmp(const char* name) {
    const char* d = std::getenv("TMPDIR"); std::string dir = d ? d : "/tmp";
    return dir + "/" + name;
}

TEST(corpus, tokenizer_roundtrip) {
    std::string s = "#version 450\nvoid main(){ gl_FragColor = vec4(1.0); }\n";
    auto t = tokenize(s);
    REQUIRE_EQ(t.size(), s.size());
    CHECK(detokenize(t) == s);
    // Special tokens are dropped on detokenize.
    std::vector<uint16_t> with_special = {TOK_BOS, 'h', 'i', TOK_SEP};
    CHECK(detokenize(with_special) == "hi");
}

TEST(corpus, npy_write_read_roundtrip) {
    std::vector<uint16_t> data;
    for (int i = 0; i < 1000; ++i) data.push_back((uint16_t)(i * 7 % VOCAB_SIZE));
    std::string path = tmp("omni_tokens_test.npy");
    REQUIRE(npy_write_u16(path, data));
    std::vector<uint16_t> back;
    REQUIRE(npy_read_u16(path, back));
    REQUIRE_EQ(back.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) REQUIRE_EQ(back[i], data[i]);
    std::remove(path.c_str());
}

TEST(corpus, build_token_stream) {
    std::vector<std::string> shaders = {
        "#version 450\nvoid main(){}\n",
        "vec3 f(vec2 uv){ return vec3(uv, 0.5); }",
        "float sdf(vec3 p){ return length(p) - 1.0; }",
    };
    std::string path = tmp("omni_corpus_test.npy");
    auto st = build_token_npy(shaders, path);
    CHECK_EQ(st.num_shaders, 3);

    // num_tokens == sum(len) + 2 specials (BOS+SEP) per shader.
    uint64_t expect = 0; for (auto& s : shaders) expect += s.size() + 2;
    CHECK_EQ(st.num_tokens, expect);

    std::vector<uint16_t> back;
    REQUIRE(npy_read_u16(path, back));
    REQUIRE_EQ((uint64_t)back.size(), st.num_tokens);
    CHECK_EQ(back[0], (uint16_t)TOK_BOS);            // stream starts with BOS
    std::printf("    built %d shaders -> %llu tokens (.npy ready for olmo_cpp::TokenDataset)\n",
                st.num_shaders, (unsigned long long)st.num_tokens);
    std::remove(path.c_str());
}

TEST(corpus, ingest_real_shader_dir) {
    // Our own sample shaders dir always has at least simple.frag.
    auto shaders = ingest_dir(std::string(OMNI_SOURCE_DIR) + "/data/shaders");
    std::printf("    ingested %zu shader files from data/shaders\n", shaders.size());
    CHECK(!shaders.empty());
    for (auto& s : shaders) CHECK(!s.empty());
}
