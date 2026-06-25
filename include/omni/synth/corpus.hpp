// omni/synth/corpus.hpp — shader corpus -> tokens -> .npy for llm-cpp's TokenDataset.
//
// From-scratch byte-level tokenizer + NumPy .npy writer (uint16). Ingests GLSL shader
// text (e.g. the shaders21k corpus — data only) and emits a single token stream the
// vendored trainer memory-maps (PLAN.md §13 synth/data, §10 olmo_cpp::TokenDataset).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace omni::synth {

// Special tokens live just above the 256 byte values.
enum : uint16_t { TOK_BOS = 256, TOK_SEP = 257, VOCAB_SIZE = 258 };

std::vector<uint16_t> tokenize(const std::string& text);
std::string detokenize(const std::vector<uint16_t>& toks);   // drops special tokens

// NumPy .npy v1.0 writer/reader for a 1-D little-endian uint16 array.
bool npy_write_u16(const std::string& path, const std::vector<uint16_t>& data);
bool npy_read_u16(const std::string& path, std::vector<uint16_t>& out);

// Read GLSL-ish text files (by extension) from a directory tree.
std::vector<std::string> ingest_dir(const std::string& dir,
                                    const std::vector<std::string>& exts = {".frag", ".glsl", ".fs", ".comp", ".vert"});

struct CorpusStats { int num_shaders = 0; uint64_t num_tokens = 0; };

// Build one token stream: <BOS> tokens(s0) <SEP> tokens(s1) <SEP> ... -> .npy.
CorpusStats build_token_npy(const std::vector<std::string>& shaders, const std::string& npy_path);

} // namespace omni::synth
