// export_hf — convert a zwt checkpoint into a HuggingFace Llama-compatible
// directory (model.safetensors + config.json + tokenizer_config.json).
//
// The zwt on-disk checkpoint stores parameters in the collect_params() order
// with local names ("weight" / "bias"). This tool rebuilds the Transformer
// from the same config used to train, loads the checkpoint into it, then
// walks the structure in HF order and writes a safetensors file with
// Llama-style tensor names so transformers can load the result via
// AutoModelForCausalLM.from_pretrained(dir).
//
// The one non-trivial remap is the fused SwiGLU gate_up projection: the
// training layout stacks gate then up along out_features ([2H, d_model]),
// so the first H rows become gate_proj.weight and the last H rows become
// up_proj.weight.
//
// Runs on the CPU build. Weights are brought to host through the existing
// checkpoint loader's D2H staging, then written straight out.

#include "zwt/core/allocator.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/parameter.hpp"
#include "zwt/layers/transformer.hpp"
#include "zwt/optim/adamw.hpp"
#include "zwt/train/checkpoint.hpp"
#include "zwt/train/config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

namespace {

struct CliArgs {
  std::string config_path;
  std::string ckpt_path;
  std::string out_dir;
};

CliArgs parse_cli(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    std::string k = s.substr(0, eq);
    std::string v = (eq == std::string::npos) ? std::string{}
                                              : s.substr(eq + 1);
    if (k == "--config") a.config_path = v;
    else if (k == "--ckpt")   a.ckpt_path = v;
    else if (k == "--out")    a.out_dir   = v;
    else throw std::runtime_error("export_hf: unknown arg: " + k);
  }
  if (a.config_path.empty() || a.ckpt_path.empty() || a.out_dir.empty()) {
    throw std::runtime_error(
        "export_hf: need --config=PATH --ckpt=PATH --out=DIR");
  }
  return a;
}

// Read a tensor's raw bytes onto the host. On CUDA-built binaries this does
// a D2H copy; on CPU it's a straight pointer.
std::vector<uint8_t> tensor_to_host(const Tensor& t) {
  std::vector<uint8_t> buf(t.nbytes());
#ifdef USE_CUDA
  if (t.device().is_cuda()) {
    cudaMemcpy(buf.data(), t.data(), t.nbytes(), cudaMemcpyDeviceToHost);
    return buf;
  }
#endif
  std::memcpy(buf.data(), t.data(), t.nbytes());
  return buf;
}

const char* safetensors_dtype(DType d) {
  switch (d) {
    case DType::F32:  return "F32";
    case DType::F16:  return "F16";
    case DType::BF16: return "BF16";
    default:
      throw std::runtime_error("export_hf: unsupported dtype for safetensors");
  }
}

struct STEntry {
  std::string        name;
  DType              dtype;
  std::vector<int64_t> shape;   // HF/PyTorch order
  std::vector<uint8_t> data;
};

// Manually-built safetensors JSON header. We don't pull in nlohmann_json —
// the schema is simple enough and staying dependency-free is worth it. The
// JSON spec wants no trailing comma; the loop below handles that.
std::string build_header(const std::vector<STEntry>& ents,
                         std::vector<uint64_t>& offsets_out) {
  std::string j = "{";
  uint64_t cursor = 0;
  offsets_out.clear();
  offsets_out.reserve(ents.size());
  for (size_t i = 0; i < ents.size(); ++i) {
    const auto& e = ents[i];
    j += "\"" + e.name + "\":{";
    j += "\"dtype\":\"" + std::string(safetensors_dtype(e.dtype)) + "\",";
    j += "\"shape\":[";
    for (size_t k = 0; k < e.shape.size(); ++k) {
      if (k) j += ",";
      j += std::to_string(e.shape[k]);
    }
    j += "],";
    uint64_t start = cursor;
    uint64_t end   = cursor + e.data.size();
    j += "\"data_offsets\":[" + std::to_string(start) + "," +
         std::to_string(end) + "]}";
    if (i + 1 < ents.size()) j += ",";
    offsets_out.push_back(start);
    cursor = end;
  }
  j += "}";
  // Pad header to 8-byte alignment with spaces — the spec is tolerant of
  // trailing whitespace in the JSON value.
  while (j.size() % 8 != 0) j.push_back(' ');
  return j;
}

void write_safetensors(const std::string& path,
                       const std::vector<STEntry>& ents) {
  std::vector<uint64_t> offsets;
  std::string header = build_header(ents, offsets);
  uint64_t header_len = header.size();

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("export_hf: cannot open " + path);
  f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& e : ents) {
    f.write(reinterpret_cast<const char*>(e.data.data()),
            static_cast<std::streamsize>(e.data.size()));
  }
  if (!f) throw std::runtime_error("export_hf: write failed on " + path);
}

STEntry make_entry(std::string name, const Tensor& t) {
  STEntry e;
  e.name = std::move(name);
  e.dtype = t.dtype();
  e.shape.resize(t.rank());
  for (int i = 0; i < t.rank(); ++i) e.shape[i] = t.dim(i);
  e.data = tensor_to_host(t);
  return e;
}

// Slice the first/second half of a [2H, D] weight into its own [H, D] entry.
// HF stores Linear weights as [out_features, in_features] — same layout zwt
// uses — so splitting is a clean row slice.
STEntry slice_first_half(std::string name, const Tensor& w) {
  if (w.rank() != 2 || w.dim(0) % 2 != 0) {
    throw std::runtime_error("export_hf: expected [2H, D] for gate_up");
  }
  int64_t two_h = w.dim(0), d = w.dim(1);
  int64_t h = two_h / 2;
  size_t row_bytes = static_cast<size_t>(d) * dtype_size(w.dtype());
  std::vector<uint8_t> host = tensor_to_host(w);
  STEntry e;
  e.name  = std::move(name);
  e.dtype = w.dtype();
  e.shape = {h, d};
  e.data.assign(host.data(), host.data() + row_bytes * static_cast<size_t>(h));
  return e;
}

STEntry slice_second_half(std::string name, const Tensor& w) {
  if (w.rank() != 2 || w.dim(0) % 2 != 0) {
    throw std::runtime_error("export_hf: expected [2H, D] for gate_up");
  }
  int64_t two_h = w.dim(0), d = w.dim(1);
  int64_t h = two_h / 2;
  size_t row_bytes = static_cast<size_t>(d) * dtype_size(w.dtype());
  std::vector<uint8_t> host = tensor_to_host(w);
  STEntry e;
  e.name  = std::move(name);
  e.dtype = w.dtype();
  e.shape = {h, d};
  e.data.assign(host.data() + row_bytes * static_cast<size_t>(h),
                host.data() + row_bytes * static_cast<size_t>(two_h));
  return e;
}

std::string write_config_json(const Transformer::Config& c) {
  std::string j = "{\n";
  j += "  \"architectures\": [\"LlamaForCausalLM\"],\n";
  j += "  \"model_type\": \"llama\",\n";
  j += "  \"vocab_size\": " + std::to_string(c.vocab_size) + ",\n";
  j += "  \"hidden_size\": " + std::to_string(c.d_model) + ",\n";
  j += "  \"intermediate_size\": " + std::to_string(c.d_ffn) + ",\n";
  j += "  \"num_hidden_layers\": " + std::to_string(c.n_layers) + ",\n";
  j += "  \"num_attention_heads\": " + std::to_string(c.n_heads) + ",\n";
  int64_t kv = (c.n_kv_heads > 0) ? c.n_kv_heads : c.n_heads;
  j += "  \"num_key_value_heads\": " + std::to_string(kv) + ",\n";
  j += "  \"hidden_act\": \"silu\",\n";
  j += "  \"max_position_embeddings\": " + std::to_string(c.max_seq) + ",\n";
  j += "  \"rms_norm_eps\": " + std::to_string(c.norm_eps) + ",\n";
  j += "  \"rope_theta\": " + std::to_string(c.rope_base) + ",\n";
  j += "  \"tie_word_embeddings\": ";
  j += (c.tie_embeddings ? "true" : "false");
  j += ",\n  \"torch_dtype\": \"bfloat16\"\n";
  j += "}\n";
  return j;
}

// Minimal tokenizer_config.json pointing at GPT-2 BPE. A real deploy should
// also ship vocab.json + merges.txt (or a tokenizer.json) — we leave that to
// downstream tooling because zwt trains against pre-tokenized shards and
// the tokenizer artifacts live with the data prep pipeline, not the model.
std::string write_tokenizer_config_json() {
  return
    "{\n"
    "  \"tokenizer_class\": \"GPT2Tokenizer\",\n"
    "  \"model_max_length\": 2048,\n"
    "  \"bos_token\": \"<|endoftext|>\",\n"
    "  \"eos_token\": \"<|endoftext|>\",\n"
    "  \"unk_token\": \"<|endoftext|>\"\n"
    "}\n";
}

void write_text(const std::string& path, const std::string& body) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("export_hf: cannot open " + path);
  f.write(body.data(), static_cast<std::streamsize>(body.size()));
  if (!f) throw std::runtime_error("export_hf: write failed on " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliArgs cli = parse_cli(argc, argv);
    train::TrainConfig cfg = train::load_train_config(cli.config_path);

#ifdef USE_CUDA
    Device dev = Device::cuda(0);
    DType  dtype = DType::BF16;
#else
    Device dev = Device::cpu();
    DType  dtype = DType::F32;
#endif

    size_t arena_bytes = static_cast<size_t>(std::max<int64_t>(cfg.arena_mb, 256)) << 20;
    set_activation_arena_capacity(arena_bytes);

    Transformer model(cfg.model, dtype, dev, cfg.init_seed);
    std::vector<Parameter*> params;
    model.collect_params(params);

    // AdamW is required by load_checkpoint to restore moments — we don't use
    // them but the loader's signature demands a matching optimizer.
    optim::AdamW opt(params, cfg.adamw);
    (void)train::load_checkpoint(cli.ckpt_path, params, opt);

    // Walk collect_params order and rename to HF Llama layout.
    //   0: tok_emb.weight
    //   per block i (i in [0, n_layers)):
    //     norm1.weight, q_proj.weight, k_proj.weight, v_proj.weight,
    //     o_proj.weight, norm2.weight, gate_up.weight, down.weight
    //   last: final_norm.weight, lm_head.weight
    std::vector<STEntry> ents;
    size_t idx = 0;
    ents.push_back(make_entry("model.embed_tokens.weight", params[idx++]->value));

    for (int64_t i = 0; i < cfg.model.n_layers; ++i) {
      const std::string p = "model.layers." + std::to_string(i) + ".";
      ents.push_back(make_entry(p + "input_layernorm.weight",         params[idx++]->value));
      ents.push_back(make_entry(p + "self_attn.q_proj.weight",        params[idx++]->value));
      ents.push_back(make_entry(p + "self_attn.k_proj.weight",        params[idx++]->value));
      ents.push_back(make_entry(p + "self_attn.v_proj.weight",        params[idx++]->value));
      ents.push_back(make_entry(p + "self_attn.o_proj.weight",        params[idx++]->value));
      ents.push_back(make_entry(p + "post_attention_layernorm.weight",params[idx++]->value));
      const Tensor& gate_up = params[idx++]->value;
      ents.push_back(slice_first_half (p + "mlp.gate_proj.weight", gate_up));
      ents.push_back(slice_second_half(p + "mlp.up_proj.weight",   gate_up));
      ents.push_back(make_entry(p + "mlp.down_proj.weight",           params[idx++]->value));
    }
    ents.push_back(make_entry("model.norm.weight",  params[idx++]->value));
    ents.push_back(make_entry("lm_head.weight",     params[idx++]->value));

    if (idx != params.size()) {
      std::fprintf(stderr,
          "export_hf: walked %zu params, Transformer exposed %zu — config "
          "does not match the loaded checkpoint.\n", idx, params.size());
      return 2;
    }

    // Write outputs. mkdir -p the out_dir is on the caller.
    write_safetensors(cli.out_dir + "/model.safetensors", ents);
    write_text(cli.out_dir + "/config.json",             write_config_json(cfg.model));
    write_text(cli.out_dir + "/tokenizer_config.json",   write_tokenizer_config_json());

    std::fprintf(stderr, "export_hf: wrote %zu tensors to %s\n",
                 ents.size(), cli.out_dir.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "export_hf: %s\n", e.what());
    return 1;
  }
}
