#pragma once

/**
 * include/olmo_cpp/nn/hf_convert.hpp
 *
 * Declares the OLMo<->HuggingFace checkpoint converter. Two responsibilities:
 *   1. Map parameter names between this codebase ("transformer.blocks.{i}.attn.w_q.weight")
 *      and the HF Llama-style layout ("model.layers.{i}.self_attn.q_proj.weight").
 *   2. Read/write the HF "safetensors" container (a stable binary format
 *      consisting of a JSON header followed by a flat blob of raw tensor bytes).
 *
 * The converter is also used (in the other direction) to load Llama/OLMo
 * weights published on the HF Hub for warm starts and continued pretraining.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: torch::Tensor for state-dict values and serialization.
 *   - <string>, <unordered_map>, <vector>: std containers used across the API.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/nn/hf_convert.cpp: implementation of every method declared here.
 *   - tools/convert_hf.cpp: standalone CLI tool that wraps HFConverter.
 *
 * --- Role in training pipeline ---
 *   Used at the start (loading HF weights as a warm start) and end (exporting
 *   our trained checkpoints to HF format for evaluation harnesses and the Hub)
 *   of training. Not on the hot path.
 */

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace olmo_cpp {

/// Bidirectional name mapping between OLMo-cpp and HuggingFace state-dict keys.
/// Built once and consulted per-tensor during convert().
struct StateMapping {
  /// olmo-name -> hf-name
  std::unordered_map<std::string, std::string> olmo_to_hf;
  /// hf-name -> olmo-name (inverse of olmo_to_hf, populated alongside it)
  std::unordered_map<std::string, std::string> hf_to_olmo;
  /// List of keys whose 2D weight tensor must be transposed during conversion.
  /// Empty for OLMo2/Llama models (HF and OLMo agree on Linear layouts).
  std::vector<std::string> transpose_keys;

  /// Build the mapping for OLMo2/Llama-style models. num_heads/num_kv_heads
  /// are accepted for API symmetry with future variants but are unused today.
  static StateMapping create_olmo2_mapping(int num_layers, int num_heads, int num_kv_heads);

  /// Build the mapping for hybrid (e.g., Mamba+Attn) models. Currently equals
  /// create_olmo2_mapping; will diverge as hybrid layer types are added.
  static StateMapping create_hybrid_mapping(int num_layers);
};

/// Top-level converter. Takes a StateMapping and uses it to translate state
/// dicts and on-disk checkpoints in either direction.
class HFConverter {
 public:
  /// Direction of conversion. Determines which side of the StateMapping is
  /// used as the lookup table.
  enum class Direction { TO_HF, FROM_HF };

  /// Construct from a prebuilt StateMapping (made via StateMapping::create_*).
  explicit HFConverter(const StateMapping& mapping);

  /// Pure-tensor conversion: given a state dict, returns a new state dict
  /// with renamed (and optionally transposed) tensors. Tensor data is shared
  /// when possible — only metadata is rewritten.
  std::unordered_map<std::string, torch::Tensor> convert(
      const std::unordered_map<std::string, torch::Tensor>& state_dict,
      Direction direction) const;

  /// Load an OLMo .pt checkpoint and save it as an HF directory. `format`
  /// is "safetensors" (default) or "bin" (legacy pickle-style).
  void convert_checkpoint_to_hf(const std::string& olmo_path,
                                 const std::string& hf_output_path,
                                 const std::string& format = "safetensors") const;

  /// Load an HF directory (safetensors preferred, falls back to .bin) and
  /// save it as a single OLMo .pt file.
  void convert_checkpoint_from_hf(const std::string& hf_path,
                                   const std::string& olmo_output_path) const;

  /// Write a minimal HF config.json describing the model. Required by HF
  /// loading code (transformers expects this file alongside weights).
  static void write_hf_config(const std::string& output_path,
                               int vocab_size, int hidden_size, int num_layers,
                               int num_heads, int num_kv_heads, int intermediate_size,
                               double rope_theta, const std::string& model_type = "llama");

  /// Write a minimal HF tokenizer_config.json. Tokenizer files (vocab/merges)
  /// must be copied separately by the caller.
  static void write_tokenizer_config(const std::string& output_path,
                                      const std::string& tokenizer_class = "GPT2Tokenizer");

 private:
  /// The name mapping captured at construction time.
  StateMapping mapping_;
};

/// safetensors I/O helpers. Implements the file format spec at
/// https://github.com/huggingface/safetensors: 8-byte little-endian header
/// length, JSON header (one entry per tensor), then concatenated raw bytes.
namespace safetensors {
/// Load all tensors in `path` into a state dict. Tensors are eager-allocated
/// on CPU; move them to the target device after loading.
std::unordered_map<std::string, torch::Tensor> load(const std::string& path);
/// Save a state dict to a single safetensors file. Keys are written in
/// sorted order so two equivalent dicts hash identically.
void save(const std::unordered_map<std::string, torch::Tensor>& state_dict,
          const std::string& path);
}  // namespace safetensors

}  // namespace olmo_cpp
