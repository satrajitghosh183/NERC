/**
 * tools/dump_params.cpp
 *
 * Tiny diagnostic utility: instantiates a small `Transformer` with a
 * hard-coded TransformerConfig, then walks `named_parameters()` and
 * prints each parameter's name and shape on its own line. Useful when
 * you want to know the exact module-tree key strings that are produced
 * by the C++ model — the same strings the HF→C++ name mapper in
 * `convert_hf.cpp` targets, and that any custom checkpoint editor would
 * need to know.
 *
 * Example (no flags — just run it):
 *   ./build/dump_params
 *
 * Example output (truncated):
 *   embeddings.weight  [1000, 256]
 *   blocks.0.attention.w_q.weight  [256, 256]
 *   ...
 *   mtp_heads.0.proj.weight  [256, 256]
 *
 * --- Build target ---
 *   dump_params (CMakeLists.txt:526). Links olmo_cpp + LibTorch.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp           : TransformerConfig struct
 *   - olmo_cpp/model/transformer.hpp: Transformer module class
 *
 * --- Reads / Writes ---
 *   - reads:  nothing (config is hard-coded in main)
 *   - writes: nothing — output goes to stdout.
 *
 * --- Role in workflow ---
 *   Pure debugging aid. Run when adding a new sub-module to the model
 *   (e.g., a new MTP head variant) to verify the expected parameter
 *   keys appear, before updating `convert_hf.cpp`'s name map.
 */
#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include <iostream>

int main() {
  // Phase 1: build a small but representative config. d_model=256, 2
  // layers, 4 heads, 1k vocab and 2 MTP heads is enough to exercise
  // every parameter family that may appear in a full-size run.
  olmo_cpp::TransformerConfig cfg;
  cfg.d_model = 256;
  cfg.n_layers = 2;
  cfg.n_heads = 4;
  cfg.vocab_size = 1000;
  cfg.num_mtp_heads = 2;
  cfg.validate();

  // Phase 2: instantiate the model so torch::nn allocates and registers
  // every parameter with its hierarchical name.
  auto model = olmo_cpp::Transformer(cfg);
  // Phase 3: iterate over named_parameters() and print "<key>  <shape>"
  // for each tensor. This is the canonical list the rest of the codebase
  // uses to look up weights.
  for (const auto& p : model->named_parameters()) {
    std::cout << p.key() << "  " << p.value().sizes() << "\n";
  }
  return 0;
}
