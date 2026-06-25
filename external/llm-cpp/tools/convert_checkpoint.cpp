/**
 * tools/convert_checkpoint.cpp
 *
 * Minimal helper that re-saves a torch::save'd .pt file through the
 * LibTorch C++ serializer. The use case: a Python checkpoint produced
 * with `torch.save(model.state_dict(), "model.pt")` is technically a
 * pickle that LibTorch's C++ loader cannot always consume directly —
 * loading it as a `std::vector<torch::Tensor>` and writing it back from
 * C++ produces a file that is unambiguously readable by LibTorch C++.
 *
 * Example:
 *   ./build/convert_checkpoint python_model.pt cpp_model.pt
 *
 * --- Build target ---
 *   convert_checkpoint (CMakeLists.txt:512). Links olmo_cpp + LibTorch.
 *   Compiled with -O3.
 *
 * --- Includes from this project ---
 *   (none directly — this tool only needs torch/torch.h. olmo_cpp is
 *   linked just to share the same LibTorch ABI as the rest of the
 *   project, not because any olmo_cpp headers are used.)
 *
 * --- Reads / Writes ---
 *   - reads:  argv[1] (input .pt produced by python torch.save)
 *   - writes: argv[2] (output .pt re-saved via LibTorch C++)
 *
 * --- Role in workflow ---
 *   Run once when ingesting a Python checkpoint. Tools like `chat` and
 *   `convert_hf` then load the converted file via torch::load(model, ...).
 *   Safetensors support is provided separately by `convert_hf`.
 */
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  // Phase 1: argument check — we expect exactly two positional paths.
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.pt> <output.pt>\n";
    return 1;
  }
  std::string input_path = argv[1];
  std::string output_path = argv[2];

  try {
    // Phase 2: load the input as a flat list of tensors. This works for
    // checkpoints saved via `torch::save(std::vector<torch::Tensor>, ...)`
    // or Python's `torch.save([t for t in state_dict.values()], ...)`.
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, input_path);
    // Phase 3: write the same tensors back out through LibTorch C++.
    torch::save(tensors, output_path);
    std::cout << "Converted " << tensors.size() << " tensors to " << output_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    // Phase 4: error handling — catch malformed pickles, missing files,
    // or version mismatches and report cleanly.
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
