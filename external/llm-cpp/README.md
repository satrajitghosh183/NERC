# llm-cpp

C++ reimplementation of [AI2 OLMo-core](https://github.com/allenai/OLMo-core) transformer training framework built on LibTorch. The upstream Python OLMo-core is vendored under `olmo-python/` as the reference comparison stack.

## Features

- **Fused operators** — fused QKV projections, fused gate+up FFN, fused RMSNorm/SiLU/RoPE kernels
- **Muon optimizer** — momentum + Newton-Schulz orthogonalization
- **µP initialization** — maximal update parameterization for stable scaling
- **DC-MRE embedding** — dual-codebook multi-resolution embedding (semantic + syntactic role + char trigrams + phrase context)
- **Multi-Token Prediction** — auxiliary MTP heads for improved training signal
- **SIMD backend** — vectorized CPU kernels with arena allocator
- **CUDA backend** — fused GPU kernels for RMSNorm, SiLU×Mul, RoPE
- **Single .conf config** — INI-style configuration, no CLI flag soup

## Build

    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

## Usage

    ./build/olmo_train conf/olmo.conf

## Config

All settings live in a single .conf file (see conf/olmo.conf):

    [model]
    d_model     256
    vocab_size  50257
    n_layers    4
    n_heads     8

    [training]
    steps       5000
    batch_size  4
    seq_len     256
    lr          3e-4
    optimizer   muon

    [optimization]
    fused       1
    mup         1
    multi_res   1

    [device]
    device      auto

Presets: conf/olmo.conf (30M), conf/olmo_125M.conf (125M+MTP).

## Data Preparation

    python tools/prepare_data.py --tokenizer gpt2 --input <text_file> --output data/tinystories_gpt2.npy

## License

Apache 2.0
