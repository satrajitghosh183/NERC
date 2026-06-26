# omnishader — local CLI for the OmniTrace debugger + generator

```bash
./omnishader debug  <shader.frag>     # debug a shader AND render it (local GPU/MoltenVK)
./omnishader render <shader.frag>     # just render it to an image and open it
./omnishader loop   "<prompt>"        # generate -> debug -> render, the whole pipeline
./omnishader gen    "<prompt>" -o f   # just generate
./omnishader eval   <dir>             # compile@1 over a directory of shaders
```

Everything runs **locally** on your machine:
- **omni_reward** (`tools/omni_reward.cpp`) — compiles, lifts SPIR-V → UIR, runs the shader on the
  CPU reference, reports compile/execute/reward. Catches shaders that compile but NaN when run.
- **omni_render** (`tools/omni_render.cpp`) — renders a `mainImage` shader on the real GPU
  (Vulkan/MoltenVK), reads back the image, writes a PNG and opens it.
- Build once: `cmake -S . -B build && cmake --build build`

Generation backends: local `llm-cpp` chat (your from-scratch model, set via flags) or the GPU box
(`OMNISHADER_BOX=user@ip` for the RL-refined DoRA-3B). The debugger + renderer are always local.
