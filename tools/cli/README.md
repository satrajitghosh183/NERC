# omnishader — CLI for the OmniTrace debugger + generator

```bash
./omnishader debug <shader.frag>      # run the OmniTrace debugger on a shader (local, C++)
./omnishader gen   "<prompt>" -o f    # generate a shader (RL-refined DoRA-3B, on the GPU box)
./omnishader loop  "<prompt>"         # generate -> debug, the whole pipeline
./omnishader eval  <dir>              # compile@1 over a directory of shaders
```

- **debug** runs locally — `omni_reward` (built from `tools/omni_reward.cpp`) compiles the shader,
  lifts SPIR-V to UIR, runs it on the CPU SIMT reference, and reports compile / execute / reward.
  It catches shaders that compile but produce NaN/Inf when run — what a compile check misses.
- **gen / loop** generate on the GPU box (`tools/cli/gen_one.py`, the RL-refined DoRA-3B) over ssh.
  Point at your box with `OMNISHADER_BOX=user@ip OMNISHADER_KEY=~/.ssh/key` (IP changes on unshelve).

Build the debugger once: `cmake -S . -B build && cmake --build build --target omni_reward`
