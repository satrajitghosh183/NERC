// Emits a sample token .npy so external tools (NumPy / olmo_cpp::TokenDataset) can verify interop.
#include "omni/synth/corpus.hpp"
#include <cstdio>
int main() {
    std::vector<std::string> shaders = {
        "#version 450\nvoid main(){ gl_FragColor = vec4(1.0); }\n",
        "vec3 palette(float t){ return 0.5+0.5*cos(6.2831*(t+vec3(0.0,0.33,0.67))); }\n",
    };
    auto st = omni::synth::build_token_npy(shaders, "data/bench/sample_tokens.npy");
    std::printf("wrote %d shaders, %llu tokens -> data/bench/sample_tokens.npy\n",
                st.num_shaders, (unsigned long long)st.num_tokens);
    return 0;
}
