#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform P { float k; } pc;
void main() {
    vec3 c = vec3(uv, 0.5);
    outColor = vec4(c * pc.k, 1.0);
}
