#version 430
layout(location = 0) out vec2 result;

in vec3 ShadowUV;

void main () {
    result = vec2(ShadowUV.z, 1.0f);
}
