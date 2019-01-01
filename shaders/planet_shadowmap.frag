#version 430
layout(location = 0) out vec2 result;

void main () {
    result = vec2(gl_FragCoord.z, 1.0f);
}
