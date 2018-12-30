#version 430

layout(location = 0) in vec3 pos;

uniform mat4 mvp;

void main () 
{
    gl_Position = mvp * vec4(500.0*pos, 1.0);
}