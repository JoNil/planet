#version 400
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex;

out vec3 Position;
out vec3 vPos;
out vec3 Normal;
out vec2 UV;

uniform mat4 MV;
uniform mat4 P;

//////////////////////////////////////////////////////////////////////////////////////////

void main () 
{	
//	float oceanHeight = 0.65f;

//	vec3 noicePos = VertexPosition + VertexNormal * fbm(VertexPosition);
//	Altitude = length(noicePos);
//	vec3 surfacePos = Altitude < oceanHeight ? VertexPosition : noicePos;

	Position =  vec3( MV * vec4(pos, 1.0));
	vPos = pos;
	Normal = normalize(mat3(MV) * normal);
	UV = tex;

	//! Convert position to clip coordinates and pass along to fragment shader
	gl_Position =  (P * MV) * vec4(pos, 1.0);

}