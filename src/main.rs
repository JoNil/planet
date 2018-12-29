use cgmath::{perspective, vec3, Deg, Matrix4, Rad};
use glium::glutin::{Api, GlProfile, GlRequest};
use glium::index::PrimitiveType;
use glium::{glutin, implement_vertex, program, uniform, Surface};
use std::f32::consts::PI;

#[derive(Copy, Clone, Default)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    tex: [f32; 2],
}
implement_vertex!(Vertex, pos, normal, tex);

#[derive(Copy, Clone, Default)]
struct Triangle {
    ind: [i32; 3],
}

fn create_sphere(vertices: &mut [Vertex], indices: &mut [Triangle], radius: f32, segments: usize) {
    let vsegs = if segments < 2 { 2 } else { segments };
    let hsegs = vsegs * 2;
    let nverts = (1 + (vsegs - 1) * (hsegs + 1) + 1) as i32;

    // Top
    vertices[0].pos[0] = 0.0;
    vertices[0].pos[1] = 0.0;
    vertices[0].pos[2] = radius;
    vertices[0].normal[0] = 0.0;
    vertices[0].normal[1] = 0.0;
    vertices[0].normal[2] = 1.0;
    vertices[0].tex[0] = 0.5;
    vertices[0].tex[1] = 1.0;

    // Bottom
    let base = (nverts as usize) - 1;
    vertices[base].pos[0] = 0.0;
    vertices[base].pos[1] = 0.0;
    vertices[base].pos[2] = -radius;
    vertices[base].normal[0] = 0.0;
    vertices[base].normal[1] = 0.0;
    vertices[base].normal[2] = -1.0;
    vertices[base].tex[0] = 0.5;
    vertices[base].tex[1] = 0.0;

    for j in 0..(vsegs - 1) {
        let theta = ((j + 1) as f32) / (vsegs as f32) * PI;
        let z = theta.cos();
        let r = theta.sin();
        for i in 0..hsegs {
            let phi = i as f32 / hsegs as f32 * 2.0 * PI;
            let x = r * phi.cos();
            let y = r * phi.sin();
            let base = 1 + j * (hsegs + 1) + i;
            vertices[base].pos[0] = radius * x;
            vertices[base].pos[1] = radius * y;
            vertices[base].pos[2] = radius * z;
            vertices[base].normal[0] = x;
            vertices[base].normal[1] = y;
            vertices[base].normal[2] = z;
            vertices[base].tex[0] = i as f32 / hsegs as f32;
            vertices[base].tex[1] = 1.0 - (j as f32 + 1.0) / vsegs as f32;
        }
    }

    // Top cap
    for i in 0..hsegs {
        indices[i].ind[0] = 0;
        indices[i].ind[1] = 1 + (i as i32);
        indices[i].ind[2] = 2 + (i as i32);
    }
    // Middle part (possibly empty if vsegs=2)
    for j in 0..(vsegs - 2) {
        for i in 0..hsegs {
            let base = hsegs + 2 * (j * hsegs + i);
            let i0 = (1 + j * (hsegs + 1) + i) as i32;
            if i == hsegs - 1 {
                let i00 = (j * (hsegs + 1)) as i32;

                indices[base].ind[0] = i0;
                indices[base].ind[1] = i0 + (hsegs as i32) + 1;
                indices[base].ind[2] = i00 + 1;
                indices[base + 1].ind[0] = i00 + 1;
                indices[base + 1].ind[1] = i0 + (hsegs as i32) + 1;
                indices[base + 1].ind[2] = i00 + (hsegs as i32) + 2;
            } else {
                indices[base].ind[0] = i0;
                indices[base].ind[1] = i0 + (hsegs as i32) + 1;
                indices[base].ind[2] = i0 + 1;
                indices[base + 1].ind[0] = i0 + 1;
                indices[base + 1].ind[1] = i0 + (hsegs as i32) + 1;
                indices[base + 1].ind[2] = i0 + (hsegs as i32) + 2;
            }
        }
    }

    // // Bottom cap
    let base = hsegs + 2 * (vsegs - 2) * hsegs;
    for i in 0..hsegs {
        indices[base + i].ind[0] = nverts - 1;
        indices[base + i].ind[1] = nverts - 2 - (i as i32);
        indices[base + i].ind[2] = nverts - 3 - (i as i32);
    }
}

fn main() {
    let mut events_loop = glutin::EventsLoop::new();
    let display = {
        let window = glutin::WindowBuilder::new().with_title("Cool yo! ;P");
        let context = glutin::ContextBuilder::new()
            .with_gl_profile(GlProfile::Core)
            .with_gl(GlRequest::Specific(Api::OpenGl, (4, 3)));
        glium::Display::new(window, context, &events_loop).unwrap()
    };

    let (vertex_buffer, index_buffer) = {
        const VSEGS: usize = 1024;
        const HSEGS: usize = VSEGS * 2;
        const NVERTS: usize = 1 + (VSEGS - 1) * (HSEGS + 1) + 1; // top + middle + bottom
        const NTRIS: usize = HSEGS + (VSEGS - 2) * HSEGS * 2 + HSEGS; // top + middle + bottom

        let mut vertex_list: Vec<Vertex> = vec![Default::default(); NVERTS];
        let mut index_list: Vec<Triangle> = vec![Default::default(); NTRIS];

        create_sphere(&mut vertex_list, &mut index_list, 0.65, VSEGS);

        let mut flat_index_list = Vec::new();

        for tri in index_list {
            flat_index_list.push(tri.ind[0] as u32);
            flat_index_list.push(tri.ind[1] as u32);
            flat_index_list.push(tri.ind[2] as u32);
        }

        let index_buffer =
            glium::IndexBuffer::new(&display, PrimitiveType::TrianglesList, &flat_index_list)
                .unwrap();

        let vertex_buffer = glium::VertexBuffer::new(&display, &vertex_list).unwrap();

        (vertex_buffer, index_buffer)
    };

    let program = program!(&display,
        400 => {
            vertex: "
                #version 400
                layout(location = 0) in vec3 pos;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec2 tex;

                out vec3 Position;
                out vec3 vPos;
                out vec3 Normal;
                out vec2 UV;
                out float Altitude;

                uniform mat4 MV;
                uniform mat4 P;

                //
                // Description : Array and textureless GLSL 2D/3D/4D simplex 
                //               noise functions.
                //      Author : Ian McEwan, Ashima Arts.
                //  Maintainer : ijm
                //     Lastmod : 20110822 (ijm)
                //     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
                //               Distributed under the MIT License. See LICENSE file.
                //               https://github.com/ashima/webgl-noise
                // 

                vec3 mod289(vec3 x) {
                  return x - floor(x * (1.0 / 289.0)) * 289.0;
                }

                vec4 mod289(vec4 x) {
                  return x - floor(x * (1.0 / 289.0)) * 289.0;
                }

                vec4 permute(vec4 x) {
                  return mod289(((x*34.0)+1.0)*x);
                }

                vec4 taylorInvSqrt(vec4 r) {
                  return 1.79284291400159 - 0.85373472095314 * r;
                }

                float snoise(vec3 v) {
                  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
                  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

                  // First corner
                  vec3 i  = floor(v + dot(v, C.yyy) );
                  vec3 x0 =   v - i + dot(i, C.xxx) ;

                  // Other corners
                  vec3 g = step(x0.yzx, x0.xyz);
                  vec3 l = 1.0 - g;
                  vec3 i1 = min( g.xyz, l.zxy );
                  vec3 i2 = max( g.xyz, l.zxy );

                  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
                  //   x1 = x0 - i1  + 1.0 * C.xxx;
                  //   x2 = x0 - i2  + 2.0 * C.xxx;
                  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
                  vec3 x1 = x0 - i1 + C.xxx;
                  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
                  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

                  // Permutations
                  i = mod289(i); 
                  vec4 p = permute(permute(permute(
                             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
                           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

                  // Gradients: 7x7 points over a square, mapped onto an octahedron.
                  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
                  float n_ = 0.142857142857; // 1.0/7.0
                  vec3  ns = n_ * D.wyz - D.xzx;

                  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

                  vec4 x_ = floor(j * ns.z);
                  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

                  vec4 x = x_ *ns.x + ns.yyyy;
                  vec4 y = y_ *ns.x + ns.yyyy;
                  vec4 h = 1.0 - abs(x) - abs(y);

                  vec4 b0 = vec4( x.xy, y.xy );
                  vec4 b1 = vec4( x.zw, y.zw );

                  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
                  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
                  vec4 s0 = floor(b0)*2.0 + 1.0;
                  vec4 s1 = floor(b1)*2.0 + 1.0;
                  vec4 sh = -step(h, vec4(0.0));

                  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
                  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

                  vec3 p0 = vec3(a0.xy,h.x);
                  vec3 p1 = vec3(a0.zw,h.y);
                  vec3 p2 = vec3(a1.xy,h.z);
                  vec3 p3 = vec3(a1.zw,h.w);

                  //Normalise gradients
                  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
                  p0 *= norm.x;
                  p1 *= norm.y;
                  p2 *= norm.z;
                  p3 *= norm.w;

                  // Mix final noise value
                  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                  m = m * m;

                  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                                dot(p2,x2), dot(p3,x3) ) );
                }

                //  <https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83>
                #define NUM_OCTAVES 5
                float fbm(vec3 x) {
                    float freq = 0.7f;
                    float amp = 0.15f;
                    float lacunarity = 1.8715f;
                    float gain = 0.5f;

                    float sum = 0.0f;
                    for (int i = 0; i < NUM_OCTAVES; ++i) {
                        sum += amp * snoise(x * freq);
                        freq *= lacunarity;
                        amp *= gain;
                    }
                    return sum;
                }

                //////////////////////////////////////////////////////////////////////////////////////////

                void main () 
                {   
                    float oceanHeight = 0.65f;

                    vec3 noicePos = pos + normal * fbm(pos);
                    Altitude = length(noicePos);
                    vec3 surfacePos = Altitude < oceanHeight ? pos : noicePos;

                    Position =  vec3( MV * vec4(surfacePos, 1.0));
                    vPos = surfacePos;
                    Normal = normalize(mat3(MV) * normal);
                    UV = tex;

                    //! Convert position to clip coordinates and pass along to fragment shader
                    gl_Position =  (P * MV) * vec4(surfacePos, 1.0);

                }
            ",

            fragment: "
                #version 400
                layout( location = 0 ) out vec4 FragColor;

                in vec3 Position;
                in vec3 vPos;
                in vec3 Normal;
                in vec2 UV;
                in float Altitude;

                //uniform vec3 LP;
                uniform sampler2D tex;

                const float shininess = 4.0;

                //uniform float oceanHeight;

                //  Classic Perlin 3D Noise 
                //  by Stefan Gustavson
                //

                vec4 mod289(vec4 x)
                {
                  return x - floor(x * (1.0 / 289.0)) * 289.0;
                }

                vec4 permute(vec4 x)
                {
                  return mod289(((x*34.0)+1.0)*x);
                }

                vec4 taylorInvSqrt(vec4 r)
                {
                  return 1.79284291400159 - 0.85373472095314 * r;
                }

                vec4 fade(vec4 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

                float cnoise(vec4 P){
                  vec4 Pi0 = floor(P); // Integer part for indexing
                  vec4 Pi1 = Pi0 + 1.0; // Integer part + 1
                  Pi0 = mod(Pi0, 289.0);
                  Pi1 = mod(Pi1, 289.0);
                  vec4 Pf0 = fract(P); // Fractional part for interpolation
                  vec4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
                  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
                  vec4 iy = vec4(Pi0.yy, Pi1.yy);
                  vec4 iz0 = vec4(Pi0.zzzz);
                  vec4 iz1 = vec4(Pi1.zzzz);
                  vec4 iw0 = vec4(Pi0.wwww);
                  vec4 iw1 = vec4(Pi1.wwww);

                  vec4 ixy = permute(permute(ix) + iy);
                  vec4 ixy0 = permute(ixy + iz0);
                  vec4 ixy1 = permute(ixy + iz1);
                  vec4 ixy00 = permute(ixy0 + iw0);
                  vec4 ixy01 = permute(ixy0 + iw1);
                  vec4 ixy10 = permute(ixy1 + iw0);
                  vec4 ixy11 = permute(ixy1 + iw1);

                  vec4 gx00 = ixy00 / 7.0;
                  vec4 gy00 = floor(gx00) / 7.0;
                  vec4 gz00 = floor(gy00) / 6.0;
                  gx00 = fract(gx00) - 0.5;
                  gy00 = fract(gy00) - 0.5;
                  gz00 = fract(gz00) - 0.5;
                  vec4 gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
                  vec4 sw00 = step(gw00, vec4(0.0));
                  gx00 -= sw00 * (step(0.0, gx00) - 0.5);
                  gy00 -= sw00 * (step(0.0, gy00) - 0.5);

                  vec4 gx01 = ixy01 / 7.0;
                  vec4 gy01 = floor(gx01) / 7.0;
                  vec4 gz01 = floor(gy01) / 6.0;
                  gx01 = fract(gx01) - 0.5;
                  gy01 = fract(gy01) - 0.5;
                  gz01 = fract(gz01) - 0.5;
                  vec4 gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
                  vec4 sw01 = step(gw01, vec4(0.0));
                  gx01 -= sw01 * (step(0.0, gx01) - 0.5);
                  gy01 -= sw01 * (step(0.0, gy01) - 0.5);

                  vec4 gx10 = ixy10 / 7.0;
                  vec4 gy10 = floor(gx10) / 7.0;
                  vec4 gz10 = floor(gy10) / 6.0;
                  gx10 = fract(gx10) - 0.5;
                  gy10 = fract(gy10) - 0.5;
                  gz10 = fract(gz10) - 0.5;
                  vec4 gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
                  vec4 sw10 = step(gw10, vec4(0.0));
                  gx10 -= sw10 * (step(0.0, gx10) - 0.5);
                  gy10 -= sw10 * (step(0.0, gy10) - 0.5);

                  vec4 gx11 = ixy11 / 7.0;
                  vec4 gy11 = floor(gx11) / 7.0;
                  vec4 gz11 = floor(gy11) / 6.0;
                  gx11 = fract(gx11) - 0.5;
                  gy11 = fract(gy11) - 0.5;
                  gz11 = fract(gz11) - 0.5;
                  vec4 gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
                  vec4 sw11 = step(gw11, vec4(0.0));
                  gx11 -= sw11 * (step(0.0, gx11) - 0.5);
                  gy11 -= sw11 * (step(0.0, gy11) - 0.5);

                  vec4 g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
                  vec4 g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
                  vec4 g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
                  vec4 g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
                  vec4 g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
                  vec4 g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
                  vec4 g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
                  vec4 g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
                  vec4 g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
                  vec4 g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
                  vec4 g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
                  vec4 g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
                  vec4 g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
                  vec4 g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
                  vec4 g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
                  vec4 g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

                  vec4 norm00 = taylorInvSqrt(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
                  g0000 *= norm00.x;
                  g0100 *= norm00.y;
                  g1000 *= norm00.z;
                  g1100 *= norm00.w;

                  vec4 norm01 = taylorInvSqrt(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
                  g0001 *= norm01.x;
                  g0101 *= norm01.y;
                  g1001 *= norm01.z;
                  g1101 *= norm01.w;

                  vec4 norm10 = taylorInvSqrt(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
                  g0010 *= norm10.x;
                  g0110 *= norm10.y;
                  g1010 *= norm10.z;
                  g1110 *= norm10.w;

                  vec4 norm11 = taylorInvSqrt(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
                  g0011 *= norm11.x;
                  g0111 *= norm11.y;
                  g1011 *= norm11.z;
                  g1111 *= norm11.w;

                  float n0000 = dot(g0000, Pf0);
                  float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
                  float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
                  float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
                  float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
                  float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
                  float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
                  float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
                  float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
                  float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
                  float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
                  float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
                  float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
                  float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
                  float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
                  float n1111 = dot(g1111, Pf1);

                  vec4 fade_xyzw = fade(Pf0);
                  vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
                  vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
                  vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
                  vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
                  float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
                  return 2.2 * n_xyzw;
                }

                //  <https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83>
                #define NUM_OCTAVES 5
                float fbm(vec4 x) {
                    float freq = 0.6f;
                    float amp = 0.5f;
                    float lacunarity = 1.8715f;
                    float gain = 0.87f;

                    float sum = 0.0f;
                    for (int i = 0; i < NUM_OCTAVES; ++i) {
                        sum += amp * cnoise(x * freq);
                        freq *= lacunarity;
                        amp *= gain;
                    }
                    return sum;
                }

                //  <https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83>
                #define NUM_OCTAVES 5
                float fbm2(vec4 x) {
                    float freq = 1.2f;
                    float amp = 0.2f;
                    float lacunarity = 1.8715f;
                    float gain = 0.65f;

                    float sum = 0.0f;
                    for (int i = 0; i < NUM_OCTAVES; ++i) {
                        sum += amp * cnoise(x * freq);
                        freq *= lacunarity;
                        amp *= gain;
                    }
                    return sum;
                }

                ///////////////////////////////////////////////////////////////////////////////////////////////////////////

                void main () {
                    float oceanHeight = 0.65f;
                    float sandHeight = oceanHeight + 0.015f;
                    float snowHeight = oceanHeight + 0.13f;

                    vec3 X = dFdx(Position);
                    vec3 Y = dFdy(Position);
                    vec3 normal = normalize(cross(X,Y));

                    ///////////////////////////////////////////////////////////////////////////
                    // Color

                    float noise = abs(fbm(vec4(vPos, Altitude)));
                    float snowNoise = clamp(fbm2(vec4(vPos, 1.0f)), -0.01f, 0.01f);

                    float red = step(sandHeight, Altitude) *  step(Altitude, snowHeight  + snowNoise) * noise;
                    float blue = step(Altitude, oceanHeight) * (1.f - (oceanHeight - Altitude));
                    float green = step(sandHeight, Altitude) *  step(Altitude, snowHeight + snowNoise) * (1 - noise);
                    vec3 sandColor = vec3(255.f, 224.f, 158.f) * step(oceanHeight, Altitude) * step(Altitude, sandHeight);
                    
                    float snowHeightNoise = snowHeight + snowNoise;
                    vec3 snowColor = vec3(1.f, 1.f, 1.f);// * max(smoothstep(snowHeight - 0.04, snowHeightNoise, Altitude), step(snowHeightNoise, Altitude)); 
                    
                    vec3 color = mix(vec3(red, green, pow(blue, 5.f)), snowColor, vec3(smoothstep(snowHeight - 0.04, snowHeightNoise, Altitude)));
                    color += sandColor;
                    clamp(color, vec3(0.f), vec3(1.0f));

                    ////////////////////////////////////////////////////////////////////////////
                    // Lighting
                    vec3 sunPos = vec3(0.f, -1.0f, 0.0f);
                    vec3 lightDir = normalize(sunPos - Position);
                    vec3 viewDir  = normalize(-Position);
                        
                    float lightIntensity = 0.6f/length(lightDir);
                    lightDir = normalize(lightDir);

                    vec3 white = vec3(1.0f, 1.0f, 1.0f);

                    //Diffuse part-----------
                    float diff = max(dot(lightDir, normal), 0.0);
                    vec3 diffuse = diff * color * lightIntensity;

                    //specular part-------------
                    vec3 H = normalize(lightDir + viewDir);
                    float NdH = max(dot(H, normal), 0.0);
                    float spec = pow(NdH, shininess);
                    vec3 specular = spec * white;

                    // Ambient-------------
                    vec3 ambient = 0.3*lightIntensity * white;// * texcolor * lightIntensity;
                    
                    vec3 resultLight = ambient + diffuse;// + specular;
                    FragColor = vec4(resultLight, 1.0f);
                }

            "
        }
    )
    .unwrap();

    let mut run = true;

    while run {
        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => {
                    run = false;
                }
                _ => (),
            },
            _ => (),
        });

        let (width, height) = display.get_framebuffer_dimensions();

        let mv: [[f32; 4]; 4] = (Matrix4::from_translation(vec3(0.0, 0.0, -3.0))
            * Matrix4::from_axis_angle(vec3(0.0, 1.0, 0.0), Rad(0.1 * PI)))
        .into();

        let p: [[f32; 4]; 4] =
            perspective(Deg(90.0), width as f32 / height as f32, 0.01, 1000.0).into();

        let uniforms = uniform! {
            MV: mv,
            P: p,
        };

        let params = glium::DrawParameters {
            ..Default::default()
        };

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 0.0);
        target
            .draw(&vertex_buffer, &index_buffer, &program, &uniforms, &params)
            .unwrap();
        target.finish().unwrap();
    }
}
