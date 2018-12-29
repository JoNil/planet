use glium::glutin::{Api, GlProfile, GlRequest};
use glium::index::PrimitiveType;
use glium::{glutin, implement_vertex, program, uniform, Surface};

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
    let pi = 3.14159265358979323846;
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
        let theta = ((j + 1) as f32) / (vsegs as f32) * pi;
        let z = theta.cos();
        let r = theta.sin();
        for i in 0..hsegs {
            let phi = i as f32 / hsegs as f32 * 2.0 * pi;
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
    let window = glutin::WindowBuilder::new().with_title("Cool yo! ;P");
    let context = glutin::ContextBuilder::new()
        .with_gl_profile(GlProfile::Core)
        .with_gl(GlRequest::Specific(Api::OpenGl, (4, 3)));
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let (vertex_buffer, index_buffer) = {
        const VSEGS: usize = 32;
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
        140 => {
            vertex: "
                #version 140
                uniform mat4 matrix;
                in vec3 pos;
                in vec3 normal;
                in vec2 tex;
                void main() {
                    gl_Position = vec4(pos, 1.0) * matrix;
                }
            ",

            fragment: "
                #version 140
                
                out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
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

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]
            ]
        };

        let params = glium::DrawParameters {
            polygon_mode: glium::PolygonMode::Line,
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
