use cgmath::{perspective, vec3, Deg, Matrix4, Vector3};
use glium::glutin::{dpi::LogicalPosition, Api, GlProfile, GlRequest};
use glium::{
    backend::Facade,
    draw_parameters::{BackfaceCullingMode, Blend},
    framebuffer::{DepthRenderBuffer, SimpleFrameBuffer},
    glutin, implement_vertex,
    index::PrimitiveType,
    texture::{texture2d::Texture2d, DepthFormat, MipmapsOption, UncompressedFloatFormat},
    uniform, Depth, DepthTest, Display, DrawParameters, Program, Surface,
};
use imgui::{im_str, FrameSize, ImGui, ImGuiCond, ImGuiKey, Ui};
use rand::distributions::{Distribution, UnitSphereSurface};
use std::borrow::Cow;
use std::cmp::max;
use std::error;
use std::f32::consts::PI;
use std::fs;
use std::time::{Instant, SystemTime};

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

#[derive(Copy, Clone, Default)]
struct StarVertex {
    pos: [f32; 3],
}
implement_vertex!(StarVertex, pos);

#[derive(Debug, Copy, Clone)]
struct MouseState {
    pos: (i32, i32),
    pressed: (bool, bool, bool),
    wheel: f32,
}

impl MouseState {
    fn new() -> MouseState {
        MouseState {
            pos: (0, 0),
            pressed: (false, false, false),
            wheel: 0.0,
        }
    }
}

fn get_shader_change_time(
    frag_path: &str,
    vert_path: &str,
) -> Result<SystemTime, Box<error::Error>> {
    let metadata_vert = fs::metadata(frag_path)?;
    let metadata_frag = fs::metadata(vert_path)?;
    Ok(max(metadata_vert.modified()?, metadata_frag.modified()?))
}

struct Shader {
    program: Program,
    program_time: SystemTime,
    frag_path: String,
    vert_path: String,
}

impl Shader {
    fn load_shadowmap<F: Facade>(facade: &F, name: &str) -> Result<Shader, Box<error::Error>> {
        let frag_path = format!("shaders/{}_shadowmap.frag", name);
        let vert_path = format!("shaders/{}.vert", name);
        let program_time = get_shader_change_time(&frag_path, &vert_path)?;
        Shader::new(
            facade,
            program_time,
            Cow::Owned(frag_path),
            Cow::Owned(vert_path),
        )
    }

    fn load<F: Facade>(facade: &F, name: &str) -> Result<Shader, Box<error::Error>> {
        let frag_path = format!("shaders/{}.frag", name);
        let vert_path = format!("shaders/{}.vert", name);
        let program_time = get_shader_change_time(&frag_path, &vert_path)?;
        Shader::new(
            facade,
            program_time,
            Cow::Owned(frag_path),
            Cow::Owned(vert_path),
        )
    }

    fn new<F: Facade>(
        facade: &F,
        program_time: SystemTime,
        frag_path: Cow<str>,
        vert_path: Cow<str>,
    ) -> Result<Shader, Box<error::Error>> {
        let input = glium::program::ProgramCreationInput::SourceCode {
            vertex_shader: &fs::read_to_string(&*vert_path)?,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            geometry_shader: None,
            fragment_shader: &fs::read_to_string(&*frag_path)?,
            transform_feedback_varyings: None,
            outputs_srgb: false,
            uses_point_size: true,
        };

        Ok(Shader {
            program: Program::new(facade, input)?,
            program_time: program_time,
            frag_path: frag_path.into_owned(),
            vert_path: vert_path.into_owned(),
        })
    }

    fn reload_if_changed<F: Facade>(&mut self, facade: &F) {
        if let Ok(new_time) = get_shader_change_time(&self.frag_path, &self.vert_path) {
            if new_time > self.program_time {
                match Shader::new(
                    facade,
                    new_time,
                    Cow::Borrowed(&self.frag_path),
                    Cow::Borrowed(&self.vert_path),
                ) {
                    Ok(program) => {
                        *self = program;
                    }
                    Err(e) => {
                        print!("{}", e);
                    }
                }
            }
        }
    }
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

struct State {
    vertex_buffer: glium::VertexBuffer<Vertex>,
    index_buffer: glium::IndexBuffer<u32>,
    star_buffer: glium::VertexBuffer<StarVertex>,

    sun_pos: Vector3<f32>,
    sun_angle: f32,

    planet_program: Shader,
    planet_shadowmap_program: Shader,
    cloud_program: Shader,
    cloud_shadowmap_program: Shader,
    star_program: Shader,

    run: bool,
    right_pressed: bool,
    left_pressed: bool,
    rot: f32,
    start_time: Instant,
    last_time: Instant,
    average_frame_time: f32,
    mouse_state: MouseState,
}

impl State {
    fn new<F: Facade>(facade: &F) -> Result<State, Box<error::Error>> {
        let (vertex_buffer, index_buffer) = {
            const VSEGS: usize = 512;
            const HSEGS: usize = VSEGS * 2;
            const NVERTS: usize = 1 + (VSEGS - 1) * (HSEGS + 1) + 1; // top + middle + bottom
            const NTRIS: usize = HSEGS + (VSEGS - 2) * HSEGS * 2 + HSEGS; // top + middle + bottom

            let mut vertex_list = vec![Default::default(); NVERTS];
            let mut index_list = vec![Default::default(); NTRIS];

            create_sphere(&mut vertex_list, &mut index_list, 0.65, VSEGS);

            let mut flat_index_list = Vec::new();

            for tri in index_list {
                flat_index_list.push(tri.ind[0] as u32);
                flat_index_list.push(tri.ind[1] as u32);
                flat_index_list.push(tri.ind[2] as u32);
            }

            let index_buffer =
                glium::IndexBuffer::new(facade, PrimitiveType::TrianglesList, &flat_index_list)?;

            let vertex_buffer = glium::VertexBuffer::new(facade, &vertex_list)?;

            (vertex_buffer, index_buffer)
        };

        let star_buffer = {
            let mut star_list = Vec::new();

            let sphere = UnitSphereSurface::new();
            let mut rng = rand::thread_rng();

            for _ in 0..10000 {
                let v = sphere.sample(&mut rng);
                star_list.push(StarVertex {
                    pos: [v[0] as f32, v[1] as f32, v[2] as f32],
                });
            }

            let vertex_buffer = glium::VertexBuffer::new(facade, &star_list)?;

            vertex_buffer
        };

        Ok(State {
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            star_buffer: star_buffer,

            sun_pos: vec3(0.0, 0.0, -1.0),
            sun_angle: 0.0,

            planet_program: Shader::load(facade, "planet")?,
            planet_shadowmap_program: Shader::load_shadowmap(facade, "planet")?,
            cloud_program: Shader::load(facade, "cloud")?,
            cloud_shadowmap_program: Shader::load_shadowmap(facade, "cloud")?,
            star_program: Shader::load(facade, "stars")?,

            run: true,
            right_pressed: false,
            left_pressed: false,
            rot: 0.0,
            start_time: Instant::now(),
            last_time: Instant::now(),
            average_frame_time: 0.0,
            mouse_state: MouseState::new(),
        })
    }
}

fn update_ui<'a>(ui: &Ui<'a>, p: &mut State) {
    ui.window(im_str!("Planet"))
        .size((300.0, 100.0), ImGuiCond::FirstUseEver)
        .build(|| {
            ui.text(im_str!(
                "{:.1} fps ({:.1} ms)",
                1.0 / p.average_frame_time,
                p.average_frame_time * 1000.0,
            ));

            if ui
                .slider_float(im_str!("Sun Angle"), &mut p.sun_angle, -180.0, 180.0)
                .build()
            {
                let x = p.sun_angle.to_radians().cos();
                let y = p.sun_angle.to_radians().sin();
                p.sun_pos = vec3(10000.0 * y, 0.0, 10000.0 * -x);
            }

            ui.text(im_str!("Sun Pos: {:?}", &p.sun_pos));
        });
}

fn main() -> Result<(), Box<error::Error>> {
    let mut event_loop = glutin::EventsLoop::new();

    let window = glutin::WindowBuilder::new().with_title("Planet");
    let context = glutin::ContextBuilder::new()
        .with_gl_profile(GlProfile::Core)
        .with_gl(GlRequest::Specific(Api::OpenGl, (4, 3)));
    let display = Display::new(window, context, &event_loop)?;

    let mut imgui = ImGui::init();
    imgui.set_ini_filename(None);
    imgui.style_mut().window_rounding = 0.0;
    imgui.set_imgui_key(ImGuiKey::Tab, 0);
    imgui.set_imgui_key(ImGuiKey::LeftArrow, 1);
    imgui.set_imgui_key(ImGuiKey::RightArrow, 2);
    imgui.set_imgui_key(ImGuiKey::UpArrow, 3);
    imgui.set_imgui_key(ImGuiKey::DownArrow, 4);
    imgui.set_imgui_key(ImGuiKey::PageUp, 5);
    imgui.set_imgui_key(ImGuiKey::PageDown, 6);
    imgui.set_imgui_key(ImGuiKey::Home, 7);
    imgui.set_imgui_key(ImGuiKey::End, 8);
    imgui.set_imgui_key(ImGuiKey::Delete, 9);
    imgui.set_imgui_key(ImGuiKey::Backspace, 10);
    imgui.set_imgui_key(ImGuiKey::Enter, 11);
    imgui.set_imgui_key(ImGuiKey::Escape, 12);
    imgui.set_imgui_key(ImGuiKey::A, 13);
    imgui.set_imgui_key(ImGuiKey::C, 14);
    imgui.set_imgui_key(ImGuiKey::V, 15);
    imgui.set_imgui_key(ImGuiKey::X, 16);
    imgui.set_imgui_key(ImGuiKey::Y, 17);
    imgui.set_imgui_key(ImGuiKey::Z, 18);

    let mut imgui_renderer = imgui_glium_renderer::Renderer::init(&mut imgui, &display).unwrap();

    let lightmap_texture = {
        let (width, height) = display.get_framebuffer_dimensions();
        Texture2d::empty_with_format(
            &display,
            UncompressedFloatFormat::F32F32,
            MipmapsOption::NoMipmap,
            width,
            height,
        )?
    };
    let shadowmap_depthbuffer = {
        let (width, height) = display.get_framebuffer_dimensions();
        DepthRenderBuffer::new(&display, DepthFormat::F32, width, height)?
    };
    let mut shadowmap_framebuffer =
        SimpleFrameBuffer::with_depth_buffer(&display, &lightmap_texture, &shadowmap_depthbuffer)?;

    let mut p = State::new(&display)?;

    while p.run {
        let dt = {
            let new_time = Instant::now();
            let duration = new_time.duration_since(p.last_time);
            p.last_time = new_time;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };

        p.average_frame_time = p.average_frame_time * 0.95 + dt * 0.05;

        p.planet_program.reload_if_changed(&display);
        p.planet_shadowmap_program.reload_if_changed(&display);
        p.cloud_program.reload_if_changed(&display);
        p.cloud_shadowmap_program.reload_if_changed(&display);
        p.star_program.reload_if_changed(&display);

        event_loop.poll_events(|event| {
            use glium::glutin::{
                ElementState, Event, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent,
            };

            match event {
                Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => {
                        p.run = false;
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        use glium::glutin::VirtualKeyCode as Key;

                        let pressed = input.state == ElementState::Pressed;
                        match input.virtual_keycode {
                            Some(Key::Tab) => imgui.set_key(0, pressed),
                            Some(Key::Left) => {
                                imgui.set_key(1, pressed);
                                p.left_pressed = pressed;
                            }
                            Some(Key::Right) => {
                                imgui.set_key(2, pressed);
                                p.right_pressed = pressed;
                            }
                            Some(Key::Up) => imgui.set_key(3, pressed),
                            Some(Key::Down) => imgui.set_key(4, pressed),
                            Some(Key::PageUp) => imgui.set_key(5, pressed),
                            Some(Key::PageDown) => imgui.set_key(6, pressed),
                            Some(Key::Home) => imgui.set_key(7, pressed),
                            Some(Key::End) => imgui.set_key(8, pressed),
                            Some(Key::Delete) => imgui.set_key(9, pressed),
                            Some(Key::Back) => imgui.set_key(10, pressed),
                            Some(Key::Return) => imgui.set_key(11, pressed),
                            Some(Key::Escape) => imgui.set_key(12, pressed),
                            Some(Key::A) => imgui.set_key(13, pressed),
                            Some(Key::C) => imgui.set_key(14, pressed),
                            Some(Key::V) => imgui.set_key(15, pressed),
                            Some(Key::X) => imgui.set_key(16, pressed),
                            Some(Key::Y) => imgui.set_key(17, pressed),
                            Some(Key::Z) => imgui.set_key(18, pressed),
                            Some(Key::LControl) | Some(Key::RControl) => {
                                imgui.set_key_ctrl(pressed)
                            }
                            Some(Key::LShift) | Some(Key::RShift) => imgui.set_key_shift(pressed),
                            Some(Key::LAlt) | Some(Key::RAlt) => imgui.set_key_alt(pressed),
                            Some(Key::LWin) | Some(Key::RWin) => imgui.set_key_super(pressed),
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved {
                        position: LogicalPosition { x, y },
                        ..
                    } => {
                        if x as i32 != 0 && y as i32 != 0 {
                            p.mouse_state.pos = (x as i32, y as i32);
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => match button {
                        MouseButton::Left => {
                            p.mouse_state.pressed.0 = state == ElementState::Pressed
                        }
                        MouseButton::Right => {
                            p.mouse_state.pressed.1 = state == ElementState::Pressed
                        }
                        MouseButton::Middle => {
                            p.mouse_state.pressed.2 = state == ElementState::Pressed
                        }
                        _ => {}
                    },
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        phase: TouchPhase::Moved,
                        ..
                    } => {
                        p.mouse_state.wheel = y;
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::PixelDelta(LogicalPosition { y, .. }),
                        phase: TouchPhase::Moved,
                        ..
                    } => {
                        p.mouse_state.wheel = y as f32;
                    }
                    WindowEvent::ReceivedCharacter(c) => imgui.add_input_character(c),
                    _ => (),
                },
                _ => (),
            }
        });

        {
            let scale = imgui.display_framebuffer_scale();

            imgui.set_mouse_pos(
                p.mouse_state.pos.0 as f32 / scale.0,
                p.mouse_state.pos.1 as f32 / scale.1,
            );

            imgui.set_mouse_down([
                p.mouse_state.pressed.0,
                p.mouse_state.pressed.1,
                p.mouse_state.pressed.2,
                false,
                false,
            ]);

            imgui.set_mouse_wheel(p.mouse_state.wheel / scale.1);
        }

        if p.right_pressed {
            p.rot += dt * 45.0;
        }

        if p.left_pressed {
            p.rot -= dt * 45.0;
        }

        let (width, height) = display.get_framebuffer_dimensions();

        let ui = imgui.frame(FrameSize::new(width as f64, height as f64, 1.0), dt);
        update_ui(&ui, &mut p);

        let planet_matrix = Matrix4::from_translation(vec3(0.0, 0.0, -3.0))
            * Matrix4::from_axis_angle(vec3(0.0, 1.0, 0.0), Deg(p.rot));
        let cloud_matrix = Matrix4::from_translation(vec3(0.0, 0.0, -3.0))
            * Matrix4::from_scale(1.2)
            * Matrix4::from_axis_angle(vec3(0.0, 1.0, 0.0), Deg(p.rot));

        let projection = perspective(Deg(90.0), width as f32 / height as f32, 0.01, 1000.0);

        let time = {
            let duration = Instant::now().duration_since(p.start_time);
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };

        {
            let planet_uniforms = {
                let mv: [[f32; 4]; 4] = planet_matrix.into();
                let projection: [[f32; 4]; 4] = projection.into();
                let sun_pos: [f32; 3] = p.sun_pos.into();
                uniform! {
                    MV: mv,
                    P: projection,
                    sunPos: sun_pos,
                }
            };

            let cloud_uniforms = {
                let mv: [[f32; 4]; 4] = cloud_matrix.into();
                let projection: [[f32; 4]; 4] = projection.into();
                let sun_pos: [f32; 3] = p.sun_pos.into();
                uniform! {
                    MV: mv,
                    P: projection,
                    time: time,
                    sunPos: sun_pos,
                }
            };

            let clockwise_params = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                backface_culling: BackfaceCullingMode::CullClockwise,
                ..Default::default()
            };

            let counter_clockwise_params = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                backface_culling: BackfaceCullingMode::CullCounterClockwise,
                ..Default::default()
            };

            shadowmap_framebuffer.clear_color(0.0, 0.0, 0.0, 0.0);
            shadowmap_framebuffer.clear_depth(1.0);

            shadowmap_framebuffer.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.planet_shadowmap_program.program,
                &planet_uniforms,
                &clockwise_params,
            )?;

            shadowmap_framebuffer.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.cloud_shadowmap_program.program,
                &cloud_uniforms,
                &counter_clockwise_params,
            )?;

            shadowmap_framebuffer.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.cloud_shadowmap_program.program,
                &cloud_uniforms,
                &clockwise_params,
            )?;
        }

        {
            let planet_uniforms = {
                let mv: [[f32; 4]; 4] = planet_matrix.into();
                let projection: [[f32; 4]; 4] = projection.into();
                let sun_pos: [f32; 3] = p.sun_pos.into();
                uniform! {
                    MV: mv,
                    P: projection,
                    sunPos: sun_pos,
                }
            };

            let cloud_uniforms = {
                let mv: [[f32; 4]; 4] = cloud_matrix.into();
                let projection: [[f32; 4]; 4] = projection.into();
                let sun_pos: [f32; 3] = p.sun_pos.into();
                uniform! {
                    MV: mv,
                    P: projection,
                    time: time,
                    sunPos: sun_pos,
                }
            };

            let star_uniforms = {
                let mvp: [[f32; 4]; 4] = (projection * planet_matrix).into();
                uniform! {
                    mvp: mvp,
                }
            };

            let planet_params = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                backface_culling: BackfaceCullingMode::CullClockwise,
                ..Default::default()
            };

            let cloud_params_back = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                blend: Blend::alpha_blending(),
                backface_culling: BackfaceCullingMode::CullCounterClockwise,
                ..Default::default()
            };

            let cloud_params_forward = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                blend: Blend::alpha_blending(),
                backface_culling: BackfaceCullingMode::CullClockwise,
                ..Default::default()
            };

            let star_params = DrawParameters {
                depth: Depth {
                    test: DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 0.0);
            target.clear_depth(1.0);

            target.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.planet_program.program,
                &planet_uniforms,
                &planet_params,
            )?;

            target.draw(
                &p.star_buffer,
                &glium::index::NoIndices(PrimitiveType::Points),
                &p.star_program.program,
                &star_uniforms,
                &star_params,
            )?;

            target.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.cloud_program.program,
                &cloud_uniforms,
                &cloud_params_back,
            )?;

            target.draw(
                &p.vertex_buffer,
                &p.index_buffer,
                &p.cloud_program.program,
                &cloud_uniforms,
                &cloud_params_forward,
            )?;

            target.blit_from_simple_framebuffer(
                &shadowmap_framebuffer,
                &glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: width,
                    height: height,
                },
                &glium::BlitTarget {
                    left: 0,
                    bottom: 0,
                    width: width as i32,
                    height: height as i32,
                },
                glium::uniforms::MagnifySamplerFilter::Linear,
            );

            imgui_renderer.render(&mut target, ui).unwrap();
            target.finish()?;
        }
    }

    Ok(())
}
