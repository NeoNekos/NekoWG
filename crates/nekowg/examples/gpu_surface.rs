#![cfg_attr(target_family = "wasm", no_main)]

use nekowg::{
    App, Bounds, Context, GpuBinding, GpuBufferDesc, GpuBufferHandle, GpuComputePassDesc,
    GpuComputeProgramDesc, GpuComputeProgramHandle, GpuDrawCall, GpuExtent, GpuGraphContext,
    GpuInitContext, GpuRenderPassDesc, GpuRenderProgramDesc, GpuRenderProgramHandle,
    GpuResizeContext, GpuSamplerDesc, GpuSamplerHandle, GpuSurfaceRedrawMode, GpuSurfaceRenderer,
    GpuTextureDesc, GpuTextureFormat, GpuTextureHandle, Window, WindowBounds, WindowOptions, div,
    gpu_surface, prelude::*, px, size,
};
use nekowg_platform::application;

const COMPUTE_WGSL: &str = r#"
struct GpuSurfaceFrame {
    metrics: vec4<f32>,
    extent_cursor: vec4<f32>,
    surface_cursor: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: GpuSurfaceFrame;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let extent = max(frame.extent_cursor.xy, vec2<f32>(1.0, 1.0));
    let pixel = gid.xy;
    if (pixel.x >= u32(extent.x) || pixel.y >= u32(extent.y)) {
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5, 0.5)) / extent;
    let time = frame.metrics.x;
    let cursor_uv = frame.surface_cursor.xy / extent;
    let has_cursor = frame.metrics.w;
    let cursor_glow = has_cursor * smoothstep(0.22, 0.0, distance(uv, cursor_uv));
    let swirl = 0.5 + 0.5 * sin((uv.x * 10.0) + time * 1.6);
    let wave = 0.5 + 0.5 * cos((uv.y * 8.0) - time * 1.4);
    let ring = 0.5 + 0.5 * sin(distance(uv, vec2<f32>(0.5, 0.5)) * 18.0 - time * 2.2);
    let color = vec3<f32>(
        0.08 + 0.50 * swirl + 0.30 * cursor_glow,
        0.18 + 0.42 * wave + 0.20 * ring,
        0.24 + 0.48 * ring + 0.22 * cursor_glow
    );
    textureStore(output_tex, vec2<i32>(pixel), vec4<f32>(color, 1.0));
}
"#;

const COMPOSITE_WGSL: &str = r#"
struct GpuSurfaceFrame {
    metrics: vec4<f32>,
    extent_cursor: vec4<f32>,
    surface_cursor: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: GpuSurfaceFrame;

@group(0) @binding(1)
var source_tex: texture_2d<f32>;

@group(0) @binding(2)
var source_sampler: sampler;

struct CompositeParams {
    tint: vec4<f32>,
};

@group(0) @binding(3)
var<uniform> params: CompositeParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );
    let position = positions[vertex_index];
    return VertexOutput(vec4<f32>(position, 0.0, 1.0), uvs[vertex_index]);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(source_tex, source_sampler, input.uv);
    let cursor = frame.surface_cursor.xy / max(frame.extent_cursor.xy, vec2<f32>(1.0, 1.0));
    let glow = smoothstep(0.20, 0.0, distance(input.uv, cursor)) * frame.metrics.w;
    return vec4<f32>(
        sampled.rgb + params.tint.rgb + glow * vec3<f32>(0.55, 0.18, 0.75),
        1.0,
    );
}
"#;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CompositeParams {
    tint: [f32; 4],
}

struct ClearSurfaceRenderer {
    generated: Option<GpuTextureHandle>,
    output: Option<GpuTextureHandle>,
    compute_program: Option<GpuComputeProgramHandle>,
    composite_program: Option<GpuRenderProgramHandle>,
    params_buffer: Option<GpuBufferHandle>,
    sampler: Option<GpuSamplerHandle>,
    extent: GpuExtent,
    time: f32,
}

impl GpuSurfaceRenderer for ClearSurfaceRenderer {
    fn update(&mut self, _next_renderer: Self) {}

    fn init(&mut self, cx: &mut GpuInitContext<'_>) {
        self.compute_program = Some(
            cx.compute_program(
                GpuComputeProgramDesc {
                    label: Some("gpu_surface_generate".into()),
                    wgsl: COMPUTE_WGSL.into(),
                    entry: "cs_main".into(),
                }
                .with_label("gpu_surface_generate"),
            ),
        );
        self.composite_program = Some(
            cx.render_program(
                GpuRenderProgramDesc {
                    label: Some("gpu_surface_composite".into()),
                    wgsl: COMPOSITE_WGSL.into(),
                    vertex_entry: "vs_main".into(),
                    fragment_entry: "fs_main".into(),
                }
                .with_label("gpu_surface_composite"),
            ),
        );
        self.params_buffer = Some(cx.uniform_buffer(GpuBufferDesc {
            label: Some("composite_params".into()),
            size: std::mem::size_of::<CompositeParams>() as u64,
            ..GpuBufferDesc::default()
        }));
        self.sampler = Some(cx.sampler(GpuSamplerDesc::default().with_label("surface_sampler")));
    }

    fn resize(
        &mut self,
        cx: &mut GpuResizeContext<'_>,
        _old: Option<GpuExtent>,
        new: GpuExtent,
        _scale_factor: f32,
    ) {
        self.extent = new;
        self.generated = Some(
            cx.persistent_texture(
                GpuTextureDesc {
                    extent: new,
                    format: GpuTextureFormat::Rgba8Unorm,
                    sampled: true,
                    storage: true,
                    ..GpuTextureDesc::default()
                }
                .with_label("gpu_surface_generated"),
            ),
        );
        self.output = Some(
            cx.persistent_texture(
                GpuTextureDesc {
                    extent: new,
                    format: GpuTextureFormat::Bgra8Unorm,
                    sampled: true,
                    render_attachment: true,
                    ..GpuTextureDesc::default()
                }
                .with_label("gpu_surface_output"),
            ),
        );
    }

    fn prepare(&mut self, frame: &nekowg::GpuFrameContext) {
        self.time = frame.time.as_secs_f32();
    }

    fn encode(&mut self, graph: &mut GpuGraphContext<'_>) {
        let (
            Some(generated),
            Some(output),
            Some(compute_program),
            Some(composite_program),
            Some(params_buffer),
            Some(sampler),
        ) = (
            self.generated,
            self.output,
            self.compute_program,
            self.composite_program,
            self.params_buffer,
            self.sampler,
        )
        else {
            return;
        };

        let tint = 0.5 + 0.5 * self.time.sin();
        graph.write_buffer_value(
            params_buffer,
            &CompositeParams {
                tint: [0.08 * tint, 0.02 * (1.0 - tint), 0.12 * tint, 1.0],
            },
        );

        graph.compute_pass(GpuComputePassDesc {
            label: Some("gpu_surface_generate_pass".into()),
            program: compute_program,
            bindings: vec![GpuBinding::StorageTexture(generated)],
            workgroups: [
                self.extent.width.div_ceil(8),
                self.extent.height.div_ceil(8),
                1,
            ],
        });
        graph.render_pass(GpuRenderPassDesc {
            label: Some("gpu_surface_composite_pass".into()),
            program: composite_program,
            target: output,
            clear_color: None,
            bindings: vec![
                GpuBinding::SampledTexture(generated),
                GpuBinding::Sampler(sampler),
                GpuBinding::UniformBuffer(params_buffer),
            ],
            draw: GpuDrawCall::FullScreenTriangle,
        });
        graph.present(output);
    }
}

struct GpuSurfaceClearDemo;

impl Render for GpuSurfaceClearDemo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .justify_center()
            .items_center()
            .child(
                div()
                    .w(px(480.0))
                    .h(px(320.0))
                    .rounded_xl()
                    .overflow_hidden()
                    .child(
                        gpu_surface(ClearSurfaceRenderer {
                            generated: None,
                            output: None,
                            compute_program: None,
                            composite_program: None,
                            params_buffer: None,
                            sampler: None,
                            extent: GpuExtent::default(),
                            time: 0.0,
                        })
                        .redraw_mode(GpuSurfaceRedrawMode::Animated)
                        .size_full(),
                    ),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(640.0), px(480.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| GpuSurfaceClearDemo),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    let _ = env_logger::try_init();
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    nekowg_platform::web_init();
    run_example();
}
