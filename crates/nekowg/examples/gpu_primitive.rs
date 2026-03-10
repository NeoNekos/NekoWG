#![cfg_attr(target_family = "wasm", no_main)]

use nekowg::wgpu;
use nekowg::{
    App, AppContext, Bounds, ContentMask, Context, DevicePixels, GpuEncodeContext, GpuError,
    GpuNode, GpuNodeId, GpuPhase, GpuPrepareContext, GpuPrimitiveDescriptor, GpuResult, Pixels,
    Render, ScaledPixels, Size, TransformationMatrix, Window, WindowBounds, WindowOptions, black,
    canvas, div, prelude::*, px, size, white,
};
use nekowg_platform::application;

const FULLSCREEN_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    let position = positions[vertex_index];
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = position * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let v = clamp(in.uv.y, 0.0, 1.0);
    let gradient = vec3<f32>(0.11 + 0.18 * v, 0.34 + 0.25 * v, 0.68 + 0.22 * (1.0 - v));
    return vec4<f32>(gradient, 1.0);
}
"#;

#[derive(Default)]
struct FullscreenGradientNode {
    pipeline: Option<wgpu::RenderPipeline>,
}

impl GpuNode for FullscreenGradientNode {
    fn prepare(&mut self, cx: &mut GpuPrepareContext<'_>) -> GpuResult<()> {
        if self.pipeline.is_none() {
            let shader = cx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("nekowg_gpu_example_shader"),
                    source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER.into()),
                });

            let layout = cx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nekowg_gpu_example_layout"),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });

            let pipeline = cx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("nekowg_gpu_example_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: cx.target_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

            self.pipeline = Some(pipeline);
        }

        Ok(())
    }

    fn encode(&mut self, cx: &mut GpuEncodeContext<'_, '_>) -> GpuResult<()> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(GpuError::Backend(
                "pipeline was not initialized before encode".to_string(),
            ));
        };

        if let Some((x, y, w, h)) =
            scissor_for_mask(cx.primitive().content_mask.clone(), cx.frame_target().size)
        {
            let pass = cx.pass();
            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(pipeline);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn on_device_lost(&mut self) {
        self.pipeline = None;
    }
}

fn scissor_for_mask(
    mask: ContentMask<ScaledPixels>,
    target_size: Size<DevicePixels>,
) -> Option<(u32, u32, u32, u32)> {
    let width = target_size.width.0.max(0) as f32;
    let height = target_size.height.0.max(0) as f32;

    let x0 = mask.bounds.origin.x.0.clamp(0.0, width);
    let y0 = mask.bounds.origin.y.0.clamp(0.0, height);
    let x1 = (mask.bounds.origin.x.0 + mask.bounds.size.width.0).clamp(0.0, width);
    let y1 = (mask.bounds.origin.y.0 + mask.bounds.size.height.0).clamp(0.0, height);

    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    let scissor_x = x0.floor() as u32;
    let scissor_y = y0.floor() as u32;
    let scissor_w = (x1.ceil() as i32 - scissor_x as i32).max(0) as u32;
    let scissor_h = (y1.ceil() as i32 - scissor_y as i32).max(0) as u32;
    Some((scissor_x, scissor_y, scissor_w, scissor_h))
}

struct GpuPrimitiveExample {
    node: GpuNodeId,
}

impl Render for GpuPrimitiveExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let node = self.node;
        let supported = window.supports_gpu_primitives();
        let status = if supported {
            "GPU primitive path active"
        } else {
            "GPU primitives unsupported on this backend"
        };

        div()
            .size_full()
            .relative()
            .child(
                canvas(
                    |bounds: Bounds<Pixels>, _, _| bounds,
                    move |bounds, _, window, _| {
                        let descriptor = GpuPrimitiveDescriptor {
                            bounds,
                            node,
                            transformation: TransformationMatrix::unit(),
                            opacity: 1.0,
                            phase: GpuPhase::Underlay,
                        };

                        if let Err(error) = window.push_gpu_primitive(descriptor) {
                            log::warn!("push_gpu_primitive failed: {error}");
                        }
                    },
                )
                .size_full(),
            )
            .child(
                div()
                    .absolute()
                    .top_4()
                    .left_4()
                    .p_2()
                    .rounded_md()
                    .bg(black().opacity(0.55))
                    .text_color(white())
                    .child(status),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(900.), px(560.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| {
                let node = window.insert_gpu_node(FullscreenGradientNode::default());
                let view = cx.new(|_| GpuPrimitiveExample { node });
                window
                    .observe_release(&view, cx, |this, window, _| {
                        if let Err(error) = window.remove_gpu_node(this.node) {
                            log::warn!("remove_gpu_node failed: {error}");
                        }
                    })
                    .detach();
                view
            },
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    env_logger::init();
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    nekowg_platform::web_init();
    run_example();
}
