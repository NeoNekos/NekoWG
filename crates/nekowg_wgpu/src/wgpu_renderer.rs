use crate::{CompositorGpuHint, WgpuAtlas, WgpuContext};
use anyhow::Context as _;
use bytemuck::{Pod, Zeroable};
use log::warn;
use nekowg::{
    AtlasTextureId, BackdropFilter, Background, Bounds, Corners, DevicePixels, GpuBinding,
    GpuBufferDesc, GpuBufferHandle, GpuBufferUsage, GpuBufferWrite, GpuClearColor,
    GpuComputePassDesc, GpuComputeProgramDesc, GpuComputeProgramHandle, GpuDrawCall,
    GpuFrameContext, GpuGraphOperation, GpuRecordedGraph, GpuRenderPassDesc, GpuRenderProgramDesc,
    GpuRenderProgramHandle, GpuSamplerDesc, GpuSamplerHandle, GpuSpecs, GpuSurfaceExecutionInput,
    GpuTextureDesc, GpuTextureFormat, GpuTextureHandle, Hsla, MonochromeSprite, PaintSurface,
    Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene, Shadow, Size,
    SubpixelSprite, Underline, get_gamma_correction_ratios, point, px, size,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashMap;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
    corner_radii: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSurfaceFrameUniform {
    metrics: [f32; 4],
    extent_cursor: [f32; 4],
    surface_cursor: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    _pad: [f32; 2],
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BackdropBlurInstance {
    bounds: Bounds<ScaledPixels>,
    content_mask: Bounds<ScaledPixels>,
    corner_radii: Corners<ScaledPixels>,
    opacity: f32,
    saturation: f32,
    _pad0: [f32; 2],
    tint: Hsla,
    direction: [f32; 2],
    texel_step: [f32; 2],
    viewport_size: [f32; 2],
    _pad1: [f32; 2],
    weights0: [f32; 4],
    weights1: [f32; 4],
}

impl BackdropBlurInstance {
    fn blur_instance(
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        opacity: f32,
        tint: Hsla,
        saturation: f32,
        direction: [f32; 2],
        texel_step: [f32; 2],
        viewport_size: [f32; 2],
        weights0: [f32; 4],
        weights1: [f32; 4],
    ) -> Self {
        Self {
            bounds,
            content_mask,
            corner_radii,
            opacity,
            saturation,
            _pad0: [0.0; 2],
            tint,
            direction,
            texel_step,
            viewport_size,
            _pad1: [0.0; 2],
            weights0,
            weights1,
        }
    }

    fn blit_instance(bounds: Bounds<ScaledPixels>, viewport_size: [f32; 2]) -> Self {
        Self::blur_instance(
            bounds,
            bounds,
            Corners::default(),
            1.0,
            Hsla::default(),
            1.0,
            [1.0, 0.0],
            [0.0, 0.0],
            viewport_size,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )
    }
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
}

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    shadows: wgpu::RenderPipeline,
    backdrop_blur_h: wgpu::RenderPipeline,
    backdrop_blur_composite: wgpu::RenderPipeline,
    backdrop_blur_blit: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
    gpu_surfaces: wgpu::RenderPipeline,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    instances_with_texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    backdrop_main_texture: Option<wgpu::Texture>,
    backdrop_main_view: Option<wgpu::TextureView>,
    backdrop_temp_texture: Option<wgpu::Texture>,
    backdrop_temp_view: Option<wgpu::TextureView>,
    backdrop_down2_texture: Option<wgpu::Texture>,
    backdrop_down2_view: Option<wgpu::TextureView>,
    backdrop_temp2_texture: Option<wgpu::Texture>,
    backdrop_temp2_view: Option<wgpu::TextureView>,
    backdrop_down4_texture: Option<wgpu::Texture>,
    backdrop_down4_view: Option<wgpu::TextureView>,
    backdrop_temp4_texture: Option<wgpu::Texture>,
    backdrop_temp4_view: Option<wgpu::TextureView>,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
}

struct WgpuGpuSurfaceState {
    textures: HashMap<GpuTextureHandle, WgpuGpuSurfaceTexture>,
    buffers: HashMap<GpuBufferHandle, WgpuGpuSurfaceBuffer>,
    samplers: HashMap<GpuSamplerHandle, WgpuGpuSurfaceSampler>,
    render_programs: HashMap<GpuRenderProgramHandle, WgpuGpuSurfaceRenderProgram>,
    compute_programs: HashMap<GpuComputeProgramHandle, WgpuGpuSurfaceComputeProgram>,
    frame_uniform_buffer: wgpu::Buffer,
    presented: Option<GpuTextureHandle>,
    last_used_frame: u64,
}

struct WgpuGpuSurfaceTexture {
    desc: GpuTextureDesc,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct WgpuGpuSurfaceBuffer {
    desc: GpuBufferDesc,
    buffer: wgpu::Buffer,
}

struct WgpuGpuSurfaceSampler {
    desc: GpuSamplerDesc,
    sampler: wgpu::Sampler,
}

struct WgpuGpuSurfaceRenderProgram {
    desc: GpuRenderProgramDesc,
    module: wgpu::ShaderModule,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

struct WgpuGpuSurfaceComputeProgram {
    desc: GpuComputeProgramDesc,
    pipeline: wgpu::ComputePipeline,
}

impl WgpuGpuSurfaceState {
    fn new(device: &wgpu::Device) -> Self {
        let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_surface_frame_uniform"),
            size: std::mem::size_of::<GpuSurfaceFrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            textures: HashMap::default(),
            buffers: HashMap::default(),
            samplers: HashMap::default(),
            render_programs: HashMap::default(),
            compute_programs: HashMap::default(),
            frame_uniform_buffer,
            presented: None,
            last_used_frame: 0,
        }
    }

    fn sync_textures(
        &mut self,
        device: &wgpu::Device,
        textures: &HashMap<GpuTextureHandle, GpuTextureDesc>,
    ) -> anyhow::Result<()> {
        self.textures.retain(|handle, _| textures.contains_key(handle));
        for (&handle, desc) in textures {
            let needs_recreate = self
                .textures
                .get(&handle)
                .is_none_or(|texture| texture.desc != *desc);
            if needs_recreate {
                self.textures
                    .insert(handle, create_gpu_surface_texture(device, desc)?);
            }
        }
        Ok(())
    }

    fn sync_buffers(
        &mut self,
        device: &wgpu::Device,
        buffers: &HashMap<GpuBufferHandle, GpuBufferDesc>,
    ) -> anyhow::Result<()> {
        self.buffers.retain(|handle, _| buffers.contains_key(handle));
        for (&handle, desc) in buffers {
            let needs_recreate = self
                .buffers
                .get(&handle)
                .is_none_or(|buffer| buffer.desc != *desc);
            if needs_recreate {
                self.buffers
                    .insert(handle, create_gpu_surface_buffer(device, desc)?);
            }
        }
        Ok(())
    }

    fn sync_samplers(
        &mut self,
        device: &wgpu::Device,
        samplers: &HashMap<GpuSamplerHandle, GpuSamplerDesc>,
    ) {
        self.samplers.retain(|handle, _| samplers.contains_key(handle));
        for (&handle, desc) in samplers {
            let needs_recreate = self
                .samplers
                .get(&handle)
                .is_none_or(|sampler| sampler.desc != *desc);
            if needs_recreate {
                self.samplers.insert(
                    handle,
                    WgpuGpuSurfaceSampler {
                        desc: desc.clone(),
                        sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                            label: desc.label.as_ref().map(|label| label.as_ref()),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::MipmapFilterMode::Linear,
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            ..Default::default()
                        }),
                    },
                );
            }
        }
    }

    fn sync_render_programs(
        &mut self,
        device: &wgpu::Device,
        render_programs: &HashMap<GpuRenderProgramHandle, GpuRenderProgramDesc>,
    ) {
        self.render_programs
            .retain(|handle, _| render_programs.contains_key(handle));
        for (&handle, desc) in render_programs {
            let needs_recreate = self
                .render_programs
                .get(&handle)
                .is_none_or(|program| program.desc != *desc);
            if needs_recreate {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: desc.label.as_ref().map(|label| label.as_ref()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(desc.wgsl.as_ref())),
                });
                self.render_programs.insert(
                    handle,
                    WgpuGpuSurfaceRenderProgram {
                        desc: desc.clone(),
                        module,
                        pipelines: HashMap::default(),
                    },
                );
            }
        }
    }

    fn sync_compute_programs(
        &mut self,
        device: &wgpu::Device,
        compute_programs: &HashMap<GpuComputeProgramHandle, GpuComputeProgramDesc>,
    ) {
        self.compute_programs
            .retain(|handle, _| compute_programs.contains_key(handle));
        for (&handle, desc) in compute_programs {
            let needs_recreate = self
                .compute_programs
                .get(&handle)
                .is_none_or(|program| program.desc != *desc);
            if needs_recreate {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: desc.label.as_ref().map(|label| label.as_ref()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(desc.wgsl.as_ref())),
                });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: desc.label.as_ref().map(|label| label.as_ref()),
                    layout: None,
                    module: &module,
                    entry_point: Some(desc.entry.as_ref()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
                self.compute_programs.insert(
                    handle,
                    WgpuGpuSurfaceComputeProgram {
                        desc: desc.clone(),
                        pipeline,
                    },
                );
            }
        }
    }

    fn execute_graph(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &GpuFrameContext,
        scale_factor: f32,
        graph: &GpuRecordedGraph,
    ) -> anyhow::Result<()> {
        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&gpu_surface_frame_uniform(frame, scale_factor)),
        );
        self.apply_buffer_writes(queue, graph.buffer_writes())?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_surface_graph_encoder"),
        });

        for operation in &graph.operations {
            match operation {
                GpuGraphOperation::RenderPass(pass) => {
                    self.execute_render_pass(device, &mut encoder, pass)?
                }
                GpuGraphOperation::ComputePass(pass) => {
                    self.execute_compute_pass(device, &mut encoder, pass)?
                }
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        self.presented = graph.presented;
        Ok(())
    }

    fn apply_buffer_writes(
        &self,
        queue: &wgpu::Queue,
        writes: &[GpuBufferWrite],
    ) -> anyhow::Result<()> {
        for write in writes {
            let buffer = self
                .buffers
                .get(&write.buffer)
                .context("GpuSurface buffer write target missing in WGPU state")?;
            let end = write.offset + write.data.len() as u64;
            if end > buffer.desc.size {
                anyhow::bail!(
                    "GpuSurface buffer write exceeds buffer bounds on WGPU execution: offset {}, size {}, capacity {}",
                    write.offset,
                    write.data.len(),
                    buffer.desc.size
                );
            }
            queue.write_buffer(&buffer.buffer, write.offset, &write.data);
        }
        Ok(())
    }

    fn execute_render_pass(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: &GpuRenderPassDesc,
    ) -> anyhow::Result<()> {
        let (target_desc, target_view) = {
            let target = self
                .textures
                .get(&pass.target)
                .context("GpuSurface render pass target missing in WGPU state")?;
            (target.desc.clone(), target.view.clone())
        };

        if !target_desc.render_attachment {
            anyhow::bail!("GpuSurface render target is not renderable on WGPU");
        }

        let pipeline = self
            .render_pipeline(device, pass.program, gpu_texture_format_to_wgpu(target_desc.format))?
            .clone();
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group_entries = self.bind_group_entries(&self.frame_uniform_buffer, &pass.bindings)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: pass.label.as_ref().map(|label| label.as_ref()),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let load = if let Some(clear) = pass.clear_color {
            wgpu::LoadOp::Clear(gpu_clear_color_to_wgpu(clear))
        } else {
            wgpu::LoadOp::Load
        };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: pass.label.as_ref().map(|label| label.as_ref()),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_viewport(
            0.0,
            0.0,
            target_desc.extent.width.max(1) as f32,
            target_desc.extent.height.max(1) as f32,
            0.0,
            1.0,
        );
        render_pass.set_scissor_rect(
            0,
            0,
            target_desc.extent.width.max(1),
            target_desc.extent.height.max(1),
        );
        match pass.draw {
            GpuDrawCall::FullScreenTriangle => render_pass.draw(0..3, 0..1),
            GpuDrawCall::Triangles {
                vertex_count,
                instance_count,
            } => render_pass.draw(0..vertex_count, 0..instance_count),
        }
        Ok(())
    }

    fn execute_compute_pass(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: &GpuComputePassDesc,
    ) -> anyhow::Result<()> {
        if pass.workgroups.contains(&0) {
            return Ok(());
        }

        let pipeline = self
            .compute_programs
            .get(&pass.program)
            .context("GpuSurface compute program missing in WGPU state")?
            .pipeline
            .clone();
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group_entries = self.bind_group_entries(&self.frame_uniform_buffer, &pass.bindings)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: pass.label.as_ref().map(|label| label.as_ref()),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: pass.label.as_ref().map(|label| label.as_ref()),
            ..Default::default()
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(pass.workgroups[0], pass.workgroups[1], pass.workgroups[2]);
        Ok(())
    }

    fn bind_group_entries<'a>(
        &'a self,
        frame_uniform_buffer: &'a wgpu::Buffer,
        bindings: &[GpuBinding],
    ) -> anyhow::Result<Vec<wgpu::BindGroupEntry<'a>>> {
        let mut entries = Vec::with_capacity(bindings.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: frame_uniform_buffer,
                offset: 0,
                size: NonZeroU64::new(std::mem::size_of::<GpuSurfaceFrameUniform>() as u64),
            }),
        });

        for (index, binding) in bindings.iter().enumerate() {
            let slot = (index + 1) as u32;
            match *binding {
                GpuBinding::UniformBuffer(handle) => {
                    let buffer = self
                        .buffers
                        .get(&handle)
                        .context("GpuSurface uniform buffer binding missing in WGPU state")?;
                    if buffer.desc.usage != GpuBufferUsage::Uniform {
                        anyhow::bail!("GpuSurface uniform binding points to a non-uniform buffer");
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer.buffer,
                            offset: 0,
                            size: None,
                        }),
                    });
                }
                GpuBinding::StorageBuffer(handle) => {
                    let buffer = self
                        .buffers
                        .get(&handle)
                        .context("GpuSurface storage buffer binding missing in WGPU state")?;
                    if buffer.desc.usage != GpuBufferUsage::Storage {
                        anyhow::bail!("GpuSurface storage binding points to a non-storage buffer");
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer.buffer,
                            offset: 0,
                            size: None,
                        }),
                    });
                }
                GpuBinding::SampledTexture(handle) => {
                    let texture = self
                        .textures
                        .get(&handle)
                        .context("GpuSurface sampled texture binding missing in WGPU state")?;
                    if !texture.desc.sampled {
                        anyhow::bail!("GpuSurface sampled binding points to a non-sampled texture");
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    });
                }
                GpuBinding::StorageTexture(handle) => {
                    let texture = self
                        .textures
                        .get(&handle)
                        .context("GpuSurface storage texture binding missing in WGPU state")?;
                    if !texture.desc.storage {
                        anyhow::bail!("GpuSurface storage binding points to a non-storage texture");
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    });
                }
                GpuBinding::Sampler(handle) => {
                    let sampler = self
                        .samplers
                        .get(&handle)
                        .context("GpuSurface sampler binding missing in WGPU state")?;
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot,
                        resource: wgpu::BindingResource::Sampler(&sampler.sampler),
                    });
                }
            }
        }

        Ok(entries)
    }

    fn render_pipeline(
        &mut self,
        device: &wgpu::Device,
        handle: GpuRenderProgramHandle,
        target_format: wgpu::TextureFormat,
    ) -> anyhow::Result<&wgpu::RenderPipeline> {
        let program = self
            .render_programs
            .get_mut(&handle)
            .context("GpuSurface render program missing in WGPU state")?;
        if let std::collections::hash_map::Entry::Vacant(entry) = program.pipelines.entry(target_format)
        {
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: program.desc.label.as_ref().map(|label| label.as_ref()),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &program.module,
                    entry_point: Some(program.desc.vertex_entry.as_ref()),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &program.module,
                    entry_point: Some(program.desc.fragment_entry.as_ref()),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            entry.insert(pipeline);
        }
        Ok(program
            .pipelines
            .get(&target_format)
            .expect("render pipeline must exist after insertion"))
    }

    fn presented_view(&self) -> anyhow::Result<&wgpu::TextureView> {
        let presented = self
            .presented
            .context("GpuSurface present texture missing in WGPU state")?;
        let texture = self
            .textures
            .get(&presented)
            .context("GpuSurface present texture missing in WGPU state")?;
        if !texture.desc.sampled {
            anyhow::bail!("GpuSurface present texture is not sampleable on WGPU");
        }
        Ok(&texture.view)
    }
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    resources: Option<WgpuResources>,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_buffer_capacity: u64,
    max_buffer_size: u64,
    storage_buffer_alignment: u64,
    rendering_params: RenderingParameters,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    last_error: Arc<Mutex<Option<String>>>,
    failed_frame_count: u32,
    gpu_surfaces: HashMap<u64, WgpuGpuSurfaceState>,
    gpu_surface_frame_serial: u64,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl WgpuRenderer {
    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    pub fn new<W: HasWindowHandle + HasDisplayHandle>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;
        let display_handle = window
            .display_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get display handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: display_handle.as_raw(),
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(WgpuContext::instance);

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(instance, &surface, compositor_gpu)?),
        };

        let atlas = Arc::new(WgpuAtlas::new(
            Arc::clone(&context.device),
            Arc::clone(&context.queue),
        ));

        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            atlas,
        )
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        // Configure the surface immediately. The adapter selection process already validated
        // that this adapter can successfully configure this surface.
        surface.configure(&context.device, &surface_config);

        let queue = Arc::clone(&context.queue);
        let dual_source_blending = context.supports_dual_source_blending();

        let rendering_params = RenderingParameters::new(&context.adapter, surface_format);
        let bind_group_layouts = Self::create_bind_group_layouts(&device);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            surface_format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
        );

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_buffer_size = device.limits().max_buffer_size;
        let storage_buffer_alignment = device.limits().min_storage_buffer_offset_alignment as u64;
        let initial_instance_buffer_capacity = 2 * 1024 * 1024;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: initial_instance_buffer_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = context.adapter.get_info();

        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let last_error_clone = Arc::clone(&last_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            let mut guard = last_error_clone.lock().unwrap();
            *guard = Some(error.to_string());
        }));

        let resources = WgpuResources {
            device,
            queue,
            surface,
            pipelines,
            bind_group_layouts,
            atlas_sampler,
            globals_buffer,
            globals_bind_group,
            path_globals_bind_group,
            instance_buffer,
            backdrop_main_texture: None,
            backdrop_main_view: None,
            backdrop_temp_texture: None,
            backdrop_temp_view: None,
            backdrop_down2_texture: None,
            backdrop_down2_view: None,
            backdrop_temp2_texture: None,
            backdrop_temp2_view: None,
            backdrop_down4_texture: None,
            backdrop_down4_view: None,
            backdrop_temp4_texture: None,
            backdrop_temp4_view: None,
            // Defer intermediate texture creation to first draw call via ensure_intermediate_textures().
            // This avoids panics when the device/surface is in an invalid state during initialization.
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
        };

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            resources: Some(resources),
            surface_config,
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_buffer_capacity: initial_instance_buffer_capacity,
            max_buffer_size,
            storage_buffer_alignment,
            rendering_params,
            dual_source_blending,
            adapter_info,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            last_error,
            failed_frame_count: 0,
            gpu_surfaces: HashMap::default(),
            gpu_surface_frame_serial: 0,
            device_lost: context.device_lost_flag(),
        })
    }

    fn create_bind_group_layouts(device: &wgpu::Device) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let storage_buffer_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[storage_buffer_entry(0)],
        });

        let instances_with_texture =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("instances_with_texture_layout"),
                entries: &[
                    storage_buffer_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            instances_with_texture,
            surfaces,
        }
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
    ) -> WgpuPipelines {
        let base_shader_source = include_str!("shaders.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nekowg_shaders"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(base_shader_source)),
        });

        let subpixel_shader_source = include_str!("shaders_subpixel.wgsl");
        let subpixel_shader_module = if dual_source_blending {
            let combined = format!(
                "enable dual_source_blending;\n{base_shader_source}\n{subpixel_shader_source}"
            );
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nekowg_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(combined)),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let color_target_no_blend = wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &[globals_layout, data_layout],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let backdrop_blur_h = create_pipeline(
            "backdrop_blur_h",
            "vs_backdrop_blur",
            "fs_backdrop_blur_h",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target_no_blend.clone())],
            1,
            &shader_module,
        );

        let backdrop_blur_composite = create_pipeline(
            "backdrop_blur_composite",
            "vs_backdrop_blur",
            "fs_backdrop_blur_composite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let backdrop_blur_blit = create_pipeline(
            "backdrop_blur_blit",
            "vs_backdrop_blur",
            "fs_backdrop_blur_blit",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target_no_blend)],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(paths_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances_with_texture,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        let gpu_surfaces = create_pipeline(
            "gpu_surfaces",
            "vs_surface",
            "fs_gpu_surface",
            &layouts.globals,
            &layouts.surfaces,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(blend_mode),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        WgpuPipelines {
            quads,
            shadows,
            backdrop_blur_h,
            backdrop_blur_composite,
            backdrop_blur_blit,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
            gpu_surfaces,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_backdrop_texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width != self.surface_config.width || height != self.surface_config.height {
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            self.surface_config.width = clamped_width.max(1);
            self.surface_config.height = clamped_height.max(1);
            let surface_config = self.surface_config.clone();

            let resources = self.resources_mut();

            // Wait for any in-flight GPU work to complete before destroying textures
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                warn!("Failed to poll device during resize: {e:?}");
            }

            // Destroy old textures before allocating new ones to avoid GPU memory spikes
            if let Some(ref texture) = resources.path_intermediate_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.path_msaa_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_main_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_temp_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_down2_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_temp2_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_down4_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.backdrop_temp4_texture {
                texture.destroy();
            }

            resources
                .surface
                .configure(&resources.device, &surface_config);

            // Invalidate intermediate textures - they will be lazily recreated
            // in draw() after we confirm the surface is healthy. This avoids
            // panics when the device/surface is in an invalid state during resize.
            resources.path_intermediate_texture = None;
            resources.path_intermediate_view = None;
            resources.path_msaa_texture = None;
            resources.path_msaa_view = None;
            resources.backdrop_main_texture = None;
            resources.backdrop_main_view = None;
            resources.backdrop_temp_texture = None;
            resources.backdrop_temp_view = None;
            resources.backdrop_down2_texture = None;
            resources.backdrop_down2_view = None;
            resources.backdrop_temp2_texture = None;
            resources.backdrop_temp2_view = None;
            resources.backdrop_down4_texture = None;
            resources.backdrop_down4_view = None;
            resources.backdrop_temp4_texture = None;
            resources.backdrop_temp4_view = None;
        }
    }

    fn ensure_intermediate_textures(&mut self) {
        if self.resources().path_intermediate_texture.is_some() {
            return;
        }

        let format = self.surface_config.format;
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let path_sample_count = self.rendering_params.path_sample_count;
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(t);
        resources.path_intermediate_view = Some(v);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;
    }

    fn ensure_backdrop_textures(&mut self) {
        if self.resources().backdrop_main_texture.is_some() {
            return;
        }

        let format = self.surface_config.format;
        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);

        let down2_width = width.div_ceil(2);
        let down2_height = height.div_ceil(2);
        let down4_width = width.div_ceil(4);
        let down4_height = height.div_ceil(4);

        let resources = self.resources_mut();

        let (main_t, main_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            width,
            height,
            "backdrop_main",
        );
        resources.backdrop_main_texture = Some(main_t);
        resources.backdrop_main_view = Some(main_v);

        let (temp_t, temp_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            width,
            height,
            "backdrop_temp",
        );
        resources.backdrop_temp_texture = Some(temp_t);
        resources.backdrop_temp_view = Some(temp_v);

        let (down2_t, down2_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            down2_width,
            down2_height,
            "backdrop_down2",
        );
        resources.backdrop_down2_texture = Some(down2_t);
        resources.backdrop_down2_view = Some(down2_v);

        let (temp2_t, temp2_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            down2_width,
            down2_height,
            "backdrop_temp2",
        );
        resources.backdrop_temp2_texture = Some(temp2_t);
        resources.backdrop_temp2_view = Some(temp2_v);

        let (down4_t, down4_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            down4_width,
            down4_height,
            "backdrop_down4",
        );
        resources.backdrop_down4_texture = Some(down4_t);
        resources.backdrop_down4_view = Some(down4_v);

        let (temp4_t, temp4_v) = Self::create_backdrop_texture(
            &resources.device,
            format,
            down4_width,
            down4_height,
            "backdrop_temp4",
        );
        resources.backdrop_temp4_texture = Some(temp4_t);
        resources.backdrop_temp4_view = Some(temp4_v);
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        if new_alpha_mode != self.surface_config.alpha_mode {
            self.surface_config.alpha_mode = new_alpha_mode;
            let surface_config = self.surface_config.clone();
            let path_sample_count = self.rendering_params.path_sample_count;
            let dual_source_blending = self.dual_source_blending;
            let resources = self.resources_mut();
            resources
                .surface
                .configure(&resources.device, &surface_config);
            resources.pipelines = Self::create_pipelines(
                &resources.device,
                &resources.bind_group_layouts,
                surface_config.format,
                surface_config.alpha_mode,
                path_sample_count,
                dual_source_blending,
            );
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    fn prune_stale_gpu_surfaces(&mut self) {
        const RETAIN_FRAMES: u64 = 120;
        let min_frame = self.gpu_surface_frame_serial.saturating_sub(RETAIN_FRAMES);
        self.gpu_surfaces
            .retain(|_, state| state.last_used_frame >= min_frame);
    }

    pub fn paint_gpu_surface(
        &mut self,
        input: GpuSurfaceExecutionInput<'_>,
    ) -> anyhow::Result<Option<PaintSurface>> {
        let Some(_) = input.graph.presented else {
            return Ok(None);
        };
        let frame = input
            .frame
            .context("GpuSurface WGPU executor requires a prepared frame context")?;
        let (device, queue) = {
            let resources = self.resources();
            (Arc::clone(&resources.device), Arc::clone(&resources.queue))
        };
        let state = self
            .gpu_surfaces
            .entry(input.surface_id)
            .or_insert_with(|| WgpuGpuSurfaceState::new(&device));
        state.last_used_frame = self.gpu_surface_frame_serial + 1;
        state.sync_textures(&device, input.textures)?;
        state.sync_buffers(&device, input.buffers)?;
        state.sync_samplers(&device, input.samplers);
        state.sync_render_programs(&device, input.render_programs);
        state.sync_compute_programs(&device, input.compute_programs);
        state.execute_graph(&device, &queue, frame, input.scale_factor, input.graph)?;
        let _ = state.presented_view()?;

        Ok(Some(PaintSurface {
            order: 0,
            bounds: input.bounds,
            content_mask: input.content_mask,
            corner_radii: input.corner_radii,
            #[cfg(target_os = "macos")]
            image_buffer: None,
            #[cfg(target_os = "windows")]
            texture_view: None,
            gpu_surface_id: Some(input.surface_id),
        }))
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) {
        let last_error = self.last_error.lock().unwrap().take();
        if let Some(error) = last_error {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 20): {error}",
                self.failed_frame_count
            );
            if self.failed_frame_count > 20 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            }
        } else {
            self.failed_frame_count = 0;
        }

        self.atlas.before_frame();
        self.gpu_surface_frame_serial = self.gpu_surface_frame_serial.wrapping_add(1);
        self.prune_stale_gpu_surfaces();

        let texture_result = self.resources().surface.get_current_texture();
        let frame = match texture_result {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let surface_config = self.surface_config.clone();
                let resources = self.resources_mut();
                resources
                    .surface
                    .configure(&resources.device, &surface_config);
                return;
            }
            Err(e) => {
                *self.last_error.lock().unwrap() =
                    Some(format!("Failed to acquire surface texture: {e}"));
                return;
            }
        };

        // Now that we know the surface is healthy, ensure intermediate textures exist
        self.ensure_intermediate_textures();

        let use_backdrop = !scene.backdrop_filters.is_empty();
        if use_backdrop {
            self.ensure_backdrop_textures();
        }

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let main_view = if use_backdrop {
            self.resources()
                .backdrop_main_view
                .as_ref()
                .expect("backdrop main view missing")
                .clone()
        } else {
            frame_view.clone()
        };

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            _pad: [0.0; 2],
        };

        let globals = GlobalParams {
            viewport_size: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            premultiplied_alpha: if self.surface_config.alpha_mode
                == wgpu::CompositeAlphaMode::PreMultiplied
            {
                1
            } else {
                0
            },
            pad: 0,
        };

        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };

        {
            let resources = self.resources();
            resources.queue.write_buffer(
                &resources.globals_buffer,
                0,
                bytemuck::bytes_of(&globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.path_globals_offset,
                bytemuck::bytes_of(&path_globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.gamma_offset,
                bytemuck::bytes_of(&gamma_params),
            );
        }

        loop {
            let mut instance_offset: u64 = 0;
            let mut overflow = false;

            let mut encoder =
                self.resources()
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("main_encoder"),
                    });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &main_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                for batch in scene.batches() {
                    let ok = match batch {
                        PrimitiveBatch::Quads(range) => {
                            self.draw_quads(&scene.quads[range], &mut instance_offset, &mut pass)
                        }
                        PrimitiveBatch::Shadows(range) => self.draw_shadows(
                            &scene.shadows[range],
                            &mut instance_offset,
                            &mut pass,
                        ),
                        PrimitiveBatch::BackdropFilters(range) => {
                            drop(pass);
                            let did_draw = self.draw_backdrop_filters(
                                &mut encoder,
                                &scene.backdrop_filters[range],
                                &mut instance_offset,
                            );
                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main_pass_continued"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &main_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });
                            did_draw
                        }
                        PrimitiveBatch::Paths(range) => {
                            let paths = &scene.paths[range];
                            if paths.is_empty() {
                                continue;
                            }

                            drop(pass);

                            let did_draw = self.draw_paths_to_intermediate(
                                &mut encoder,
                                paths,
                                &mut instance_offset,
                            );

                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main_pass_continued"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &main_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });

                            if did_draw {
                                self.draw_paths_from_intermediate(
                                    paths,
                                    &mut instance_offset,
                                    &mut pass,
                                )
                            } else {
                                false
                            }
                        }
                        PrimitiveBatch::Underlines(range) => self.draw_underlines(
                            &scene.underlines[range],
                            &mut instance_offset,
                            &mut pass,
                        ),
                        PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                            .draw_monochrome_sprites(
                                &scene.monochrome_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::SubpixelSprites { texture_id, range } => self
                            .draw_subpixel_sprites(
                                &scene.subpixel_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                            .draw_polychrome_sprites(
                                &scene.polychrome_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                            &scene.surfaces[range],
                            &mut instance_offset,
                            &mut pass,
                        ),
                    };
                    if !ok {
                        overflow = true;
                        break;
                    }
                }
            }

            if !overflow && use_backdrop {
                if let Some(view) = self.resources().backdrop_main_view.as_ref() {
                    let full_bounds = Bounds {
                        origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                        size: size(
                            ScaledPixels(self.surface_config.width as f32),
                            ScaledPixels(self.surface_config.height as f32),
                        ),
                    };
                    let instance = BackdropBlurInstance::blit_instance(
                        full_bounds,
                        [
                            self.surface_config.width as f32,
                            self.surface_config.height as f32,
                        ],
                    );
                    self.blit_backdrop_to_frame(
                        &mut encoder,
                        &frame_view,
                        view,
                        &instance,
                        &mut instance_offset,
                    );
                }
            }

            if overflow {
                drop(encoder);
                if self.instance_buffer_capacity >= self.max_buffer_size {
                    log::error!(
                        "instance buffer size grew too large: {}",
                        self.instance_buffer_capacity
                    );
                    frame.present();
                    return;
                }
                self.grow_instance_buffer();
                continue;
            }

            self.resources()
                .queue
                .submit(std::iter::once(encoder.finish()));
            frame.present();
            return;
        }
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(quads) };
        self.draw_instances(
            data,
            quads.len() as u32,
            &self.resources().pipelines.quads,
            instance_offset,
            pass,
        )
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(shadows) };
        self.draw_instances(
            data,
            shadows.len() as u32,
            &self.resources().pipelines.shadows,
            instance_offset,
            pass,
        )
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(underlines) };
        self.draw_instances(
            data,
            underlines.len() as u32,
            &self.resources().pipelines.underlines,
            instance_offset,
            pass,
        )
    }

    fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.resources().pipelines.mono_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        let resources = self.resources();
        let pipeline = resources
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&resources.pipelines.mono_sprites);
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            pipeline,
            instance_offset,
            pass,
        )
    }

    fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.resources().pipelines.poly_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_surfaces(
        &self,
        surfaces: &[PaintSurface],
        _instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if surfaces.is_empty() {
            return true;
        }

        let resources = self.resources();
        for surface in surfaces {
            let Some(surface_id) = surface.gpu_surface_id else {
                continue;
            };
            let Some(state) = self.gpu_surfaces.get(&surface_id) else {
                continue;
            };
            let Ok(texture_view) = state.presented_view() else {
                continue;
            };

            let params = SurfaceParams {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.bounds.into(),
                corner_radii: [
                    surface.corner_radii.top_left.0,
                    surface.corner_radii.top_right.0,
                    surface.corner_radii.bottom_right.0,
                    surface.corner_radii.bottom_left.0,
                ],
            };
            let params_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surface_params_buffer"),
                size: std::mem::size_of::<SurfaceParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            resources
                .queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            let bind_group = resources
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_surface_bind_group"),
                    layout: &resources.bind_group_layouts.surfaces,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &params_buffer,
                                offset: 0,
                                size: NonZeroU64::new(std::mem::size_of::<SurfaceParams>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        // Binding 2 is unused by the GPU surface path, but required by layout.
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                        },
                    ],
                });
            pass.set_pipeline(&resources.pipelines.gpu_surfaces);
            pass.set_bind_group(0, &resources.globals_bind_group, &[]);
            pass.set_bind_group(1, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }

        true
    }

    fn draw_instances(
        &self,
        data: &[u8],
        instance_count: u32,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(offset, size),
                }],
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    fn draw_instances_with_texture(
        &self,
        data: &[u8],
        instance_count: u32,
        texture_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        self.draw_instances_with_texture_scissored(
            data,
            instance_count,
            texture_view,
            pipeline,
            instance_offset,
            pass,
            None,
        )
    }

    fn draw_instances_with_texture_scissored(
        &self,
        data: &[u8],
        instance_count: u32,
        texture_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
        scissor: Option<(u32, u32, u32, u32)>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances_with_texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instance_binding(offset, size),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            });
        if let Some((x, y, width, height)) = scissor {
            pass.set_scissor_rect(x, y, width, height);
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn scale_bounds(bounds: Bounds<ScaledPixels>, factor: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(bounds.origin.x * factor, bounds.origin.y * factor),
            size: size(bounds.size.width * factor, bounds.size.height * factor),
        }
    }

    fn scissor_from_bounds(
        bounds: Bounds<ScaledPixels>,
        target_size: [u32; 2],
    ) -> Option<(u32, u32, u32, u32)> {
        let min_x = bounds.origin.x.0.max(0.0).floor() as i32;
        let min_y = bounds.origin.y.0.max(0.0).floor() as i32;
        let max_x = (bounds.origin.x.0 + bounds.size.width.0)
            .min(target_size[0] as f32)
            .ceil() as i32;
        let max_y = (bounds.origin.y.0 + bounds.size.height.0)
            .min(target_size[1] as f32)
            .ceil() as i32;

        if max_x <= min_x || max_y <= min_y {
            return None;
        }

        Some((
            min_x as u32,
            min_y as u32,
            (max_x - min_x) as u32,
            (max_y - min_y) as u32,
        ))
    }

    fn blur_passes(radius: f32, scale: u32) -> u32 {
        if scale == 1 {
            return 1;
        }
        let radius_ds = (radius / scale as f32).max(0.0);
        let ratio = radius_ds / 8.0;
        let passes = (ratio * ratio).ceil() as u32;
        passes.clamp(1, 6)
    }

    fn gaussian_weights(radius: f32, scale: f32) -> ([f32; 4], [f32; 4], f32) {
        let radius_ds = (radius / scale).max(0.0);
        let step = (radius_ds / 4.0).max(1.0);
        let sigma = (radius_ds / 2.0).max(0.5);

        let mut weights = [0.0f32; 5];
        for i in 0..5 {
            let x = i as f32 * step;
            weights[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        }
        let mut sum = weights[0];
        for i in 1..5 {
            sum += 2.0 * weights[i];
        }
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        (
            [weights[0], weights[1], weights[2], weights[3]],
            [weights[4], 0.0, 0.0, 0.0],
            step,
        )
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let first_path = &paths[0];
        let sprites: Vec<PathSprite> = if paths.last().map(|p| &p.order) == Some(&first_path.order)
        {
            paths
                .iter()
                .map(|p| PathSprite {
                    bounds: p.clipped_bounds(),
                })
                .collect()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return true;
        };

        let sprite_data = unsafe { Self::instance_bytes(&sprites) };
        self.draw_instances_with_texture(
            sprite_data,
            sprites.len() as u32,
            path_intermediate_view,
            &resources.pipelines.paths,
            instance_offset,
            pass,
        )
    }

    fn draw_paths_to_intermediate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
    ) -> bool {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return true;
        }

        let vertex_data = unsafe { Self::instance_bytes(&vertices) };
        let Some((vertex_offset, vertex_size)) =
            self.write_to_instance_buffer(instance_offset, vertex_data)
        else {
            return false;
        };

        let resources = self.resources();
        let data_bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("path_rasterization_bind_group"),
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(vertex_offset, vertex_size),
                }],
            });

        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return true;
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &data_bind_group, &[]);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        true
    }

    fn draw_backdrop_filters(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        filters: &[BackdropFilter],
        instance_offset: &mut u64,
    ) -> bool {
        if filters.is_empty() {
            return true;
        }

        let resources = self.resources();
        let Some(main_view) = resources.backdrop_main_view.as_ref() else {
            return true;
        };

        let full_size = [
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        ];
        let full_size_u32 = [self.surface_config.width, self.surface_config.height];

        for filter in filters {
            let radius = filter.blur_radius.as_f32().max(0.0);
            if radius <= 0.0 {
                continue;
            }

            let scale = if radius <= 8.0 {
                1u32
            } else if radius <= 16.0 {
                2u32
            } else {
                4u32
            };

            let kernel = (radius * 1.5).ceil();
            let expanded_bounds = filter.bounds.dilate(ScaledPixels::from(kernel + 2.0));
            let expanded_scaled = if scale == 1 {
                expanded_bounds
            } else {
                Self::scale_bounds(expanded_bounds, 1.0 / scale as f32)
            };

            let target_size_u32 = match scale {
                1 => full_size_u32,
                2 => [full_size_u32[0].div_ceil(2), full_size_u32[1].div_ceil(2)],
                _ => [full_size_u32[0].div_ceil(4), full_size_u32[1].div_ceil(4)],
            };
            let target_size = [target_size_u32[0] as f32, target_size_u32[1] as f32];

            let scissor_expanded = Self::scissor_from_bounds(expanded_scaled, target_size_u32);
            let scissor_full = Self::scissor_from_bounds(filter.bounds, full_size_u32);

            let passes = Self::blur_passes(radius, scale);
            let pass_radius = if passes > 1 {
                radius / (passes as f32).sqrt()
            } else {
                radius
            };
            let (weights0, weights1, step) = Self::gaussian_weights(pass_radius, scale as f32);
            let texel_step = [step / target_size[0], step / target_size[1]];

            let (down_view, temp_view) = match scale {
                1 => (
                    None,
                    resources
                        .backdrop_temp_view
                        .as_ref()
                        .expect("backdrop temp view missing"),
                ),
                2 => (
                    Some(
                        resources
                            .backdrop_down2_view
                            .as_ref()
                            .expect("backdrop down2 view missing"),
                    ),
                    resources
                        .backdrop_temp2_view
                        .as_ref()
                        .expect("backdrop temp2 view missing"),
                ),
                _ => (
                    Some(
                        resources
                            .backdrop_down4_view
                            .as_ref()
                            .expect("backdrop down4 view missing"),
                    ),
                    resources
                        .backdrop_temp4_view
                        .as_ref()
                        .expect("backdrop temp4 view missing"),
                ),
            };

            if scale > 1 {
                let Some(scissor) = scissor_expanded else {
                    continue;
                };
                let blit_instance =
                    BackdropBlurInstance::blit_instance(expanded_scaled, target_size);
                let blit_instances = [blit_instance];
                let blit_data = unsafe { Self::instance_bytes(&blit_instances) };
                let dest_view = down_view.expect("downsample view missing");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("backdrop_blur_downsample"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dest_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                let ok = self.draw_instances_with_texture_scissored(
                    blit_data,
                    1,
                    main_view,
                    &resources.pipelines.backdrop_blur_blit,
                    instance_offset,
                    &mut pass,
                    Some(scissor),
                );
                if !ok {
                    return false;
                }
            }

            let mut source_view = down_view.unwrap_or(main_view);
            let Some(scissor) = scissor_expanded else {
                continue;
            };
            for pass_index in 0..passes {
                let blur_instance = BackdropBlurInstance::blur_instance(
                    expanded_scaled,
                    expanded_scaled,
                    Corners::default(),
                    1.0,
                    Hsla::default(),
                    1.0,
                    [1.0, 0.0],
                    texel_step,
                    target_size,
                    weights0,
                    weights1,
                );
                let blur_instances = [blur_instance];
                let blur_data = unsafe { Self::instance_bytes(&blur_instances) };
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("backdrop_blur_h"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: temp_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    let ok = self.draw_instances_with_texture_scissored(
                        blur_data,
                        1,
                        source_view,
                        &resources.pipelines.backdrop_blur_h,
                        instance_offset,
                        &mut pass,
                        Some(scissor),
                    );
                    if !ok {
                        return false;
                    }
                }

                if pass_index + 1 == passes {
                    let Some(scissor_full) = scissor_full else {
                        continue;
                    };
                    let composite_instance = BackdropBlurInstance::blur_instance(
                        filter.bounds,
                        filter.content_mask.bounds,
                        filter.corner_radii,
                        filter.opacity,
                        filter.tint,
                        filter.saturation,
                        [0.0, 1.0],
                        texel_step,
                        full_size,
                        weights0,
                        weights1,
                    );
                    let composite_instances = [composite_instance];
                    let composite_data = unsafe { Self::instance_bytes(&composite_instances) };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("backdrop_blur_composite"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: main_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    let ok = self.draw_instances_with_texture_scissored(
                        composite_data,
                        1,
                        temp_view,
                        &resources.pipelines.backdrop_blur_composite,
                        instance_offset,
                        &mut pass,
                        Some(scissor_full),
                    );
                    if !ok {
                        return false;
                    }
                } else {
                    let Some(down_view) = down_view else {
                        return false;
                    };
                    let blur_instance = BackdropBlurInstance::blur_instance(
                        expanded_scaled,
                        expanded_scaled,
                        Corners::default(),
                        1.0,
                        Hsla::default(),
                        1.0,
                        [0.0, 1.0],
                        texel_step,
                        target_size,
                        weights0,
                        weights1,
                    );
                    let blur_instances = [blur_instance];
                    let blur_data = unsafe { Self::instance_bytes(&blur_instances) };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("backdrop_blur_v"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: down_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    let ok = self.draw_instances_with_texture_scissored(
                        blur_data,
                        1,
                        temp_view,
                        &resources.pipelines.backdrop_blur_h,
                        instance_offset,
                        &mut pass,
                        Some(scissor),
                    );
                    if !ok {
                        return false;
                    }
                    source_view = down_view;
                }
            }
        }

        true
    }

    fn blit_backdrop_to_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
        source_view: &wgpu::TextureView,
        instance: &BackdropBlurInstance,
        instance_offset: &mut u64,
    ) {
        let data = unsafe { Self::instance_bytes(std::slice::from_ref(instance)) };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("backdrop_blit_frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        let _ = self.draw_instances_with_texture(
            data,
            1,
            source_view,
            &self.resources().pipelines.backdrop_blur_blit,
            instance_offset,
            &mut pass,
        );
    }

    fn grow_instance_buffer(&mut self) {
        let new_capacity = (self.instance_buffer_capacity * 2).min(self.max_buffer_size);
        log::info!("increased instance buffer size to {}", new_capacity);
        let resources = self.resources_mut();
        resources.instance_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: new_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_buffer_capacity = new_capacity;
    }

    fn write_to_instance_buffer(
        &self,
        instance_offset: &mut u64,
        data: &[u8],
    ) -> Option<(u64, NonZeroU64)> {
        let offset = (*instance_offset).next_multiple_of(self.storage_buffer_alignment);
        let size = (data.len() as u64).max(16);
        if offset + size > self.instance_buffer_capacity {
            return None;
        }
        let resources = self.resources();
        resources
            .queue
            .write_buffer(&resources.instance_buffer, offset, data);
        *instance_offset = offset + size;
        Some((offset, NonZeroU64::new(size).expect("size is at least 16")))
    }

    fn instance_binding(&self, offset: u64, size: NonZeroU64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.resources().instance_buffer,
            offset,
            size: Some(size),
        })
    }

    pub fn destroy(&mut self) {
        // wgpu resources are automatically cleaned up when dropped
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Recovers from a lost GPU device by recreating the renderer with a new context.
    ///
    /// Call this after detecting `device_lost()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    pub fn recover(
        &mut self,
        raw_display_handle: raw_window_handle::RawDisplayHandle,
        raw_window_handle: raw_window_handle::RawWindowHandle,
    ) -> anyhow::Result<()> {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        // Check if another window already recovered the context
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost());

        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");

            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.resources = None;
            *gpu_context.borrow_mut() = None;

            // Wait for GPU driver to stabilize (350ms copied from windows :shrug:)
            std::thread::sleep(std::time::Duration::from_millis(350));

            let instance = WgpuContext::instance();
            let surface = create_surface(&instance, raw_display_handle, raw_window_handle)?;
            let new_context = WgpuContext::new(instance, &surface, self.compositor_gpu)?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, raw_display_handle, raw_window_handle)?
        };

        let config = WgpuSurfaceConfig {
            size: nekowg::Size {
                width: nekowg::DevicePixels(self.surface_config.width as i32),
                height: nekowg::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
        };
        let gpu_context = Rc::clone(gpu_context);
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.resources = None;
        self.atlas
            .handle_device_lost(Arc::clone(&context.device), Arc::clone(&context.queue));

        *self = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            surface,
            config,
            self.compositor_gpu,
            self.atlas.clone(),
        )?;

        log::info!("GPU recovery complete");
        Ok(())
    }
}

fn create_gpu_surface_texture(
    device: &wgpu::Device,
    desc: &GpuTextureDesc,
) -> anyhow::Result<WgpuGpuSurfaceTexture> {
    let mut usage = wgpu::TextureUsages::empty();
    if desc.sampled {
        usage |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if desc.storage {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if desc.render_attachment {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if desc.copy_src {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    if desc.copy_dst {
        usage |= wgpu::TextureUsages::COPY_DST;
    }
    if usage.is_empty() {
        anyhow::bail!("GpuSurface texture usage cannot be empty for WGPU");
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: desc.label.as_ref().map(|label| label.as_ref()),
        size: wgpu::Extent3d {
            width: desc.extent.width.max(1),
            height: desc.extent.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu_texture_format_to_wgpu(desc.format),
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(WgpuGpuSurfaceTexture {
        desc: desc.clone(),
        _texture: texture,
        view,
    })
}

fn create_gpu_surface_buffer(
    device: &wgpu::Device,
    desc: &GpuBufferDesc,
) -> anyhow::Result<WgpuGpuSurfaceBuffer> {
    let usage = match desc.usage {
        GpuBufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        GpuBufferUsage::Storage => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    };
    let size = desc.size.max(16);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: desc.label.as_ref().map(|label| label.as_ref()),
        size,
        usage,
        mapped_at_creation: false,
    });
    Ok(WgpuGpuSurfaceBuffer {
        desc: desc.clone(),
        buffer,
    })
}

fn gpu_texture_format_to_wgpu(format: GpuTextureFormat) -> wgpu::TextureFormat {
    match format {
        GpuTextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        GpuTextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        GpuTextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        GpuTextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        GpuTextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}

fn gpu_surface_frame_uniform(frame: &GpuFrameContext, scale_factor: f32) -> GpuSurfaceFrameUniform {
    let surface_cursor_position = frame
        .surface_cursor_position
        .unwrap_or(point(px(0.0), px(0.0)))
        .scale(scale_factor);
    let cursor_position = frame.cursor_position.scale(scale_factor);
    GpuSurfaceFrameUniform {
        metrics: [
            frame.time.as_secs_f32(),
            frame.delta_time.as_secs_f32(),
            frame.frame_index as f32,
            if frame.surface_cursor_position.is_some() {
                1.0
            } else {
                0.0
            },
        ],
        extent_cursor: [
            frame.extent.width as f32,
            frame.extent.height as f32,
            cursor_position.x.as_f32(),
            cursor_position.y.as_f32(),
        ],
        surface_cursor: [
            surface_cursor_position.x.as_f32(),
            surface_cursor_position.y.as_f32(),
            0.0,
            0.0,
        ],
    }
}

fn gpu_clear_color_to_wgpu(clear: GpuClearColor) -> wgpu::Color {
    wgpu::Color {
        r: clear.r as f64,
        g: clear.g as f64,
        b: clear.b as f64,
        a: clear.a as f64,
    }
}

fn create_surface(
    instance: &wgpu::Instance,
    raw_display_handle: raw_window_handle::RawDisplayHandle,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}
