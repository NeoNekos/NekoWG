use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;
use nekowg::{
    AtlasTextureId, BackdropFilter, Background, Bounds, ContentMask, Corners, DevicePixels,
    GpuBinding, GpuBufferDesc, GpuBufferHandle, GpuBufferUsage, GpuBufferWrite, GpuClearColor,
    GpuComputePassDesc, GpuComputeProgramDesc, GpuComputeProgramHandle, GpuDrawCall,
    GpuFrameContext, GpuGraphOperation, GpuRecordedGraph, GpuRenderPassDesc, GpuRenderProgramDesc,
    GpuRenderProgramHandle, GpuSamplerDesc, GpuSamplerHandle, GpuSurfaceExecutionInput,
    GpuTextureDesc, GpuTextureFormat, GpuTextureHandle, Hsla, MonochromeSprite, PaintSurface, Path,
    Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene, Shadow, Size, Underline,
    point, px, size,
};

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    RenderPassColorAttachmentDescriptorRef,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{
    cell::Cell,
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    mem, ptr,
    sync::Arc,
};

// Exported to metal
pub(crate) type PointF = nekowg::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

pub(crate) type Context = Arc<Mutex<InstanceBufferPool>>;
pub(crate) type Renderer = MetalRenderer;

pub(crate) unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: nekowg::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: metal::MetalLayer,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_h_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_composite_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_blit_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    backdrop_main_texture: Option<metal::Texture>,
    backdrop_temp_texture: Option<metal::Texture>,
    backdrop_down2_texture: Option<metal::Texture>,
    backdrop_temp2_texture: Option<metal::Texture>,
    backdrop_down4_texture: Option<metal::Texture>,
    backdrop_temp4_texture: Option<metal::Texture>,
    path_sample_count: u32,
    gpu_surfaces_pipeline_state: metal::RenderPipelineState,
    gpu_surfaces: HashMap<u64, MetalGpuSurfaceState>,
    gpu_surface_frame_serial: u64,
}

#[repr(C)]
struct MetalGpuSurfaceFrameUniform {
    metrics: [f32; 4],
    extent_cursor: [f32; 4],
    surface_cursor: [f32; 4],
}

struct MetalGpuSurfaceState {
    textures: HashMap<GpuTextureHandle, MetalGpuSurfaceTexture>,
    buffers: HashMap<GpuBufferHandle, MetalGpuSurfaceBuffer>,
    samplers: HashMap<GpuSamplerHandle, MetalGpuSurfaceSampler>,
    render_programs: HashMap<GpuRenderProgramHandle, MetalGpuSurfaceRenderProgram>,
    compute_programs: HashMap<GpuComputeProgramHandle, MetalGpuSurfaceComputeProgram>,
    frame_uniform_buffer: metal::Buffer,
    presented: Option<GpuTextureHandle>,
    last_used_frame: u64,
}

struct MetalGpuSurfaceTexture {
    desc: GpuTextureDesc,
    texture: metal::Texture,
}

struct MetalGpuSurfaceBuffer {
    desc: GpuBufferDesc,
    buffer: metal::Buffer,
}

struct MetalGpuSurfaceSampler {
    desc: GpuSamplerDesc,
    sampler: metal::SamplerState,
}

struct MetalGpuSurfaceRenderProgram {
    desc: GpuRenderProgramDesc,
    library: metal::Library,
    bindings: Vec<MetalGpuSurfaceProgramBinding>,
    pipelines: HashMap<MTLPixelFormat, metal::RenderPipelineState>,
}

struct MetalGpuSurfaceComputeProgram {
    desc: GpuComputeProgramDesc,
    pipeline: metal::ComputePipelineState,
    bindings: Vec<MetalGpuSurfaceProgramBinding>,
    workgroup_size: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetalGpuSurfaceProgramBinding {
    slot: u32,
    kind: MetalGpuSurfaceProgramBindingKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalGpuSurfaceProgramBindingKind {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    StorageTexture,
    Sampler,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BackdropBlurInstance {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub opacity: f32,
    pub saturation: f32,
    pub _pad0: [f32; 2],
    pub tint: Hsla,
    pub direction: [f32; 2],
    pub texel_step: [f32; 2],
    pub viewport_size: [f32; 2],
    pub _pad1: [f32; 2],
    pub weights0: [f32; 4],
    pub weights1: [f32; 4],
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

impl MetalGpuSurfaceState {
    fn new(device: &metal::DeviceRef, is_unified_memory: bool) -> Self {
        let frame_uniform_buffer = create_gpu_surface_buffer(
            device,
            &GpuBufferDesc {
                label: Some("gpu_surface_frame_uniform".into()),
                usage: GpuBufferUsage::Uniform,
                size: mem::size_of::<MetalGpuSurfaceFrameUniform>() as u64,
            },
            is_unified_memory,
        )
        .expect("creating the Metal GPU surface frame uniform buffer should succeed")
        .buffer;
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
        device: &metal::DeviceRef,
        textures: &HashMap<GpuTextureHandle, GpuTextureDesc>,
    ) -> anyhow::Result<()> {
        self.textures
            .retain(|handle, _| textures.contains_key(handle));
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
        device: &metal::DeviceRef,
        buffers: &HashMap<GpuBufferHandle, GpuBufferDesc>,
        is_unified_memory: bool,
    ) -> anyhow::Result<()> {
        self.buffers
            .retain(|handle, _| buffers.contains_key(handle));
        for (&handle, desc) in buffers {
            let needs_recreate = self
                .buffers
                .get(&handle)
                .is_none_or(|buffer| buffer.desc != *desc);
            if needs_recreate {
                self.buffers.insert(
                    handle,
                    create_gpu_surface_buffer(device, desc, is_unified_memory)?,
                );
            }
        }
        Ok(())
    }

    fn sync_samplers(
        &mut self,
        device: &metal::DeviceRef,
        samplers: &HashMap<GpuSamplerHandle, GpuSamplerDesc>,
    ) {
        self.samplers
            .retain(|handle, _| samplers.contains_key(handle));
        for (&handle, desc) in samplers {
            let needs_recreate = self
                .samplers
                .get(&handle)
                .is_none_or(|sampler| sampler.desc != *desc);
            if needs_recreate {
                self.samplers
                    .insert(handle, create_gpu_surface_sampler(device, desc));
            }
        }
    }

    fn sync_render_programs(
        &mut self,
        device: &metal::DeviceRef,
        render_programs: &HashMap<GpuRenderProgramHandle, GpuRenderProgramDesc>,
    ) -> anyhow::Result<()> {
        self.render_programs
            .retain(|handle, _| render_programs.contains_key(handle));
        for (&handle, desc) in render_programs {
            let needs_recreate = self
                .render_programs
                .get(&handle)
                .is_none_or(|program| program.desc != *desc);
            if needs_recreate {
                self.render_programs
                    .insert(handle, create_gpu_surface_render_program(device, desc)?);
            }
        }
        Ok(())
    }

    fn sync_compute_programs(
        &mut self,
        device: &metal::DeviceRef,
        compute_programs: &HashMap<GpuComputeProgramHandle, GpuComputeProgramDesc>,
    ) -> anyhow::Result<()> {
        self.compute_programs
            .retain(|handle, _| compute_programs.contains_key(handle));
        for (&handle, desc) in compute_programs {
            let needs_recreate = self
                .compute_programs
                .get(&handle)
                .is_none_or(|program| program.desc != *desc);
            if needs_recreate {
                self.compute_programs
                    .insert(handle, create_gpu_surface_compute_program(device, desc)?);
            }
        }
        Ok(())
    }

    fn execute_graph(
        &mut self,
        device: &metal::DeviceRef,
        command_queue: &metal::CommandQueueRef,
        frame: &GpuFrameContext,
        scale_factor: f32,
        graph: &GpuRecordedGraph,
        is_unified_memory: bool,
    ) -> anyhow::Result<()> {
        write_gpu_surface_buffer(
            &self.frame_uniform_buffer,
            0,
            unsafe {
                std::slice::from_raw_parts(
                    (&gpu_surface_frame_uniform(frame, scale_factor)
                        as *const MetalGpuSurfaceFrameUniform)
                        .cast::<u8>(),
                    mem::size_of::<MetalGpuSurfaceFrameUniform>(),
                )
            },
            is_unified_memory,
        );
        self.apply_buffer_writes(graph.buffer_writes(), is_unified_memory)?;

        let command_buffer = command_queue.new_command_buffer();
        for operation in &graph.operations {
            match operation {
                GpuGraphOperation::RenderPass(pass) => {
                    self.execute_render_pass(device, command_buffer, pass)?
                }
                GpuGraphOperation::ComputePass(pass) => {
                    self.execute_compute_pass(command_buffer, pass)?
                }
            }
        }
        command_buffer.commit();
        self.presented = graph.presented;
        Ok(())
    }

    fn apply_buffer_writes(
        &self,
        writes: &[GpuBufferWrite],
        is_unified_memory: bool,
    ) -> anyhow::Result<()> {
        for write in writes {
            let buffer = self
                .buffers
                .get(&write.buffer)
                .context("GpuSurface buffer write target missing in Metal state")?;
            let end = write.offset + write.data.len() as u64;
            if end > buffer.desc.size {
                anyhow::bail!(
                    "GpuSurface buffer write exceeds buffer bounds on Metal execution: offset {}, size {}, capacity {}",
                    write.offset,
                    write.data.len(),
                    buffer.desc.size
                );
            }
            write_gpu_surface_buffer(&buffer.buffer, write.offset, &write.data, is_unified_memory);
        }
        Ok(())
    }

    fn execute_render_pass(
        &mut self,
        device: &metal::DeviceRef,
        command_buffer: &metal::CommandBufferRef,
        pass: &GpuRenderPassDesc,
    ) -> anyhow::Result<()> {
        let (target_desc, target_texture) = {
            let target = self
                .textures
                .get(&pass.target)
                .context("GpuSurface render pass target missing in Metal state")?;
            (target.desc.clone(), target.texture.clone())
        };
        if !target_desc.render_attachment {
            anyhow::bail!("GpuSurface render target is not renderable on Metal");
        }

        let pipeline = self
            .render_pipeline(
                device,
                pass.program,
                gpu_texture_format_to_metal(target_desc.format),
            )?
            .clone();
        let bindings = self
            .render_programs
            .get(&pass.program)
            .context("GpuSurface render program missing in Metal state")?
            .bindings
            .clone();
        if bindings.len() != pass.bindings.len() {
            anyhow::bail!(
                "GpuSurface render pass binding count mismatch: shader expects {}, graph provides {}",
                bindings.len(),
                pass.bindings.len()
            );
        }

        let viewport_size = size(
            DevicePixels(target_desc.extent.width.max(1) as i32),
            DevicePixels(target_desc.extent.height.max(1) as i32),
        );
        let load_action = if pass.clear_color.is_some() {
            metal::MTLLoadAction::Clear
        } else {
            metal::MTLLoadAction::Load
        };
        let clear_color = pass
            .clear_color
            .map(gpu_clear_color_to_metal)
            .unwrap_or_else(|| metal::MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
        let encoder = new_texture_command_encoder(
            command_buffer,
            &target_texture,
            viewport_size,
            load_action,
            clear_color,
        );
        encoder.set_render_pipeline_state(&pipeline);
        self.bind_render_resources(encoder, &bindings, &pass.bindings)?;
        match pass.draw {
            GpuDrawCall::FullScreenTriangle => {
                encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
            }
            GpuDrawCall::Triangles {
                vertex_count,
                instance_count,
            } => {
                encoder.draw_primitives_instanced(
                    metal::MTLPrimitiveType::Triangle,
                    0,
                    vertex_count as u64,
                    instance_count as u64,
                );
            }
        }
        encoder.end_encoding();
        Ok(())
    }

    fn execute_compute_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        pass: &GpuComputePassDesc,
    ) -> anyhow::Result<()> {
        if pass.workgroups.contains(&0) {
            return Ok(());
        }
        let program = self
            .compute_programs
            .get(&pass.program)
            .context("GpuSurface compute program missing in Metal state")?;
        if program.bindings.len() != pass.bindings.len() {
            anyhow::bail!(
                "GpuSurface compute pass binding count mismatch: shader expects {}, graph provides {}",
                program.bindings.len(),
                pass.bindings.len()
            );
        }

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&program.pipeline);
        self.bind_compute_resources(encoder, &program.bindings, &pass.bindings)?;
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: pass.workgroups[0] as u64,
                height: pass.workgroups[1] as u64,
                depth: pass.workgroups[2] as u64,
            },
            metal::MTLSize {
                width: program.workgroup_size[0].max(1) as u64,
                height: program.workgroup_size[1].max(1) as u64,
                depth: program.workgroup_size[2].max(1) as u64,
            },
        );
        encoder.end_encoding();
        Ok(())
    }

    fn bind_render_resources(
        &self,
        encoder: &metal::RenderCommandEncoderRef,
        program_bindings: &[MetalGpuSurfaceProgramBinding],
        pass_bindings: &[GpuBinding],
    ) -> anyhow::Result<()> {
        encoder.set_vertex_buffer(0, Some(&self.frame_uniform_buffer), 0);
        encoder.set_fragment_buffer(0, Some(&self.frame_uniform_buffer), 0);

        for (binding, program_binding) in pass_bindings.iter().zip(program_bindings) {
            match (*binding, program_binding.kind) {
                (
                    GpuBinding::UniformBuffer(handle),
                    MetalGpuSurfaceProgramBindingKind::UniformBuffer,
                )
                | (
                    GpuBinding::StorageBuffer(handle),
                    MetalGpuSurfaceProgramBindingKind::StorageBuffer,
                ) => {
                    let buffer = self.buffers.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface buffer binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    let expected_usage = if program_binding.kind
                        == MetalGpuSurfaceProgramBindingKind::UniformBuffer
                    {
                        GpuBufferUsage::Uniform
                    } else {
                        GpuBufferUsage::Storage
                    };
                    if buffer.desc.usage != expected_usage {
                        anyhow::bail!("GpuSurface render pass binding type does not match shader");
                    }
                    encoder.set_vertex_buffer(program_binding.slot as u64, Some(&buffer.buffer), 0);
                    encoder.set_fragment_buffer(
                        program_binding.slot as u64,
                        Some(&buffer.buffer),
                        0,
                    );
                }
                (
                    GpuBinding::SampledTexture(handle),
                    MetalGpuSurfaceProgramBindingKind::SampledTexture,
                )
                | (
                    GpuBinding::StorageTexture(handle),
                    MetalGpuSurfaceProgramBindingKind::StorageTexture,
                ) => {
                    let texture = self.textures.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface texture binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    let valid = if program_binding.kind
                        == MetalGpuSurfaceProgramBindingKind::SampledTexture
                    {
                        texture.desc.sampled
                    } else {
                        texture.desc.storage
                    };
                    if !valid {
                        anyhow::bail!("GpuSurface render pass binding type does not match shader");
                    }
                    encoder.set_vertex_texture(program_binding.slot as u64, Some(&texture.texture));
                    encoder
                        .set_fragment_texture(program_binding.slot as u64, Some(&texture.texture));
                }
                (GpuBinding::Sampler(handle), MetalGpuSurfaceProgramBindingKind::Sampler) => {
                    let sampler = self.samplers.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface sampler binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    encoder.set_vertex_sampler_state(
                        program_binding.slot as u64,
                        Some(&sampler.sampler),
                    );
                    encoder.set_fragment_sampler_state(
                        program_binding.slot as u64,
                        Some(&sampler.sampler),
                    );
                }
                _ => anyhow::bail!("GpuSurface render pass binding type does not match shader"),
            }
        }

        Ok(())
    }

    fn bind_compute_resources(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        program_bindings: &[MetalGpuSurfaceProgramBinding],
        pass_bindings: &[GpuBinding],
    ) -> anyhow::Result<()> {
        encoder.set_buffer(0, Some(&self.frame_uniform_buffer), 0);

        for (binding, program_binding) in pass_bindings.iter().zip(program_bindings) {
            match (*binding, program_binding.kind) {
                (
                    GpuBinding::UniformBuffer(handle),
                    MetalGpuSurfaceProgramBindingKind::UniformBuffer,
                )
                | (
                    GpuBinding::StorageBuffer(handle),
                    MetalGpuSurfaceProgramBindingKind::StorageBuffer,
                ) => {
                    let buffer = self.buffers.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface buffer binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    let expected_usage = if program_binding.kind
                        == MetalGpuSurfaceProgramBindingKind::UniformBuffer
                    {
                        GpuBufferUsage::Uniform
                    } else {
                        GpuBufferUsage::Storage
                    };
                    if buffer.desc.usage != expected_usage {
                        anyhow::bail!("GpuSurface compute pass binding type does not match shader");
                    }
                    encoder.set_buffer(program_binding.slot as u64, Some(&buffer.buffer), 0);
                }
                (
                    GpuBinding::SampledTexture(handle),
                    MetalGpuSurfaceProgramBindingKind::SampledTexture,
                )
                | (
                    GpuBinding::StorageTexture(handle),
                    MetalGpuSurfaceProgramBindingKind::StorageTexture,
                ) => {
                    let texture = self.textures.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface texture binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    let valid = if program_binding.kind
                        == MetalGpuSurfaceProgramBindingKind::SampledTexture
                    {
                        texture.desc.sampled
                    } else {
                        texture.desc.storage
                    };
                    if !valid {
                        anyhow::bail!("GpuSurface compute pass binding type does not match shader");
                    }
                    encoder.set_texture(program_binding.slot as u64, Some(&texture.texture));
                }
                (GpuBinding::Sampler(handle), MetalGpuSurfaceProgramBindingKind::Sampler) => {
                    let sampler = self.samplers.get(&handle).with_context(|| {
                        format!(
                            "GpuSurface sampler binding {} missing in Metal state",
                            program_binding.slot
                        )
                    })?;
                    encoder.set_sampler_state(program_binding.slot as u64, Some(&sampler.sampler));
                }
                _ => anyhow::bail!("GpuSurface compute pass binding type does not match shader"),
            }
        }

        Ok(())
    }

    fn render_pipeline(
        &mut self,
        device: &metal::DeviceRef,
        handle: GpuRenderProgramHandle,
        target_format: MTLPixelFormat,
    ) -> anyhow::Result<&metal::RenderPipelineState> {
        let program = self
            .render_programs
            .get_mut(&handle)
            .context("GpuSurface render program missing in Metal state")?;
        if let std::collections::hash_map::Entry::Vacant(entry) =
            program.pipelines.entry(target_format)
        {
            entry.insert(build_gpu_surface_pipeline_state(
                device,
                &program.library,
                program.desc.label.as_ref().map(|label| label.as_ref()),
                program.desc.vertex_entry.as_ref(),
                program.desc.fragment_entry.as_ref(),
                target_format,
            )?);
        }
        Ok(program
            .pipelines
            .get(&target_format)
            .expect("Metal GpuSurface render pipeline must exist after insertion"))
    }

    fn presented_texture(&self) -> anyhow::Result<&metal::TextureRef> {
        let presented = self
            .presented
            .context("GpuSurface present texture missing in Metal state")?;
        let texture = self
            .textures
            .get(&presented)
            .context("GpuSurface present texture missing in Metal state")?;
        if !texture.desc.sampled {
            anyhow::bail!("GpuSurface present texture is not sampleable on Metal");
        }
        Ok(texture.texture.as_ref())
    }
}

impl MetalRenderer {
    fn prune_stale_gpu_surfaces(&mut self) {
        const RETAIN_FRAMES: u64 = 120;
        let min_frame = self.gpu_surface_frame_serial.saturating_sub(RETAIN_FRAMES);
        self.gpu_surfaces
            .retain(|_, state| state.last_used_frame >= min_frame);
    }

    pub(crate) fn paint_gpu_surface(
        &mut self,
        input: GpuSurfaceExecutionInput<'_>,
    ) -> anyhow::Result<Option<PaintSurface>> {
        let Some(_) = input.graph.presented else {
            return Ok(None);
        };
        let frame = input
            .frame
            .context("GpuSurface Metal executor requires a prepared frame context")?;
        let state = self
            .gpu_surfaces
            .entry(input.surface_id)
            .or_insert_with(|| MetalGpuSurfaceState::new(&self.device, self.is_unified_memory));
        state.last_used_frame = self.gpu_surface_frame_serial + 1;
        state.sync_textures(&self.device, input.textures)?;
        state.sync_buffers(&self.device, input.buffers, self.is_unified_memory)?;
        state.sync_samplers(&self.device, input.samplers);
        state.sync_render_programs(&self.device, input.render_programs)?;
        state.sync_compute_programs(&self.device, input.compute_programs)?;
        state.execute_graph(
            &self.device,
            &self.command_queue,
            frame,
            input.scale_factor,
            input.graph,
            self.is_unified_memory,
        )?;
        let _ = state.presented_texture()?;

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
}

impl MetalRenderer {
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        let device = if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        };

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let gpu_surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "gpu_surfaces",
            "surface_vertex",
            "surface_rgb_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blur_h_pipeline_state = build_pipeline_state_no_blend(
            &device,
            &library,
            "backdrop_blur_h",
            "backdrop_blur_h_vertex",
            "backdrop_blur_h_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blur_composite_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "backdrop_blur_composite",
            "backdrop_blur_composite_vertex",
            "backdrop_blur_composite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blur_blit_pipeline_state = build_pipeline_state_no_blend(
            &device,
            &library,
            "backdrop_blur_blit",
            "backdrop_blur_blit_vertex",
            "backdrop_blur_blit_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            backdrop_blur_h_pipeline_state,
            backdrop_blur_composite_pipeline_state,
            backdrop_blur_blit_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            gpu_surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            backdrop_main_texture: None,
            backdrop_temp_texture: None,
            backdrop_down2_texture: None,
            backdrop_temp2_texture: None,
            backdrop_down4_texture: None,
            backdrop_temp4_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            gpu_surfaces: HashMap::default(),
            gpu_surface_frame_serial: 0,
        }
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![
                self.layer(),
                setDrawableSize: size
            ];
        }
        let device_pixels_size = Size {
            width: DevicePixels(size.width as i32),
            height: DevicePixels(size.height as i32),
        };
        self.update_path_intermediate_textures(device_pixels_size);
        self.invalidate_backdrop_textures();
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    fn invalidate_backdrop_textures(&mut self) {
        self.backdrop_main_texture = None;
        self.backdrop_temp_texture = None;
        self.backdrop_down2_texture = None;
        self.backdrop_temp2_texture = None;
        self.backdrop_down4_texture = None;
        self.backdrop_temp4_texture = None;
    }

    fn ensure_backdrop_textures(&mut self, size: Size<DevicePixels>) -> bool {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.invalidate_backdrop_textures();
            return false;
        }

        let width = size.width.0 as u64;
        let height = size.height.0 as u64;
        let needs_recreate = self.backdrop_main_texture.as_ref().map_or(true, |texture| {
            texture.width() != width || texture.height() != height
        });

        if !needs_recreate {
            return true;
        }

        let down2 = Size {
            width: DevicePixels(((size.width.0 + 1) / 2).max(1)),
            height: DevicePixels(((size.height.0 + 1) / 2).max(1)),
        };
        let down4 = Size {
            width: DevicePixels(((size.width.0 + 3) / 4).max(1)),
            height: DevicePixels(((size.height.0 + 3) / 4).max(1)),
        };

        self.backdrop_main_texture = Some(self.create_backdrop_texture(size, "backdrop_main"));
        self.backdrop_temp_texture = Some(self.create_backdrop_texture(size, "backdrop_temp"));
        self.backdrop_down2_texture = Some(self.create_backdrop_texture(down2, "backdrop_down2"));
        self.backdrop_temp2_texture = Some(self.create_backdrop_texture(down2, "backdrop_temp2"));
        self.backdrop_down4_texture = Some(self.create_backdrop_texture(down4, "backdrop_down4"));
        self.backdrop_temp4_texture = Some(self.create_backdrop_texture(down4, "backdrop_temp4"));

        true
    }

    fn create_backdrop_texture(&self, size: Size<DevicePixels>, label: &str) -> metal::Texture {
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0.max(1) as u64);
        texture_descriptor.set_height(size.height.0.max(1) as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);

        let texture = self.device.new_texture(&texture_descriptor);
        texture.set_label(label);
        texture
    }

    pub fn update_transparency(&self, transparent: bool) {
        self.layer.set_opaque(!transparent);
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        self.gpu_surface_frame_serial = self.gpu_surface_frame_serial.wrapping_add(1);
        self.prune_stale_gpu_surfaces();

        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    return;
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        self.gpu_surface_frame_serial = self.gpu_surface_frame_serial.wrapping_add(1);
        self.prune_stale_gpu_surfaces();

        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // Commit and wait for completion without presenting
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    // Read pixels from the texture
                    let texture = drawable.texture();
                    let width = texture.width() as u32;
                    let height = texture.height() as u32;
                    let bytes_per_row = width as usize * 4;
                    let buffer_size = height as usize * bytes_per_row;

                    let mut pixels = vec![0u8; buffer_size];

                    let region = metal::MTLRegion {
                        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: metal::MTLSize {
                            width: width as u64,
                            height: height as u64,
                            depth: 1,
                        },
                    };

                    texture.get_bytes(
                        pixels.as_mut_ptr() as *mut std::ffi::c_void,
                        bytes_per_row as u64,
                        region,
                        0,
                    );

                    // Convert BGRA to RGBA (swap B and R channels)
                    for chunk in pixels.chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }

                    return RgbaImage::from_raw(width, height, pixels).ok_or_else(|| {
                        anyhow::anyhow!("Failed to create RgbaImage from pixel data")
                    });
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.layer.is_opaque() { 1. } else { 0. };
        let mut instance_offset = 0;
        let use_backdrop = !scene.backdrop_filters.is_empty();

        if use_backdrop && !self.ensure_backdrop_textures(viewport_size) {
            return Ok(command_buffer.to_owned());
        }

        let main_texture_owned = if use_backdrop {
            Some(
                self.backdrop_main_texture
                    .as_ref()
                    .expect("backdrop main texture missing")
                    .to_owned(),
            )
        } else {
            None
        };
        let main_texture = main_texture_owned.as_ref().map(|texture| texture.as_ref());

        let clear_color = metal::MTLClearColor::new(0.0, 0.0, 0.0, alpha);

        let mut command_encoder = if let Some(main_texture) = main_texture {
            new_texture_command_encoder(
                command_buffer,
                main_texture,
                viewport_size,
                metal::MTLLoadAction::Clear,
                clear_color,
            )
        } else {
            new_command_encoder(
                command_buffer,
                drawable,
                viewport_size,
                |color_attachment| {
                    color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                    color_attachment.set_clear_color(clear_color);
                },
            )
        };

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(
                    &scene.shadows[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::BackdropFilters(range) => {
                    let Some(main_texture) = main_texture else {
                        return Err(anyhow::anyhow!("backdrop filter without main texture"));
                    };
                    command_encoder.end_encoding();
                    let ok = self.draw_backdrop_filters(
                        &scene.backdrop_filters[range],
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_buffer,
                        main_texture,
                    );
                    command_encoder = new_texture_command_encoder(
                        command_buffer,
                        main_texture,
                        viewport_size,
                        metal::MTLLoadAction::Load,
                        clear_color,
                    );
                    ok
                }
                PrimitiveBatch::Quads(range) => self.draw_quads(
                    &scene.quads[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_buffer,
                    );

                    command_encoder = if let Some(main_texture) = main_texture {
                        new_texture_command_encoder(
                            command_buffer,
                            main_texture,
                            viewport_size,
                            metal::MTLLoadAction::Load,
                            clear_color,
                        )
                    } else {
                        new_command_encoder(
                            command_buffer,
                            drawable,
                            viewport_size,
                            |color_attachment| {
                                color_attachment.set_load_action(metal::MTLLoadAction::Load);
                            },
                        )
                    };

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            command_encoder,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(
                    &scene.underlines[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        &scene.monochrome_sprites[range],
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        &scene.polychrome_sprites[range],
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();

        if let Some(main_texture) = main_texture {
            self.blit_backdrop_to_drawable(
                instance_buffer,
                &mut instance_offset,
                viewport_size,
                command_buffer,
                drawable,
                main_texture,
            )?;
        }

        if !self.is_unified_memory {
            // Sync the instance buffer to the GPU
            instance_buffer.metal_buffer.did_modify_range(NSRange {
                location: 0,
                length: instance_offset as NSUInteger,
            });
        }

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);

        align_offset(instance_offset);
        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.end_encoding();
            return false;
        }
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                buffer_contents,
                vertices_bytes_len,
            );
        }
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );
        *instance_offset = next_offset;

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(ref intermediate_texture) = self.path_intermediate_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;

        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for surface in surfaces {
            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<SurfaceBounds>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_fragment_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );

            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut SurfaceBounds;
                ptr::write(
                    buffer_contents,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask.clone(),
                        corner_radii: surface.corner_radii,
                    },
                );
            }

            if let Some(gpu_surface_id) = surface.gpu_surface_id {
                let Some(surface_state) = self.gpu_surfaces.get(&gpu_surface_id) else {
                    log::error!(
                        "skipping macOS GpuSurface draw because surface state {} is missing",
                        gpu_surface_id
                    );
                    *instance_offset = next_offset;
                    continue;
                };
                let texture = match surface_state.presented_texture() {
                    Ok(texture) => texture.to_owned(),
                    Err(err) => {
                        log::error!(
                            "skipping macOS GpuSurface draw because the presented texture is unavailable: {err:#}"
                        );
                        *instance_offset = next_offset;
                        continue;
                    }
                };
                let texture_size = size(
                    DevicePixels::from(texture.width() as i32),
                    DevicePixels::from(texture.height() as i32),
                );
                command_encoder.set_render_pipeline_state(&self.gpu_surfaces_pipeline_state);
                command_encoder.set_vertex_bytes(
                    SurfaceInputIndex::TextureSize as u64,
                    mem::size_of_val(&texture_size) as u64,
                    &texture_size as *const Size<DevicePixels> as *const _,
                );
                command_encoder.set_fragment_texture(
                    SurfaceInputIndex::YTexture as u64,
                    Some(texture.as_ref()),
                );
                command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, None);
            } else {
                let image_buffer = surface
                    .image_buffer
                    .as_ref()
                    .expect("macOS surface batch requires a pixel buffer");
                let texture_size = size(
                    DevicePixels::from(image_buffer.get_width() as i32),
                    DevicePixels::from(image_buffer.get_height() as i32),
                );

                assert_eq!(
                    image_buffer.get_pixel_format(),
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                );

                let y_texture = self
                    .core_video_texture_cache
                    .create_texture_from_image(
                        image_buffer.as_concrete_TypeRef(),
                        None,
                        MTLPixelFormat::R8Unorm,
                        image_buffer.get_width_of_plane(0),
                        image_buffer.get_height_of_plane(0),
                        0,
                    )
                    .unwrap();
                let cb_cr_texture = self
                    .core_video_texture_cache
                    .create_texture_from_image(
                        image_buffer.as_concrete_TypeRef(),
                        None,
                        MTLPixelFormat::RG8Unorm,
                        image_buffer.get_width_of_plane(1),
                        image_buffer.get_height_of_plane(1),
                        1,
                    )
                    .unwrap();
                command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
                command_encoder.set_vertex_bytes(
                    SurfaceInputIndex::TextureSize as u64,
                    mem::size_of_val(&texture_size) as u64,
                    &texture_size as *const Size<DevicePixels> as *const _,
                );
                command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                    let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                    Some(metal::TextureRef::from_ptr(texture as *mut _))
                });
                command_encoder.set_fragment_texture(
                    SurfaceInputIndex::CbCrTexture as u64,
                    unsafe {
                        let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                        Some(metal::TextureRef::from_ptr(texture as *mut _))
                    },
                );
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }

    fn draw_backdrop_filters(
        &self,
        filters: &[BackdropFilter],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
        main_texture: &metal::TextureRef,
    ) -> bool {
        let full_size = [
            viewport_size.width.0.max(1) as u32,
            viewport_size.height.0.max(1) as u32,
        ];
        let full_size_f = [full_size[0] as f32, full_size[1] as f32];

        let Some(temp_full) = self.backdrop_temp_texture.as_ref() else {
            return false;
        };

        for filter in filters {
            let radius = filter.blur_radius.0.max(0.0);
            if radius <= 0.0 {
                continue;
            }

            let scale = if radius <= 8.0 {
                1
            } else if radius <= 16.0 {
                2
            } else {
                4
            };

            let kernel = (radius * 1.5).ceil();
            let expanded_bounds = filter.bounds.dilate(ScaledPixels::from(kernel + 2.0));
            let expanded_scaled = if scale == 1 {
                expanded_bounds
            } else {
                Self::scale_bounds(expanded_bounds, 1.0 / scale as f32)
            };

            let target_size_u32 = match scale {
                1 => full_size,
                2 => [full_size[0].div_ceil(2), full_size[1].div_ceil(2)],
                _ => [full_size[0].div_ceil(4), full_size[1].div_ceil(4)],
            };
            let target_size_f = [target_size_u32[0] as f32, target_size_u32[1] as f32];
            let scissor_expanded = Self::scissor_from_bounds(expanded_scaled, target_size_u32);
            let scissor_full = Self::scissor_from_bounds(filter.bounds, full_size);

            let passes = Self::blur_passes(radius, scale);
            let pass_radius = if passes > 1 {
                radius / (passes as f32).sqrt()
            } else {
                radius
            };
            let (weights0, weights1, step) = Self::gaussian_weights(pass_radius, scale as f32);
            let texel_step = [
                step / target_size_f[0].max(1.0),
                step / target_size_f[1].max(1.0),
            ];

            let (down_texture, temp_texture): (Option<&metal::TextureRef>, &metal::TextureRef) =
                match scale {
                    1 => (None, temp_full),
                    2 => (
                        Some(
                            self.backdrop_down2_texture
                                .as_ref()
                                .expect("backdrop down2 texture missing"),
                        ),
                        self.backdrop_temp2_texture
                            .as_ref()
                            .expect("backdrop temp2 texture missing"),
                    ),
                    _ => (
                        Some(
                            self.backdrop_down4_texture
                                .as_ref()
                                .expect("backdrop down4 texture missing"),
                        ),
                        self.backdrop_temp4_texture
                            .as_ref()
                            .expect("backdrop temp4 texture missing"),
                    ),
                };

            if scale > 1 {
                let Some(scissor) = scissor_expanded else {
                    continue;
                };
                let blit_instance =
                    BackdropBlurInstance::blit_instance(expanded_scaled, target_size_f);
                let ok = self.draw_backdrop_pass(
                    &blit_instance,
                    instance_buffer,
                    instance_offset,
                    command_buffer,
                    &self.backdrop_blur_blit_pipeline_state,
                    down_texture.expect("downsample texture missing"),
                    main_texture,
                    Size {
                        width: DevicePixels(target_size_u32[0] as i32),
                        height: DevicePixels(target_size_u32[1] as i32),
                    },
                    Some(scissor),
                );
                if !ok {
                    return false;
                }
            }

            let Some(scissor) = scissor_expanded else {
                continue;
            };
            let mut source_texture = down_texture.unwrap_or(main_texture);
            let target_size = Size {
                width: DevicePixels(target_size_u32[0] as i32),
                height: DevicePixels(target_size_u32[1] as i32),
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
                    target_size_f,
                    weights0,
                    weights1,
                );
                let ok = self.draw_backdrop_pass(
                    &blur_instance,
                    instance_buffer,
                    instance_offset,
                    command_buffer,
                    &self.backdrop_blur_h_pipeline_state,
                    temp_texture,
                    source_texture,
                    target_size,
                    Some(scissor),
                );
                if !ok {
                    return false;
                }

                if pass_index + 1 == passes {
                    let Some(scissor) = scissor_full else {
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
                        full_size_f,
                        weights0,
                        weights1,
                    );
                    let ok = self.draw_backdrop_pass(
                        &composite_instance,
                        instance_buffer,
                        instance_offset,
                        command_buffer,
                        &self.backdrop_blur_composite_pipeline_state,
                        main_texture,
                        temp_texture,
                        viewport_size,
                        Some(scissor),
                    );
                    if !ok {
                        return false;
                    }
                } else {
                    let down_texture = down_texture.expect("downsample texture missing");
                    let blur_instance = BackdropBlurInstance::blur_instance(
                        expanded_scaled,
                        expanded_scaled,
                        Corners::default(),
                        1.0,
                        Hsla::default(),
                        1.0,
                        [0.0, 1.0],
                        texel_step,
                        target_size_f,
                        weights0,
                        weights1,
                    );
                    let ok = self.draw_backdrop_pass(
                        &blur_instance,
                        instance_buffer,
                        instance_offset,
                        command_buffer,
                        &self.backdrop_blur_h_pipeline_state,
                        down_texture,
                        temp_texture,
                        target_size,
                        Some(scissor),
                    );
                    if !ok {
                        return false;
                    }
                    source_texture = down_texture;
                }
            }
        }

        true
    }

    fn draw_backdrop_pass(
        &self,
        instance: &BackdropBlurInstance,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
        pipeline_state: &metal::RenderPipelineState,
        target_texture: &metal::TextureRef,
        source_texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        scissor: Option<metal::MTLScissorRect>,
    ) -> bool {
        align_offset(instance_offset);
        let instance_bytes_len = mem::size_of_val(instance);
        let next_offset = *instance_offset + instance_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            let buffer_contents =
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset);
            ptr::copy_nonoverlapping(
                instance as *const BackdropBlurInstance as *const u8,
                buffer_contents,
                instance_bytes_len,
            );
        }

        let command_encoder = new_texture_command_encoder(
            command_buffer,
            target_texture,
            viewport_size,
            metal::MTLLoadAction::Load,
            metal::MTLClearColor::new(0.0, 0.0, 0.0, 0.0),
        );
        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(
            BackdropBlurInputIndex::SourceTexture as u64,
            Some(source_texture),
        );
        if let Some(scissor) = scissor {
            command_encoder.set_scissor_rect(scissor);
        }
        command_encoder.draw_primitives_instanced(metal::MTLPrimitiveType::Triangle, 0, 6, 1);
        command_encoder.end_encoding();

        *instance_offset = next_offset;
        true
    }

    fn blit_backdrop_to_drawable(
        &self,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
        drawable: &metal::MetalDrawableRef,
        main_texture: &metal::TextureRef,
    ) -> Result<()> {
        let bounds = Bounds {
            origin: point(ScaledPixels::from(0.0), ScaledPixels::from(0.0)),
            size: size(
                ScaledPixels::from(viewport_size.width.0 as f32),
                ScaledPixels::from(viewport_size.height.0 as f32),
            ),
        };
        let blit_instance = BackdropBlurInstance::blit_instance(
            bounds,
            [viewport_size.width.0 as f32, viewport_size.height.0 as f32],
        );

        align_offset(instance_offset);
        let instance_bytes_len = mem::size_of_val(&blit_instance);
        let next_offset = *instance_offset + instance_bytes_len;
        if next_offset > instance_buffer.size {
            anyhow::bail!("scene too large for instance buffer");
        }

        unsafe {
            let buffer_contents =
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset);
            ptr::copy_nonoverlapping(
                &blit_instance as *const BackdropBlurInstance as *const u8,
                buffer_contents,
                instance_bytes_len,
            );
        }

        let alpha = if self.layer.is_opaque() { 1.0 } else { 0.0 };
        let command_encoder = new_command_encoder(
            command_buffer,
            drawable,
            viewport_size,
            |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(0.0, 0.0, 0.0, alpha));
            },
        );
        command_encoder.set_render_pipeline_state(&self.backdrop_blur_blit_pipeline_state);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(
            BackdropBlurInputIndex::SourceTexture as u64,
            Some(main_texture),
        );
        command_encoder.draw_primitives_instanced(metal::MTLPrimitiveType::Triangle, 0, 6, 1);
        command_encoder.end_encoding();
        *instance_offset = next_offset;
        Ok(())
    }

    fn scale_bounds(bounds: Bounds<ScaledPixels>, factor: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(
                ScaledPixels::from(bounds.origin.x.0 * factor),
                ScaledPixels::from(bounds.origin.y.0 * factor),
            ),
            size: size(
                ScaledPixels::from(bounds.size.width.0 * factor),
                ScaledPixels::from(bounds.size.height.0 * factor),
            ),
        }
    }

    fn scissor_from_bounds(
        bounds: Bounds<ScaledPixels>,
        target_size: [u32; 2],
    ) -> Option<metal::MTLScissorRect> {
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

        Some(metal::MTLScissorRect {
            x: min_x as u64,
            y: min_y as u64,
            width: (max_x - min_x) as u64,
            height: (max_y - min_y) as u64,
        })
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
}

fn new_command_encoder<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    drawable: &'a metal::MetalDrawableRef,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(drawable.texture()));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn new_texture_command_encoder<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    load_action: metal::MTLLoadAction,
    clear_color: metal::MTLClearColor,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    color_attachment.set_load_action(load_action);
    if matches!(load_action, metal::MTLLoadAction::Clear) {
        color_attachment.set_clear_color(clear_color);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn create_gpu_surface_texture(
    device: &metal::DeviceRef,
    desc: &GpuTextureDesc,
) -> anyhow::Result<MetalGpuSurfaceTexture> {
    let texture_descriptor = metal::TextureDescriptor::new();
    texture_descriptor.set_texture_type(metal::MTLTextureType::D2);
    texture_descriptor.set_width(desc.extent.width.max(1) as u64);
    texture_descriptor.set_height(desc.extent.height.max(1) as u64);
    texture_descriptor.set_depth(1);
    texture_descriptor.set_mipmap_level_count(1);
    texture_descriptor.set_sample_count(1);
    texture_descriptor.set_pixel_format(gpu_texture_format_to_metal(desc.format));
    texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);

    let mut usage = metal::MTLTextureUsage::Unknown;
    if desc.sampled {
        usage |= metal::MTLTextureUsage::ShaderRead;
    }
    if desc.storage {
        usage |= metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::ShaderWrite;
    }
    if desc.render_attachment {
        usage |= metal::MTLTextureUsage::RenderTarget;
    }
    texture_descriptor.set_usage(usage);

    let texture = device.new_texture(&texture_descriptor);
    if let Some(label) = desc.label.as_ref() {
        texture.set_label(label.as_ref());
    }

    Ok(MetalGpuSurfaceTexture {
        desc: desc.clone(),
        texture,
    })
}

fn create_gpu_surface_buffer(
    device: &metal::DeviceRef,
    desc: &GpuBufferDesc,
    is_unified_memory: bool,
) -> anyhow::Result<MetalGpuSurfaceBuffer> {
    let options = match desc.usage {
        GpuBufferUsage::Uniform => {
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            }
        }
        GpuBufferUsage::Storage => {
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
            } else {
                MTLResourceOptions::StorageModeManaged
            }
        }
    };

    let buffer = device.new_buffer(gpu_surface_buffer_size(desc), options);
    if let Some(label) = desc.label.as_ref() {
        buffer.set_label(label.as_ref());
    }

    Ok(MetalGpuSurfaceBuffer {
        desc: desc.clone(),
        buffer,
    })
}

fn create_gpu_surface_sampler(
    device: &metal::DeviceRef,
    desc: &GpuSamplerDesc,
) -> MetalGpuSurfaceSampler {
    let sampler_descriptor = metal::SamplerDescriptor::new();
    sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mip_filter(metal::MTLSamplerMipFilter::NotMipmapped);
    sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_r(metal::MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_lod_min_clamp(0.0);
    sampler_descriptor.set_lod_max_clamp(f32::MAX);
    if let Some(label) = desc.label.as_ref() {
        sampler_descriptor.set_label(label.as_ref());
    }

    MetalGpuSurfaceSampler {
        desc: desc.clone(),
        sampler: device.new_sampler(&sampler_descriptor),
    }
}

fn create_gpu_surface_render_program(
    device: &metal::DeviceRef,
    desc: &GpuRenderProgramDesc,
) -> anyhow::Result<MetalGpuSurfaceRenderProgram> {
    let module =
        naga::front::wgsl::parse_str(desc.wgsl.as_ref()).context("Parsing GpuSurface WGSL")?;
    let bindings = gpu_surface_program_bindings(&module)?;
    let module_info = validate_gpu_surface_module(&module)?;
    let source = build_gpu_surface_msl_source(&module, &module_info, None)?;
    let library = device
        .new_library_with_source(&source, &metal::CompileOptions::new())
        .map_err(anyhow::Error::msg)
        .context("Compiling GpuSurface MSL render library")?;
    if let Some(label) = desc.label.as_ref() {
        library.set_label(label.as_ref());
    }

    Ok(MetalGpuSurfaceRenderProgram {
        desc: desc.clone(),
        library,
        bindings,
        pipelines: HashMap::default(),
    })
}

fn create_gpu_surface_compute_program(
    device: &metal::DeviceRef,
    desc: &GpuComputeProgramDesc,
) -> anyhow::Result<MetalGpuSurfaceComputeProgram> {
    let module =
        naga::front::wgsl::parse_str(desc.wgsl.as_ref()).context("Parsing GpuSurface WGSL")?;
    let bindings = gpu_surface_program_bindings(&module)?;
    let workgroup_size = gpu_surface_compute_workgroup_size(&module, desc.entry.as_ref())?;
    let module_info = validate_gpu_surface_module(&module)?;
    let source = build_gpu_surface_msl_source(
        &module,
        &module_info,
        Some((naga::ShaderStage::Compute, desc.entry.as_ref())),
    )?;
    let library = device
        .new_library_with_source(&source, &metal::CompileOptions::new())
        .map_err(anyhow::Error::msg)
        .context("Compiling GpuSurface MSL compute library")?;
    let function = library
        .get_function(desc.entry.as_ref(), None)
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "Locating GpuSurface compute entry '{}' in generated MSL",
                desc.entry
            )
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)
        .context("Creating GpuSurface Metal compute pipeline")?;

    Ok(MetalGpuSurfaceComputeProgram {
        desc: desc.clone(),
        pipeline,
        bindings,
        workgroup_size,
    })
}

fn validate_gpu_surface_module(module: &naga::Module) -> anyhow::Result<naga::valid::ModuleInfo> {
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.subgroup_stages(naga::valid::ShaderStages::all());
    validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());
    validator
        .validate(module)
        .context("Validating GpuSurface WGSL module")
}

fn gpu_surface_program_bindings(
    module: &naga::Module,
) -> anyhow::Result<Vec<MetalGpuSurfaceProgramBinding>> {
    let mut bindings = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            anyhow::bail!("GpuSurface Metal supports only @group(0) bindings");
        }
        let kind = gpu_surface_binding_kind(module, variable)
            .with_context(|| format!("Resolving WGSL binding {}", binding.binding))?;
        if binding.binding == 0 {
            if kind != MetalGpuSurfaceProgramBindingKind::UniformBuffer {
                anyhow::bail!(
                    "GpuSurface Metal reserves @group(0) @binding(0) for the frame uniform buffer"
                );
            }
            continue;
        }
        bindings.push(MetalGpuSurfaceProgramBinding {
            slot: binding.binding,
            kind,
        });
    }
    bindings.sort_by_key(|binding| binding.slot);
    Ok(bindings)
}

fn gpu_surface_binding_kind(
    module: &naga::Module,
    variable: &naga::GlobalVariable,
) -> anyhow::Result<MetalGpuSurfaceProgramBindingKind> {
    use naga::{AddressSpace, TypeInner};

    let inner = &module.types[variable.ty].inner;
    if let TypeInner::BindingArray { .. } = inner {
        anyhow::bail!("GpuSurface Metal does not support binding arrays yet");
    }
    if matches!(inner, TypeInner::Sampler { .. }) {
        return Ok(MetalGpuSurfaceProgramBindingKind::Sampler);
    }
    if let TypeInner::Image { class, .. } = inner {
        return Ok(match class {
            naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. } => {
                MetalGpuSurfaceProgramBindingKind::SampledTexture
            }
            naga::ImageClass::Storage { .. } => MetalGpuSurfaceProgramBindingKind::StorageTexture,
            naga::ImageClass::External => {
                anyhow::bail!("GpuSurface Metal does not support external image bindings")
            }
        });
    }

    match variable.space {
        AddressSpace::Uniform => Ok(MetalGpuSurfaceProgramBindingKind::UniformBuffer),
        AddressSpace::Storage { .. } => Ok(MetalGpuSurfaceProgramBindingKind::StorageBuffer),
        _ => anyhow::bail!(
            "GpuSurface Metal does not support bindings from address space {:?}",
            variable.space
        ),
    }
}

fn gpu_surface_compute_workgroup_size(
    module: &naga::Module,
    entry_name: &str,
) -> anyhow::Result<[u32; 3]> {
    module
        .entry_points
        .iter()
        .find(|entry| entry.stage == naga::ShaderStage::Compute && entry.name == entry_name)
        .map(|entry| entry.workgroup_size)
        .with_context(|| format!("GpuSurface compute entry point '{entry_name}' missing"))
}

fn build_gpu_surface_msl_source(
    module: &naga::Module,
    module_info: &naga::valid::ModuleInfo,
    entry_point: Option<(naga::ShaderStage, &str)>,
) -> anyhow::Result<String> {
    let options = naga::back::msl::Options {
        lang_version: (2, 1),
        per_entry_point_map: gpu_surface_msl_resource_map(module)?,
        fake_missing_bindings: false,
        ..Default::default()
    };
    let pipeline_options = naga::back::msl::PipelineOptions {
        entry_point: entry_point.map(|(stage, name)| (stage, name.to_string())),
        vertex_pulling_transform: false,
        ..Default::default()
    };
    let (source, _) =
        naga::back::msl::write_string(module, module_info, &options, &pipeline_options)
            .context("Generating MSL for GpuSurface program")?;
    Ok(source)
}

fn gpu_surface_msl_resource_map(
    module: &naga::Module,
) -> anyhow::Result<naga::back::msl::EntryPointResourceMap> {
    let resources = naga::back::msl::EntryPointResources {
        resources: gpu_surface_msl_binding_map(module)?,
        ..Default::default()
    };
    Ok(module
        .entry_points
        .iter()
        .map(|entry| (entry.name.clone(), resources.clone()))
        .collect::<BTreeMap<_, _>>())
}

fn gpu_surface_msl_binding_map(
    module: &naga::Module,
) -> anyhow::Result<naga::back::msl::BindingMap> {
    let mut binding_map = naga::back::msl::BindingMap::default();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            anyhow::bail!("GpuSurface Metal supports only @group(0) bindings");
        }

        let slot = gpu_surface_msl_slot(binding.binding)?;
        let mut target = naga::back::msl::BindTarget::default();
        match &module.types[variable.ty].inner {
            naga::TypeInner::BindingArray { .. } => {
                anyhow::bail!("GpuSurface Metal does not support binding arrays yet");
            }
            naga::TypeInner::Sampler { .. } => {
                target.sampler = Some(naga::back::msl::BindSamplerTarget::Resource(slot));
            }
            naga::TypeInner::Image { class, .. } => {
                target.texture = Some(slot);
                if let naga::ImageClass::Storage { access, .. } = class {
                    target.mutable = access.contains(naga::StorageAccess::STORE);
                }
            }
            _ => {
                target.buffer = Some(slot);
                if let naga::AddressSpace::Storage { access } = variable.space {
                    target.mutable = access.contains(naga::StorageAccess::STORE);
                }
            }
        }
        binding_map.insert(binding, target);
    }
    Ok(binding_map)
}

fn gpu_surface_msl_slot(slot: u32) -> anyhow::Result<u8> {
    u8::try_from(slot).context("GpuSurface Metal binding slot exceeds MSL slot limit")
}

fn build_gpu_surface_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: Option<&str>,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> anyhow::Result<metal::RenderPipelineState> {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Locating GpuSurface vertex entry '{vertex_fn_name}'"))?;
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Locating GpuSurface fragment entry '{fragment_fn_name}'"))?;

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label.unwrap_or("gpu_surface_pipeline"));
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .map_err(anyhow::Error::msg)
        .context("Creating GpuSurface Metal render pipeline")
}

fn write_gpu_surface_buffer(
    buffer: &metal::BufferRef,
    offset: u64,
    data: &[u8],
    is_unified_memory: bool,
) {
    if data.is_empty() {
        return;
    }

    unsafe {
        ptr::copy_nonoverlapping(
            data.as_ptr(),
            (buffer.contents() as *mut u8).add(offset as usize),
            data.len(),
        );
    }

    if !is_unified_memory {
        buffer.did_modify_range(NSRange {
            location: offset as NSUInteger,
            length: data.len() as NSUInteger,
        });
    }
}

fn gpu_surface_buffer_size(desc: &GpuBufferDesc) -> u64 {
    match desc.usage {
        GpuBufferUsage::Uniform => desc.size.max(16).div_ceil(16) * 16,
        GpuBufferUsage::Storage => desc.size.max(4).div_ceil(4) * 4,
    }
}

fn gpu_surface_frame_uniform(
    frame: &GpuFrameContext,
    scale_factor: f32,
) -> MetalGpuSurfaceFrameUniform {
    let surface_cursor_position = frame
        .surface_cursor_position
        .unwrap_or(point(px(0.0), px(0.0)))
        .scale(scale_factor);
    let cursor_position = frame.cursor_position.scale(scale_factor);
    MetalGpuSurfaceFrameUniform {
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

fn gpu_texture_format_to_metal(format: GpuTextureFormat) -> MTLPixelFormat {
    match format {
        GpuTextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        GpuTextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        GpuTextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        GpuTextureFormat::R32Float => MTLPixelFormat::R32Float,
        GpuTextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
    }
}

fn gpu_clear_color_to_metal(clear: GpuClearColor) -> metal::MTLClearColor {
    metal::MTLClearColor::new(
        clear.r as f64,
        clear.g as f64,
        clear.b as f64,
        clear.a as f64,
    )
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_pipeline_state_no_blend(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[repr(C)]
enum BackdropBlurInputIndex {
    Vertices = 0,
    BackdropBlurs = 1,
    SourceTexture = 2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}
