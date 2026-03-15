use std::{
    collections::{BTreeSet, HashMap},
    slice,
    sync::{Arc, OnceLock},
};

use ::util::ResultExt;
use anyhow::{Context, Result};
use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
        },
    },
    core::{Interface, PCSTR},
};

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::*;
use nekowg::*;

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "NEKOWG_DISABLE_DIRECT_COMPOSITION";
const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    atlas: Arc<DirectXAtlas>,
    devices: Option<DirectXRendererDevices>,
    resources: Option<DirectXResources>,
    globals: DirectXGlobalElements,
    pipelines: DirectXRenderPipelines,
    direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,

    width: u32,
    height: u32,

    /// Whether we want to skip drwaing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    skip_draws: bool,
    gpu_surfaces: HashMap<u64, DirectXGpuSurfaceState>,
    frame_serial: u64,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    dxgi_device: Option<IDXGIDevice>,
}

struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,

    // Backdrop blur textures
    backdrop_main_texture: ID3D11Texture2D,
    backdrop_main_rtv: Option<ID3D11RenderTargetView>,
    backdrop_main_srv: Option<ID3D11ShaderResourceView>,
    backdrop_temp_texture: ID3D11Texture2D,
    backdrop_temp_rtv: Option<ID3D11RenderTargetView>,
    backdrop_temp_srv: Option<ID3D11ShaderResourceView>,
    backdrop_down2_texture: ID3D11Texture2D,
    backdrop_down2_rtv: Option<ID3D11RenderTargetView>,
    backdrop_down2_srv: Option<ID3D11ShaderResourceView>,
    backdrop_temp2_texture: ID3D11Texture2D,
    backdrop_temp2_rtv: Option<ID3D11RenderTargetView>,
    backdrop_temp2_srv: Option<ID3D11ShaderResourceView>,
    backdrop_down4_texture: ID3D11Texture2D,
    backdrop_down4_rtv: Option<ID3D11RenderTargetView>,
    backdrop_down4_srv: Option<ID3D11ShaderResourceView>,
    backdrop_temp4_texture: ID3D11Texture2D,
    backdrop_temp4_rtv: Option<ID3D11RenderTargetView>,
    backdrop_temp4_srv: Option<ID3D11ShaderResourceView>,

    // Path intermediate textures (with MSAA)
    path_intermediate_texture: ID3D11Texture2D,
    path_intermediate_srv: Option<ID3D11ShaderResourceView>,
    path_intermediate_msaa_texture: ID3D11Texture2D,
    path_intermediate_msaa_view: Option<ID3D11RenderTargetView>,

    // Cached viewport
    viewport: D3D11_VIEWPORT,
}

struct DirectXRenderPipelines {
    shadow_pipeline: PipelineState<Shadow>,
    quad_pipeline: PipelineState<Quad>,
    backdrop_blur_h_pipeline: PipelineState<BackdropBlurInstance>,
    backdrop_blur_composite_pipeline: PipelineState<BackdropBlurInstance>,
    backdrop_blur_blit_pipeline: PipelineState<BackdropBlurInstance>,
    path_rasterization_pipeline: PipelineState<PathRasterizationSprite>,
    path_sprite_pipeline: PipelineState<PathSprite>,
    underline_pipeline: PipelineState<Underline>,
    mono_sprites: PipelineState<MonochromeSprite>,
    subpixel_sprites: PipelineState<SubpixelSprite>,
    poly_sprites: PipelineState<PolychromeSprite>,
    surface_pipeline: PipelineState<SurfaceSprite>,
}

struct DirectXGlobalElements {
    global_params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
    surface_sampler: Option<ID3D11SamplerState>,
}

struct DirectComposition {
    comp_device: IDCompositionDevice,
    comp_target: IDCompositionTarget,
    comp_visual: IDCompositionVisual,
}

struct DirectXGpuSurfaceState {
    textures: HashMap<GpuTextureHandle, DirectXGpuSurfaceTexture>,
    texture_pool: DirectXGpuSurfaceTexturePool<DirectXGpuSurfaceTexture>,
    buffers: HashMap<GpuBufferHandle, DirectXGpuSurfaceBuffer>,
    samplers: HashMap<GpuSamplerHandle, DirectXGpuSurfaceSampler>,
    render_programs: HashMap<GpuRenderProgramHandle, DirectXGpuSurfaceRenderProgram>,
    compute_programs: HashMap<GpuComputeProgramHandle, DirectXGpuSurfaceComputeProgram>,
    frame_uniform_buffer: ID3D11Buffer,
    last_used_frame: u64,
}

struct DirectXGpuSurfaceTexture {
    desc: GpuTextureDesc,
    _texture: ID3D11Texture2D,
    render_target_view: Option<ID3D11RenderTargetView>,
    shader_resource_view: Option<ID3D11ShaderResourceView>,
    unordered_access_view: Option<ID3D11UnorderedAccessView>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DirectXGpuSurfaceTexturePoolKey {
    extent: GpuExtent,
    format: GpuTextureFormat,
    sampled: bool,
    storage: bool,
    render_attachment: bool,
    copy_src: bool,
    copy_dst: bool,
}

impl From<&GpuTextureDesc> for DirectXGpuSurfaceTexturePoolKey {
    fn from(desc: &GpuTextureDesc) -> Self {
        Self {
            extent: desc.extent,
            format: desc.format,
            sampled: desc.sampled,
            storage: desc.storage,
            render_attachment: desc.render_attachment,
            copy_src: desc.copy_src,
            copy_dst: desc.copy_dst,
        }
    }
}

struct DirectXGpuSurfaceTexturePool<T> {
    textures: HashMap<DirectXGpuSurfaceTexturePoolKey, Vec<T>>,
}

impl<T> DirectXGpuSurfaceTexturePool<T> {
    fn recycle(&mut self, key: DirectXGpuSurfaceTexturePoolKey, texture: T) {
        self.textures.entry(key).or_default().push(texture);
    }

    fn take(&mut self, desc: &GpuTextureDesc) -> Option<T> {
        let key = DirectXGpuSurfaceTexturePoolKey::from(desc);
        let mut value = None;
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.textures.entry(key) {
            value = entry.get_mut().pop();
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        value
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.textures.values().map(Vec::len).sum()
    }
}

impl<T> Default for DirectXGpuSurfaceTexturePool<T> {
    fn default() -> Self {
        Self {
            textures: HashMap::default(),
        }
    }
}

struct DirectXGpuSurfaceRenderProgram {
    desc: GpuRenderProgramDesc,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    bindings: Vec<DirectXGpuSurfaceProgramBinding>,
}

struct DirectXGpuSurfaceComputeProgram {
    desc: GpuComputeProgramDesc,
    compute: ID3D11ComputeShader,
    bindings: Vec<DirectXGpuSurfaceProgramBinding>,
}

struct DirectXGpuSurfaceBuffer {
    desc: GpuBufferDesc,
    buffer: ID3D11Buffer,
    data: Vec<u8>,
    shader_resource_view: Option<ID3D11ShaderResourceView>,
    unordered_access_view: Option<ID3D11UnorderedAccessView>,
}

struct DirectXGpuSurfaceSampler {
    desc: GpuSamplerDesc,
    sampler: ID3D11SamplerState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectXGpuSurfaceProgramBindingKind {
    UniformBuffer,
    StorageBufferReadOnly,
    StorageBufferReadWrite,
    SampledTexture,
    StorageTexture,
    Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectXGpuSurfaceProgramBinding {
    slot: u32,
    kind: DirectXGpuSurfaceProgramBindingKind,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct GpuSurfaceFrameUniform {
    metrics: [f32; 4],
    extent_cursor: [f32; 4],
    surface_cursor: [f32; 4],
}

impl DirectXGpuSurfaceState {
    fn new(device: &ID3D11Device) -> Result<Self> {
        Ok(Self {
            textures: HashMap::default(),
            texture_pool: DirectXGpuSurfaceTexturePool::default(),
            buffers: HashMap::default(),
            samplers: HashMap::default(),
            render_programs: HashMap::default(),
            compute_programs: HashMap::default(),
            frame_uniform_buffer: create_constant_buffer(
                device,
                std::mem::size_of::<GpuSurfaceFrameUniform>(),
            )?,
            last_used_frame: 0,
        })
    }

    fn sync_textures(
        &mut self,
        device: &ID3D11Device,
        textures: &HashMap<GpuTextureHandle, GpuTextureDesc>,
    ) -> Result<()> {
        let stale_handles = self
            .textures
            .keys()
            .copied()
            .filter(|handle| !textures.contains_key(handle))
            .collect::<Vec<_>>();
        for handle in stale_handles {
            if let Some(texture) = self.textures.remove(&handle) {
                let key = DirectXGpuSurfaceTexturePoolKey::from(&texture.desc);
                self.texture_pool.recycle(key, texture);
            }
        }

        for (&handle, desc) in textures {
            let needs_recreate = self
                .textures
                .get(&handle)
                .is_none_or(|texture| texture.desc != *desc);
            if needs_recreate {
                if let Some(texture) = self.textures.remove(&handle) {
                    let key = DirectXGpuSurfaceTexturePoolKey::from(&texture.desc);
                    self.texture_pool.recycle(key, texture);
                }
                let mut texture = self
                    .texture_pool
                    .take(desc)
                    .unwrap_or(create_gpu_surface_texture(device, desc)?);
                texture.desc = desc.clone();
                self.textures.insert(handle, texture);
            }
        }

        Ok(())
    }

    fn sync_buffers(
        &mut self,
        device: &ID3D11Device,
        buffers: &HashMap<GpuBufferHandle, GpuBufferDesc>,
    ) -> Result<()> {
        self.buffers
            .retain(|handle, _| buffers.contains_key(handle));

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
        device: &ID3D11Device,
        samplers: &HashMap<GpuSamplerHandle, GpuSamplerDesc>,
    ) -> Result<()> {
        self.samplers
            .retain(|handle, _| samplers.contains_key(handle));

        for (&handle, desc) in samplers {
            let needs_recreate = self
                .samplers
                .get(&handle)
                .is_none_or(|sampler| sampler.desc != *desc);
            if needs_recreate {
                self.samplers
                    .insert(handle, create_gpu_surface_sampler(device, desc)?);
            }
        }

        Ok(())
    }

    fn sync_render_programs(
        &mut self,
        device: &ID3D11Device,
        render_programs: &HashMap<GpuRenderProgramHandle, GpuRenderProgramDesc>,
    ) -> Result<()> {
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
        device: &ID3D11Device,
        compute_programs: &HashMap<GpuComputeProgramHandle, GpuComputeProgramDesc>,
    ) -> Result<()> {
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
        device_context: &ID3D11DeviceContext,
        frame: &GpuFrameContext,
        scale_factor: f32,
        graph: &GpuRecordedGraph,
    ) -> Result<()> {
        self.apply_buffer_writes(device_context, graph.buffer_writes())?;
        for operation in &graph.operations {
            match operation {
                GpuGraphOperation::RenderPass(pass) => {
                    self.execute_render_pass(device_context, frame, scale_factor, pass)?
                }
                GpuGraphOperation::ComputePass(pass) => {
                    self.execute_compute_pass(device_context, frame, scale_factor, pass)?
                }
            }
        }

        Ok(())
    }

    fn apply_buffer_writes(
        &mut self,
        device_context: &ID3D11DeviceContext,
        writes: &[GpuBufferWrite],
    ) -> Result<()> {
        let mut dirty_buffers = Vec::new();
        for write in writes {
            let buffer = self
                .buffers
                .get_mut(&write.buffer)
                .context("GpuSurface buffer write target missing in DX11 state")?;
            let start = write.offset as usize;
            let end = start + write.data.len();
            if end > buffer.desc.size as usize {
                anyhow::bail!(
                    "GpuSurface buffer write exceeds buffer bounds on DX11 execution: offset {}, size {}, capacity {}",
                    write.offset,
                    write.data.len(),
                    buffer.desc.size
                );
            }
            buffer.data[start..end].copy_from_slice(&write.data);
            if !dirty_buffers.contains(&write.buffer) {
                dirty_buffers.push(write.buffer);
            }
        }

        for handle in dirty_buffers {
            let buffer = self
                .buffers
                .get(&handle)
                .context("GpuSurface dirty buffer missing in DX11 state")?;
            upload_gpu_surface_buffer(device_context, buffer)?;
        }

        Ok(())
    }

    fn execute_render_pass(
        &mut self,
        device_context: &ID3D11DeviceContext,
        frame: &GpuFrameContext,
        scale_factor: f32,
        pass: &GpuRenderPassDesc,
    ) -> Result<()> {
        let target = self
            .textures
            .get(&pass.target)
            .context("GpuSurface render pass target missing in DX11 state")?;
        let program = self
            .render_programs
            .get(&pass.program)
            .context("GpuSurface render program missing in DX11 state")?;

        if let Some(clear_color) = pass.clear_color {
            let render_target_view = target
                .render_target_view
                .as_ref()
                .context("GpuSurface render target is not renderable on DX11")?;
            unsafe {
                device_context.ClearRenderTargetView(
                    render_target_view,
                    &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                );
            }
        }

        if pass.bindings.len() != program.bindings.len() {
            anyhow::bail!(
                "GpuSurface render pass binding count mismatch: shader expects {}, graph provides {}",
                program.bindings.len(),
                pass.bindings.len()
            );
        }
        let render_target_view = target
            .render_target_view
            .as_ref()
            .context("GpuSurface render target is not renderable on DX11")?;
        let frame_uniform = gpu_surface_frame_uniform(frame, scale_factor);
        update_buffer(
            device_context,
            &self.frame_uniform_buffer,
            std::slice::from_ref(&frame_uniform),
        )?;

        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: target.desc.extent.width.max(1) as f32,
            Height: target.desc.extent.height.max(1) as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: target.desc.extent.width.max(1) as i32,
            bottom: target.desc.extent.height.max(1) as i32,
        };
        let render_target = [Some(render_target_view.clone())];
        let frame_uniform_buffer = [Some(self.frame_uniform_buffer.clone())];
        let mut extra_constant_buffers = Vec::new();
        let mut extra_shader_resources = Vec::new();
        let mut extra_samplers = Vec::new();
        for (binding, layout) in pass.bindings.iter().zip(program.bindings.iter()) {
            match (binding, layout.kind) {
                (
                    GpuBinding::UniformBuffer(handle),
                    DirectXGpuSurfaceProgramBindingKind::UniformBuffer,
                ) => {
                    let buffer = self
                        .buffers
                        .get(handle)
                        .context("GpuSurface uniform buffer binding missing in DX11 state")?;
                    if buffer.desc.usage != GpuBufferUsage::Uniform {
                        anyhow::bail!("GpuSurface uniform binding points to a non-uniform buffer");
                    }
                    extra_constant_buffers.push((layout.slot, buffer.buffer.clone()));
                }
                (
                    GpuBinding::SampledTexture(handle),
                    DirectXGpuSurfaceProgramBindingKind::SampledTexture,
                ) => {
                    let view = self
                        .textures
                        .get(handle)
                        .context("GpuSurface sampled texture binding missing in DX11 state")?
                        .shader_resource_view
                        .clone()
                        .context("GpuSurface sampled texture is not sampleable on DX11")?;
                    extra_shader_resources.push((layout.slot, view));
                }
                (GpuBinding::Sampler(handle), DirectXGpuSurfaceProgramBindingKind::Sampler) => {
                    let sampler = self
                        .samplers
                        .get(handle)
                        .context("GpuSurface sampler binding missing in DX11 state")?
                        .sampler
                        .clone();
                    extra_samplers.push((layout.slot, sampler));
                }
                (
                    GpuBinding::StorageBuffer(_),
                    DirectXGpuSurfaceProgramBindingKind::StorageBufferReadOnly,
                )
                | (
                    GpuBinding::StorageBuffer(_),
                    DirectXGpuSurfaceProgramBindingKind::StorageBufferReadWrite,
                )
                | (
                    GpuBinding::StorageTexture(_),
                    DirectXGpuSurfaceProgramBindingKind::StorageTexture,
                ) => {
                    anyhow::bail!("GpuSurface DX11 executor does not support storage bindings yet");
                }
                _ => {
                    anyhow::bail!("GpuSurface render pass binding type does not match shader");
                }
            }
        }
        let vertex_count = match pass.draw {
            GpuDrawCall::FullScreenTriangle => 3,
            GpuDrawCall::Triangles { .. } => {
                anyhow::bail!(
                    "GpuSurface DX11 executor supports only FullScreenTriangle draws for now"
                );
            }
        };
        unsafe {
            device_context.OMSetRenderTargets(Some(&render_target), None);
            device_context.RSSetViewports(Some(std::slice::from_ref(&viewport)));
            device_context.RSSetScissorRects(Some(std::slice::from_ref(&scissor)));
            device_context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            device_context.VSSetShader(&program.vertex, None);
            device_context.PSSetShader(&program.fragment, None);
            device_context.VSSetConstantBuffers(0, Some(&frame_uniform_buffer));
            device_context.PSSetConstantBuffers(0, Some(&frame_uniform_buffer));
            for (slot, buffer) in &extra_constant_buffers {
                let constant_buffer = [Some(buffer.clone())];
                device_context.VSSetConstantBuffers(*slot, Some(&constant_buffer));
                device_context.PSSetConstantBuffers(*slot, Some(&constant_buffer));
            }
            for (slot, view) in &extra_shader_resources {
                let shader_resource = [Some(view.clone())];
                device_context.VSSetShaderResources(*slot, Some(&shader_resource));
                device_context.PSSetShaderResources(*slot, Some(&shader_resource));
            }
            for (slot, sampler) in &extra_samplers {
                let sampler_state = [Some(sampler.clone())];
                device_context.VSSetSamplers(*slot, Some(&sampler_state));
                device_context.PSSetSamplers(*slot, Some(&sampler_state));
            }
            device_context.Draw(vertex_count, 0);

            let empty_constant_buffer: [Option<ID3D11Buffer>; 1] = [None];
            let empty_shader_resource: [Option<ID3D11ShaderResourceView>; 1] = [None];
            let empty_sampler: [Option<ID3D11SamplerState>; 1] = [None];
            for (slot, _) in &extra_constant_buffers {
                device_context.VSSetConstantBuffers(*slot, Some(&empty_constant_buffer));
                device_context.PSSetConstantBuffers(*slot, Some(&empty_constant_buffer));
            }
            for (slot, _) in &extra_shader_resources {
                device_context.VSSetShaderResources(*slot, Some(&empty_shader_resource));
                device_context.PSSetShaderResources(*slot, Some(&empty_shader_resource));
            }
            for (slot, _) in &extra_samplers {
                device_context.VSSetSamplers(*slot, Some(&empty_sampler));
                device_context.PSSetSamplers(*slot, Some(&empty_sampler));
            }
        }
        Ok(())
    }

    fn execute_compute_pass(
        &mut self,
        device_context: &ID3D11DeviceContext,
        frame: &GpuFrameContext,
        scale_factor: f32,
        pass: &GpuComputePassDesc,
    ) -> Result<()> {
        if pass.workgroups.contains(&0) {
            return Ok(());
        }

        let program = self
            .compute_programs
            .get(&pass.program)
            .context("GpuSurface compute program missing in DX11 state")?;

        if pass.bindings.len() != program.bindings.len() {
            anyhow::bail!(
                "GpuSurface compute pass binding count mismatch: shader expects {}, graph provides {}",
                program.bindings.len(),
                pass.bindings.len()
            );
        }

        let frame_uniform = gpu_surface_frame_uniform(frame, scale_factor);
        update_buffer(
            device_context,
            &self.frame_uniform_buffer,
            std::slice::from_ref(&frame_uniform),
        )?;

        let frame_uniform_buffer = [Some(self.frame_uniform_buffer.clone())];
        let mut extra_constant_buffers = Vec::new();
        let mut extra_shader_resources = Vec::new();
        let mut extra_unordered_access_views = Vec::new();
        let mut extra_samplers = Vec::new();
        for (binding, layout) in pass.bindings.iter().zip(program.bindings.iter()) {
            match (binding, layout.kind) {
                (
                    GpuBinding::UniformBuffer(handle),
                    DirectXGpuSurfaceProgramBindingKind::UniformBuffer,
                ) => {
                    let buffer = self
                        .buffers
                        .get(handle)
                        .context("GpuSurface uniform buffer binding missing in DX11 state")?;
                    if buffer.desc.usage != GpuBufferUsage::Uniform {
                        anyhow::bail!("GpuSurface uniform binding points to a non-uniform buffer");
                    }
                    extra_constant_buffers.push((layout.slot, buffer.buffer.clone()));
                }
                (
                    GpuBinding::StorageBuffer(handle),
                    DirectXGpuSurfaceProgramBindingKind::StorageBufferReadOnly,
                ) => {
                    let buffer = self
                        .buffers
                        .get(handle)
                        .context("GpuSurface storage buffer binding missing in DX11 state")?;
                    if buffer.desc.usage != GpuBufferUsage::Storage {
                        anyhow::bail!("GpuSurface storage binding points to a non-storage buffer");
                    }
                    let view = buffer
                        .shader_resource_view
                        .clone()
                        .context("GpuSurface storage buffer is not readable on DX11")?;
                    extra_shader_resources.push((layout.slot, view));
                }
                (
                    GpuBinding::StorageBuffer(handle),
                    DirectXGpuSurfaceProgramBindingKind::StorageBufferReadWrite,
                ) => {
                    let buffer = self
                        .buffers
                        .get(handle)
                        .context("GpuSurface storage buffer binding missing in DX11 state")?;
                    if buffer.desc.usage != GpuBufferUsage::Storage {
                        anyhow::bail!("GpuSurface storage binding points to a non-storage buffer");
                    }
                    let view = buffer
                        .unordered_access_view
                        .clone()
                        .context("GpuSurface storage buffer is not writable on DX11")?;
                    extra_unordered_access_views.push((layout.slot, view));
                }
                (
                    GpuBinding::SampledTexture(handle),
                    DirectXGpuSurfaceProgramBindingKind::SampledTexture,
                ) => {
                    let view = self
                        .textures
                        .get(handle)
                        .context("GpuSurface sampled texture binding missing in DX11 state")?
                        .shader_resource_view
                        .clone()
                        .context("GpuSurface sampled texture is not sampleable on DX11")?;
                    extra_shader_resources.push((layout.slot, view));
                }
                (
                    GpuBinding::StorageTexture(handle),
                    DirectXGpuSurfaceProgramBindingKind::StorageTexture,
                ) => {
                    let view = self
                        .textures
                        .get(handle)
                        .context("GpuSurface storage texture binding missing in DX11 state")?
                        .unordered_access_view
                        .clone()
                        .context("GpuSurface storage texture is not writable on DX11")?;
                    extra_unordered_access_views.push((layout.slot, view));
                }
                (GpuBinding::Sampler(handle), DirectXGpuSurfaceProgramBindingKind::Sampler) => {
                    let sampler = self
                        .samplers
                        .get(handle)
                        .context("GpuSurface sampler binding missing in DX11 state")?
                        .sampler
                        .clone();
                    extra_samplers.push((layout.slot, sampler));
                }
                _ => {
                    anyhow::bail!("GpuSurface compute pass binding type does not match shader");
                }
            }
        }

        unsafe {
            device_context.CSSetShader(&program.compute, None);
            device_context.CSSetConstantBuffers(0, Some(&frame_uniform_buffer));
            for (slot, buffer) in &extra_constant_buffers {
                let constant_buffer = [Some(buffer.clone())];
                device_context.CSSetConstantBuffers(*slot, Some(&constant_buffer));
            }
            for (slot, view) in &extra_shader_resources {
                let shader_resource = [Some(view.clone())];
                device_context.CSSetShaderResources(*slot, Some(&shader_resource));
            }
            for (slot, sampler) in &extra_samplers {
                let sampler_state = [Some(sampler.clone())];
                device_context.CSSetSamplers(*slot, Some(&sampler_state));
            }
            for (slot, view) in &extra_unordered_access_views {
                let unordered_access_view = [Some(view.clone())];
                device_context.CSSetUnorderedAccessViews(
                    *slot,
                    1,
                    Some(unordered_access_view.as_ptr()),
                    None,
                );
            }
            device_context.Dispatch(pass.workgroups[0], pass.workgroups[1], pass.workgroups[2]);

            let empty_constant_buffer: [Option<ID3D11Buffer>; 1] = [None];
            let empty_shader_resource: [Option<ID3D11ShaderResourceView>; 1] = [None];
            let empty_sampler: [Option<ID3D11SamplerState>; 1] = [None];
            let empty_unordered_access_view: [Option<ID3D11UnorderedAccessView>; 1] = [None];
            for (slot, _) in &extra_constant_buffers {
                device_context.CSSetConstantBuffers(*slot, Some(&empty_constant_buffer));
            }
            for (slot, _) in &extra_shader_resources {
                device_context.CSSetShaderResources(*slot, Some(&empty_shader_resource));
            }
            for (slot, _) in &extra_samplers {
                device_context.CSSetSamplers(*slot, Some(&empty_sampler));
            }
            for (slot, _) in &extra_unordered_access_views {
                device_context.CSSetUnorderedAccessViews(
                    *slot,
                    1,
                    Some(empty_unordered_access_view.as_ptr()),
                    None,
                );
            }
            device_context.CSSetShader(Option::<&ID3D11ComputeShader>::None, None);
        }

        Ok(())
    }

    fn presented_view(&self, handle: GpuTextureHandle) -> Result<ID3D11ShaderResourceView> {
        self.textures
            .get(&handle)
            .context("GpuSurface present texture missing in DX11 state")?
            .shader_resource_view
            .clone()
            .context("GpuSurface present texture is not sampleable on DX11")
    }
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            dxgi_device,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            direct_composition,
            font_info: Self::get_font_info(),
            width: 1,
            height: 1,
            skip_draws: false,
            gpu_surfaces: HashMap::default(),
            frame_serial: 0,
        })
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(
        &self,
        clear_color: &[f32; 4],
        render_target_view: &ID3D11RenderTargetView,
    ) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let device_context = &self
            .devices
            .as_ref()
            .expect("devices missing")
            .device_context;
        update_buffer(
            device_context,
            self.globals.global_params_buffer.as_ref().unwrap(),
            &[GlobalParams {
                gamma_ratios: self.font_info.gamma_ratios,
                viewport_size: [resources.viewport.Width, resources.viewport.Height],
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
            }],
        )?;
        unsafe {
            device_context.ClearRenderTargetView(render_target_view, clear_color);
            let rtv = Some(render_target_view.clone());
            device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
            let scissor = RECT {
                left: 0,
                top: 0,
                right: resources.viewport.Width as i32,
                bottom: resources.viewport.Height as i32,
            };
            device_context.RSSetScissorRects(Some(slice::from_ref(&scissor)));
        }
        Ok(())
    }

    #[inline]
    fn present(&mut self) -> Result<()> {
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
        };
        result.ok().context("Presenting swap chain failed")
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        let disable_direct_composition = self.direct_composition.is_none();

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
            self.gpu_surfaces.clear();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.globals = globals;
        self.pipelines = pipelines;
        self.direct_composition = direct_composition;
        self.skip_draws = true;
        Ok(())
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        self.frame_serial += 1;
        if self.skip_draws {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            self.prune_stale_gpu_surfaces();
            return Ok(());
        }
        let clear_color = match background_appearance {
            WindowBackgroundAppearance::Opaque => [1.0f32; 4],
            _ => [0.0f32; 4],
        };

        let use_backdrop = !scene.backdrop_filters.is_empty();
        let main_rtv = {
            let resources = self.resources.as_ref().context("resources missing")?;
            if use_backdrop {
                resources.backdrop_main_rtv.clone()
            } else {
                resources.render_target_view.clone()
            }
        };
        let main_rtv = main_rtv.as_ref().context("missing render target view")?;

        self.pre_draw(&clear_color, main_rtv)?;

        self.upload_scene_buffers(scene)?;

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(range.start, range.len()),
                PrimitiveBatch::BackdropFilters(range) => self.draw_backdrop_filters(
                    &scene.backdrop_filters[range],
                    main_rtv,
                ),
                PrimitiveBatch::Quads(range) => self.draw_quads(range.start, range.len()),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    self.draw_paths_to_intermediate(paths, main_rtv)?;
                    self.draw_paths_from_intermediate(paths)
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(range.start, range.len()),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_monochrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    self.draw_subpixel_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_polychrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(&scene.surfaces[range]),
            }
            .context(format!(
                "scene too large:\
                {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.subpixel_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            ))?;
        }

        if use_backdrop {
            self.blit_backdrop_to_frame()?;
        }
        let result = self.present();
        self.prune_stale_gpu_surfaces();
        result
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    RENDER_TARGET_FORMAT,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }

        resources.recreate_resources(devices, width, height)?;

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }

        Ok(())
    }

    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;

        if !scene.shadows.is_empty() {
            self.pipelines.shadow_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.shadows,
            )?;
        }

        if !scene.quads.is_empty() {
            self.pipelines.quad_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.quads,
            )?;
        }

        if !scene.underlines.is_empty() {
            self.pipelines.underline_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.underlines,
            )?;
        }

        if !scene.monochrome_sprites.is_empty() {
            self.pipelines.mono_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.monochrome_sprites,
            )?;
        }

        if !scene.subpixel_sprites.is_empty() {
            self.pipelines.subpixel_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.subpixel_sprites,
            )?;
        }

        if !scene.polychrome_sprites.is_empty() {
            self.pipelines.poly_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.polychrome_sprites,
            )?;
        }

        Ok(())
    }

    fn draw_shadows(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.shadow_pipeline.draw_range(
            &devices.device,
            &devices.device_context,
            slice::from_ref(
                &self
                    .resources
                    .as_ref()
                    .context("resources missing")?
                    .viewport,
            ),
            slice::from_ref(&self.globals.global_params_buffer),
            4,
            start as u32,
            len as u32,
        )
    }

    fn draw_quads(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.quad_pipeline.draw_range(
            &devices.device,
            &devices.device_context,
            slice::from_ref(
                &self
                    .resources
                    .as_ref()
                    .context("resources missing")?
                    .viewport,
            ),
            slice::from_ref(&self.globals.global_params_buffer),
            4,
            start as u32,
            len as u32,
        )
    }

    fn draw_paths_to_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        render_target_view: &ID3D11RenderTargetView,
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        // Clear intermediate MSAA texture
        unsafe {
            devices.device_context.ClearRenderTargetView(
                resources.path_intermediate_msaa_view.as_ref().unwrap(),
                &[0.0; 4],
            );
            // Set intermediate MSAA texture as render target
            devices.device_context.OMSetRenderTargets(
                Some(slice::from_ref(&resources.path_intermediate_msaa_view)),
                None,
            );
        }

        // Collect all vertices and sprites for a single draw call
        let mut vertices = Vec::new();

        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationSprite {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
            }));
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &vertices,
        )?;

        self.pipelines.path_rasterization_pipeline.draw(
            &devices.device_context,
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            vertices.len() as u32,
            1,
        )?;

        // Resolve MSAA to non-MSAA intermediate texture
        unsafe {
            devices.device_context.ResolveSubresource(
                &resources.path_intermediate_texture,
                0,
                &resources.path_intermediate_msaa_texture,
                0,
                RENDER_TARGET_FORMAT,
            );
            // Restore main render target
            let rtv = Some(render_target_view.clone());
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
        }

        Ok(())
    }

    fn draw_paths_from_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites = if paths.last().unwrap().order == first_path.order {
            paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect::<Vec<_>>()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &sprites,
        )?;

        // Draw the sprites with the path texture
        self.pipelines.path_sprite_pipeline.draw_with_texture(
            &devices.device_context,
            slice::from_ref(&resources.path_intermediate_srv),
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            slice::from_ref(&self.globals.sampler),
            sprites.len() as u32,
        )
    }

    fn draw_backdrop_filters(
        &mut self,
        filters: &[BackdropFilter],
        main_rtv: &ID3D11RenderTargetView,
    ) -> Result<()> {
        if filters.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let device = &devices.device;
        let device_context = &devices.device_context;

        let full_width = self.width;
        let full_height = self.height;
        let full_viewport = resources.viewport;

        let set_viewport = |width: u32, height: u32| {
            let viewport = D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            unsafe { device_context.RSSetViewports(Some(slice::from_ref(&viewport))) };
            viewport
        };

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
            let expanded = filter.bounds.dilate(ScaledPixels::from(kernel + 2.0));
            let expanded_scaled = if scale == 1 {
                expanded
            } else {
                scale_bounds(expanded, 1.0 / scale as f32)
            };

            let target_width = match scale {
                1 => full_width,
                2 => full_width.div_ceil(2),
                _ => full_width.div_ceil(4),
            };
            let target_height = match scale {
                1 => full_height,
                2 => full_height.div_ceil(2),
                _ => full_height.div_ceil(4),
            };

            let scissor_expanded =
                scissor_from_bounds(expanded_scaled, target_width, target_height);
            let scissor_full = scissor_from_bounds(filter.bounds, full_width, full_height);

            let passes = blur_passes(radius, scale);
            let pass_radius = if passes > 1 {
                radius / (passes as f32).sqrt()
            } else {
                radius
            };
            let (weights0, weights1, step) = gaussian_weights(pass_radius, scale as f32);
            let texel_step = [step / target_width as f32, step / target_height as f32];

            let (down_rtv, down_srv, temp_rtv, temp_srv) = match scale {
                1 => (
                    None,
                    None,
                    resources
                        .backdrop_temp_rtv
                        .as_ref()
                        .context("missing backdrop temp rtv")?,
                    resources
                        .backdrop_temp_srv
                        .as_ref()
                        .context("missing backdrop temp srv")?,
                ),
                2 => (
                    Some(
                        resources
                            .backdrop_down2_rtv
                            .as_ref()
                            .context("missing backdrop down2 rtv")?,
                    ),
                    Some(
                        resources
                            .backdrop_down2_srv
                            .as_ref()
                            .context("missing backdrop down2 srv")?,
                    ),
                    resources
                        .backdrop_temp2_rtv
                        .as_ref()
                        .context("missing backdrop temp2 rtv")?,
                    resources
                        .backdrop_temp2_srv
                        .as_ref()
                        .context("missing backdrop temp2 srv")?,
                ),
                _ => (
                    Some(
                        resources
                            .backdrop_down4_rtv
                            .as_ref()
                            .context("missing backdrop down4 rtv")?,
                    ),
                    Some(
                        resources
                            .backdrop_down4_srv
                            .as_ref()
                            .context("missing backdrop down4 srv")?,
                    ),
                    resources
                        .backdrop_temp4_rtv
                        .as_ref()
                        .context("missing backdrop temp4 rtv")?,
                    resources
                        .backdrop_temp4_srv
                        .as_ref()
                        .context("missing backdrop temp4 srv")?,
                ),
            };

            let main_srv = resources
                .backdrop_main_srv
                .as_ref()
                .context("missing backdrop main srv")?;

            if scale > 1 {
                let Some(scissor) = scissor_expanded else {
                    continue;
                };
                unsafe {
                    let rtv = Some(down_rtv.unwrap().clone());
                    device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
                    device_context.RSSetScissorRects(Some(slice::from_ref(&scissor)));
                }
                let down_viewport = set_viewport(target_width, target_height);

                let instance = BackdropBlurInstance {
                    bounds: expanded_scaled,
                    content_mask: expanded_scaled,
                    corner_radii: Corners::default(),
                    opacity: 1.0,
                    saturation: 1.0,
                    _pad0: [0.0; 2],
                    tint: Hsla::default(),
                    direction: [1.0, 0.0],
                    texel_step: [0.0, 0.0],
                    viewport_size: [target_width as f32, target_height as f32],
                    _pad1: [0.0; 2],
                    weights0: [1.0, 0.0, 0.0, 0.0],
                    weights1: [0.0, 0.0, 0.0, 0.0],
                };
                self.pipelines.backdrop_blur_blit_pipeline.update_buffer(
                    device,
                    device_context,
                    &[instance],
                )?;
                self.pipelines
                    .backdrop_blur_blit_pipeline
                    .draw_with_texture(
                        device_context,
                        slice::from_ref(&Some(main_srv.clone())),
                        slice::from_ref(&down_viewport),
                        slice::from_ref(&self.globals.global_params_buffer),
                        slice::from_ref(&self.globals.sampler),
                        1,
                    )?;
            }

            let mut source_srv = down_srv.unwrap_or(main_srv);
            let Some(scissor) = scissor_expanded else {
                continue;
            };
            let blur_viewport = if scale == 1 {
                full_viewport
            } else {
                set_viewport(target_width, target_height)
            };

            for pass_index in 0..passes {
                unsafe {
                    let rtv = Some(temp_rtv.clone());
                    device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
                    device_context.RSSetScissorRects(Some(slice::from_ref(&scissor)));
                }

                let blur_instance = BackdropBlurInstance {
                    bounds: expanded_scaled,
                    content_mask: expanded_scaled,
                    corner_radii: Corners::default(),
                    opacity: 1.0,
                    saturation: 1.0,
                    _pad0: [0.0; 2],
                    tint: Hsla::default(),
                    direction: [1.0, 0.0],
                    texel_step,
                    viewport_size: [target_width as f32, target_height as f32],
                    _pad1: [0.0; 2],
                    weights0,
                    weights1,
                };
                self.pipelines.backdrop_blur_h_pipeline.update_buffer(
                    device,
                    device_context,
                    &[blur_instance],
                )?;
                self.pipelines.backdrop_blur_h_pipeline.draw_with_texture(
                    device_context,
                    slice::from_ref(&Some(source_srv.clone())),
                    slice::from_ref(&blur_viewport),
                    slice::from_ref(&self.globals.global_params_buffer),
                    slice::from_ref(&self.globals.sampler),
                    1,
                )?;

                if pass_index + 1 == passes {
                    let Some(scissor_full) = scissor_full else {
                        continue;
                    };
                    unsafe {
                        let rtv = Some(main_rtv.clone());
                        device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
                        device_context.RSSetScissorRects(Some(slice::from_ref(&scissor_full)));
                    }
                    set_viewport(full_width, full_height);

                    let composite_instance = BackdropBlurInstance {
                        bounds: filter.bounds,
                        content_mask: filter.content_mask.bounds,
                        corner_radii: filter.corner_radii,
                        opacity: filter.opacity,
                        saturation: filter.saturation,
                        _pad0: [0.0; 2],
                        tint: filter.tint,
                        direction: [0.0, 1.0],
                        texel_step,
                        viewport_size: [full_width as f32, full_height as f32],
                        _pad1: [0.0; 2],
                        weights0,
                        weights1,
                    };
                    self.pipelines
                        .backdrop_blur_composite_pipeline
                        .update_buffer(device, device_context, &[composite_instance])?;
                    self.pipelines
                        .backdrop_blur_composite_pipeline
                        .draw_with_texture(
                            device_context,
                            slice::from_ref(&Some(temp_srv.clone())),
                            slice::from_ref(&full_viewport),
                            slice::from_ref(&self.globals.global_params_buffer),
                            slice::from_ref(&self.globals.sampler),
                            1,
                        )?;
                } else {
                    let down_rtv = down_rtv.expect("missing backdrop down rtv");
                    let down_srv = down_srv.expect("missing backdrop down srv");
                    unsafe {
                        let rtv = Some(down_rtv.clone());
                        device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
                        device_context.RSSetScissorRects(Some(slice::from_ref(&scissor)));
                    }
                    self.pipelines.backdrop_blur_h_pipeline.update_buffer(
                        device,
                        device_context,
                        &[BackdropBlurInstance {
                            bounds: expanded_scaled,
                            content_mask: expanded_scaled,
                            corner_radii: Corners::default(),
                            opacity: 1.0,
                            saturation: 1.0,
                            _pad0: [0.0; 2],
                            tint: Hsla::default(),
                            direction: [0.0, 1.0],
                            texel_step,
                            viewport_size: [target_width as f32, target_height as f32],
                            _pad1: [0.0; 2],
                            weights0,
                            weights1,
                        }],
                    )?;
                    self.pipelines.backdrop_blur_h_pipeline.draw_with_texture(
                        device_context,
                        slice::from_ref(&Some(temp_srv.clone())),
                        slice::from_ref(&blur_viewport),
                        slice::from_ref(&self.globals.global_params_buffer),
                        slice::from_ref(&self.globals.sampler),
                        1,
                    )?;
                    source_srv = down_srv;
                }
            }
        }

        unsafe {
            let rtv = Some(main_rtv.clone());
            device_context.OMSetRenderTargets(Some(slice::from_ref(&rtv)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&full_viewport)));
            let full_scissor = RECT {
                left: 0,
                top: 0,
                right: full_width as i32,
                bottom: full_height as i32,
            };
            device_context.RSSetScissorRects(Some(slice::from_ref(&full_scissor)));
        }

        Ok(())
    }

    fn blit_backdrop_to_frame(&mut self) -> Result<()> {
        let resources = self.resources.as_ref().context("resources missing")?;
        let devices = self.devices.as_ref().context("devices missing")?;
        let main_srv = resources
            .backdrop_main_srv
            .as_ref()
            .context("missing backdrop main srv")?;
        let backbuffer_rtv = resources
            .render_target_view
            .as_ref()
            .context("missing render target view")?;

        let instance = BackdropBlurInstance {
            bounds: Bounds {
                origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size: size(
                    ScaledPixels(self.width as f32),
                    ScaledPixels(self.height as f32),
                ),
            },
            content_mask: Bounds {
                origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size: size(
                    ScaledPixels(self.width as f32),
                    ScaledPixels(self.height as f32),
                ),
            },
            corner_radii: Corners::default(),
            opacity: 1.0,
            saturation: 1.0,
            _pad0: [0.0; 2],
            tint: Hsla::default(),
            direction: [1.0, 0.0],
            texel_step: [0.0, 0.0],
            viewport_size: [self.width as f32, self.height as f32],
            _pad1: [0.0; 2],
            weights0: [1.0, 0.0, 0.0, 0.0],
            weights1: [0.0, 0.0, 0.0, 0.0],
        };

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&Some(backbuffer_rtv.clone()))), None);
            devices
                .device_context
                .RSSetViewports(Some(slice::from_ref(&resources.viewport)));
            let full_scissor = RECT {
                left: 0,
                top: 0,
                right: self.width as i32,
                bottom: self.height as i32,
            };
            devices
                .device_context
                .RSSetScissorRects(Some(slice::from_ref(&full_scissor)));
        }

        self.pipelines.backdrop_blur_blit_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &[instance],
        )?;
        self.pipelines
            .backdrop_blur_blit_pipeline
            .draw_with_texture(
                &devices.device_context,
                slice::from_ref(&Some(main_srv.clone())),
                slice::from_ref(&resources.viewport),
                slice::from_ref(&self.globals.global_params_buffer),
                slice::from_ref(&self.globals.sampler),
                1,
            )
    }

    fn draw_underlines(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        self.pipelines.underline_pipeline.draw_range(
            &devices.device,
            &devices.device_context,
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            4,
            start as u32,
            len as u32,
        )
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.mono_sprites.draw_range_with_texture(
            &devices.device,
            &devices.device_context,
            &texture_view,
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.subpixel_sprites.draw_range_with_texture(
            &devices.device,
            &devices.device_context,
            &texture_view,
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.poly_sprites.draw_range_with_texture(
            &devices.device,
            &devices.device_context,
            &texture_view,
            slice::from_ref(&resources.viewport),
            slice::from_ref(&self.globals.global_params_buffer),
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_surfaces(&mut self, surfaces: &[PaintSurface]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let sprites = surfaces
            .iter()
            .map(|surface| SurfaceSprite {
                bounds: surface.bounds,
                content_mask: surface.content_mask.bounds,
                corner_radii: surface.corner_radii,
            })
            .collect::<Vec<_>>();
        self.pipelines.surface_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &sprites,
        )?;

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_view: ID3D11ShaderResourceView = surface
                .texture_view
                .as_ref()
                .context("DX11 surface batch requires a texture view")?
                .cast()
                .context("Casting GpuSurface texture view to DX11 SRV")?;
            self.pipelines.surface_pipeline.draw_range_with_texture(
                &devices.device,
                &devices.device_context,
                &[Some(texture_view)],
                slice::from_ref(&resources.viewport),
                slice::from_ref(&self.globals.global_params_buffer),
                slice::from_ref(&self.globals.surface_sampler),
                index as u32,
                1,
            )?;
        }

        Ok(())
    }

    pub(crate) fn paint_gpu_surface(
        &mut self,
        input: GpuSurfaceExecutionInput<'_>,
    ) -> Result<Option<PaintSurface>> {
        let Some(presented) = input.graph.presented else {
            return Ok(None);
        };
        let frame = input
            .frame
            .context("GpuSurface DX11 executor requires a prepared frame context")?;
        let devices = self.devices.as_ref().context("devices missing")?;
        let state = self
            .gpu_surfaces
            .entry(input.surface_id)
            .or_insert(DirectXGpuSurfaceState::new(&devices.device)?);
        state.last_used_frame = self.frame_serial + 1;
        state.sync_textures(&devices.device, input.textures)?;
        state.sync_buffers(&devices.device, input.buffers)?;
        state.sync_samplers(&devices.device, input.samplers)?;
        state.sync_render_programs(&devices.device, input.render_programs)?;
        state.sync_compute_programs(&devices.device, input.compute_programs)?;
        state.execute_graph(
            &devices.device_context,
            frame,
            input.scale_factor,
            input.graph,
        )?;
        let texture_view = state.presented_view(presented)?;

        Ok(Some(PaintSurface {
            order: 0,
            bounds: input.bounds,
            content_mask: input.content_mask,
            corner_radii: input.corner_radii,
            texture_view: Some(texture_view.cast()?),
            gpu_surface_id: None,
        }))
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
            let render_params: IDWriteRenderingParams1 =
                factory.CreateRenderingParams().unwrap().cast().unwrap();
            FontInfo {
                gamma_ratios: nekowg::get_gamma_correction_ratios(render_params.GetGamma()),
                grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }

    fn prune_stale_gpu_surfaces(&mut self) {
        let frame_serial = self.frame_serial;
        self.gpu_surfaces
            .retain(|_, state| state.last_used_frame >= frame_serial);
    }
}

impl DirectXResources {
    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (
            render_target,
            render_target_view,
            path_intermediate_texture,
            path_intermediate_srv,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            viewport,
        ) = create_resources(devices, &swap_chain, width, height)?;

        let (backdrop_main_texture, backdrop_main_rtv, backdrop_main_srv) =
            create_backdrop_texture(&devices.device, width, height)?;
        let (backdrop_temp_texture, backdrop_temp_rtv, backdrop_temp_srv) =
            create_backdrop_texture(&devices.device, width, height)?;

        let down2_width = width.div_ceil(2);
        let down2_height = height.div_ceil(2);
        let down4_width = width.div_ceil(4);
        let down4_height = height.div_ceil(4);

        let (backdrop_down2_texture, backdrop_down2_rtv, backdrop_down2_srv) =
            create_backdrop_texture(&devices.device, down2_width, down2_height)?;
        let (backdrop_temp2_texture, backdrop_temp2_rtv, backdrop_temp2_srv) =
            create_backdrop_texture(&devices.device, down2_width, down2_height)?;
        let (backdrop_down4_texture, backdrop_down4_rtv, backdrop_down4_srv) =
            create_backdrop_texture(&devices.device, down4_width, down4_height)?;
        let (backdrop_temp4_texture, backdrop_temp4_rtv, backdrop_temp4_srv) =
            create_backdrop_texture(&devices.device, down4_width, down4_height)?;
        set_rasterizer_state(&devices.device, &devices.device_context)?;

        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
            backdrop_main_texture,
            backdrop_main_rtv,
            backdrop_main_srv,
            backdrop_temp_texture,
            backdrop_temp_rtv,
            backdrop_temp_srv,
            backdrop_down2_texture,
            backdrop_down2_rtv,
            backdrop_down2_srv,
            backdrop_temp2_texture,
            backdrop_temp2_rtv,
            backdrop_temp2_srv,
            backdrop_down4_texture,
            backdrop_down4_rtv,
            backdrop_down4_srv,
            backdrop_temp4_texture,
            backdrop_temp4_rtv,
            backdrop_temp4_srv,
            path_intermediate_texture,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            path_intermediate_srv,
            viewport,
        })
    }

    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (
            render_target,
            render_target_view,
            path_intermediate_texture,
            path_intermediate_srv,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            viewport,
        ) = create_resources(devices, &self.swap_chain, width, height)?;

        let (backdrop_main_texture, backdrop_main_rtv, backdrop_main_srv) =
            create_backdrop_texture(&devices.device, width, height)?;
        let (backdrop_temp_texture, backdrop_temp_rtv, backdrop_temp_srv) =
            create_backdrop_texture(&devices.device, width, height)?;

        let down2_width = width.div_ceil(2);
        let down2_height = height.div_ceil(2);
        let down4_width = width.div_ceil(4);
        let down4_height = height.div_ceil(4);

        let (backdrop_down2_texture, backdrop_down2_rtv, backdrop_down2_srv) =
            create_backdrop_texture(&devices.device, down2_width, down2_height)?;
        let (backdrop_temp2_texture, backdrop_temp2_rtv, backdrop_temp2_srv) =
            create_backdrop_texture(&devices.device, down2_width, down2_height)?;
        let (backdrop_down4_texture, backdrop_down4_rtv, backdrop_down4_srv) =
            create_backdrop_texture(&devices.device, down4_width, down4_height)?;
        let (backdrop_temp4_texture, backdrop_temp4_rtv, backdrop_temp4_srv) =
            create_backdrop_texture(&devices.device, down4_width, down4_height)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        self.backdrop_main_texture = backdrop_main_texture;
        self.backdrop_main_rtv = backdrop_main_rtv;
        self.backdrop_main_srv = backdrop_main_srv;
        self.backdrop_temp_texture = backdrop_temp_texture;
        self.backdrop_temp_rtv = backdrop_temp_rtv;
        self.backdrop_temp_srv = backdrop_temp_srv;
        self.backdrop_down2_texture = backdrop_down2_texture;
        self.backdrop_down2_rtv = backdrop_down2_rtv;
        self.backdrop_down2_srv = backdrop_down2_srv;
        self.backdrop_temp2_texture = backdrop_temp2_texture;
        self.backdrop_temp2_rtv = backdrop_temp2_rtv;
        self.backdrop_temp2_srv = backdrop_temp2_srv;
        self.backdrop_down4_texture = backdrop_down4_texture;
        self.backdrop_down4_rtv = backdrop_down4_rtv;
        self.backdrop_down4_srv = backdrop_down4_srv;
        self.backdrop_temp4_texture = backdrop_temp4_texture;
        self.backdrop_temp4_rtv = backdrop_temp4_rtv;
        self.backdrop_temp4_srv = backdrop_temp4_srv;
        self.path_intermediate_texture = path_intermediate_texture;
        self.path_intermediate_msaa_texture = path_intermediate_msaa_texture;
        self.path_intermediate_msaa_view = path_intermediate_msaa_view;
        self.path_intermediate_srv = path_intermediate_srv;
        self.viewport = viewport;
        Ok(())
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?;
        let backdrop_blur_h_pipeline = PipelineState::new(
            device,
            "backdrop_blur_h_pipeline",
            ShaderModule::BackdropBlurH,
            4,
            create_blend_state_no_blend(device)?,
        )?;
        let backdrop_blur_composite_pipeline = PipelineState::new(
            device,
            "backdrop_blur_composite_pipeline",
            ShaderModule::BackdropBlurComposite,
            4,
            create_blend_state(device)?,
        )?;
        let backdrop_blur_blit_pipeline = PipelineState::new(
            device,
            "backdrop_blur_blit_pipeline",
            ShaderModule::BackdropBlurBlit,
            4,
            create_blend_state_no_blend(device)?,
        )?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_blend_state_for_path_rasterization(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_blend_state_for_path_sprite(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?;
        let surface_pipeline = PipelineState::new(
            device,
            "surface_pipeline",
            ShaderModule::Surface,
            16,
            create_blend_state(device)?,
        )?;

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            backdrop_blur_h_pipeline,
            backdrop_blur_composite_pipeline,
            backdrop_blur_blit_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surface_pipeline,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, true) }?;
        let comp_visual = unsafe { comp_device.CreateVisual() }?;

        Ok(Self {
            comp_device,
            comp_target,
            comp_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.comp_visual.SetContent(swap_chain)?;
            self.comp_target.SetRoot(&self.comp_visual)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let global_params_buffer = unsafe {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<GlobalParams>() as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                ..Default::default()
            };
            let mut buffer = None;
            device.CreateBuffer(&desc, None, Some(&mut buffer))?;
            buffer
        };

        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        let surface_sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        Ok(Self {
            global_params_buffer,
            sampler,
            surface_sampler,
        })
    }
}

#[derive(Debug, Default)]
#[repr(C)]
struct GlobalParams {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

struct PipelineState<T> {
    label: &'static str,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    buffer: ID3D11Buffer,
    buffer_size: usize,
    view: Option<ID3D11ShaderResourceView>,
    blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            vertex,
            fragment,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        if self.buffer_size < data.len() {
            let new_buffer_size = data.len().next_power_of_two();
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer(device_context, &self.buffer, data)
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        viewport: &[D3D11_VIEWPORT],
        global_params: &[Option<ID3D11Buffer>],
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            topology,
            viewport,
            &self.vertex,
            &self.fragment,
            global_params,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        viewport: &[D3D11_VIEWPORT],
        global_params: &[Option<ID3D11Buffer>],
        sampler: &[Option<ID3D11SamplerState>],
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            viewport,
            &self.vertex,
            &self.fragment,
            global_params,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));

            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        viewport: &[D3D11_VIEWPORT],
        global_params: &[Option<ID3D11Buffer>],
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        let view = create_buffer_view_range(device, &self.buffer, first_instance, instance_count)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            viewport,
            &self.vertex,
            &self.fragment,
            global_params,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range_with_texture(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        viewport: &[D3D11_VIEWPORT],
        global_params: &[Option<ID3D11Buffer>],
        sampler: &[Option<ID3D11SamplerState>],
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        let view = create_buffer_view_range(device, &self.buffer, first_instance, instance_count)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            viewport,
            &self.vertex,
            &self.fragment,
            global_params,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathRasterizationSprite {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy)]
#[repr(C)]
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

#[derive(Clone, Copy)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SurfaceSprite {
    bounds: Bounds<ScaledPixels>,
    content_mask: Bounds<ScaledPixels>,
    corner_radii: Corners<ScaledPixels>,
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    ID3D11Texture2D,
    Option<ID3D11ShaderResourceView>,
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device)?;
    let (path_intermediate_texture, path_intermediate_srv) =
        create_path_intermediate_texture(&devices.device, width, height)?;
    let (path_intermediate_msaa_texture, path_intermediate_msaa_view) =
        create_path_intermediate_msaa_texture_and_view(&devices.device, width, height)?;
    let viewport = set_viewport(&devices.device_context, width as f32, height as f32);
    Ok((
        render_target,
        render_target_view,
        path_intermediate_texture,
        path_intermediate_srv,
        path_intermediate_msaa_texture,
        path_intermediate_msaa_view,
        viewport,
    ))
}

#[inline]
fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0) }?;
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };

    Ok((texture, Some(shader_resource_view.unwrap())))
}

#[inline]
fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

#[inline]
fn create_backdrop_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    Option<ID3D11ShaderResourceView>,
)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width.max(1),
            Height: height.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut rtv = None;
    let mut srv = None;
    unsafe {
        device.CreateRenderTargetView(&texture, None, Some(&mut rtv))?;
        device.CreateShaderResourceView(&texture, None, Some(&mut srv))?;
    }

    Ok((texture, rtv, srv))
}

fn create_gpu_surface_texture(
    device: &ID3D11Device,
    desc: &GpuTextureDesc,
) -> Result<DirectXGpuSurfaceTexture> {
    let bind_flags = if desc.sampled {
        D3D11_BIND_SHADER_RESOURCE.0 as u32
    } else {
        0
    } | if desc.render_attachment {
        D3D11_BIND_RENDER_TARGET.0 as u32
    } else {
        0
    } | if desc.storage {
        D3D11_BIND_UNORDERED_ACCESS.0 as u32
    } else {
        0
    };
    let texture = unsafe {
        let mut output = None;
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: desc.extent.width.max(1),
            Height: desc.extent.height.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: gpu_texture_format_to_dxgi(desc.format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flags,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&texture_desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    if desc.sampled {
        unsafe {
            device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))?;
        }
    }

    let mut render_target_view = None;
    if desc.render_attachment {
        unsafe {
            device.CreateRenderTargetView(&texture, None, Some(&mut render_target_view))?;
        }
    }

    let mut unordered_access_view = None;
    if desc.storage {
        unsafe {
            device.CreateUnorderedAccessView(&texture, None, Some(&mut unordered_access_view))?;
        }
    }

    Ok(DirectXGpuSurfaceTexture {
        desc: desc.clone(),
        _texture: texture,
        render_target_view,
        shader_resource_view,
        unordered_access_view,
    })
}

fn create_gpu_surface_buffer(
    device: &ID3D11Device,
    desc: &GpuBufferDesc,
) -> Result<DirectXGpuSurfaceBuffer> {
    let (buffer, shader_resource_view, unordered_access_view) = match desc.usage {
        GpuBufferUsage::Uniform => (
            create_constant_buffer(device, desc.size as usize)?,
            None,
            None,
        ),
        GpuBufferUsage::Storage => {
            let buffer_desc = D3D11_BUFFER_DESC {
                ByteWidth: gpu_surface_buffer_byte_width(desc),
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32
                    | D3D11_BIND_UNORDERED_ACCESS.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
                StructureByteStride: 0,
            };
            let mut output = None;
            unsafe { device.CreateBuffer(&buffer_desc, None, Some(&mut output))? };
            let buffer = output.unwrap();
            let shader_resource_view = Some(
                create_gpu_surface_storage_buffer_shader_resource_view(device, &buffer, desc)?,
            );
            let unordered_access_view = Some(
                create_gpu_surface_storage_buffer_unordered_access_view(device, &buffer, desc)?,
            );
            (buffer, shader_resource_view, unordered_access_view)
        }
    };

    Ok(DirectXGpuSurfaceBuffer {
        desc: desc.clone(),
        buffer,
        data: vec![0; gpu_surface_buffer_byte_width(desc) as usize],
        shader_resource_view,
        unordered_access_view,
    })
}

fn create_gpu_surface_sampler(
    device: &ID3D11Device,
    desc: &GpuSamplerDesc,
) -> Result<DirectXGpuSurfaceSampler> {
    let state = unsafe {
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_ALWAYS,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: D3D11_FLOAT32_MAX,
        };
        let mut output = None;
        device.CreateSamplerState(&sampler_desc, Some(&mut output))?;
        output.unwrap()
    };

    Ok(DirectXGpuSurfaceSampler {
        desc: desc.clone(),
        sampler: state,
    })
}

fn gpu_surface_buffer_byte_width(desc: &GpuBufferDesc) -> u32 {
    match desc.usage {
        GpuBufferUsage::Uniform => ((desc.size as u32).max(1)).div_ceil(16) * 16,
        GpuBufferUsage::Storage => ((desc.size as u32).max(4)).div_ceil(4) * 4,
    }
}

fn upload_gpu_surface_buffer(
    device_context: &ID3D11DeviceContext,
    buffer: &DirectXGpuSurfaceBuffer,
) -> Result<()> {
    match buffer.desc.usage {
        GpuBufferUsage::Uniform => {
            update_buffer(device_context, &buffer.buffer, buffer.data.as_slice())
        }
        GpuBufferUsage::Storage => {
            unsafe {
                device_context.UpdateSubresource(
                    &buffer.buffer,
                    0,
                    None,
                    buffer.data.as_ptr() as _,
                    0,
                    0,
                );
            }
            Ok(())
        }
    }
}

fn create_gpu_surface_storage_buffer_shader_resource_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    desc: &GpuBufferDesc,
) -> Result<ID3D11ShaderResourceView> {
    let byte_width = gpu_surface_buffer_byte_width(desc);
    let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D_SRV_DIMENSION_BUFFEREX,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            BufferEx: D3D11_BUFFEREX_SRV {
                FirstElement: 0,
                NumElements: byte_width / 4,
                Flags: D3D11_BUFFEREX_SRV_FLAG_RAW.0 as u32,
            },
        },
    };
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, Some(&view_desc), Some(&mut view))? };
    Ok(view.expect("DX11 storage buffer SRV creation should populate the view"))
}

fn create_gpu_surface_storage_buffer_unordered_access_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    desc: &GpuBufferDesc,
) -> Result<ID3D11UnorderedAccessView> {
    let byte_width = gpu_surface_buffer_byte_width(desc);
    let view_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
        Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D11_BUFFER_UAV {
                FirstElement: 0,
                NumElements: byte_width / 4,
                Flags: D3D11_BUFFER_UAV_FLAG_RAW.0 as u32,
            },
        },
    };
    let mut view = None;
    unsafe { device.CreateUnorderedAccessView(buffer, Some(&view_desc), Some(&mut view))? };
    Ok(view.expect("DX11 storage buffer UAV creation should populate the view"))
}

fn create_gpu_surface_render_program(
    device: &ID3D11Device,
    desc: &GpuRenderProgramDesc,
) -> Result<DirectXGpuSurfaceRenderProgram> {
    let module =
        naga::front::wgsl::parse_str(desc.wgsl.as_ref()).context("Parsing GpuSurface WGSL")?;
    let bindings = gpu_surface_program_bindings(&module)?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.subgroup_stages(naga::valid::ShaderStages::all());
    validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());
    let module_info = validator
        .validate(&module)
        .context("Validating GpuSurface WGSL module")?;
    let sampler_bindings = gpu_surface_hlsl_sampler_bindings(&module)?;
    let vertex_hlsl = build_gpu_surface_hlsl_source(
        &module,
        &module_info,
        naga::ShaderStage::Vertex,
        desc.vertex_entry.as_ref(),
        desc.fragment_entry.as_ref(),
        &sampler_bindings,
    )?;
    let fragment_hlsl = build_gpu_surface_hlsl_source(
        &module,
        &module_info,
        naga::ShaderStage::Fragment,
        desc.fragment_entry.as_ref(),
        desc.fragment_entry.as_ref(),
        &sampler_bindings,
    )?;
    let vertex_blob = compile_hlsl_blob(
        &vertex_hlsl,
        desc.vertex_entry.as_ref(),
        "vs_5_0",
        desc.label.as_ref().map(|label| label.as_ref()),
    )?;
    let fragment_blob = compile_hlsl_blob(
        &fragment_hlsl,
        desc.fragment_entry.as_ref(),
        "ps_5_0",
        desc.label.as_ref().map(|label| label.as_ref()),
    )?;
    let vertex = create_vertex_shader(device, blob_bytes(&vertex_blob))?;
    let fragment = create_fragment_shader(device, blob_bytes(&fragment_blob))?;

    Ok(DirectXGpuSurfaceRenderProgram {
        desc: desc.clone(),
        vertex,
        fragment,
        bindings,
    })
}

fn create_gpu_surface_compute_program(
    device: &ID3D11Device,
    desc: &GpuComputeProgramDesc,
) -> Result<DirectXGpuSurfaceComputeProgram> {
    let module =
        naga::front::wgsl::parse_str(desc.wgsl.as_ref()).context("Parsing GpuSurface WGSL")?;
    let bindings = gpu_surface_program_bindings(&module)?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.subgroup_stages(naga::valid::ShaderStages::all());
    validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());
    let module_info = validator
        .validate(&module)
        .context("Validating GpuSurface WGSL module")?;
    let sampler_bindings = gpu_surface_hlsl_sampler_bindings(&module)?;
    let compute_hlsl = build_gpu_surface_hlsl_source(
        &module,
        &module_info,
        naga::ShaderStage::Compute,
        desc.entry.as_ref(),
        desc.entry.as_ref(),
        &sampler_bindings,
    )?;
    let compute_blob = compile_hlsl_blob(
        &compute_hlsl,
        desc.entry.as_ref(),
        "cs_5_0",
        desc.label.as_ref().map(|label| label.as_ref()),
    )?;
    let compute = create_compute_shader(device, blob_bytes(&compute_blob))?;

    Ok(DirectXGpuSurfaceComputeProgram {
        desc: desc.clone(),
        compute,
        bindings,
    })
}

fn gpu_surface_program_bindings(
    module: &naga::Module,
) -> Result<Vec<DirectXGpuSurfaceProgramBinding>> {
    let mut bindings = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            anyhow::bail!("GpuSurface DX11 supports only @group(0) bindings");
        }
        let kind = gpu_surface_binding_kind(module, variable)
            .with_context(|| format!("Failed to resolve WGSL binding {}", binding.binding))?;
        if binding.binding == 0 {
            if kind != DirectXGpuSurfaceProgramBindingKind::UniformBuffer {
                anyhow::bail!(
                    "GpuSurface DX11 reserves @group(0) @binding(0) for the frame uniform buffer"
                );
            }
            continue;
        }
        bindings.push(DirectXGpuSurfaceProgramBinding {
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
) -> Result<DirectXGpuSurfaceProgramBindingKind> {
    use naga::{AddressSpace, StorageAccess, TypeInner};

    let inner = &module.types[variable.ty].inner;
    if let TypeInner::BindingArray { .. } = inner {
        anyhow::bail!("GpuSurface DX11 does not support binding arrays yet");
    }
    if matches!(inner, TypeInner::Sampler { .. }) {
        return Ok(DirectXGpuSurfaceProgramBindingKind::Sampler);
    }
    if let TypeInner::Image { class, .. } = inner {
        return Ok(match class {
            naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. } => {
                DirectXGpuSurfaceProgramBindingKind::SampledTexture
            }
            naga::ImageClass::Storage { .. } => DirectXGpuSurfaceProgramBindingKind::StorageTexture,
            naga::ImageClass::External => {
                anyhow::bail!("GpuSurface DX11 does not support external image bindings")
            }
        });
    }

    match variable.space {
        AddressSpace::Uniform => Ok(DirectXGpuSurfaceProgramBindingKind::UniformBuffer),
        AddressSpace::Storage { access } => {
            if access.contains(StorageAccess::STORE) {
                Ok(DirectXGpuSurfaceProgramBindingKind::StorageBufferReadWrite)
            } else {
                Ok(DirectXGpuSurfaceProgramBindingKind::StorageBufferReadOnly)
            }
        }
        _ => anyhow::bail!(
            "GpuSurface DX11 does not support bindings from address space {:?}",
            variable.space
        ),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GpuSurfaceHlslSamplerBinding {
    name: String,
    group: u32,
    slot: u32,
    comparison: bool,
}

fn gpu_surface_hlsl_sampler_bindings(
    module: &naga::Module,
) -> Result<Vec<GpuSurfaceHlslSamplerBinding>> {
    let mut sampler_bindings = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            anyhow::bail!("GpuSurface DX11 supports only @group(0) bindings");
        }
        let naga::TypeInner::Sampler { comparison } = module.types[variable.ty].inner else {
            continue;
        };
        let name = variable
            .name
            .clone()
            .context("GpuSurface DX11 requires named WGSL sampler bindings")?;
        sampler_bindings.push(GpuSurfaceHlslSamplerBinding {
            name,
            group: binding.group,
            slot: binding.binding,
            comparison,
        });
    }
    Ok(sampler_bindings)
}

fn build_gpu_surface_hlsl_source(
    module: &naga::Module,
    module_info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_point: &str,
    fragment_entry_point: &str,
    sampler_bindings: &[GpuSurfaceHlslSamplerBinding],
) -> Result<String> {
    let mut binding_map = naga::back::hlsl::BindingMap::default();
    let mut sampler_groups = BTreeSet::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            anyhow::bail!("GpuSurface DX11 supports only @group(0) bindings");
        }
        if matches!(
            module.types[variable.ty].inner,
            naga::TypeInner::Sampler { .. }
        ) {
            sampler_groups.insert(binding.group);
        }
        binding_map.insert(
            binding,
            naga::back::hlsl::BindTarget {
                space: 0,
                register: binding.binding,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
    }
    let mut sampler_buffer_binding_map = naga::back::hlsl::SamplerIndexBufferBindingMap::default();
    let mut next_sampler_buffer_register = binding_map
        .values()
        .map(|target| target.register)
        .max()
        .unwrap_or(0)
        + 1;
    for group in sampler_groups {
        sampler_buffer_binding_map.insert(
            naga::back::hlsl::SamplerIndexBufferKey { group },
            naga::back::hlsl::BindTarget {
                space: 0,
                register: next_sampler_buffer_register,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
        next_sampler_buffer_register += 1;
    }
    let options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_0,
        binding_map,
        sampler_buffer_binding_map,
        fake_missing_bindings: false,
        ..Default::default()
    };
    let pipeline_options = naga::back::hlsl::PipelineOptions {
        entry_point: Some((stage, entry_point.to_string())),
    };
    let vertex_fragment_entry =
        naga::back::hlsl::FragmentEntryPoint::new(module, fragment_entry_point);
    let mut output = String::new();
    naga::back::hlsl::Writer::new(&mut output, &options, &pipeline_options)
        .write(
            module,
            module_info,
            if stage == naga::ShaderStage::Vertex {
                vertex_fragment_entry.as_ref()
            } else {
                None
            },
        )
        .context("Generating HLSL for GpuSurface render program")?;
    strip_gpu_surface_hlsl_register_spaces(rewrite_gpu_surface_hlsl_sampler_bindings(
        output,
        sampler_bindings,
    )?)
}

fn rewrite_gpu_surface_hlsl_sampler_bindings(
    source: String,
    sampler_bindings: &[GpuSurfaceHlslSamplerBinding],
) -> Result<String> {
    if sampler_bindings.is_empty() {
        return Ok(source);
    }

    let mut rewritten = Vec::new();
    let mut replaced_samplers = 0usize;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("SamplerState nagaSamplerHeap[2048]: register(")
            || trimmed
                .starts_with("SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(")
            || (trimmed.starts_with("StructuredBuffer<uint> nagaGroup")
                && trimmed.contains("SamplerIndexArray : register("))
        {
            continue;
        }

        let mut replaced = false;
        for binding in sampler_bindings {
            let sampler_type = if binding.comparison {
                "SamplerComparisonState"
            } else {
                "SamplerState"
            };
            let heap_var = if binding.comparison {
                "nagaComparisonSamplerHeap"
            } else {
                "nagaSamplerHeap"
            };
            let expected = format!(
                "static const {sampler_type} {} = {heap_var}[nagaGroup{}SamplerIndexArray[{}]];",
                binding.name, binding.group, binding.slot
            );
            if trimmed == expected {
                rewritten.push(format!(
                    "{sampler_type} {} : register(s{});",
                    binding.name, binding.slot
                ));
                replaced_samplers += 1;
                replaced = true;
                break;
            }
        }
        if !replaced {
            rewritten.push(line.to_string());
        }
    }

    if replaced_samplers != sampler_bindings.len() {
        anyhow::bail!("Failed to rewrite all GpuSurface sampler bindings for DX11 HLSL generation");
    }

    Ok(rewritten.join("\n"))
}

fn strip_gpu_surface_hlsl_register_spaces(source: String) -> Result<String> {
    if source.contains("space1") {
        anyhow::bail!(
            "GpuSurface DX11 HLSL generation left an unsupported non-zero register space"
        );
    }
    Ok(source.replace(", space0)", ")"))
}

fn compile_hlsl_blob(
    source: &str,
    entry_point: &str,
    target: &str,
    source_name: Option<&str>,
) -> Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    let source_name = source_name.map(std::ffi::CString::new).transpose()?;
    let entry_point = std::ffi::CString::new(entry_point)?;
    let target = std::ffi::CString::new(target)?;
    let mut shader_blob = None;
    let mut error_blob = None;
    let result = unsafe {
        windows::Win32::Graphics::Direct3D::Fxc::D3DCompile(
            source.as_ptr() as *const _,
            source.len(),
            source_name
                .as_ref()
                .map(|name| PCSTR::from_raw(name.as_ptr() as *const u8))
                .unwrap_or(PCSTR::null()),
            None,
            None,
            PCSTR::from_raw(entry_point.as_ptr() as *const u8),
            PCSTR::from_raw(target.as_ptr() as *const u8),
            windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &mut shader_blob,
            Some(&mut error_blob),
        )
    };
    if let Err(error) = result {
        if let Some(error_blob) = error_blob {
            let error_string =
                unsafe { std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8) }
                    .to_string_lossy()
                    .into_owned();
            anyhow::bail!("Compiling GpuSurface HLSL failed: {error_string}");
        }
        return Err(error).context("Compiling GpuSurface HLSL failed");
    }

    Ok(shader_blob.expect("D3DCompile should populate the shader blob"))
}

fn blob_bytes(blob: &windows::Win32::Graphics::Direct3D::ID3DBlob) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
    }
}

fn create_constant_buffer(device: &ID3D11Device, size: usize) -> Result<ID3D11Buffer> {
    let byte_width = size.div_ceil(16) * 16;
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: byte_width as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer))? };
    Ok(buffer.unwrap())
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

fn gpu_texture_format_to_dxgi(format: GpuTextureFormat) -> DXGI_FORMAT {
    match format {
        GpuTextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        GpuTextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        GpuTextureFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        GpuTextureFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        GpuTextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
    }
}

#[inline]
fn set_viewport(device_context: &ID3D11DeviceContext, width: f32, height: f32) -> D3D11_VIEWPORT {
    let viewport = [D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width,
        Height: height,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }];
    unsafe { device_context.RSSetViewports(Some(&viewport)) };
    viewport[0]
}

fn scale_bounds(bounds: Bounds<ScaledPixels>, factor: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(bounds.origin.x * factor, bounds.origin.y * factor),
        size: size(bounds.size.width * factor, bounds.size.height * factor),
    }
}

fn scissor_from_bounds(
    bounds: Bounds<ScaledPixels>,
    target_width: u32,
    target_height: u32,
) -> Option<RECT> {
    let min_x = bounds.origin.x.0.max(0.0).floor() as i32;
    let min_y = bounds.origin.y.0.max(0.0).floor() as i32;
    let max_x = (bounds.origin.x.0 + bounds.size.width.0)
        .min(target_width as f32)
        .ceil() as i32;
    let max_y = (bounds.origin.y.0 + bounds.size.height.0)
        .min(target_height as f32)
        .ceil() as i32;

    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some(RECT {
        left: min_x,
        top: min_y,
        right: max_x,
        bottom: max_y,
    })
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

fn blur_passes(radius: f32, scale: u32) -> u32 {
    if scale == 1 {
        return 1;
    }
    let radius_ds = (radius / scale as f32).max(0.0);
    let ratio = radius_ds / 8.0;
    let passes = (ratio * ratio).ceil() as u32;
    passes.clamp(1, 6)
}

#[inline]
fn set_rasterizer_state(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Result<()> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: true.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let rasterizer_state = unsafe {
        let mut state = None;
        device.CreateRasterizerState(&desc, Some(&mut state))?;
        state.unwrap()
    };
    unsafe { device_context.RSSetState(&rasterizer_state) };
    Ok(())
}

// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_blend_desc
#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_no_blend(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = false.into();
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_rasterization(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_sprite(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_vertex_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_fragment_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_compute_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11ComputeShader> {
    unsafe {
        let mut shader = None;
        device.CreateComputeShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: (element_size * buffer_size) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
        StructureByteStride: element_size as u32,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, None, Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn create_buffer_view_range(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    first_element: u32,
    num_elements: u32,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D11_SRV_DIMENSION_BUFFER,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D11_BUFFER_SRV {
                Anonymous1: D3D11_BUFFER_SRV_0 {
                    FirstElement: first_element,
                },
                Anonymous2: D3D11_BUFFER_SRV_1 {
                    NumElements: num_elements,
                },
            },
        },
    };
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, Some(&desc), Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DirectXGpuSurfaceProgramBinding, DirectXGpuSurfaceProgramBindingKind,
        DirectXGpuSurfaceTexturePool, GpuSurfaceHlslSamplerBinding,
        rewrite_gpu_surface_hlsl_sampler_bindings,
    };
    use nekowg::{GpuExtent, GpuTextureDesc, GpuTextureFormat};

    #[test]
    fn gpu_surface_texture_pool_reuses_matching_desc_ignoring_labels() {
        let mut pool = DirectXGpuSurfaceTexturePool::default();
        let desc = GpuTextureDesc {
            label: Some("first".into()),
            extent: GpuExtent {
                width: 96,
                height: 64,
            },
            format: GpuTextureFormat::Rgba8Unorm,
            sampled: true,
            storage: true,
            render_attachment: false,
            copy_src: false,
            copy_dst: false,
        };
        let same_texture_different_label = GpuTextureDesc {
            label: Some("second".into()),
            ..desc.clone()
        };

        pool.recycle((&desc).into(), 7u32);

        assert_eq!(pool.len(), 1);
        assert_eq!(pool.take(&same_texture_different_label), Some(7));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn gpu_surface_texture_pool_keeps_distinct_descs_separate() {
        let mut pool = DirectXGpuSurfaceTexturePool::default();
        let base = GpuTextureDesc {
            extent: GpuExtent {
                width: 64,
                height: 64,
            },
            format: GpuTextureFormat::Rgba8Unorm,
            sampled: true,
            storage: true,
            render_attachment: false,
            copy_src: false,
            copy_dst: false,
            ..GpuTextureDesc::default()
        };
        let different_extent = GpuTextureDesc {
            extent: GpuExtent {
                width: 128,
                height: 64,
            },
            ..base.clone()
        };

        pool.recycle((&base).into(), 11u32);

        assert_eq!(pool.take(&different_extent), None);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.take(&base), Some(11));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn rewrites_sampler_heap_bindings_into_direct_sampler_registers() {
        let source = r#"
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t3, space0);
Texture2D<float4> source_tex : register(t1, space0);
static const SamplerState source_sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[2]];
"#
        .trim()
        .to_string();

        let rewritten = rewrite_gpu_surface_hlsl_sampler_bindings(
            source,
            &[GpuSurfaceHlslSamplerBinding {
                name: "source_sampler".into(),
                group: 0,
                slot: 2,
                comparison: false,
            }],
        )
        .expect("sampler rewrite should succeed");

        let rewritten = super::strip_gpu_surface_hlsl_register_spaces(rewritten)
            .expect("space stripping should work");

        assert!(rewritten.contains("Texture2D<float4> source_tex : register(t1);"));
        assert!(rewritten.contains("SamplerState source_sampler : register(s2);"));
        assert!(!rewritten.contains("nagaSamplerHeap"));
        assert!(!rewritten.contains("nagaGroup0SamplerIndexArray"));
    }

    #[test]
    fn generated_gpu_surface_hlsl_strips_register_spaces_for_shader_model_5_0() {
        let wgsl = r#"
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
    return textureSample(source_tex, source_sampler, input.uv);
}
"#;
        let module = naga::front::wgsl::parse_str(wgsl).expect("WGSL should parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.subgroup_stages(naga::valid::ShaderStages::all());
        validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());
        let module_info = validator.validate(&module).expect("WGSL should validate");
        let sampler_bindings = super::gpu_surface_hlsl_sampler_bindings(&module)
            .expect("sampler bindings should resolve");
        let source = super::build_gpu_surface_hlsl_source(
            &module,
            &module_info,
            naga::ShaderStage::Fragment,
            "fs_main",
            "fs_main",
            &sampler_bindings,
        )
        .expect("HLSL generation should succeed");

        assert!(
            !source.contains("space"),
            "generated HLSL still contains register spaces:\n{source}"
        );
    }

    #[test]
    fn storage_buffer_bindings_track_access_mode() {
        let wgsl = r#"
struct GpuSurfaceFrame {
    metrics: vec4<f32>,
    extent_cursor: vec4<f32>,
    surface_cursor: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: GpuSurfaceFrame;

@group(0) @binding(1)
var<storage, read> input_data: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output_data: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn cs_main() {}
"#;
        let module = naga::front::wgsl::parse_str(wgsl).expect("WGSL should parse");
        let bindings =
            super::gpu_surface_program_bindings(&module).expect("bindings should resolve");

        assert_eq!(
            bindings,
            vec![
                DirectXGpuSurfaceProgramBinding {
                    slot: 1,
                    kind: DirectXGpuSurfaceProgramBindingKind::StorageBufferReadOnly,
                },
                DirectXGpuSurfaceProgramBinding {
                    slot: 2,
                    kind: DirectXGpuSurfaceProgramBindingKind::StorageBufferReadWrite,
                },
            ]
        );
    }
}

#[inline]
fn set_pipeline_state(
    device_context: &ID3D11DeviceContext,
    buffer_view: &[Option<ID3D11ShaderResourceView>],
    topology: D3D_PRIMITIVE_TOPOLOGY,
    viewport: &[D3D11_VIEWPORT],
    vertex_shader: &ID3D11VertexShader,
    fragment_shader: &ID3D11PixelShader,
    global_params: &[Option<ID3D11Buffer>],
    blend_state: &ID3D11BlendState,
) {
    unsafe {
        device_context.VSSetShaderResources(1, Some(buffer_view));
        device_context.PSSetShaderResources(1, Some(buffer_view));
        device_context.IASetPrimitiveTopology(topology);
        device_context.RSSetViewports(Some(viewport));
        device_context.VSSetShader(vertex_shader, None);
        device_context.PSSetShader(fragment_shader, None);
        device_context.VSSetConstantBuffers(0, Some(global_params));
        device_context.PSSetConstantBuffers(0, Some(global_params));
        device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
    }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

const BUFFER_COUNT: usize = 3;

pub(crate) mod shader_resources {
    use anyhow::Result;

    #[cfg(debug_assertions)]
    use windows::{
        Win32::Graphics::Direct3D::{
            Fxc::{D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompileFromFile},
            ID3DBlob,
        },
        core::{HSTRING, PCSTR},
    };

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        Shadow,
        BackdropBlurH,
        BackdropBlurComposite,
        BackdropBlurBlit,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        Surface,
        EmojiRasterization,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderTarget {
        Vertex,
        Fragment,
    }

    pub(crate) struct RawShaderBytes<'t> {
        inner: &'t [u8],

        #[cfg(debug_assertions)]
        _blob: ID3DBlob,
    }

    impl<'t> RawShaderBytes<'t> {
        pub(crate) fn new(module: ShaderModule, target: ShaderTarget) -> Result<Self> {
            #[cfg(not(debug_assertions))]
            {
                Ok(Self::from_bytes(module, target))
            }
            #[cfg(debug_assertions)]
            {
                let blob = build_shader_blob(module, target)?;
                let inner = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    )
                };
                Ok(Self { inner, _blob: blob })
            }
        }

        pub(crate) fn as_bytes(&'t self) -> &'t [u8] {
            self.inner
        }

        #[cfg(not(debug_assertions))]
        fn from_bytes(module: ShaderModule, target: ShaderTarget) -> Self {
            let bytes = match module {
                ShaderModule::Quad => match target {
                    ShaderTarget::Vertex => QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::Shadow => match target {
                    ShaderTarget::Vertex => SHADOW_VERTEX_BYTES,
                    ShaderTarget::Fragment => SHADOW_FRAGMENT_BYTES,
                },
                ShaderModule::BackdropBlurH => match target {
                    ShaderTarget::Vertex => BACKDROP_BLUR_H_VERTEX_BYTES,
                    ShaderTarget::Fragment => BACKDROP_BLUR_H_FRAGMENT_BYTES,
                },
                ShaderModule::BackdropBlurComposite => match target {
                    ShaderTarget::Vertex => BACKDROP_BLUR_COMPOSITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => BACKDROP_BLUR_COMPOSITE_FRAGMENT_BYTES,
                },
                ShaderModule::BackdropBlurBlit => match target {
                    ShaderTarget::Vertex => BACKDROP_BLUR_BLIT_VERTEX_BYTES,
                    ShaderTarget::Fragment => BACKDROP_BLUR_BLIT_FRAGMENT_BYTES,
                },
                ShaderModule::Underline => match target {
                    ShaderTarget::Vertex => UNDERLINE_VERTEX_BYTES,
                    ShaderTarget::Fragment => UNDERLINE_FRAGMENT_BYTES,
                },
                ShaderModule::PathRasterization => match target {
                    ShaderTarget::Vertex => PATH_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::PathSprite => match target {
                    ShaderTarget::Vertex => PATH_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::MonochromeSprite => match target {
                    ShaderTarget::Vertex => MONOCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MONOCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::SubpixelSprite => match target {
                    ShaderTarget::Vertex => SUBPIXEL_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SUBPIXEL_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::PolychromeSprite => match target {
                    ShaderTarget::Vertex => POLYCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => POLYCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::Surface => match target {
                    ShaderTarget::Vertex => SURFACE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SURFACE_FRAGMENT_BYTES,
                },
                ShaderModule::EmojiRasterization => match target {
                    ShaderTarget::Vertex => EMOJI_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => EMOJI_RASTERIZATION_FRAGMENT_BYTES,
                },
            };
            Self { inner: bytes }
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn build_shader_blob(entry: ShaderModule, target: ShaderTarget) -> Result<ID3DBlob> {
        unsafe {
            use windows::Win32::Graphics::{
                Direct3D::ID3DInclude, Hlsl::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            };

            let shader_name = if matches!(entry, ShaderModule::EmojiRasterization) {
                "color_text_raster.hlsl"
            } else {
                "shaders.hlsl"
            };

            let entry = format!(
                "{}_{}\0",
                entry.as_str(),
                match target {
                    ShaderTarget::Vertex => "vertex",
                    ShaderTarget::Fragment => "fragment",
                }
            );
            let target = match target {
                ShaderTarget::Vertex => "vs_4_1\0",
                ShaderTarget::Fragment => "ps_4_1\0",
            };

            let mut compile_blob = None;
            let mut error_blob = None;
            let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(&format!("src/{}", shader_name))
                .canonicalize()?;

            let entry_point = PCSTR::from_raw(entry.as_ptr());
            let target_cstr = PCSTR::from_raw(target.as_ptr());

            // really dirty trick because winapi bindings are unhappy otherwise
            let include_handler = &std::mem::transmute::<usize, ID3DInclude>(
                D3D_COMPILE_STANDARD_FILE_INCLUDE as usize,
            );

            let ret = D3DCompileFromFile(
                &HSTRING::from(shader_path.to_str().unwrap()),
                None,
                include_handler,
                entry_point,
                target_cstr,
                D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
                0,
                &mut compile_blob,
                Some(&mut error_blob),
            );
            if ret.is_err() {
                let Some(error_blob) = error_blob else {
                    return Err(anyhow::anyhow!("{ret:?}"));
                };

                let error_string =
                    std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8)
                        .to_string_lossy();
                log::error!("Shader compile error: {}", error_string);
                return Err(anyhow::anyhow!("Compile error: {}", error_string));
            }
            Ok(compile_blob.unwrap())
        }
    }

    #[cfg(not(debug_assertions))]
    include!(concat!(env!("OUT_DIR"), "/shaders_bytes.rs"));

    #[cfg(debug_assertions)]
    impl ShaderModule {
        pub fn as_str(self) -> &'static str {
            match self {
                ShaderModule::Quad => "quad",
                ShaderModule::Shadow => "shadow",
                ShaderModule::BackdropBlurH => "backdrop_blur_h",
                ShaderModule::BackdropBlurComposite => "backdrop_blur_composite",
                ShaderModule::BackdropBlurBlit => "backdrop_blur_blit",
                ShaderModule::Underline => "underline",
                ShaderModule::PathRasterization => "path_rasterization",
                ShaderModule::PathSprite => "path_sprite",
                ShaderModule::MonochromeSprite => "monochrome_sprite",
                ShaderModule::SubpixelSprite => "subpixel_sprite",
                ShaderModule::PolychromeSprite => "polychrome_sprite",
                ShaderModule::Surface => "surface",
                ShaderModule::EmojiRasterization => "emoji_rasterization",
            }
        }
    }
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}
