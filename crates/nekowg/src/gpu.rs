use crate::{App, Bounds, ContentMask, ElementId, Pixels, Point, ScaledPixels, SharedString};
use bytemuck::Pod;
use scheduler::Instant;
use std::{collections::HashMap, mem, time::Duration};

/// A GPU surface extent in physical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuExtent {
    /// Surface width in physical pixels.
    pub width: u32,
    /// Surface height in physical pixels.
    pub height: u32,
}

impl GpuExtent {
    /// Returns true when either dimension is zero.
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Determines how a GPU surface requests future frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GpuSurfaceRedrawMode {
    /// Only redraw when the surrounding view tree is refreshed.
    #[default]
    OnDemand,
    /// Request a frame every refresh cycle while the surface remains mounted.
    Animated,
    /// Refresh is externally driven by the owning view/application.
    ExternalWake,
}

/// A GPU texture format supported by the portable surface API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GpuTextureFormat {
    /// 8-bit normalized RGBA texture.
    #[default]
    Rgba8Unorm,
    /// 8-bit normalized BGRA texture.
    Bgra8Unorm,
    /// 16-bit floating-point RGBA texture.
    Rgba16Float,
    /// Single-channel 32-bit floating-point texture.
    R32Float,
    /// 32-bit floating-point RGBA texture.
    Rgba32Float,
}

/// A coarse usage description for a GPU buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GpuBufferUsage {
    /// Uniform buffer usage.
    #[default]
    Uniform,
    /// Storage buffer usage.
    Storage,
}

/// A portable texture descriptor for `GpuSurface`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuTextureDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// Texture extent.
    pub extent: GpuExtent,
    /// Texture format.
    pub format: GpuTextureFormat,
    /// Whether the texture can be sampled.
    pub sampled: bool,
    /// Whether the texture can be used as a storage texture.
    pub storage: bool,
    /// Whether the texture can be used as a render target.
    pub render_attachment: bool,
    /// Whether the texture can be copied from.
    pub copy_src: bool,
    /// Whether the texture can be copied to.
    pub copy_dst: bool,
}

impl GpuTextureDesc {
    /// Sets a debug label for the texture.
    pub fn with_label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A portable buffer descriptor for `GpuSurface`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuBufferDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// Buffer usage class.
    pub usage: GpuBufferUsage,
    /// Buffer size in bytes.
    pub size: u64,
}

impl GpuBufferDesc {
    /// Sets a debug label for the buffer.
    pub fn with_label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A portable sampler descriptor for `GpuSurface`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuSamplerDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
}

impl GpuSamplerDesc {
    /// Sets a debug label for the sampler.
    pub fn with_label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A render program written in WGSL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuRenderProgramDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// WGSL shader source.
    pub wgsl: SharedString,
    /// Vertex entry point.
    pub vertex_entry: SharedString,
    /// Fragment entry point.
    pub fragment_entry: SharedString,
}

impl GpuRenderProgramDesc {
    /// Sets a debug label for the program.
    pub fn with_label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A compute program written in WGSL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuComputeProgramDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// WGSL shader source.
    pub wgsl: SharedString,
    /// Compute entry point.
    pub entry: SharedString,
}

impl GpuComputeProgramDesc {
    /// Sets a debug label for the program.
    pub fn with_label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A texture handle owned by a `GpuSurface`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuTextureHandle(pub u64);

/// A persistent ping-pong texture pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuTexturePair {
    /// First texture.
    pub a: GpuTextureHandle,
    /// Second texture.
    pub b: GpuTextureHandle,
}

/// A buffer handle owned by a `GpuSurface`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuBufferHandle(pub u64);

/// A sampler handle owned by a `GpuSurface`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuSamplerHandle(pub u64);

/// A render program handle owned by a `GpuSurface`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuRenderProgramHandle(pub u64);

/// A compute program handle owned by a `GpuSurface`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuComputeProgramHandle(pub u64);

/// A resource binding used by render/compute passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuBinding {
    /// Uniform buffer binding.
    UniformBuffer(GpuBufferHandle),
    /// Storage buffer binding.
    StorageBuffer(GpuBufferHandle),
    /// Sampled texture binding.
    SampledTexture(GpuTextureHandle),
    /// Storage texture binding.
    StorageTexture(GpuTextureHandle),
    /// Sampler binding.
    Sampler(GpuSamplerHandle),
}

/// How a render pass issues geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuDrawCall {
    /// Draw a full-screen triangle.
    FullScreenTriangle,
    /// Draw N vertices with M instances.
    Triangles {
        /// Vertex count.
        vertex_count: u32,
        /// Instance count.
        instance_count: u32,
    },
}

/// A clear color for a render pass target.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuClearColor {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

/// A recorded render pass.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuRenderPassDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// Render program.
    pub program: GpuRenderProgramHandle,
    /// Target texture.
    pub target: GpuTextureHandle,
    /// Optional clear color.
    pub clear_color: Option<GpuClearColor>,
    /// Resource bindings.
    pub bindings: Vec<GpuBinding>,
    /// Geometry submission mode.
    pub draw: GpuDrawCall,
}

/// A recorded compute pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuComputePassDesc {
    /// Optional debug label.
    pub label: Option<SharedString>,
    /// Compute program.
    pub program: GpuComputeProgramHandle,
    /// Resource bindings.
    pub bindings: Vec<GpuBinding>,
    /// Workgroup counts.
    pub workgroups: [u32; 3],
}

/// A recorded graph operation.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuGraphOperation {
    /// Render pass operation.
    RenderPass(GpuRenderPassDesc),
    /// Compute pass operation.
    ComputePass(GpuComputePassDesc),
}

/// A frame graph recorded for a single `GpuSurface` frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuRecordedGraph {
    /// Operations in submission order.
    pub operations: Vec<GpuGraphOperation>,
    /// Transient textures allocated for this frame only.
    pub transient_textures: Vec<GpuTextureHandle>,
    /// Final presented texture, if any.
    pub presented: Option<GpuTextureHandle>,
    buffer_writes: Vec<GpuBufferWrite>,
}

/// A buffer upload recorded for the current frame graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBufferWrite {
    /// Target buffer handle.
    pub buffer: GpuBufferHandle,
    /// Byte offset into the target buffer.
    pub offset: u64,
    /// Uploaded bytes.
    pub data: Vec<u8>,
}

/// Internal execution input passed from `GpuSurface` to a platform backend.
#[doc(hidden)]
pub struct GpuSurfaceExecutionInput<'a> {
    /// Stable per-surface backend cache key.
    pub surface_id: u64,
    /// Surface bounds in physical pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// Final visible content mask in physical pixels.
    pub content_mask: ContentMask<ScaledPixels>,
    /// Surface corner radii in physical pixels.
    pub corner_radii: crate::Corners<ScaledPixels>,
    /// Window scale factor used to convert logical frame input into physical units.
    pub scale_factor: f32,
    /// Recorded frame graph for the current frame.
    pub graph: &'a GpuRecordedGraph,
    /// Most recently prepared frame context, if available.
    pub frame: Option<&'a GpuFrameContext>,
    /// Texture descriptors visible to the recorded graph.
    pub textures: &'a HashMap<GpuTextureHandle, GpuTextureDesc>,
    /// Buffer descriptors visible to the recorded graph.
    pub buffers: &'a HashMap<GpuBufferHandle, GpuBufferDesc>,
    /// Sampler descriptors visible to the recorded graph.
    pub samplers: &'a HashMap<GpuSamplerHandle, GpuSamplerDesc>,
    /// Render program descriptors visible to the recorded graph.
    pub render_programs: &'a HashMap<GpuRenderProgramHandle, GpuRenderProgramDesc>,
    /// Compute program descriptors visible to the recorded graph.
    pub compute_programs: &'a HashMap<GpuComputeProgramHandle, GpuComputeProgramDesc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuGraphValidationError {
    MissingTexture(GpuTextureHandle),
    MissingBuffer(GpuBufferHandle),
    MissingSampler(GpuSamplerHandle),
    MissingRenderProgram(GpuRenderProgramHandle),
    MissingComputeProgram(GpuComputeProgramHandle),
    NonRenderableTarget(GpuTextureHandle),
    NonSampledPresentTexture(GpuTextureHandle),
    NonUniformBufferBinding(GpuBufferHandle),
    NonStorageBufferBinding(GpuBufferHandle),
    NonSampledTextureBinding(GpuTextureHandle),
    NonStorageTextureBinding(GpuTextureHandle),
    RenderTargetSampledConflict(GpuTextureHandle),
    RenderTargetStorageConflict(GpuTextureHandle),
    SampledStorageTextureConflict(GpuTextureHandle),
    UniformStorageBufferConflict(GpuBufferHandle),
    BufferWriteOutOfBounds {
        handle: GpuBufferHandle,
        offset: u64,
        size: u64,
        buffer_size: u64,
    },
}

/// Frame timing and pointer data exposed to `GpuSurfaceRenderer::prepare`.
#[derive(Clone, Copy, Debug)]
pub struct GpuFrameContext {
    /// Time elapsed since the surface was initialized.
    pub time: Duration,
    /// Delta since the previous prepared frame.
    pub delta_time: Duration,
    /// Frame counter.
    pub frame_index: u64,
    /// Surface extent in physical pixels.
    pub extent: GpuExtent,
    /// Window-space cursor position.
    pub cursor_position: Point<Pixels>,
    /// Surface-local cursor position when the pointer is inside the surface.
    pub surface_cursor_position: Option<Point<Pixels>>,
}

#[derive(Default)]
pub(crate) struct GpuResourceRegistry {
    next_id: u64,
    pub textures: HashMap<GpuTextureHandle, GpuTextureDesc>,
    pub buffers: HashMap<GpuBufferHandle, GpuBufferDesc>,
    pub samplers: HashMap<GpuSamplerHandle, GpuSamplerDesc>,
    pub render_programs: HashMap<GpuRenderProgramHandle, GpuRenderProgramDesc>,
    pub compute_programs: HashMap<GpuComputeProgramHandle, GpuComputeProgramDesc>,
}

impl GpuResourceRegistry {
    fn next_handle(&mut self) -> u64 {
        let next = self.next_id;
        self.next_id += 1;
        next
    }

    fn create_texture(&mut self, desc: GpuTextureDesc) -> GpuTextureHandle {
        let handle = GpuTextureHandle(self.next_handle());
        self.textures.insert(handle, desc);
        handle
    }

    fn create_buffer(&mut self, desc: GpuBufferDesc) -> GpuBufferHandle {
        let handle = GpuBufferHandle(self.next_handle());
        self.buffers.insert(handle, desc);
        handle
    }

    fn create_sampler(&mut self, desc: GpuSamplerDesc) -> GpuSamplerHandle {
        let handle = GpuSamplerHandle(self.next_handle());
        self.samplers.insert(handle, desc);
        handle
    }

    fn create_render_program(&mut self, desc: GpuRenderProgramDesc) -> GpuRenderProgramHandle {
        let handle = GpuRenderProgramHandle(self.next_handle());
        self.render_programs.insert(handle, desc);
        handle
    }

    fn create_compute_program(&mut self, desc: GpuComputeProgramDesc) -> GpuComputeProgramHandle {
        let handle = GpuComputeProgramHandle(self.next_handle());
        self.compute_programs.insert(handle, desc);
        handle
    }

    fn release_transient_textures(&mut self, handles: &[GpuTextureHandle]) {
        self.release_textures(handles);
    }

    fn release_textures(&mut self, handles: &[GpuTextureHandle]) {
        for handle in handles {
            self.textures.remove(handle);
        }
    }

    fn release_buffers(&mut self, handles: &[GpuBufferHandle]) {
        for handle in handles {
            self.buffers.remove(handle);
        }
    }

    fn release_samplers(&mut self, handles: &[GpuSamplerHandle]) {
        for handle in handles {
            self.samplers.remove(handle);
        }
    }

    fn release_scoped_resources(&mut self, resources: &GpuScopedResources) {
        self.release_textures(&resources.textures);
        self.release_buffers(&resources.buffers);
        self.release_samplers(&resources.samplers);
    }

    fn has_sampler(&self, handle: GpuSamplerHandle) -> bool {
        self.samplers.contains_key(&handle)
    }

    fn has_render_program(&self, handle: GpuRenderProgramHandle) -> bool {
        self.render_programs.contains_key(&handle)
    }

    fn has_compute_program(&self, handle: GpuComputeProgramHandle) -> bool {
        self.compute_programs.contains_key(&handle)
    }
}

fn buffer_desc_with_usage(mut desc: GpuBufferDesc, usage: GpuBufferUsage) -> GpuBufferDesc {
    desc.usage = usage;
    desc
}

#[derive(Default)]
struct GpuScopedResources {
    textures: Vec<GpuTextureHandle>,
    buffers: Vec<GpuBufferHandle>,
    samplers: Vec<GpuSamplerHandle>,
}

/// Resource allocation context used during `init`.
pub struct GpuInitContext<'a> {
    resources: &'a mut GpuResourceRegistry,
}

impl<'a> GpuInitContext<'a> {
    pub(crate) fn new(resources: &'a mut GpuResourceRegistry) -> Self {
        Self { resources }
    }

    /// Creates a persistent texture.
    pub fn persistent_texture(&mut self, desc: GpuTextureDesc) -> GpuTextureHandle {
        self.resources.create_texture(desc)
    }

    /// Creates a persistent ping-pong texture pair.
    pub fn persistent_texture_pair(&mut self, desc: GpuTextureDesc) -> GpuTexturePair {
        GpuTexturePair {
            a: self.resources.create_texture(desc.clone()),
            b: self.resources.create_texture(desc),
        }
    }

    /// Creates a persistent uniform buffer.
    pub fn uniform_buffer(&mut self, desc: GpuBufferDesc) -> GpuBufferHandle {
        self.resources
            .create_buffer(buffer_desc_with_usage(desc, GpuBufferUsage::Uniform))
    }

    /// Creates a persistent storage buffer.
    pub fn storage_buffer(&mut self, desc: GpuBufferDesc) -> GpuBufferHandle {
        self.resources
            .create_buffer(buffer_desc_with_usage(desc, GpuBufferUsage::Storage))
    }

    /// Creates a persistent sampler.
    pub fn sampler(&mut self, desc: GpuSamplerDesc) -> GpuSamplerHandle {
        self.resources.create_sampler(desc)
    }

    /// Registers a WGSL render program.
    pub fn render_program(&mut self, desc: GpuRenderProgramDesc) -> GpuRenderProgramHandle {
        self.resources.create_render_program(desc)
    }

    /// Registers a WGSL compute program.
    pub fn compute_program(&mut self, desc: GpuComputeProgramDesc) -> GpuComputeProgramHandle {
        self.resources.create_compute_program(desc)
    }
}

/// Resource allocation context used during `resize`.
pub struct GpuResizeContext<'a> {
    resources: &'a mut GpuResourceRegistry,
    scoped_resources: &'a mut GpuScopedResources,
    /// Previous extent, if the surface had already been initialized.
    pub old_extent: Option<GpuExtent>,
    /// New extent.
    pub new_extent: GpuExtent,
    /// Window scale factor.
    pub scale_factor: f32,
}

impl<'a> GpuResizeContext<'a> {
    fn new(
        resources: &'a mut GpuResourceRegistry,
        scoped_resources: &'a mut GpuScopedResources,
        old_extent: Option<GpuExtent>,
        new_extent: GpuExtent,
        scale_factor: f32,
    ) -> Self {
        Self {
            resources,
            scoped_resources,
            old_extent,
            new_extent,
            scale_factor,
        }
    }

    /// Creates a resize-scoped persistent texture.
    ///
    /// All resources created by one `resize` call replace the resources created by the previous
    /// `resize` call once the callback returns.
    pub fn persistent_texture(&mut self, desc: GpuTextureDesc) -> GpuTextureHandle {
        let handle = self.resources.create_texture(desc);
        self.scoped_resources.textures.push(handle);
        handle
    }

    /// Creates a resize-scoped persistent ping-pong texture pair.
    ///
    /// All resources created by one `resize` call replace the resources created by the previous
    /// `resize` call once the callback returns.
    pub fn persistent_texture_pair(&mut self, desc: GpuTextureDesc) -> GpuTexturePair {
        GpuTexturePair {
            a: self.persistent_texture(desc.clone()),
            b: self.persistent_texture(desc),
        }
    }

    /// Creates a resize-scoped persistent uniform buffer.
    ///
    /// All resources created by one `resize` call replace the resources created by the previous
    /// `resize` call once the callback returns.
    pub fn uniform_buffer(&mut self, desc: GpuBufferDesc) -> GpuBufferHandle {
        let handle = self
            .resources
            .create_buffer(buffer_desc_with_usage(desc, GpuBufferUsage::Uniform));
        self.scoped_resources.buffers.push(handle);
        handle
    }

    /// Creates a resize-scoped persistent storage buffer.
    ///
    /// All resources created by one `resize` call replace the resources created by the previous
    /// `resize` call once the callback returns.
    pub fn storage_buffer(&mut self, desc: GpuBufferDesc) -> GpuBufferHandle {
        let handle = self
            .resources
            .create_buffer(buffer_desc_with_usage(desc, GpuBufferUsage::Storage));
        self.scoped_resources.buffers.push(handle);
        handle
    }

    /// Creates a resize-scoped persistent sampler.
    ///
    /// All resources created by one `resize` call replace the resources created by the previous
    /// `resize` call once the callback returns.
    pub fn sampler(&mut self, desc: GpuSamplerDesc) -> GpuSamplerHandle {
        let handle = self.resources.create_sampler(desc);
        self.scoped_resources.samplers.push(handle);
        handle
    }
}

/// Teardown context used when a `GpuSurface` state is dropped.
pub struct GpuTeardownContext<'a> {
    resources: &'a mut GpuResourceRegistry,
}

impl<'a> GpuTeardownContext<'a> {
    pub(crate) fn new(resources: &'a mut GpuResourceRegistry) -> Self {
        Self { resources }
    }

    /// Returns the number of tracked textures at teardown time.
    pub fn texture_count(&self) -> usize {
        self.resources.textures.len()
    }
}

/// Frame graph recording context used during `encode`.
pub struct GpuGraphContext<'a> {
    resources: &'a mut GpuResourceRegistry,
    graph: &'a mut GpuRecordedGraph,
}

impl<'a> GpuGraphContext<'a> {
    pub(crate) fn new(
        resources: &'a mut GpuResourceRegistry,
        graph: &'a mut GpuRecordedGraph,
    ) -> Self {
        Self { resources, graph }
    }

    /// Allocates a transient texture for the current frame graph.
    pub fn transient_texture(&mut self, desc: GpuTextureDesc) -> GpuTextureHandle {
        let handle = self.resources.create_texture(desc);
        self.graph.transient_textures.push(handle);
        handle
    }

    /// Records a full-buffer upload at byte offset `0`.
    pub fn write_buffer(&mut self, buffer: GpuBufferHandle, data: impl AsRef<[u8]>) {
        self.write_buffer_with_offset(buffer, 0, data);
    }

    /// Records a byte upload into an existing buffer.
    pub fn write_buffer_with_offset(
        &mut self,
        buffer: GpuBufferHandle,
        offset: u64,
        data: impl AsRef<[u8]>,
    ) {
        self.graph.buffer_writes.push(GpuBufferWrite {
            buffer,
            offset,
            data: data.as_ref().to_vec(),
        });
    }

    /// Records a typed upload for a single plain-old-data value.
    pub fn write_buffer_value<T: Pod>(&mut self, buffer: GpuBufferHandle, value: &T) {
        self.write_buffer(buffer, bytemuck::bytes_of(value));
    }

    /// Records a typed upload for a slice of plain-old-data values.
    pub fn write_buffer_slice<T: Pod>(&mut self, buffer: GpuBufferHandle, values: &[T]) {
        self.write_buffer(buffer, bytemuck::cast_slice(values));
    }

    /// Records a render pass.
    pub fn render_pass(&mut self, pass: GpuRenderPassDesc) {
        self.graph
            .operations
            .push(GpuGraphOperation::RenderPass(pass));
    }

    /// Records a compute pass.
    pub fn compute_pass(&mut self, pass: GpuComputePassDesc) {
        self.graph
            .operations
            .push(GpuGraphOperation::ComputePass(pass));
    }

    /// Marks the final presented texture for this frame.
    pub fn present(&mut self, texture: GpuTextureHandle) {
        self.graph.presented = Some(texture);
    }
}

impl GpuRecordedGraph {
    /// Returns the buffer uploads recorded for this frame.
    pub fn buffer_writes(&self) -> &[GpuBufferWrite] {
        &self.buffer_writes
    }

    fn validate(&self, resources: &GpuResourceRegistry) -> Result<(), GpuGraphValidationError> {
        for write in &self.buffer_writes {
            let Some(desc) = resources.buffers.get(&write.buffer) else {
                return Err(GpuGraphValidationError::MissingBuffer(write.buffer));
            };
            let size = write.data.len() as u64;
            let Some(end) = write.offset.checked_add(size) else {
                return Err(GpuGraphValidationError::BufferWriteOutOfBounds {
                    handle: write.buffer,
                    offset: write.offset,
                    size,
                    buffer_size: desc.size,
                });
            };
            if end > desc.size {
                return Err(GpuGraphValidationError::BufferWriteOutOfBounds {
                    handle: write.buffer,
                    offset: write.offset,
                    size,
                    buffer_size: desc.size,
                });
            }
        }

        for operation in &self.operations {
            match operation {
                GpuGraphOperation::RenderPass(pass) => {
                    if !resources.has_render_program(pass.program) {
                        return Err(GpuGraphValidationError::MissingRenderProgram(pass.program));
                    }
                    let Some(target_desc) = resources.textures.get(&pass.target) else {
                        return Err(GpuGraphValidationError::MissingTexture(pass.target));
                    };
                    if !target_desc.render_attachment {
                        return Err(GpuGraphValidationError::NonRenderableTarget(pass.target));
                    }
                    validate_render_pass_hazards(pass)?;
                    for binding in &pass.bindings {
                        validate_binding(*binding, resources)?;
                    }
                }
                GpuGraphOperation::ComputePass(pass) => {
                    if !resources.has_compute_program(pass.program) {
                        return Err(GpuGraphValidationError::MissingComputeProgram(pass.program));
                    }
                    validate_compute_pass_hazards(pass)?;
                    for binding in &pass.bindings {
                        validate_binding(*binding, resources)?;
                    }
                }
            }
        }

        if let Some(texture) = self.presented {
            let Some(texture_desc) = resources.textures.get(&texture) else {
                return Err(GpuGraphValidationError::MissingTexture(texture));
            };
            if !texture_desc.sampled {
                return Err(GpuGraphValidationError::NonSampledPresentTexture(texture));
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuPassTextureUsage {
    Sampled,
    Storage,
    RenderTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuPassBufferUsage {
    Uniform,
    Storage,
}

fn validate_render_pass_hazards(pass: &GpuRenderPassDesc) -> Result<(), GpuGraphValidationError> {
    let mut textures = HashMap::<GpuTextureHandle, GpuPassTextureUsage>::default();
    textures.insert(pass.target, GpuPassTextureUsage::RenderTarget);
    let mut buffers = HashMap::<GpuBufferHandle, GpuPassBufferUsage>::default();
    validate_pass_hazards(&pass.bindings, &mut textures, &mut buffers)
}

fn validate_compute_pass_hazards(pass: &GpuComputePassDesc) -> Result<(), GpuGraphValidationError> {
    let mut textures = HashMap::<GpuTextureHandle, GpuPassTextureUsage>::default();
    let mut buffers = HashMap::<GpuBufferHandle, GpuPassBufferUsage>::default();
    validate_pass_hazards(&pass.bindings, &mut textures, &mut buffers)
}

fn validate_pass_hazards(
    bindings: &[GpuBinding],
    textures: &mut HashMap<GpuTextureHandle, GpuPassTextureUsage>,
    buffers: &mut HashMap<GpuBufferHandle, GpuPassBufferUsage>,
) -> Result<(), GpuGraphValidationError> {
    for binding in bindings {
        match *binding {
            GpuBinding::UniformBuffer(handle) => {
                validate_buffer_pass_usage(handle, GpuPassBufferUsage::Uniform, buffers)?
            }
            GpuBinding::StorageBuffer(handle) => {
                validate_buffer_pass_usage(handle, GpuPassBufferUsage::Storage, buffers)?
            }
            GpuBinding::SampledTexture(handle) => {
                validate_texture_pass_usage(handle, GpuPassTextureUsage::Sampled, textures)?
            }
            GpuBinding::StorageTexture(handle) => {
                validate_texture_pass_usage(handle, GpuPassTextureUsage::Storage, textures)?
            }
            GpuBinding::Sampler(_) => {}
        }
    }
    Ok(())
}

fn validate_texture_pass_usage(
    handle: GpuTextureHandle,
    usage: GpuPassTextureUsage,
    usages: &mut HashMap<GpuTextureHandle, GpuPassTextureUsage>,
) -> Result<(), GpuGraphValidationError> {
    match usages.insert(handle, usage) {
        None => Ok(()),
        Some(previous) if previous == usage => Ok(()),
        Some(GpuPassTextureUsage::RenderTarget) => match usage {
            GpuPassTextureUsage::Sampled => {
                Err(GpuGraphValidationError::RenderTargetSampledConflict(handle))
            }
            GpuPassTextureUsage::Storage => {
                Err(GpuGraphValidationError::RenderTargetStorageConflict(handle))
            }
            GpuPassTextureUsage::RenderTarget => Ok(()),
        },
        Some(GpuPassTextureUsage::Sampled) | Some(GpuPassTextureUsage::Storage) => Err(
            GpuGraphValidationError::SampledStorageTextureConflict(handle),
        ),
    }
}

fn validate_buffer_pass_usage(
    handle: GpuBufferHandle,
    usage: GpuPassBufferUsage,
    usages: &mut HashMap<GpuBufferHandle, GpuPassBufferUsage>,
) -> Result<(), GpuGraphValidationError> {
    match usages.insert(handle, usage) {
        None => Ok(()),
        Some(previous) if previous == usage => Ok(()),
        Some(GpuPassBufferUsage::Uniform) | Some(GpuPassBufferUsage::Storage) => Err(
            GpuGraphValidationError::UniformStorageBufferConflict(handle),
        ),
    }
}

fn validate_binding(
    binding: GpuBinding,
    resources: &GpuResourceRegistry,
) -> Result<(), GpuGraphValidationError> {
    match binding {
        GpuBinding::UniformBuffer(handle) => {
            let Some(desc) = resources.buffers.get(&handle) else {
                return Err(GpuGraphValidationError::MissingBuffer(handle));
            };
            if desc.usage != GpuBufferUsage::Uniform {
                Err(GpuGraphValidationError::NonUniformBufferBinding(handle))
            } else {
                Ok(())
            }
        }
        GpuBinding::StorageBuffer(handle) => {
            let Some(desc) = resources.buffers.get(&handle) else {
                return Err(GpuGraphValidationError::MissingBuffer(handle));
            };
            if desc.usage != GpuBufferUsage::Storage {
                Err(GpuGraphValidationError::NonStorageBufferBinding(handle))
            } else {
                Ok(())
            }
        }
        GpuBinding::SampledTexture(handle) => {
            let Some(desc) = resources.textures.get(&handle) else {
                return Err(GpuGraphValidationError::MissingTexture(handle));
            };
            if !desc.sampled {
                Err(GpuGraphValidationError::NonSampledTextureBinding(handle))
            } else {
                Ok(())
            }
        }
        GpuBinding::StorageTexture(handle) => {
            let Some(desc) = resources.textures.get(&handle) else {
                return Err(GpuGraphValidationError::MissingTexture(handle));
            };
            if !desc.storage {
                Err(GpuGraphValidationError::NonStorageTextureBinding(handle))
            } else {
                Ok(())
            }
        }
        GpuBinding::Sampler(handle) => {
            if resources.has_sampler(handle) {
                Ok(())
            } else {
                Err(GpuGraphValidationError::MissingSampler(handle))
            }
        }
    }
}

/// A stateful renderer used by `GpuSurface`.
pub trait GpuSurfaceRenderer: 'static {
    /// Called when a newly-built `GpuSurface` with the same id provides a fresh renderer value.
    ///
    /// Implementations should merge external configuration from `next_renderer` while preserving
    /// any runtime state they want to keep alive across frames.
    fn update(&mut self, next_renderer: Self)
    where
        Self: Sized;

    /// Called once when the surface state is initialized.
    fn init(&mut self, _cx: &mut GpuInitContext<'_>) {}

    /// Called when the surface size or scale factor changes.
    fn resize(
        &mut self,
        _cx: &mut GpuResizeContext<'_>,
        _old: Option<GpuExtent>,
        _new: GpuExtent,
        _scale_factor: f32,
    ) {
    }

    /// Called once per frame before graph encoding.
    fn prepare(&mut self, _frame: &GpuFrameContext) {}

    /// Called during the paint phase to record the current frame graph.
    fn encode(&mut self, _graph: &mut GpuGraphContext<'_>) {}

    /// Called when the surface state is torn down.
    fn teardown(&mut self, _cx: &mut GpuTeardownContext<'_>) {}
}

/// A stable identity key for a `GpuSurface`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuSurfaceId(pub ElementId);

impl From<ElementId> for GpuSurfaceId {
    fn from(value: ElementId) -> Self {
        Self(value)
    }
}

impl From<GpuSurfaceId> for ElementId {
    fn from(value: GpuSurfaceId) -> Self {
        value.0
    }
}

pub(crate) struct GpuSurfaceRuntime<R: GpuSurfaceRenderer> {
    pub renderer: R,
    pub initialized: bool,
    pub extent: Option<GpuExtent>,
    pub scale_factor: Option<f32>,
    pub frame_index: u64,
    pub start_time: Instant,
    pub last_frame_at: Option<Instant>,
    pub last_frame: Option<GpuFrameContext>,
    pub resources: GpuResourceRegistry,
    resize_resources: GpuScopedResources,
    pub recorded_graph: GpuRecordedGraph,
}

impl<R: GpuSurfaceRenderer> GpuSurfaceRuntime<R> {
    pub fn new(renderer: R) -> Self {
        let now = Instant::now();
        Self {
            renderer,
            initialized: false,
            extent: None,
            scale_factor: None,
            frame_index: 0,
            start_time: now,
            last_frame_at: None,
            last_frame: None,
            resources: GpuResourceRegistry::default(),
            resize_resources: GpuScopedResources::default(),
            recorded_graph: GpuRecordedGraph::default(),
        }
    }

    pub fn ensure_initialized(&mut self) {
        if self.initialized {
            return;
        }

        let mut cx = GpuInitContext::new(&mut self.resources);
        self.renderer.init(&mut cx);
        self.initialized = true;
    }

    pub fn resize_if_needed(&mut self, new_extent: GpuExtent, scale_factor: f32) {
        let needs_resize =
            self.extent != Some(new_extent) || self.scale_factor != Some(scale_factor);

        if !needs_resize {
            return;
        }

        if new_extent.is_empty() {
            let old_resize_resources = mem::take(&mut self.resize_resources);
            self.resources
                .release_scoped_resources(&old_resize_resources);
            self.extent = Some(new_extent);
            self.scale_factor = Some(scale_factor);
            return;
        }

        let old_extent = self.extent;
        let mut resize_resources = GpuScopedResources::default();
        let mut cx = GpuResizeContext::new(
            &mut self.resources,
            &mut resize_resources,
            old_extent,
            new_extent,
            scale_factor,
        );
        self.renderer
            .resize(&mut cx, old_extent, new_extent, scale_factor);
        let old_resize_resources = mem::replace(&mut self.resize_resources, resize_resources);
        self.resources
            .release_scoped_resources(&old_resize_resources);
        self.extent = Some(new_extent);
        self.scale_factor = Some(scale_factor);
    }

    pub fn prepare_frame(
        &mut self,
        extent: GpuExtent,
        cursor_position: Point<Pixels>,
        surface_cursor_position: Option<Point<Pixels>>,
    ) -> GpuFrameContext {
        let now = Instant::now();
        let frame =
            surface_frame_context(self, now, extent, cursor_position, surface_cursor_position);
        self.renderer.prepare(&frame);
        self.last_frame = Some(frame);
        self.last_frame_at = Some(now);
        self.frame_index += 1;
        frame
    }

    pub fn encode_frame(&mut self) -> Result<(), GpuGraphValidationError> {
        self.begin_graph_recording();
        let mut graph = GpuGraphContext::new(&mut self.resources, &mut self.recorded_graph);
        self.renderer.encode(&mut graph);
        self.recorded_graph.validate(&self.resources)
    }

    pub fn clear_frame_graph(&mut self) {
        self.begin_graph_recording();
    }

    fn begin_graph_recording(&mut self) {
        self.resources
            .release_transient_textures(&self.recorded_graph.transient_textures);
        self.recorded_graph = GpuRecordedGraph::default();
    }
}

impl<R: GpuSurfaceRenderer> Drop for GpuSurfaceRuntime<R> {
    fn drop(&mut self) {
        let mut cx = GpuTeardownContext::new(&mut self.resources);
        self.renderer.teardown(&mut cx);
    }
}

/// Returns the current frame context for a surface runtime.
pub(crate) fn surface_frame_context(
    runtime: &GpuSurfaceRuntime<impl GpuSurfaceRenderer>,
    now: Instant,
    extent: GpuExtent,
    cursor_position: Point<Pixels>,
    surface_cursor_position: Option<Point<Pixels>>,
) -> GpuFrameContext {
    GpuFrameContext {
        time: now - runtime.start_time,
        delta_time: runtime
            .last_frame_at
            .map(|previous| now - previous)
            .unwrap_or(Duration::ZERO),
        frame_index: runtime.frame_index,
        extent,
        cursor_position,
        surface_cursor_position,
    }
}

/// Refreshes an animated surface on the next frame.
pub(crate) fn request_surface_redraw(
    redraw_mode: GpuSurfaceRedrawMode,
    window: &crate::Window,
    _cx: &mut App,
) {
    if matches!(redraw_mode, GpuSurfaceRedrawMode::Animated) {
        window.request_animation_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_textures_get_distinct_handles() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);

        let a = cx.transient_texture(GpuTextureDesc::default().with_label("a"));
        let b = cx.transient_texture(GpuTextureDesc::default().with_label("b"));

        assert_ne!(a, b);
    }

    #[test]
    fn persistent_texture_pair_allocates_two_textures() {
        let mut resources = GpuResourceRegistry::default();
        let mut cx = GpuInitContext::new(&mut resources);
        let pair = cx.persistent_texture_pair(GpuTextureDesc::default());

        assert_ne!(pair.a, pair.b);
    }

    #[test]
    fn buffer_helpers_assign_expected_usage() {
        let mut resources = GpuResourceRegistry::default();
        let mut cx = GpuInitContext::new(&mut resources);
        let uniform = cx.uniform_buffer(GpuBufferDesc::default().with_label("uniform"));
        let storage = cx.storage_buffer(GpuBufferDesc::default().with_label("storage"));

        assert_eq!(
            resources.buffers.get(&uniform).map(|desc| desc.usage),
            Some(GpuBufferUsage::Uniform)
        );
        assert_eq!(
            resources.buffers.get(&storage).map(|desc| desc.usage),
            Some(GpuBufferUsage::Storage)
        );
    }

    struct TransientTextureRenderer;

    impl GpuSurfaceRenderer for TransientTextureRenderer {
        fn update(&mut self, _next_renderer: Self) {}

        fn encode(&mut self, graph: &mut GpuGraphContext<'_>) {
            let texture = graph.transient_texture(GpuTextureDesc::default().with_label("frame"));
            graph.present(texture);
        }
    }

    #[test]
    fn runtime_recycles_transient_textures_between_frames() {
        let mut runtime = GpuSurfaceRuntime::new(TransientTextureRenderer);

        runtime.encode_frame().unwrap();
        let first = runtime.recorded_graph.transient_textures[0];
        assert_eq!(runtime.resources.textures.len(), 1);

        runtime.encode_frame().unwrap();
        let second = runtime.recorded_graph.transient_textures[0];
        assert_ne!(first, second);
        assert_eq!(runtime.resources.textures.len(), 1);
    }

    struct FrameTimingRenderer;

    impl GpuSurfaceRenderer for FrameTimingRenderer {
        fn update(&mut self, _next_renderer: Self) {}
    }

    #[test]
    fn prepare_frame_advances_frame_index_and_updates_last_frame() {
        let mut runtime = GpuSurfaceRuntime::new(FrameTimingRenderer);
        runtime.ensure_initialized();

        let extent = GpuExtent {
            width: 64,
            height: 32,
        };
        let first =
            runtime.prepare_frame(extent, crate::point(crate::px(4.0), crate::px(8.0)), None);
        assert_eq!(first.frame_index, 0);
        assert_eq!(runtime.frame_index, 1);
        assert_eq!(
            runtime.last_frame.as_ref().map(|frame| frame.extent),
            Some(extent)
        );

        let second = runtime.prepare_frame(
            extent,
            crate::point(crate::px(5.0), crate::px(9.0)),
            Some(crate::point(crate::px(1.0), crate::px(2.0))),
        );
        assert_eq!(second.frame_index, 1);
        assert_eq!(runtime.frame_index, 2);
        assert!(second.time >= first.time);
    }

    #[test]
    fn graph_records_buffer_writes() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let buffer = init.uniform_buffer(GpuBufferDesc {
            size: 16,
            ..GpuBufferDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);

        cx.write_buffer_with_offset(buffer, 4, [1u8, 2, 3, 4]);

        assert_eq!(graph.buffer_writes.len(), 1);
        assert_eq!(
            graph.buffer_writes[0],
            GpuBufferWrite {
                buffer,
                offset: 4,
                data: vec![1, 2, 3, 4],
            }
        );
    }

    #[test]
    fn graph_rejects_out_of_bounds_buffer_writes() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let buffer = init.uniform_buffer(GpuBufferDesc {
            size: 8,
            ..GpuBufferDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.write_buffer_with_offset(buffer, 4, [1u8, 2, 3, 4, 5]);

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::BufferWriteOutOfBounds {
                handle: buffer,
                offset: 4,
                size: 5,
                buffer_size: 8,
            })
        );
    }

    struct ResizeAllocatingRenderer {
        texture: Option<GpuTextureHandle>,
        resize_count: usize,
    }

    impl GpuSurfaceRenderer for ResizeAllocatingRenderer {
        fn update(&mut self, _next_renderer: Self) {}

        fn resize(
            &mut self,
            cx: &mut GpuResizeContext<'_>,
            _old: Option<GpuExtent>,
            new_extent: GpuExtent,
            _scale_factor: f32,
        ) {
            self.resize_count += 1;
            self.texture = Some(cx.persistent_texture(GpuTextureDesc {
                extent: new_extent,
                sampled: true,
                ..GpuTextureDesc::default()
            }));
        }
    }

    #[test]
    fn resize_releases_previous_resize_scoped_resources() {
        let mut runtime = GpuSurfaceRuntime::new(ResizeAllocatingRenderer {
            texture: None,
            resize_count: 0,
        });

        runtime.ensure_initialized();
        runtime.resize_if_needed(
            GpuExtent {
                width: 64,
                height: 64,
            },
            1.0,
        );
        let first = runtime.renderer.texture;
        assert_eq!(runtime.renderer.resize_count, 1);
        assert_eq!(runtime.resources.textures.len(), 1);

        runtime.resize_if_needed(
            GpuExtent {
                width: 128,
                height: 128,
            },
            1.0,
        );
        assert_ne!(first, runtime.renderer.texture);
        assert_eq!(runtime.renderer.resize_count, 2);
        assert_eq!(runtime.resources.textures.len(), 1);
    }

    #[test]
    fn zero_extent_does_not_resize_or_retain_resize_resources() {
        let mut runtime = GpuSurfaceRuntime::new(ResizeAllocatingRenderer {
            texture: None,
            resize_count: 0,
        });

        runtime.ensure_initialized();
        runtime.resize_if_needed(
            GpuExtent {
                width: 0,
                height: 0,
            },
            1.0,
        );
        assert_eq!(runtime.renderer.resize_count, 0);
        assert!(runtime.renderer.texture.is_none());
        assert!(runtime.resources.textures.is_empty());
        assert_eq!(
            runtime.extent,
            Some(GpuExtent {
                width: 0,
                height: 0,
            })
        );

        runtime.resize_if_needed(
            GpuExtent {
                width: 32,
                height: 32,
            },
            1.0,
        );
        assert_eq!(runtime.renderer.resize_count, 1);
        assert_eq!(runtime.resources.textures.len(), 1);

        runtime.resize_if_needed(
            GpuExtent {
                width: 0,
                height: 0,
            },
            1.0,
        );
        assert_eq!(runtime.renderer.resize_count, 1);
        assert!(runtime.resources.textures.is_empty());
    }

    #[test]
    fn scale_factor_change_rebuilds_resize_scoped_resources() {
        let mut runtime = GpuSurfaceRuntime::new(ResizeAllocatingRenderer {
            texture: None,
            resize_count: 0,
        });

        runtime.ensure_initialized();
        runtime.resize_if_needed(
            GpuExtent {
                width: 96,
                height: 96,
            },
            1.0,
        );
        let first = runtime.renderer.texture;
        assert_eq!(runtime.renderer.resize_count, 1);
        assert_eq!(runtime.resources.textures.len(), 1);
        assert_eq!(runtime.scale_factor, Some(1.0));

        runtime.resize_if_needed(
            GpuExtent {
                width: 96,
                height: 96,
            },
            1.5,
        );
        assert_ne!(first, runtime.renderer.texture);
        assert_eq!(runtime.renderer.resize_count, 2);
        assert_eq!(runtime.resources.textures.len(), 1);
        assert_eq!(runtime.scale_factor, Some(1.5));
    }

    struct InvalidGraphRenderer;

    impl GpuSurfaceRenderer for InvalidGraphRenderer {
        fn update(&mut self, _next_renderer: Self) {}

        fn encode(&mut self, graph: &mut GpuGraphContext<'_>) {
            graph.present(GpuTextureHandle(999));
        }
    }

    #[test]
    fn encode_frame_rejects_unknown_present_texture() {
        let mut runtime = GpuSurfaceRuntime::new(InvalidGraphRenderer);
        let error = runtime.encode_frame().unwrap_err();

        assert_eq!(
            error,
            GpuGraphValidationError::MissingTexture(GpuTextureHandle(999))
        );
    }

    #[test]
    fn render_target_must_be_renderable() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let program = init.render_program(GpuRenderProgramDesc {
            label: None,
            wgsl: "render".into(),
            vertex_entry: "vs_main".into(),
            fragment_entry: "fs_main".into(),
        });
        let target = init.persistent_texture(GpuTextureDesc {
            sampled: true,
            ..GpuTextureDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.render_pass(GpuRenderPassDesc {
            label: None,
            program,
            target,
            clear_color: None,
            bindings: Vec::new(),
            draw: GpuDrawCall::FullScreenTriangle,
        });

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::NonRenderableTarget(target))
        );
    }

    #[test]
    fn present_texture_must_be_sampleable() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let target = init.persistent_texture(GpuTextureDesc {
            render_attachment: true,
            ..GpuTextureDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.present(target);

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::NonSampledPresentTexture(target))
        );
    }

    #[test]
    fn uniform_binding_requires_uniform_buffer_usage() {
        let mut resources = GpuResourceRegistry::default();
        let mut init = GpuInitContext::new(&mut resources);
        let storage = init.storage_buffer(GpuBufferDesc {
            size: 16,
            ..GpuBufferDesc::default()
        });

        assert_eq!(
            validate_binding(GpuBinding::UniformBuffer(storage), &resources),
            Err(GpuGraphValidationError::NonUniformBufferBinding(storage))
        );
    }

    #[test]
    fn sampled_binding_requires_sampled_texture_usage() {
        let mut resources = GpuResourceRegistry::default();
        let mut init = GpuInitContext::new(&mut resources);
        let storage_only = init.persistent_texture(GpuTextureDesc {
            storage: true,
            ..GpuTextureDesc::default()
        });

        assert_eq!(
            validate_binding(GpuBinding::SampledTexture(storage_only), &resources),
            Err(GpuGraphValidationError::NonSampledTextureBinding(
                storage_only
            ))
        );
    }

    #[test]
    fn render_pass_rejects_sampling_its_own_target() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let program = init.render_program(GpuRenderProgramDesc {
            label: None,
            wgsl: "render".into(),
            vertex_entry: "vs_main".into(),
            fragment_entry: "fs_main".into(),
        });
        let target = init.persistent_texture(GpuTextureDesc {
            sampled: true,
            render_attachment: true,
            ..GpuTextureDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.render_pass(GpuRenderPassDesc {
            label: None,
            program,
            target,
            clear_color: None,
            bindings: vec![GpuBinding::SampledTexture(target)],
            draw: GpuDrawCall::FullScreenTriangle,
        });

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::RenderTargetSampledConflict(target))
        );
    }

    #[test]
    fn compute_pass_rejects_sampled_and_storage_texture_alias() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let program = init.compute_program(GpuComputeProgramDesc {
            label: None,
            wgsl: "compute".into(),
            entry: "cs_main".into(),
        });
        let texture = init.persistent_texture(GpuTextureDesc {
            sampled: true,
            storage: true,
            ..GpuTextureDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.compute_pass(GpuComputePassDesc {
            label: None,
            program,
            bindings: vec![
                GpuBinding::SampledTexture(texture),
                GpuBinding::StorageTexture(texture),
            ],
            workgroups: [1, 1, 1],
        });

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::SampledStorageTextureConflict(
                texture
            ))
        );
    }

    #[test]
    fn pass_rejects_uniform_and_storage_buffer_alias() {
        let mut resources = GpuResourceRegistry::default();
        let mut graph = GpuRecordedGraph::default();
        let mut init = GpuInitContext::new(&mut resources);
        let program = init.compute_program(GpuComputeProgramDesc {
            label: None,
            wgsl: "compute".into(),
            entry: "cs_main".into(),
        });
        let buffer = init.storage_buffer(GpuBufferDesc {
            size: 16,
            ..GpuBufferDesc::default()
        });
        let mut cx = GpuGraphContext::new(&mut resources, &mut graph);
        cx.compute_pass(GpuComputePassDesc {
            label: None,
            program,
            bindings: vec![
                GpuBinding::StorageBuffer(buffer),
                GpuBinding::UniformBuffer(buffer),
            ],
            workgroups: [1, 1, 1],
        });

        assert_eq!(
            graph.validate(&resources),
            Err(GpuGraphValidationError::UniformStorageBufferConflict(
                buffer
            ))
        );
    }

    struct MissingRenderProgramRenderer;

    impl GpuSurfaceRenderer for MissingRenderProgramRenderer {
        fn update(&mut self, _next_renderer: Self) {}

        fn encode(&mut self, graph: &mut GpuGraphContext<'_>) {
            let target = graph.transient_texture(GpuTextureDesc {
                render_attachment: true,
                ..GpuTextureDesc::default()
            });
            graph.render_pass(GpuRenderPassDesc {
                label: Some("missing_program".into()),
                program: GpuRenderProgramHandle(123),
                target,
                clear_color: None,
                bindings: Vec::new(),
                draw: GpuDrawCall::FullScreenTriangle,
            });
            graph.present(target);
        }
    }

    #[test]
    fn encode_frame_rejects_missing_render_program() {
        let mut runtime = GpuSurfaceRuntime::new(MissingRenderProgramRenderer);
        let error = runtime.encode_frame().unwrap_err();

        assert_eq!(
            error,
            GpuGraphValidationError::MissingRenderProgram(GpuRenderProgramHandle(123))
        );
    }
}
