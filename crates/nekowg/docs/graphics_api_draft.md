# Low-Level Graphics API Draft

This document proposes a low-level graphics extension surface for a NekoWG fork
that targets complex graphics, complex animation, and media-heavy workloads.

The design goal is explicit control, not high-level convenience.

## Goals

- Preserve NekoWG's existing strengths: windowing, input, entities, text, layout.
- Add a low-level rendering path for graphics-heavy components.
- Let applications own shader, texture, pass, and cache policy decisions.
- Keep the first phase backend-focused and avoid early high-level abstractions.

## Non-Goals

- Do not redesign the element/layout system.
- Do not introduce a new high-level animation DSL in phase one.
- Do not replace the existing text system.
- Do not force all rendering through a new graphics API.

## Priority Table

| Priority | Capability | Why it matters |
| --- | --- | --- |
| `P0` | `GpuContext` | Raw device/queue/encoder access is the base of all custom rendering. |
| `P0` | `CustomPrimitive` | Lets a component inject GPU work into the scene without abusing layout. |
| `P0` | `Pipeline` / `ShaderModule` registration | Needed for reusable custom draw code. |
| `P0` | `Uniform` / `Instance` / `Vertex` upload | Required for animation parameters and batching. |
| `P0` | Transform / clip / z-order propagation | Required to compose custom graphics with normal UI. |
| `P1` | `RenderTarget` / `OffscreenPass` | Required for blur, mask, glow, and post-processing. |
| `P1` | `ExternalTexture` | Allows apps to own image/video/decoded texture lifetimes. |
| `P1` | Blend / sampler / color-space control | Required for correct visual output. |
| `P1` | Resource lifetime hooks | Required to avoid leaks and backend-lost issues. |
| `P2` | Image cache policy API | Needed for media-heavy apps with strict RAM/VRAM budgets. |
| `P2` | Text-to-texture / glyph-sprite access | Needed for text effects that should not go through live glyph layout. |
| `P2` | GPU profiling hooks | Needed to tune passes and resource growth. |

## Phase Plan

### Phase 1

Provide a backend-neutral low-level API:

- `GpuContext`
- `GpuPrimitive`
- `GpuPipelineHandle`
- `GpuBufferHandle`
- `GpuTextureHandle`
- `Window::push_gpu_primitive(...)`

This is enough to move effects like fluid backgrounds out of the layout system.

### Phase 2

Provide pass and resource composition:

- `RenderTarget`
- `OffscreenPass`
- `ExternalTexture`
- explicit sampler/blend state

This is enough to support blur, glow, compositing, and media-backed rendering.

### Phase 3

Provide media- and text-specific escape hatches:

- image cache budget/trim policy
- text-to-texture support
- optional glyph sprite access

This is enough to support lyrics, masks, sweep highlights, and large image sets.

## Proposed Modules

The API surface should remain small and explicit.

```text
crates/nekowg/src/gpu.rs
crates/nekowg/src/gpu_types.rs
crates/nekowg/src/scene.rs
crates/nekowg/src/window.rs
crates/nekowg_wgpu/src/graphics_backend.rs
crates/nekowg_wgpu/src/pass.rs
crates/nekowg_wgpu/src/pipeline.rs
```

Suggested ownership:

- `nekowg` exposes backend-neutral traits and handles.
- `nekowg_wgpu` implements the actual backend objects and draw execution.
- platform crates should not expose separate graphics APIs directly.

## Core Types

### `GpuContext`

The application-facing backend context. This is intentionally low-level.

```rust
pub enum GpuContext<'a> {
    Wgpu(&'a mut dyn WgpuContext),
}

pub trait WgpuContext {
    fn device(&self) -> &wgpu::Device;
    fn queue(&self) -> &wgpu::Queue;
    fn surface_format(&self) -> wgpu::TextureFormat;
}
```

Rationale:

- Start with one backend-neutral enum.
- Avoid exposing renderer internals directly.
- Still allow applications to access native backend objects where needed.

### `GpuPrimitive`

An explicit scene primitive that participates in bounds, clip, transform, and z.

```rust
pub struct GpuPrimitive {
    pub bounds: Bounds<Pixels>,
    pub clip_bounds: Option<Bounds<Pixels>>,
    pub transform: TransformationMatrix,
    pub z_index: i32,
    pub draw: Arc<dyn GpuDrawCallback>,
}

pub trait GpuDrawCallback: Send + Sync + 'static {
    fn draw(&self, params: &mut GpuDrawParams<'_>) -> anyhow::Result<()>;
}
```

### `GpuDrawParams`

The per-draw execution state.

```rust
pub struct GpuDrawParams<'a> {
    pub gpu: GpuContext<'a>,
    pub viewport: Bounds<Pixels>,
    pub clip_bounds: Option<Bounds<Pixels>>,
    pub transform: TransformationMatrix,
    pub frame: &'a mut dyn GpuFrame,
}

pub trait GpuFrame {
    fn create_buffer(&mut self, desc: BufferDesc, data: &[u8]) -> GpuBufferHandle;
    fn create_texture(&mut self, desc: TextureDesc) -> GpuTextureHandle;
    fn write_buffer(&mut self, handle: GpuBufferHandle, offset: u64, data: &[u8]);
    fn write_texture(&mut self, handle: GpuTextureHandle, data: &[u8], layout: TextureLayout);
    fn submit_draw(&mut self, draw: DrawCall<'_>) -> anyhow::Result<()>;
}
```

Rationale:

- `GpuContext` gives backend access.
- `GpuFrame` limits what code can do inside one scene pass.
- This avoids exposing the whole renderer object graph.

## Pipeline API

Phase one should not invent a custom shading language.

The API should accept backend-native shader modules.

```rust
pub struct ShaderSource<'a> {
    pub label: Option<&'a str>,
    pub wgsl: &'a str,
}

pub struct PipelineDesc<'a> {
    pub label: Option<&'a str>,
    pub shader: ShaderSource<'a>,
    pub vertex_layouts: Vec<VertexLayoutDesc>,
    pub bind_group_layouts: Vec<BindGroupLayoutDesc>,
    pub color_targets: Vec<ColorTargetDesc>,
    pub primitive: PrimitiveStateDesc,
    pub depth_stencil: Option<DepthStencilDesc>,
    pub multisample: MultisampleDesc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuPipelineHandle(u64);

pub trait GpuPipelineCache {
    fn get_or_create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> anyhow::Result<GpuPipelineHandle>;
}
```

Design choice:

- Registration must be explicit and cacheable.
- The framework should not compile shaders every frame.

## Buffer and Texture API

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuBufferHandle(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuTextureHandle(u64);

pub struct BufferDesc {
    pub label: Option<&'static str>,
    pub usage: BufferUsage,
    pub size: u64,
}

pub struct TextureDesc {
    pub label: Option<&'static str>,
    pub size: Size<DevicePixels>,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_level_count: u32,
    pub sample_count: u32,
}
```

This handle-based design is deliberate:

- lifetime stays under the renderer
- app code can still cache handles
- backend loss can invalidate resources centrally

## Scene and Window Hooks

The minimal scene integration should be explicit:

```rust
impl Window {
    pub fn gpu_context(&mut self) -> Option<GpuContext<'_>>;

    pub fn push_gpu_primitive(&mut self, primitive: GpuPrimitive);
}
```

For elements:

```rust
pub fn gpu_canvas<S, P>(state: S, paint: P) -> impl Element
where
    S: FnOnce(Bounds<Pixels>, &mut Window, &mut App) -> GpuCanvasState + 'static,
    P: Fn(&GpuCanvasState, Bounds<Pixels>, &mut Window, &mut App) + 'static;
```

This is intentionally thin. The point is to give components a structured escape
hatch, not to replace `canvas` or `div`.

## Offscreen and Post-Processing

This is the first large feature after the primitive path is stable.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetHandle(u64);

pub struct RenderTargetDesc {
    pub label: Option<&'static str>,
    pub size: Size<DevicePixels>,
    pub format: TextureFormat,
    pub sample_count: u32,
    pub clear_color: Option<Hsla>,
}

pub trait OffscreenPass {
    fn begin_pass(&mut self, target: RenderTargetHandle) -> anyhow::Result<()>;
    fn end_pass(&mut self) -> anyhow::Result<()>;
}
```

Use cases:

- backdrop blur
- glow
- compositing image masks
- text highlight sweep from a cached text texture

## External Textures

Media-heavy applications cannot depend on the framework owning every image.

```rust
pub struct ExternalTextureDesc {
    pub label: Option<&'static str>,
    pub size: Size<DevicePixels>,
    pub format: TextureFormat,
}

pub trait ExternalTextureRegistry {
    fn import_texture(&mut self, desc: ExternalTextureDesc, native: ExternalTexture) -> anyhow::Result<GpuTextureHandle>;
    fn release_texture(&mut self, handle: GpuTextureHandle);
}
```

This is required for:

- custom image pipeline
- video frames
- text-to-texture overlays
- decoded cover art with explicit trim policy

## Image Cache Policy

This should stay separate from low-level draw APIs but still be part of the
graphics plan.

```rust
pub struct ImageCacheBudget {
    pub decoded_bytes: usize,
    pub texture_bytes: usize,
}

pub trait ImageCacheController {
    fn set_budget(&mut self, budget: ImageCacheBudget);
    fn trim_to_budget(&mut self);
    fn evict_unused(&mut self);
    fn stats(&self) -> ImageCacheStats;
}
```

This is not optional for a media app. Without it, RAM and VRAM ceilings are not
predictable.

## Text Escape Hatches

This is phase two or three, not phase one.

Two viable directions:

1. text-to-texture
2. glyph sprite access

Minimal text-to-texture API:

```rust
pub struct TextTextureDesc<'a> {
    pub text: &'a str,
    pub font_size: Pixels,
    pub runs: &'a [TextRun],
    pub wrap_width: Option<Pixels>,
}

pub trait TextTextureRenderer {
    fn render_text_texture(&mut self, desc: TextTextureDesc<'_>) -> anyhow::Result<GpuTextureHandle>;
}
```

This is the cleanest path for lyrics and animated text overlays.

## Backend Boundaries

The backend-neutral crate should expose:

- handles
- descriptors
- traits
- scene/window hooks

The backend crate should own:

- actual `wgpu` resources
- shader compilation
- pipeline cache
- pass encoder management
- resource destruction

This separation is important. It keeps the public API stable while the backend
evolves internally.

## Recommended First Implementation Order

1. `GpuContext`
2. `GpuPrimitive`
3. `GpuPipelineCache`
4. `GpuFrame` buffer/texture upload
5. `Window::push_gpu_primitive`
6. `RenderTarget`
7. `ExternalTexture`
8. `ImageCacheController`
9. `TextTextureRenderer`

## What Not To Build First

- a full material system
- a node-based effect graph
- a high-level shader widget DSL
- a new animation framework
- cross-backend raw handle exposure for every platform object

These can wait. They add maintenance cost before the core escape hatches are
proven.

## Immediate Validation Targets

The API is good enough for phase one only if it can implement both:

1. a GPU fluid background with no layout-driven animation
2. a lyrics overlay where the moving word is no longer rendered as live glyph motion

If those two use cases are not simpler after the API lands, the abstraction is
still too high-level or in the wrong place.
