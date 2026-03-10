//! Scene-integrated GPU callback API.
//!
//! This module provides a low-level, `wgpu`-first escape hatch for custom rendering that is
//! integrated with NekoWG's scene. Callbacks are given an append-only render pass for the current
//! window target only. Offscreen targets, transfer/copy operations, and compute are intentionally
//! out of scope for v1.
//!
//! ## Lifecycle
//! - Create a node with [`Window::insert_gpu_node`].
//! - Schedule draws during paint with [`Window::push_gpu_primitive`].
//! - Retire a node with [`Window::remove_gpu_node`]. Retirement is deferred until the current
//!   frame finishes encoding.
//!
//! ## Scheduling contract
//! - `prepare` is called at most once per scheduled node per frame.
//! - `encode` is called once per scheduled primitive, in scene order.
//!
//! ## Current-target pass contract
//! The pass returned from [`GpuEncodeContext::pass`] is append-only and owned by the renderer.
//! You must not clear, change load/store ops, change target format, open another pass, or access
//! surface/swapchain resources. Pass and frame-scoped objects must not be retained across frames.
//!
//! ## Threading
//! All GPU callbacks run on the owning window/renderer thread in v1.

use crate::{
    Bounds, ContentMask, DevicePixels, Pixels, ScaledPixels, Scene, Size, TransformationMatrix,
};
use slotmap::SlotMap;
use std::{cell::RefCell, rc::Rc};
use thiserror::Error;

slotmap::new_key_type! {
    /// A window-local identifier for a GPU node.
    pub struct GpuNodeId;
}

/// The scene phase used when sorting GPU primitives relative to the retained scene.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum GpuPhase {
    /// Draw before the retained-scene primitives in the same stacking layer.
    Underlay,
    /// Draw alongside the retained scene using the default phase ordering.
    #[default]
    Inline,
    /// Draw after the retained-scene primitives in the same stacking layer.
    Overlay,
}

impl GpuPhase {
    pub(crate) const fn sort_key(self) -> u32 {
        match self {
            Self::Underlay => 0,
            Self::Inline => 1,
            Self::Overlay => 2,
        }
    }
}

/// Stable error categories for GPU callback failures.
#[derive(Debug, Error)]
pub enum GpuError {
    /// The referenced GPU node id is not live in the current window.
    #[error("invalid GPU node id")]
    InvalidNodeId,
    /// The current renderer backend cannot execute the requested GPU operation.
    #[error("unsupported GPU backend: {0}")]
    UnsupportedBackend(&'static str),
    /// The node attempted an operation that violates the current-target callback contract.
    #[error("GPU pass violation: {0}")]
    PassViolation(&'static str),
    /// A backend-specific failure occurred while preparing or encoding GPU work.
    #[error("GPU backend error: {0}")]
    Backend(String),
}

/// Convenience alias for GPU callback APIs.
pub type GpuResult<T> = std::result::Result<T, GpuError>;

/// The frame target exposed to a GPU callback during encode.
#[derive(Clone, Copy)]
pub struct GpuFrameTargetRef<'a> {
    /// The current render target view.
    pub view: &'a wgpu::TextureView,
    /// The current render target format.
    pub format: wgpu::TextureFormat,
    /// The render target size in device pixels.
    pub size: Size<DevicePixels>,
}

/// Metadata captured for a scene GPU primitive.
///
/// This is the renderer snapshot used during encode. Applications should schedule GPU work using
/// [`GpuPrimitiveDescriptor`] via [`Window::push_gpu_primitive`], not by constructing this type.
#[derive(Clone, Debug)]
pub struct GpuPrimitive {
    /// The phase-adjusted draw order used by the renderer.
    pub order: u32,
    /// The primitive bounds in scaled pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip mask inherited from the retained scene.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The inherited element opacity for this primitive.
    pub opacity: f32,
    /// The transform associated with this primitive.
    pub transformation: TransformationMatrix,
    /// The node that will be prepared and encoded for this primitive.
    pub node: GpuNodeId,
    /// The scene phase used for ordering.
    pub phase: GpuPhase,
}

/// User-facing descriptor for pushing a GPU primitive during paint.
///
/// Use [`Window::push_gpu_primitive`] to schedule a GPU primitive for the current frame.
#[derive(Clone, Copy, Debug)]
pub struct GpuPrimitiveDescriptor {
    /// The bounds in window pixels before applying the window scale factor.
    pub bounds: Bounds<Pixels>,
    /// The GPU node to invoke for this primitive.
    pub node: GpuNodeId,
    /// The transform associated with this primitive.
    pub transformation: TransformationMatrix,
    /// Additional opacity multiplied with the inherited element opacity.
    pub opacity: f32,
    /// The scene phase for this primitive.
    pub phase: GpuPhase,
}

impl GpuPrimitiveDescriptor {
    /// Creates a descriptor with unit transform, full opacity, and inline phase.
    pub fn new(bounds: Bounds<Pixels>, node: GpuNodeId) -> Self {
        Self {
            bounds,
            node,
            transformation: TransformationMatrix::unit(),
            opacity: 1.0,
            phase: GpuPhase::Inline,
        }
    }
}

/// Shared per-window GPU node state referenced by each rendered frame.
#[derive(Clone)]
pub struct GpuFrameState {
    frame_id: u64,
    registry: Rc<RefCell<GpuNodeRegistry>>,
}

impl GpuFrameState {
    pub(crate) fn new(registry: Rc<RefCell<GpuNodeRegistry>>) -> Self {
        Self {
            frame_id: 0,
            registry,
        }
    }

    pub(crate) fn clear_for_reuse(&mut self) {
        self.frame_id = 0;
    }

    pub(crate) fn set_frame_id(&mut self, frame_id: u64) {
        self.frame_id = frame_id;
    }

    /// Returns the captured frame id for this rendered frame.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    pub(crate) fn insert_node(&self, node: impl GpuNode + 'static) -> GpuNodeId {
        self.registry.borrow_mut().insert(node)
    }

    pub(crate) fn retire_node(&self, id: GpuNodeId) -> GpuResult<()> {
        self.registry.borrow_mut().retire(id, self.frame_id)
    }

    pub(crate) fn can_schedule(&self, id: GpuNodeId) -> bool {
        self.registry.borrow().can_schedule(id)
    }

    #[doc(hidden)]
    pub fn begin_wgpu_frame(&self, target_size: Size<DevicePixels>) {
        self.registry.borrow_mut().begin_frame(target_size);
    }

    #[doc(hidden)]
    pub fn notify_device_lost(&self) {
        self.registry.borrow_mut().notify_device_lost();
    }

    #[doc(hidden)]
    pub fn prepare_node(&self, id: GpuNodeId, cx: &mut GpuPrepareContext<'_>) -> GpuResult<()> {
        self.registry
            .borrow_mut()
            .prepare_node(id, self.frame_id, cx)
    }

    #[doc(hidden)]
    pub fn encode_node(&self, id: GpuNodeId, cx: &mut GpuEncodeContext<'_, '_>) -> GpuResult<()> {
        self.registry
            .borrow_mut()
            .encode_node(id, self.frame_id, cx)
    }

    pub(crate) fn finish_frame(&self) {
        self.registry.borrow_mut().finish_frame(self.frame_id);
    }
}

/// The data handed to platform renderers for the current presentation.
pub struct RenderFrame<'a> {
    /// The retained scene for the frame.
    pub scene: &'a Scene,
    /// The GPU node frame state captured alongside the scene.
    pub gpu: &'a GpuFrameState,
}

/// Context available while preparing a node for the current frame.
///
/// `prepare` is called at most once per scheduled node per frame.
pub struct GpuPrepareContext<'a> {
    /// The renderer device.
    pub device: &'a wgpu::Device,
    /// The renderer queue.
    pub queue: &'a wgpu::Queue,
    /// The current render target format.
    pub target_format: wgpu::TextureFormat,
    /// The current render target size in device pixels.
    pub target_size: Size<DevicePixels>,
    /// The frame id being prepared.
    pub frame_id: u64,
}

/// Context available while encoding a single scheduled primitive.
///
/// The pass returned from [`GpuEncodeContext::pass`] is append-only and targets the current
/// window surface. It must not be retained beyond the callback.
pub struct GpuEncodeContext<'pass, 'target> {
    frame_target: GpuFrameTargetRef<'target>,
    primitive: &'pass GpuPrimitive,
    pass: &'pass mut wgpu::RenderPass<'target>,
    frame_id: u64,
}

impl<'pass, 'target> GpuEncodeContext<'pass, 'target> {
    #[doc(hidden)]
    pub fn new(
        frame_target: GpuFrameTargetRef<'target>,
        primitive: &'pass GpuPrimitive,
        pass: &'pass mut wgpu::RenderPass<'target>,
        frame_id: u64,
    ) -> Self {
        Self {
            frame_target,
            primitive,
            pass,
            frame_id,
        }
    }

    /// Returns the frame target associated with this callback.
    pub fn frame_target(&self) -> GpuFrameTargetRef<'target> {
        self.frame_target
    }

    /// Returns the scheduled primitive metadata for this invocation.
    pub fn primitive(&self) -> &GpuPrimitive {
        self.primitive
    }

    /// Returns the frame id for this invocation.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Returns the current append-only render pass.
    ///
    /// Contract:
    /// - Do not clear or change load/store ops.
    /// - Do not change the target format.
    /// - Do not use depth/stencil.
    /// - Do not open a second current-target pass.
    /// - Do not submit or present.
    /// - Do not retain pass-scoped objects beyond the callback.
    pub fn pass(&mut self) -> &mut wgpu::RenderPass<'target> {
        self.pass
    }
}

/// GPU callback interface for scene-integrated custom rendering.
///
/// Callbacks run on the owning window/renderer thread. Use `prepare` to create or update persistent
/// GPU resources, and `encode` to emit draw commands for a single scheduled primitive.
pub trait GpuNode {
    /// Prepare persistent GPU resources for the current frame.
    fn prepare(&mut self, _cx: &mut GpuPrepareContext<'_>) -> GpuResult<()> {
        Ok(())
    }

    /// Encode draw commands for a single scheduled primitive.
    ///
    /// This method may be called multiple times per frame if multiple primitives reference the
    /// same `GpuNodeId`.
    fn encode(&mut self, cx: &mut GpuEncodeContext<'_, '_>) -> GpuResult<()>;

    /// Called before the next `prepare` after the render target size changes.
    fn on_resize(&mut self, _size: Size<DevicePixels>) {}

    /// Called before the next `prepare` after device loss is detected.
    fn on_device_lost(&mut self) {}

    /// Called once after the final frame referencing this node has completed encoding.
    fn destroy(&mut self) {}
}

pub(crate) fn new_gpu_registry() -> Rc<RefCell<GpuNodeRegistry>> {
    Rc::new(RefCell::new(GpuNodeRegistry::default()))
}

#[derive(Default)]
pub(crate) struct GpuNodeRegistry {
    nodes: SlotMap<GpuNodeId, StoredGpuNode>,
    last_target_size: Option<Size<DevicePixels>>,
    device_lost: bool,
}

struct StoredGpuNode {
    node: Box<dyn GpuNode>,
    retired_after_frame: Option<u64>,
}

impl GpuNodeRegistry {
    fn insert(&mut self, node: impl GpuNode + 'static) -> GpuNodeId {
        self.nodes.insert(StoredGpuNode {
            node: Box::new(node),
            retired_after_frame: None,
        })
    }

    fn retire(&mut self, id: GpuNodeId, frame_id: u64) -> GpuResult<()> {
        let Some(node) = self.nodes.get_mut(id) else {
            return Err(GpuError::InvalidNodeId);
        };
        if node.retired_after_frame.is_none() {
            node.retired_after_frame = Some(frame_id);
        }
        Ok(())
    }

    fn can_schedule(&self, id: GpuNodeId) -> bool {
        self.nodes
            .get(id)
            .is_some_and(|node| node.retired_after_frame.is_none())
    }

    fn notify_device_lost(&mut self) {
        self.device_lost = true;
    }

    fn begin_frame(&mut self, target_size: Size<DevicePixels>) {
        if self.last_target_size != Some(target_size) {
            for node in self.nodes.values_mut() {
                node.node.on_resize(target_size);
            }
            self.last_target_size = Some(target_size);
        }

        if self.device_lost {
            for node in self.nodes.values_mut() {
                node.node.on_device_lost();
            }
            self.device_lost = false;
        }
    }

    fn prepare_node(
        &mut self,
        id: GpuNodeId,
        frame_id: u64,
        cx: &mut GpuPrepareContext<'_>,
    ) -> GpuResult<()> {
        let node = self.lookup_live_node(id, frame_id)?;
        node.prepare(cx)
    }

    fn encode_node(
        &mut self,
        id: GpuNodeId,
        frame_id: u64,
        cx: &mut GpuEncodeContext<'_, '_>,
    ) -> GpuResult<()> {
        let node = self.lookup_live_node(id, frame_id)?;
        node.encode(cx)
    }

    fn finish_frame(&mut self, frame_id: u64) {
        let retired: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                node.retired_after_frame
                    .is_some_and(|retired_after| retired_after <= frame_id)
                    .then_some(id)
            })
            .collect();

        for id in retired {
            if let Some(mut node) = self.nodes.remove(id) {
                node.node.destroy();
            }
        }
    }

    fn lookup_live_node(&mut self, id: GpuNodeId, frame_id: u64) -> GpuResult<&mut dyn GpuNode> {
        let Some(node) = self.nodes.get_mut(id) else {
            return Err(GpuError::InvalidNodeId);
        };
        if node
            .retired_after_frame
            .is_some_and(|retired_after| frame_id > retired_after)
        {
            return Err(GpuError::InvalidNodeId);
        }
        Ok(node.node.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestNode {
        destroyed: bool,
    }

    impl GpuNode for TestNode {
        fn encode(&mut self, _cx: &mut GpuEncodeContext<'_, '_>) -> GpuResult<()> {
            Ok(())
        }

        fn destroy(&mut self) {
            self.destroyed = true;
        }
    }

    #[test]
    fn retired_nodes_remain_valid_for_captured_frame_only() {
        let registry = new_gpu_registry();
        let mut first_frame = GpuFrameState::new(registry.clone());
        first_frame.set_frame_id(1);

        let id = first_frame.insert_node(TestNode::default());
        assert!(first_frame.can_schedule(id));

        first_frame.retire_node(id).unwrap();
        assert!(!first_frame.can_schedule(id));

        assert!(registry.borrow_mut().lookup_live_node(id, 1).is_ok());
        assert!(matches!(
            registry.borrow_mut().lookup_live_node(id, 2),
            Err(GpuError::InvalidNodeId)
        ));

        first_frame.finish_frame();
        assert!(matches!(
            registry.borrow_mut().lookup_live_node(id, 1),
            Err(GpuError::InvalidNodeId)
        ));
    }

    #[test]
    fn gpu_node_ids_do_not_silently_reuse_live_generation() {
        let registry = new_gpu_registry();
        let first = {
            let frame = GpuFrameState::new(registry.clone());
            frame.insert_node(TestNode::default())
        };

        {
            let mut registry = registry.borrow_mut();
            registry.retire(first, 0).unwrap();
            registry.finish_frame(0);
        }

        let second = {
            let frame = GpuFrameState::new(registry);
            frame.insert_node(TestNode::default())
        };

        assert_ne!(first, second);
    }
}
