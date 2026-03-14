use crate::window::{ClipMask, pixels_bounds_nearly_equal};
use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, GpuExtent, GpuSurfaceRedrawMode,
    GpuSurfaceRenderer, GpuSurfaceRuntime, InspectorElementId, IntoElement, LayoutId, Pixels,
    Style, StyleRefinement, Styled, Window, request_surface_redraw,
};
use refineable::Refineable as _;
use std::{
    any::TypeId,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

/// Construct a `GpuSurface` element from a renderer.
///
/// The current v1 skeleton records the surface frame graph and lifecycle state, but does not yet
/// execute the recorded graph on a backend GPU.
#[track_caller]
pub fn gpu_surface<R: GpuSurfaceRenderer>(renderer: R) -> GpuSurface<R> {
    GpuSurface::new(renderer)
}

/// A stateful element that owns a `GpuSurfaceRenderer`.
///
/// The current v1 skeleton records frame graphs during paint, while backend execution remains a
/// follow-up step.
pub struct GpuSurface<R> {
    id: ElementId,
    renderer: Option<R>,
    redraw_mode: GpuSurfaceRedrawMode,
    style: StyleRefinement,
    #[cfg(debug_assertions)]
    source: &'static core::panic::Location<'static>,
}

impl<R: GpuSurfaceRenderer> GpuSurface<R> {
    /// Create a new `GpuSurface`.
    #[track_caller]
    pub fn new(renderer: R) -> Self {
        let caller = core::panic::Location::caller();
        Self {
            id: ElementId::CodeLocation(*caller),
            renderer: Some(renderer),
            redraw_mode: GpuSurfaceRedrawMode::default(),
            style: StyleRefinement::default(),
            #[cfg(debug_assertions)]
            source: caller,
        }
    }

    /// Override the stable element id used to persist surface state across frames.
    pub fn id(mut self, id: impl Into<ElementId>) -> Self {
        self.id = id.into();
        self
    }

    /// Configure how the surface requests future frames.
    pub fn redraw_mode(mut self, redraw_mode: GpuSurfaceRedrawMode) -> Self {
        self.redraw_mode = redraw_mode;
        self
    }

    fn with_runtime<T>(
        &mut self,
        global_id: &GlobalElementId,
        window: &mut Window,
        f: impl FnOnce(&mut GpuSurfaceRuntime<R>, &mut Window) -> T,
    ) -> T {
        let incoming_renderer = self.renderer.take();
        window.with_element_state(global_id, |state: Option<GpuSurfaceRuntime<R>>, window| {
            let mut runtime = Self::runtime_from_state(state, incoming_renderer);
            let result = f(&mut runtime, window);
            (result, runtime)
        })
    }

    fn with_existing_runtime<T>(
        global_id: &GlobalElementId,
        window: &mut Window,
        f: impl FnOnce(&mut GpuSurfaceRuntime<R>, &mut Window) -> T,
    ) -> Option<T> {
        let key = (global_id.clone(), TypeId::of::<GpuSurfaceRuntime<R>>());
        let has_runtime = window.next_frame.element_states.contains_key(&key)
            || window.rendered_frame.element_states.contains_key(&key);

        has_runtime.then(|| {
            window.with_element_state(global_id, |state: Option<GpuSurfaceRuntime<R>>, window| {
                let mut runtime = state.expect("GpuSurface runtime should exist");
                let result = f(&mut runtime, window);
                (result, runtime)
            })
        })
    }

    fn runtime_from_state(
        state: Option<GpuSurfaceRuntime<R>>,
        incoming_renderer: Option<R>,
    ) -> GpuSurfaceRuntime<R> {
        match state {
            Some(mut runtime) => {
                if let Some(renderer) = incoming_renderer {
                    runtime.renderer.update(renderer);
                }
                runtime
            }
            None => GpuSurfaceRuntime::new(
                incoming_renderer
                    .expect("GpuSurface renderer was already moved into persistent state"),
            ),
        }
    }
}

impl<R: GpuSurfaceRenderer> IntoElement for GpuSurface<R> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl<R: GpuSurfaceRenderer> Element for GpuSurface<R> {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        #[cfg(debug_assertions)]
        return Some(self.source);

        #[cfg(not(debug_assertions))]
        return None;
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style.clone(), [], cx);
        (layout_id, style)
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        if style.visibility == crate::Visibility::Hidden {
            return;
        }

        let global_id = global_id.expect("GpuSurface always has a global id");
        let scale_factor = window.scale_factor();
        let extent = gpu_surface_extent(bounds, scale_factor);
        let cursor_position = window.mouse_position();
        let redraw_mode = self.redraw_mode;
        let visible_clip = gpu_surface_visible_clip(style, bounds, window);
        let surface_cursor_position =
            gpu_surface_contains_point(bounds, visible_clip, cursor_position).then_some(
                crate::point(
                    cursor_position.x - bounds.origin.x,
                    cursor_position.y - bounds.origin.y,
                ),
            );

        self.with_runtime(global_id, window, |runtime, window| {
            runtime.ensure_initialized();
            runtime.resize_if_needed(extent, scale_factor);
            if extent.is_empty() {
                return;
            }
            runtime.prepare_frame(extent, cursor_position, surface_cursor_position);
            request_surface_redraw(redraw_mode, window, cx);
        });
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if style.visibility == crate::Visibility::Hidden {
            if let Some(global_id) = global_id {
                Self::with_existing_runtime(global_id, window, |runtime, _window| {
                    runtime.clear_frame_graph();
                });
            }
            return;
        }

        let global_id = global_id.expect("GpuSurface always has a global id");
        let extent = gpu_surface_extent(bounds, window.scale_factor());
        window.with_element_opacity(style.opacity, |window| {
            style.paint(bounds, window, cx, |window, _cx| {
                window.with_clip_mask(
                    style.overflow_clip_mask(bounds, window.rem_size()),
                    |window| {
                        self.with_runtime(global_id, window, |runtime, window| {
                            if extent.is_empty() {
                                runtime.clear_frame_graph();
                            } else {
                                if let Err(error) = runtime.encode_frame() {
                                    log::error!(
                                        "GpuSurface {} recorded an invalid frame graph: {:?}",
                                        global_id,
                                        error
                                    );
                                    runtime.clear_frame_graph();
                                } else {
                                    window.paint_gpu_surface(
                                        gpu_surface_backend_id(global_id),
                                        bounds,
                                        &runtime.recorded_graph,
                                        runtime.last_frame.as_ref(),
                                        &runtime.resources.textures,
                                        &runtime.resources.buffers,
                                        &runtime.resources.samplers,
                                        &runtime.resources.render_programs,
                                        &runtime.resources.compute_programs,
                                    );
                                }
                            }
                        });
                    },
                );
            });
        });
    }
}

impl<R> Styled for GpuSurface<R> {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

fn gpu_surface_extent(bounds: Bounds<Pixels>, scale_factor: f32) -> GpuExtent {
    let physical_size = bounds.size.scale(scale_factor);
    GpuExtent {
        width: physical_size.width.ceil().into(),
        height: physical_size.height.ceil().into(),
    }
}

fn gpu_surface_visible_clip(
    style: &Style,
    bounds: Bounds<Pixels>,
    window: &Window,
) -> ClipMask<Pixels> {
    let window_clip = window.clip_mask();
    let inherited_corner_radii = if pixels_bounds_nearly_equal(window_clip.bounds, bounds) {
        window_clip.corner_radii
    } else {
        crate::Corners::default()
    };
    let self_clip = style.overflow_clip_mask(bounds, window.rem_size());

    self_clip
        .map(|mask| mask.intersect(&window_clip))
        .unwrap_or(ClipMask {
            bounds: bounds.intersect(&window_clip.bounds),
            corner_radii: inherited_corner_radii,
        })
}

fn gpu_surface_contains_point(
    bounds: Bounds<Pixels>,
    visible_clip: ClipMask<Pixels>,
    point: crate::Point<Pixels>,
) -> bool {
    visible_clip.bounds.contains(&point)
        && point_within_rounded_rect(bounds, visible_clip.corner_radii, point)
}

fn point_within_rounded_rect(
    bounds: Bounds<Pixels>,
    corner_radii: crate::Corners<Pixels>,
    point: crate::Point<Pixels>,
) -> bool {
    if !bounds.contains(&point) {
        return false;
    }

    let local_x = (point.x - bounds.origin.x).as_f32();
    let local_y = (point.y - bounds.origin.y).as_f32();
    let width = bounds.size.width.as_f32();
    let height = bounds.size.height.as_f32();
    let top_left = corner_radii.top_left.as_f32();
    let top_right = corner_radii.top_right.as_f32();
    let bottom_right = corner_radii.bottom_right.as_f32();
    let bottom_left = corner_radii.bottom_left.as_f32();

    if local_x < top_left && local_y < top_left {
        return point_inside_corner(local_x, local_y, top_left, top_left, top_left);
    }
    if local_x > width - top_right && local_y < top_right {
        return point_inside_corner(local_x, local_y, width - top_right, top_right, top_right);
    }
    if local_x > width - bottom_right && local_y > height - bottom_right {
        return point_inside_corner(
            local_x,
            local_y,
            width - bottom_right,
            height - bottom_right,
            bottom_right,
        );
    }
    if local_x < bottom_left && local_y > height - bottom_left {
        return point_inside_corner(
            local_x,
            local_y,
            bottom_left,
            height - bottom_left,
            bottom_left,
        );
    }

    true
}

fn point_inside_corner(x: f32, y: f32, center_x: f32, center_y: f32, radius: f32) -> bool {
    if radius <= 0.0 {
        return true;
    }

    let dx = x - center_x;
    let dy = y - center_y;
    dx * dx + dy * dy <= radius * radius
}

fn gpu_surface_backend_id(global_id: &GlobalElementId) -> u64 {
    let mut hasher = DefaultHasher::new();
    global_id.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UpdatingRenderer {
        config: u32,
        preserved_state: u32,
        update_calls: usize,
    }

    impl GpuSurfaceRenderer for UpdatingRenderer {
        fn update(&mut self, next_renderer: Self) {
            self.config = next_renderer.config;
            self.update_calls += 1;
        }
    }

    #[test]
    fn runtime_from_state_updates_existing_renderer_configuration() {
        let runtime = GpuSurfaceRuntime::new(UpdatingRenderer {
            config: 1,
            preserved_state: 99,
            update_calls: 0,
        });

        let runtime = GpuSurface::<UpdatingRenderer>::runtime_from_state(
            Some(runtime),
            Some(UpdatingRenderer {
                config: 2,
                preserved_state: 0,
                update_calls: 0,
            }),
        );

        assert_eq!(runtime.renderer.config, 2);
        assert_eq!(runtime.renderer.preserved_state, 99);
        assert_eq!(runtime.renderer.update_calls, 1);
    }

    #[test]
    fn rounded_surface_hit_test_excludes_transparent_corner() {
        let bounds = Bounds::new(
            crate::point(crate::px(10.0), crate::px(20.0)),
            crate::size(crate::px(100.0), crate::px(100.0)),
        );
        let corner_radii = crate::Corners {
            top_left: crate::px(24.0),
            top_right: crate::px(24.0),
            bottom_right: crate::px(24.0),
            bottom_left: crate::px(24.0),
        };

        assert!(!point_within_rounded_rect(
            bounds,
            corner_radii,
            crate::point(crate::px(12.0), crate::px(22.0)),
        ));
        assert!(point_within_rounded_rect(
            bounds,
            corner_radii,
            crate::point(crate::px(34.0), crate::px(44.0)),
        ));
    }
}
