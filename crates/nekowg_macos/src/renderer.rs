use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use anyhow::Result;
use cocoa::{
    base::{NO, YES},
    foundation::NSSize,
    quartzcore::AutoresizingMask,
};
use metal::{CAMetalLayer, MTLPixelFormat};
use nekowg::{DevicePixels, GpuSpecs, PlatformAtlas, RenderFrame, Size, size};
use nekowg_wgpu::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};
use objc::msg_send;
use raw_window_handle as rwh;

use crate::metal_renderer;

#[derive(Clone)]
pub(crate) struct Context {
    metal: metal_renderer::Context,
    wgpu: GpuContext,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            metal: Default::default(),
            wgpu: Default::default(),
        }
    }
}

pub(crate) enum Renderer {
    Metal(metal_renderer::Renderer),
    Wgpu(WgpuMacRenderer),
}

struct WgpuRawWindow {
    view: NonNull<c_void>,
}

impl rwh::HasWindowHandle for WgpuRawWindow {
    fn window_handle(&self) -> std::result::Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let handle = rwh::AppKitWindowHandle::new(self.view);
        Ok(unsafe { rwh::WindowHandle::borrow_raw(handle.into()) })
    }
}

impl rwh::HasDisplayHandle for WgpuRawWindow {
    fn display_handle(&self) -> std::result::Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        unsafe {
            Ok(rwh::DisplayHandle::borrow_raw(
                rwh::AppKitDisplayHandle::new().into(),
            ))
        }
    }
}

pub(crate) struct WgpuMacRenderer {
    renderer: WgpuRenderer,
    layer: metal::MetalLayer,
}

pub(crate) unsafe fn new_renderer(
    context: Context,
    native_window: *mut c_void,
    native_view: *mut c_void,
    bounds: nekowg::Size<f32>,
    transparent: bool,
) -> Renderer {
    if prefer_wgpu_renderer() {
        let raw_window = NonNull::new(native_view).map(|view| WgpuRawWindow { view });
        if let Some(raw_window) = raw_window {
            let config = WgpuSurfaceConfig {
                size: size(
                    DevicePixels(bounds.width.max(1.0).ceil() as i32),
                    DevicePixels(bounds.height.max(1.0).ceil() as i32),
                ),
                transparent,
            };
            match WgpuRenderer::new(context.wgpu, &raw_window, config, None) {
                Ok(mut renderer) => {
                    renderer.update_transparency(transparent);
                    return Renderer::Wgpu(WgpuMacRenderer {
                        renderer,
                        layer: create_layer(transparent),
                    });
                }
                Err(error) => {
                    log::warn!(
                        "Failed to initialize wgpu renderer on macOS; falling back to Metal: {error:#}"
                    );
                }
            }
        } else {
            log::warn!(
                "native_view was null while creating macOS wgpu renderer; falling back to Metal"
            );
        }
    }

    Renderer::Metal(metal_renderer::new_renderer(
        context.metal,
        native_window,
        native_view,
        bounds,
        transparent,
    ))
}

impl Renderer {
    pub fn layer(&self) -> &metal::MetalLayerRef {
        match self {
            Self::Metal(renderer) => renderer.layer(),
            Self::Wgpu(renderer) => &renderer.layer,
        }
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        match self {
            Self::Metal(renderer) => renderer.layer_ptr(),
            Self::Wgpu(renderer) => renderer.layer.as_ptr(),
        }
    }

    pub fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        match self {
            Self::Metal(renderer) => renderer.sprite_atlas().clone(),
            Self::Wgpu(renderer) => renderer.renderer.sprite_atlas().clone(),
        }
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        match self {
            Self::Metal(renderer) => {
                renderer.set_presents_with_transaction(presents_with_transaction)
            }
            Self::Wgpu(renderer) => renderer
                .layer
                .set_presents_with_transaction(presents_with_transaction),
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        match self {
            Self::Metal(renderer) => renderer.update_drawable_size(size),
            Self::Wgpu(renderer) => {
                let drawable_size = NSSize {
                    width: size.width.0 as f64,
                    height: size.height.0 as f64,
                };
                unsafe {
                    let _: () = msg_send![&*renderer.layer, setDrawableSize: drawable_size];
                }
                renderer.renderer.update_drawable_size(size);
            }
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        match self {
            Self::Metal(renderer) => renderer.update_transparency(transparent),
            Self::Wgpu(renderer) => {
                renderer.layer.set_opaque(!transparent);
                renderer.renderer.update_transparency(transparent);
            }
        }
    }

    pub fn destroy(&self) {
        if let Self::Metal(renderer) = self {
            renderer.destroy();
        }
    }

    pub fn draw(&mut self, frame: &RenderFrame<'_>) {
        match self {
            Self::Metal(renderer) => renderer.draw(frame.scene),
            Self::Wgpu(renderer) => renderer.renderer.draw(frame),
        }
    }

    pub fn gpu_specs(&self) -> Option<GpuSpecs> {
        match self {
            Self::Metal(_) => None,
            Self::Wgpu(renderer) => Some(renderer.renderer.gpu_specs()),
        }
    }

    pub fn supports_gpu_primitives(&self) -> bool {
        matches!(self, Self::Wgpu(_))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &nekowg::Scene) -> Result<image::RgbaImage> {
        match self {
            Self::Metal(renderer) => renderer.render_to_image(scene),
            Self::Wgpu(_) => anyhow::bail!(
                "render_to_image is currently only implemented for the Metal renderer"
            ),
        }
    }
}

fn create_layer(transparent: bool) -> metal::MetalLayer {
    let layer = metal::MetalLayer::new();
    if let Some(device) = metal::Device::system_default() {
        layer.set_device(&device);
    }
    layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    layer.set_opaque(!transparent);
    layer.set_maximum_drawable_count(3);
    unsafe {
        let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
        let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
        let _: () = msg_send![
            &*layer,
            setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                | AutoresizingMask::HEIGHT_SIZABLE
        ];
    }
    layer
}

fn prefer_wgpu_renderer() -> bool {
    !std::env::var("NEKOWG_MACOS_RENDERER").is_ok_and(|value| value.eq_ignore_ascii_case("metal"))
}
