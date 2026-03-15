# GpuSurface

GpuSurface is a UI element that lets you generate GPU content inside the normal
layout/paint pipeline. You provide a `GpuSurfaceRenderer` that records a frame
graph; NekoWG composites the output texture with standard UI semantics (clip,
opacity, z-order, scroll). It is **not** a general-purpose backend wrapper or a
`GpuEffect`.

```rust,ignore
use nekowg::{
    gpu_surface, GpuSurfaceRenderer, GpuInitContext, GpuGraphContext, GpuTextureDesc,
    GpuRenderPassDesc, GpuDrawCall,
};

struct Solid;

impl GpuSurfaceRenderer for Solid {
    fn init(&mut self, _cx: &mut GpuInitContext<'_>) {}

    fn encode(&mut self, graph: &mut GpuGraphContext<'_>) {
        let target = graph.transient_texture(GpuTextureDesc {
            render_attachment: true,
            ..GpuTextureDesc::default()
        });
        graph.render_pass(GpuRenderPassDesc {
            label: Some("solid".into()),
            program: /* handle from init */,
            target,
            clear_color: None,
            bindings: Vec::new(),
            draw: GpuDrawCall::FullScreenTriangle,
        });
        graph.present(target);
    }
}

let element = gpu_surface(Solid);
```

Limitations in v1:
- No direct access to backend device/queue/encoder/swapchain handles.
