# NekoWG

NekoWG is a fork of [GPUI](https://github.com/zed-industries/zed/tree/main/crates/gpui), oriented toward graphics-heavy desktop UI and rendering.
It embeds programmable GPU content and composites it within the standard layout/paint pipeline.

## New Features and Direction
- GpuSurface v1: write GPU programs in WGSL, record a frame graph, and hand the output texture back to the framework for composition.
- Desktop focus: no Web/WASM support for now.

## Platform and Maintenance Status
- Maintenance focus is Windows.
- Linux is not fully tested due to limited time; issues may exist.
- No macOS hardware for debugging; unexpected behavior is possible.
