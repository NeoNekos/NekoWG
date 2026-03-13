#![cfg_attr(target_family = "wasm", no_main)]

use nekowg::{
    App, Bounds, Context, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
};
use nekowg_platform::application;

struct BackdropBlurDemo;

impl Render for BackdropBlurDemo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size(px(640.0))
            .bg(nekowg::white())
            .text_color(nekowg::black())
            .relative()
            .child(
                div()
                    .absolute()
                    .left_8()
                    .top_8()
                    .w(px(240.0))
                    .h(px(240.0))
                    .rounded_xl()
                    .bg(rgb(0xff6b6b)),
            )
            .child(
                div()
                    .absolute()
                    .left_24()
                    .top_24()
                    .w(px(260.0))
                    .h(px(260.0))
                    .rounded_xl()
                    .bg(rgb(0x2dd4bf)),
            )
            .child(
                div()
                    .absolute()
                    .left_16()
                    .top_48()
                    .w(px(300.0))
                    .h(px(180.0))
                    .backdrop_blur_3xl()
                    .backdrop_tint(nekowg::white().opacity(0.08))
                    .backdrop_saturation(1.80)
                    .p_3()
                    .text_sm()
                    .child("backdrop_blur_3xl"),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(640.), px(640.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| BackdropBlurDemo),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    nekowg_platform::web_init();
    run_example();
}
