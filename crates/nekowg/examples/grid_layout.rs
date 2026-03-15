use nekowg::{
    App, Bounds, Context, Hsla, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
};
use nekowg_platform::application;

// https://en.wikipedia.org/wiki/Holy_grail_(web_design)
struct HolyGrailExample {}

impl Render for HolyGrailExample {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let block = |color: Hsla| {
            div()
                .size_full()
                .bg(color)
                .border_1()
                .border_dashed()
                .rounded_md()
                .border_color(nekowg::white())
                .items_center()
        };

        div()
            .gap_1()
            .grid()
            .bg(rgb(0x505050))
            .size(px(500.0))
            .shadow_lg()
            .border_1()
            .size_full()
            .grid_cols(5)
            .grid_rows(5)
            .child(
                block(nekowg::white())
                    .row_span(1)
                    .col_span_full()
                    .child("Header"),
            )
            .child(
                block(nekowg::red())
                    .col_span(1)
                    .h_56()
                    .child("Table of contents"),
            )
            .child(
                block(nekowg::green())
                    .col_span(3)
                    .row_span(3)
                    .child("Content"),
            )
            .child(
                block(nekowg::blue())
                    .col_span(1)
                    .row_span(3)
                    .child("AD :(")
                    .text_color(nekowg::white()),
            )
            .child(
                block(nekowg::black())
                    .row_span(1)
                    .col_span_full()
                    .text_color(nekowg::white())
                    .child("Footer"),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(500.), px(500.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| HolyGrailExample {}),
        )
        .unwrap();
        cx.activate(true);
    });
}

fn main() {
    run_example();
}
