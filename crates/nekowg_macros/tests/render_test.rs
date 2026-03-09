#[test]
fn test_derive_render() {
    use nekowg_macros::Render;

    #[derive(Render)]
    struct _Element;
}
