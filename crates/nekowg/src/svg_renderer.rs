use crate::{
    AssetSource, DevicePixels, IsZero, RenderImage, Result, SharedString, Size, hash,
    swap_rgba_pa_to_bgra,
};
use image::Frame;
use parking_lot::RwLock;
use resvg::tiny_skia::Pixmap;
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, LazyLock},
};

/// When rendering SVGs, we render them at twice the size to get a higher-quality result.
pub const SMOOTH_SVG_SCALE_FACTOR: f32 = 2.;
const SVG_RASTER_BUCKET_DEVICE_PIXELS: i32 = SMOOTH_SVG_SCALE_FACTOR as i32;

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct RenderSvgParams {
    pub path: SharedString,
    pub size: Size<DevicePixels>,
}

impl PartialEq for RenderSvgParams {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && quantize_svg_raster_size(self.size) == quantize_svg_raster_size(other.size)
    }
}

impl Eq for RenderSvgParams {}

impl Hash for RenderSvgParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        quantize_svg_raster_size(self.size).hash(state);
    }
}

#[derive(Clone)]
/// A struct holding everything necessary to render SVGs.
pub struct SvgRenderer {
    asset_source: Arc<dyn AssetSource>,
    usvg_options: Arc<usvg::Options<'static>>,
    parsed_document_cache: Arc<RwLock<HashMap<u64, Arc<ParsedSvgDocument>>>>,
}

#[derive(Clone)]
pub(crate) struct ParsedSvgDocument(Arc<usvg::Tree>);

/// The size in which to render the SVG.
pub enum SvgSize {
    /// An absolute size in device pixels.
    Size(Size<DevicePixels>),
    /// A scaling factor to apply to the size provided by the SVG.
    ScaleFactor(f32),
}

impl SvgRenderer {
    /// Creates a new SVG renderer with the provided asset source.
    pub fn new(asset_source: Arc<dyn AssetSource>) -> Self {
        static FONT_DB: LazyLock<Arc<usvg::fontdb::Database>> = LazyLock::new(|| {
            let mut db = usvg::fontdb::Database::new();
            db.load_system_fonts();
            Arc::new(db)
        });
        let default_font_resolver = usvg::FontResolver::default_font_selector();
        let font_resolver = Box::new(
            move |font: &usvg::Font, db: &mut Arc<usvg::fontdb::Database>| {
                if db.is_empty() {
                    *db = FONT_DB.clone();
                }
                default_font_resolver(font, db)
            },
        );
        let options = usvg::Options {
            font_resolver: usvg::FontResolver {
                select_font: font_resolver,
                select_fallback: usvg::FontResolver::default_fallback_selector(),
            },
            ..Default::default()
        };
        Self {
            asset_source,
            usvg_options: Arc::new(options),
            parsed_document_cache: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    /// Renders the given bytes into an image buffer.
    pub fn render_single_frame(
        &self,
        bytes: &[u8],
        scale_factor: f32,
        to_brga: bool,
    ) -> Result<Arc<RenderImage>, usvg::Error> {
        let document = self.parse_document(bytes)?;
        self.render_pixmap(
            &document,
            SvgSize::ScaleFactor(scale_factor * SMOOTH_SVG_SCALE_FACTOR),
        )
        .map(|pixmap| {
            let mut buffer =
                image::ImageBuffer::from_raw(pixmap.width(), pixmap.height(), pixmap.take())
                    .unwrap();

            if to_brga {
                for pixel in buffer.chunks_exact_mut(4) {
                    swap_rgba_pa_to_bgra(pixel);
                }
            }

            let mut image = RenderImage::new(SmallVec::from_const([Frame::new(buffer)]));
            image.scale_factor = SMOOTH_SVG_SCALE_FACTOR;
            Arc::new(image)
        })
    }

    pub(crate) fn render_alpha_mask(
        &self,
        params: &RenderSvgParams,
        bytes: Option<&[u8]>,
    ) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        anyhow::ensure!(!params.size.is_zero(), "can't render at a zero size");

        let render_pixmap = |bytes| {
            let document = self.parse_document(bytes)?;
            let pixmap = self.render_pixmap(&document, SvgSize::Size(params.size))?;

            // Convert the pixmap's pixels into an alpha mask.
            let size = Size::new(
                DevicePixels(pixmap.width() as i32),
                DevicePixels(pixmap.height() as i32),
            );
            let alpha_mask = pixmap
                .pixels()
                .iter()
                .map(|p| p.alpha())
                .collect::<Vec<_>>();

            Ok(Some((size, alpha_mask)))
        };

        if let Some(bytes) = bytes {
            render_pixmap(bytes)
        } else if let Some(bytes) = self.asset_source.load(&params.path)? {
            render_pixmap(&bytes)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn parse_document(
        &self,
        bytes: &[u8],
    ) -> Result<Arc<ParsedSvgDocument>, usvg::Error> {
        let key = hash(&bytes);
        if let Some(document) = self.parsed_document_cache.read().get(&key) {
            return Ok(document.clone());
        }

        let document = Arc::new(ParsedSvgDocument(Arc::new(usvg::Tree::from_data(
            bytes,
            &self.usvg_options,
        )?)));
        self.parsed_document_cache
            .write()
            .insert(key, document.clone());
        Ok(document)
    }

    pub(crate) fn load_document(&self, path: &str) -> Result<Option<Arc<ParsedSvgDocument>>> {
        let Some(bytes) = self.asset_source.load(path)? else {
            return Ok(None);
        };
        self.parse_document(&bytes)
            .map(Some)
            .map_err(anyhow::Error::from)
    }

    pub(crate) fn render_alpha_mask_from_document(
        &self,
        params: &RenderSvgParams,
        document: &ParsedSvgDocument,
    ) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        anyhow::ensure!(!params.size.is_zero(), "can't render at a zero size");

        let pixmap = self.render_pixmap(document, SvgSize::Size(params.size))?;
        let size = Size::new(
            DevicePixels(pixmap.width() as i32),
            DevicePixels(pixmap.height() as i32),
        );
        let alpha_mask = pixmap
            .pixels()
            .iter()
            .map(|p| p.alpha())
            .collect::<Vec<_>>();

        Ok(Some((size, alpha_mask)))
    }

    fn render_pixmap(
        &self,
        document: &ParsedSvgDocument,
        size: SvgSize,
    ) -> Result<Pixmap, usvg::Error> {
        let svg_size = document.0.size();
        let scale = match size {
            SvgSize::Size(size) => size.width.0 as f32 / svg_size.width(),
            SvgSize::ScaleFactor(scale) => scale,
        };

        // Render the SVG to a pixmap with the specified width and height.
        let mut pixmap = resvg::tiny_skia::Pixmap::new(
            (svg_size.width() * scale) as u32,
            (svg_size.height() * scale) as u32,
        )
        .ok_or(usvg::Error::InvalidSize)?;

        let transform = resvg::tiny_skia::Transform::from_scale(scale, scale);

        resvg::render(&document.0, transform, &mut pixmap.as_mut());

        Ok(pixmap)
    }
}

#[inline]
pub(crate) fn quantize_svg_raster_size(size: Size<DevicePixels>) -> Size<DevicePixels> {
    size.map(|value| DevicePixels(quantize_svg_dimension(value.0)))
}

#[inline]
fn quantize_svg_dimension(value: i32) -> i32 {
    if value <= 0 {
        return value;
    }
    let bucket = SVG_RASTER_BUCKET_DEVICE_PIXELS.max(1);
    ((value + bucket - 1) / bucket) * bucket
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size;

    #[test]
    fn parsed_svg_document_renders_without_reparsing_bytes() {
        let renderer = SvgRenderer::new(Arc::new(()));
        let bytes = br#"<svg xmlns='http://www.w3.org/2000/svg' width='8' height='4'><rect width='8' height='4' fill='black'/></svg>"#;
        let document = renderer.parse_document(bytes).unwrap();

        let result = renderer
            .render_alpha_mask_from_document(
                &RenderSvgParams {
                    path: "inline-test".into(),
                    size: size(DevicePixels(8), DevicePixels(4)),
                },
                &document,
            )
            .unwrap()
            .unwrap();

        assert_eq!(result.0, size(DevicePixels(8), DevicePixels(4)));
        assert!(result.1.iter().any(|alpha| *alpha > 0));
    }

    #[test]
    fn parsed_svg_document_cache_reuses_same_bytes() {
        let renderer = SvgRenderer::new(Arc::new(()));
        let bytes = br#"<svg xmlns='http://www.w3.org/2000/svg' width='8' height='4'><rect width='8' height='4' fill='black'/></svg>"#;

        let first = renderer.parse_document(bytes).unwrap();
        let second = renderer.parse_document(bytes).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn render_svg_params_quantize_device_sizes() {
        let a = RenderSvgParams {
            path: "icon".into(),
            size: size(DevicePixels(7), DevicePixels(5)),
        };
        let b = RenderSvgParams {
            path: "icon".into(),
            size: size(DevicePixels(8), DevicePixels(6)),
        };
        let c = RenderSvgParams {
            path: "icon".into(),
            size: size(DevicePixels(9), DevicePixels(7)),
        };

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
