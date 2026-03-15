use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame, ImageFormat};
use parking_lot::RwLock;
use std::{
    borrow::Cow,
    collections::VecDeque,
    fmt,
    hash::Hash,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// Source data that can be used to rehydrate discarded pixel buffers.
#[derive(Clone)]
pub enum RenderImageSource {
    /// Raster-encoded image bytes (PNG/JPEG/WebP/etc) used to rehydrate discarded pixels.
    Raster {
        /// The encoded image format.
        format: ImageFormat,
        /// Encoded bytes for the image.
        bytes: Arc<[u8]>,
    },
}

struct RenderFrame {
    size: Size<DevicePixels>,
    delay: Delay,
    pixels: Option<Arc<[u8]>>,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    frames: RwLock<SmallVec<[RenderFrame; 1]>>,
    frame_cache: RwLock<FrameCache>,
    source: Option<RenderImageSource>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let frames = data
            .into()
            .into_iter()
            .map(RenderFrame::from_frame)
            .collect::<SmallVec<[RenderFrame; 1]>>();

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            frames: RwLock::new(frames),
            frame_cache: RwLock::new(FrameCache::default()),
            source: None,
        }
    }

    fn from_render_frames(frames: SmallVec<[RenderFrame; 1]>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            frames: RwLock::new(frames),
            frame_cache: RwLock::new(FrameCache::default()),
            source: None,
        }
    }

    pub(crate) fn from_metadata(
        frames: &[(Size<DevicePixels>, Delay)],
        first_pixels: Option<Arc<[u8]>>,
    ) -> Self {
        let mut render_frames = SmallVec::<[RenderFrame; 1]>::new();
        for (index, (size, delay)) in frames.iter().enumerate() {
            let pixels = if index == 0 {
                first_pixels.clone()
            } else {
                None
            };
            render_frames.push(RenderFrame::new(*size, *delay, pixels));
        }
        Self::from_render_frames(render_frames)
    }

    /// Associate a source with this image so discarded pixels can be rehydrated later.
    pub fn with_source(mut self, source: RenderImageSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Returns the encoded raster source for this image, if available.
    pub fn encoded_source(&self) -> Option<(ImageFormat, Arc<[u8]>)> {
        match self.source.as_ref()? {
            RenderImageSource::Raster { format, bytes } => Some((*format, bytes.clone())),
        }
    }

    /// Convert this image into a byte buffer.
    pub fn bytes(&self, frame_index: usize) -> Option<Arc<[u8]>> {
        if let Some(bytes) = self.frame_pixels(frame_index) {
            return Some(bytes);
        }

        if let Some(bytes) = self.cached_frame(frame_index) {
            return Some(bytes);
        }

        self.rehydrate_frame(frame_index)
    }

    fn frame_pixels(&self, frame_index: usize) -> Option<Arc<[u8]>> {
        self.frames
            .read()
            .get(frame_index)
            .and_then(|frame| frame.pixels.as_ref().cloned())
    }

    fn cached_frame(&self, frame_index: usize) -> Option<Arc<[u8]>> {
        self.frame_cache.write().get(frame_index)
    }

    pub(crate) fn cache_frame_pixels(&self, frame_index: usize, pixels: Arc<[u8]>) {
        self.frame_cache.write().insert(frame_index, pixels);
    }

    fn rehydrate_frame(&self, frame_index: usize) -> Option<Arc<[u8]>> {
        let RenderImageSource::Raster { format, bytes } = self.source.as_ref()?;

        // Only support rehydration for single-frame raster images.
        if frame_index != 0 || self.frame_count() > 1 {
            return None;
        }

        let mut data = image::load_from_memory_with_format(bytes, *format)
            .ok()?
            .into_rgba8();
        for pixel in data.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        let pixels: Arc<[u8]> = Arc::from(data.into_raw().into_boxed_slice());

        let mut frames = self.frames.write();
        if let Some(frame) = frames.get_mut(frame_index) {
            frame.pixels = Some(pixels.clone());
            return Some(pixels);
        }

        None
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.frames.read()[frame_index].size
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.frames.read()[frame_index].delay
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        self.frames.read().len()
    }

    /// Returns the number of bytes currently retained for this image.
    pub fn memory_bytes_len(&self) -> usize {
        let frame_bytes: usize = self
            .frames
            .read()
            .iter()
            .filter_map(|frame| frame.pixels.as_ref())
            .map(|pixels| pixels.len())
            .sum();
        let cache_bytes = self.frame_cache.read().total_bytes();
        let source_bytes = match &self.source {
            Some(RenderImageSource::Raster { bytes, .. }) => bytes.len(),
            None => 0,
        };

        frame_bytes + cache_bytes + source_bytes
    }

    /// Discard the CPU pixel buffer for the given frame if it can be rehydrated.
    pub fn discard_frame_pixels(&self, frame_index: usize) {
        if self.frame_count() > 1 {
            return;
        }
        if self.source.is_none() {
            return;
        }
        let mut frames = self.frames.write();
        if let Some(frame) = frames.get_mut(frame_index) {
            frame.pixels = None;
        }
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.size(0))
            .finish()
    }
}

impl RenderFrame {
    pub(crate) fn new(size: Size<DevicePixels>, delay: Delay, pixels: Option<Arc<[u8]>>) -> Self {
        Self {
            size,
            delay,
            pixels,
        }
    }

    fn from_frame(frame: Frame) -> Self {
        let delay = frame.delay();
        let buffer = frame.into_buffer();
        let (width, height) = buffer.dimensions();
        let size = size(DevicePixels(width as i32), DevicePixels(height as i32));
        let pixels: Arc<[u8]> = Arc::from(buffer.into_raw().into_boxed_slice());
        Self {
            size,
            delay,
            pixels: Some(pixels),
        }
    }
}

const ANIMATED_FRAME_CACHE_FRAMES: usize = 3;

#[derive(Default)]
struct FrameCache {
    frames: VecDeque<CachedFrame>,
}

impl FrameCache {
    fn get(&mut self, frame_index: usize) -> Option<Arc<[u8]>> {
        if let Some(pos) = self
            .frames
            .iter()
            .position(|entry| entry.index == frame_index)
        {
            if let Some(entry) = self.frames.remove(pos) {
                let pixels = entry.pixels.clone();
                self.frames.push_back(entry);
                return Some(pixels);
            }
        }
        None
    }

    fn insert(&mut self, frame_index: usize, pixels: Arc<[u8]>) {
        if let Some(pos) = self
            .frames
            .iter()
            .position(|entry| entry.index == frame_index)
        {
            self.frames.remove(pos);
        }
        self.frames.push_back(CachedFrame {
            index: frame_index,
            pixels,
        });
        while self.frames.len() > ANIMATED_FRAME_CACHE_FRAMES {
            self.frames.pop_front();
        }
    }

    fn total_bytes(&self) -> usize {
        self.frames.iter().map(|entry| entry.pixels.len()).sum()
    }
}

struct CachedFrame {
    index: usize,
    pixels: Arc<[u8]>,
}
