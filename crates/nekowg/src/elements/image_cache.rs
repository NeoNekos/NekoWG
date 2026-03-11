use crate::{
    AnyElement, AnyEntity, App, AppContext, Asset, AssetLogger, Bounds, Element, ElementId, Entity,
    GlobalElementId, ImageAssetLoader, ImageCacheError, InspectorElementId, IntoElement, LayoutId,
    ParentElement, Pixels, RenderImage, Resource, Style, StyleRefinement, Styled, Task, Window,
    hash,
};

use futures::{FutureExt, future::Shared};
use refineable::Refineable;
use smallvec::SmallVec;
use std::{collections::HashMap, fmt, sync::Arc};

/// An image cache element, all its child img elements will use the cache specified by this element.
/// Note that this could as simple as passing an `Entity<T: ImageCache>`
pub fn image_cache(image_cache_provider: impl ImageCacheProvider) -> ImageCacheElement {
    ImageCacheElement {
        image_cache_provider: Box::new(image_cache_provider),
        style: StyleRefinement::default(),
        children: SmallVec::default(),
    }
}

/// A dynamically typed image cache, which can be used to store any image cache
#[derive(Clone)]
pub struct AnyImageCache {
    image_cache: AnyEntity,
    load_fn: fn(
        image_cache: &AnyEntity,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>>,
}

impl<I: ImageCache> From<Entity<I>> for AnyImageCache {
    fn from(image_cache: Entity<I>) -> Self {
        Self {
            image_cache: image_cache.into_any(),
            load_fn: any_image_cache::load::<I>,
        }
    }
}

impl AnyImageCache {
    /// Load an image given a resource
    /// returns the result of loading the image if it has finished loading, or None if it is still loading
    pub fn load(
        &self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        (self.load_fn)(&self.image_cache, resource, window, cx)
    }
}

mod any_image_cache {
    use super::*;

    pub(crate) fn load<I: 'static + ImageCache>(
        image_cache: &AnyEntity,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let image_cache = image_cache.clone().downcast::<I>().unwrap();
        image_cache.update(cx, |image_cache, cx| image_cache.load(resource, window, cx))
    }
}

/// An image cache element.
pub struct ImageCacheElement {
    image_cache_provider: Box<dyn ImageCacheProvider>,
    style: StyleRefinement,
    children: SmallVec<[AnyElement; 2]>,
}

impl ParentElement for ImageCacheElement {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements)
    }
}

impl Styled for ImageCacheElement {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl IntoElement for ImageCacheElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for ImageCacheElement {
    type RequestLayoutState = SmallVec<[LayoutId; 4]>;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let image_cache = self.image_cache_provider.provide(window, cx);
        window.with_image_cache(Some(image_cache), |window| {
            let child_layout_ids = self
                .children
                .iter_mut()
                .map(|child| child.request_layout(window, cx))
                .collect::<SmallVec<_>>();
            let mut style = Style::default();
            style.refine(&self.style);
            let layout_id = window.request_layout(style, child_layout_ids.iter().copied(), cx);
            (layout_id, child_layout_ids)
        })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        for child in &mut self.children {
            child.prepaint(window, cx);
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let image_cache = self.image_cache_provider.provide(window, cx);
        window.with_image_cache(Some(image_cache), |window| {
            for child in &mut self.children {
                child.paint(window, cx);
            }
        })
    }
}

/// An image loading task associated with an image cache.
pub type ImageLoadingTask = Shared<Task<Result<Arc<RenderImage>, ImageCacheError>>>;

/// An image cache item
pub enum ImageCacheItem {
    /// The associated image is currently loading
    Loading(ImageLoadingTask),
    /// This item has loaded an image.
    Loaded(Result<Arc<RenderImage>, ImageCacheError>),
}

impl std::fmt::Debug for ImageCacheItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self {
            ImageCacheItem::Loading(_) => &"Loading...".to_string(),
            ImageCacheItem::Loaded(render_image) => &format!("{:?}", render_image),
        };
        f.debug_struct("ImageCacheItem")
            .field("status", status)
            .finish()
    }
}

impl ImageCacheItem {
    /// Attempt to get the image from the cache item.
    pub fn get(&mut self) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        match self {
            ImageCacheItem::Loading(task) => {
                let res = task.now_or_never()?;
                *self = ImageCacheItem::Loaded(res.clone());
                Some(res)
            }
            ImageCacheItem::Loaded(res) => Some(res.clone()),
        }
    }
}

/// An object that can handle the caching and unloading of images.
/// Implementations of this trait should ensure that images are removed from all windows when they are no longer needed.
pub trait ImageCache: 'static {
    /// Load an image given a resource
    /// returns the result of loading the image if it has finished loading, or None if it is still loading
    fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>>;
}

/// An object that can create an ImageCache during the render phase.
/// See the ImageCache trait for more information.
pub trait ImageCacheProvider: 'static {
    /// Called during the request_layout phase to create an ImageCache.
    fn provide(&mut self, _window: &mut Window, _cx: &mut App) -> AnyImageCache;
}

impl<T: ImageCache> ImageCacheProvider for Entity<T> {
    fn provide(&mut self, _window: &mut Window, _cx: &mut App) -> AnyImageCache {
        self.clone().into()
    }
}

/// An implementation of ImageCache, that uses an LRU caching strategy to unload images when the cache is full
pub struct RetainAllImageCache(HashMap<u64, ImageCacheItem>);

impl fmt::Debug for RetainAllImageCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashMapImageCache")
            .field("num_images", &self.0.len())
            .finish()
    }
}

impl RetainAllImageCache {
    /// Create a new image cache.
    #[inline]
    pub fn new(cx: &mut App) -> Entity<Self> {
        let e = cx.new(|_cx| RetainAllImageCache(HashMap::new()));
        cx.observe_release(&e, |image_cache, cx| {
            for (_, mut item) in std::mem::replace(&mut image_cache.0, HashMap::new()) {
                if let Some(Ok(image)) = item.get() {
                    cx.drop_image(image, None);
                }
            }
        })
        .detach();
        e
    }

    /// Load an image from the given source.
    ///
    /// Returns `None` if the image is loading.
    pub fn load(
        &mut self,
        source: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let hash = hash(source);

        if let Some(item) = self.0.get_mut(&hash) {
            return item.get();
        }

        let fut = AssetLogger::<ImageAssetLoader>::load(source.clone(), cx);
        let task = cx.background_executor().spawn(fut).shared();
        self.0.insert(hash, ImageCacheItem::Loading(task.clone()));

        let entity = window.current_view();
        window
            .spawn(cx, {
                async move |cx| {
                    _ = task.await;
                    cx.on_next_frame(move |_, cx| {
                        cx.notify(entity);
                    });
                }
            })
            .detach();

        None
    }

    /// Clear the image cache.
    pub fn clear(&mut self, window: &mut Window, cx: &mut App) {
        for (_, mut item) in std::mem::replace(&mut self.0, HashMap::new()) {
            if let Some(Ok(image)) = item.get() {
                cx.drop_image(image, Some(window));
            }
        }
    }

    /// Remove the image from the cache by the given source.
    pub fn remove(&mut self, source: &Resource, window: &mut Window, cx: &mut App) {
        let hash = hash(source);
        if let Some(mut item) = self.0.remove(&hash)
            && let Some(Ok(image)) = item.get()
        {
            cx.drop_image(image, Some(window));
        }
    }

    /// Returns the number of images in the cache.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl ImageCache for RetainAllImageCache {
    fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        RetainAllImageCache::load(self, resource, window, cx)
    }
}

/// Default memory budget for the global LRU image cache.
pub const DEFAULT_IMAGE_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Constructs a retain-all image cache that uses the element state associated with the given ID.
pub fn retain_all(id: impl Into<ElementId>) -> RetainAllImageCacheProvider {
    RetainAllImageCacheProvider { id: id.into() }
}

/// A provider struct for creating a retain-all image cache inline
pub struct RetainAllImageCacheProvider {
    id: ElementId,
}

impl ImageCacheProvider for RetainAllImageCacheProvider {
    fn provide(&mut self, window: &mut Window, cx: &mut App) -> AnyImageCache {
        window
            .with_global_id(self.id.clone(), |global_id, window| {
                window.with_element_state::<Entity<RetainAllImageCache>, _>(
                    global_id,
                    |cache, _window| {
                        let mut cache = cache.unwrap_or_else(|| RetainAllImageCache::new(cx));
                        (cache.clone(), cache)
                    },
                )
            })
            .into()
    }
}

/// An LRU image cache that evicts entries when it exceeds a byte budget.
pub struct LruImageCache {
    entries: HashMap<u64, LruImageCacheEntry>,
    max_bytes: usize,
    current_bytes: usize,
    clock: u64,
}

struct LruImageCacheEntry {
    item: ImageCacheItem,
    last_used: u64,
    bytes: usize,
    source: Resource,
}

impl LruImageCache {
    /// Create a new LRU image cache with a maximum memory budget in bytes.
    #[inline]
    pub fn new(max_bytes: usize, cx: &mut App) -> Entity<Self> {
        let e = cx.new(|_cx| LruImageCache {
            entries: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            clock: 0,
        });
        cx.observe_release(&e, |image_cache, cx| {
            for (_, entry) in std::mem::replace(&mut image_cache.entries, HashMap::new()) {
                image_cache.evict_entry(entry, None, cx);
            }
            image_cache.current_bytes = 0;
        })
        .detach();
        e
    }

    /// Load an image from the given source.
    /// Returns `None` if the image is loading.
    pub fn load(
        &mut self,
        source: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let hash = hash(source);
        self.clock = self.clock.wrapping_add(1);
        let now = self.clock;

        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.last_used = now;
            let result = entry.item.get();
            if let Some(result) = &result {
                let new_bytes = result
                    .as_ref()
                    .map(|image| image.memory_bytes_len())
                    .unwrap_or(0);
                if new_bytes != entry.bytes {
                    let old_bytes = entry.bytes;
                    if new_bytes > entry.bytes {
                        self.current_bytes += new_bytes - entry.bytes;
                    } else {
                        self.current_bytes -= entry.bytes - new_bytes;
                    }
                    entry.bytes = new_bytes;
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "image cache bytes update: source={:?} bytes={} -> {} current={} max={}",
                            entry.source,
                            old_bytes,
                            new_bytes,
                            self.current_bytes,
                            self.max_bytes
                        );
                    }
                }
                self.evict_if_needed(window, cx);
            }
            return result;
        }

        let fut = AssetLogger::<ImageAssetLoader>::load(source.clone(), cx);
        let task = cx.background_executor().spawn(fut).shared();
        self.entries.insert(
            hash,
            LruImageCacheEntry {
                item: ImageCacheItem::Loading(task.clone()),
                last_used: now,
                bytes: 0,
                source: source.clone(),
            },
        );

        let entity = window.current_view();
        window
            .spawn(cx, {
                async move |cx| {
                    _ = task.await;
                    cx.on_next_frame(move |_, cx| {
                        cx.notify(entity);
                    });
                }
            })
            .detach();

        None
    }

    fn evict_if_needed(&mut self, window: &mut Window, cx: &mut App) {
        while self.current_bytes > self.max_bytes {
            let Some(candidate) = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.bytes > 0)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(hash, _)| *hash)
            else {
                break;
            };

            if let Some(entry) = self.entries.remove(&candidate) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "image cache evict: source={:?} bytes={} current={} max={}",
                        entry.source,
                        entry.bytes,
                        self.current_bytes,
                        self.max_bytes
                    );
                }
                self.evict_entry(entry, Some(window), cx);
            }
        }
    }

    fn evict_entry(
        &self,
        mut entry: LruImageCacheEntry,
        window: Option<&mut Window>,
        cx: &mut App,
    ) {
        if let Some(result) = entry.item.get() {
            if let Ok(image) = result {
                cx.drop_image(image, window);
            }
        }
        cx.remove_asset::<AssetLogger<ImageAssetLoader>>(&entry.source);
    }

    /// Clear the image cache.
    pub fn clear(&mut self, window: &mut Window, cx: &mut App) {
        for (_, entry) in std::mem::replace(&mut self.entries, HashMap::new()) {
            self.evict_entry(entry, Some(window), cx);
        }
        self.current_bytes = 0;
    }

    /// Remove the image from the cache by the given source.
    pub fn remove(&mut self, source: &Resource, window: &mut Window, cx: &mut App) {
        let hash = hash(source);
        if let Some(entry) = self.entries.remove(&hash) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
            self.evict_entry(entry, Some(window), cx);
        }
    }

    /// Returns the number of images in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl ImageCache for LruImageCache {
    fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        LruImageCache::load(self, resource, window, cx)
    }
}

/// Constructs an LRU image cache that uses the element state associated with the given ID.
pub fn lru(id: impl Into<ElementId>, max_bytes: usize) -> LruImageCacheProvider {
    LruImageCacheProvider {
        id: id.into(),
        max_bytes,
    }
}

/// A provider struct for creating an LRU image cache inline.
pub struct LruImageCacheProvider {
    id: ElementId,
    max_bytes: usize,
}

impl ImageCacheProvider for LruImageCacheProvider {
    fn provide(&mut self, window: &mut Window, cx: &mut App) -> AnyImageCache {
        let max_bytes = self.max_bytes;
        window
            .with_global_id(self.id.clone(), |global_id, window| {
                window.with_element_state::<Entity<LruImageCache>, _>(
                    global_id,
                    |cache, _window| {
                        let mut cache = cache.unwrap_or_else(|| LruImageCache::new(max_bytes, cx));
                        (cache.clone(), cache)
                    },
                )
            })
            .into()
    }
}
