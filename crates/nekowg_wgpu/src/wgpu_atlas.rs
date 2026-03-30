use anyhow::{Context as _, Result};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use nekowg::{
    AtlasKey, AtlasStats, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds,
    DevicePixels, PlatformAtlas, Point, Size, atlas_entry_is_stale,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

pub struct WgpuAtlas(Mutex<WgpuAtlasState>);

#[derive(Clone)]
struct AtlasEntry {
    tile: AtlasTile,
    last_used_frame: u64,
}

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    storage: WgpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasEntry>,
    removed_keys: Vec<AtlasKey>,
    pending_uploads: Vec<PendingUpload>,
    current_frame: u64,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
}

impl WgpuAtlas {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Mutex::new(WgpuAtlasState {
            device,
            queue,
            max_texture_size,
            storage: WgpuAtlasStorage::default(),
            tiles_by_key: Default::default(),
            removed_keys: Vec::new(),
            pending_uploads: Vec::new(),
            current_frame: 0,
        }))
    }

    pub fn before_frame(&self) {
        let mut lock = self.0.lock();
        lock.flush_uploads();
    }

    pub fn get_texture_info(&self, id: AtlasTextureId) -> WgpuTextureInfo {
        let lock = self.0.lock();
        let texture = &lock.storage[id];
        WgpuTextureInfo {
            view: texture.view.clone(),
        }
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
        let mut lock = self.0.lock();
        lock.device = device;
        lock.queue = queue;
        lock.storage = WgpuAtlasStorage::default();
        let removed_keys = lock.tiles_by_key.keys().cloned().collect::<Vec<_>>();
        lock.removed_keys.extend(removed_keys);
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
        lock.current_frame = 0;
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn begin_frame(&self) {
        let mut lock = self.0.lock();
        lock.current_frame = lock.current_frame.wrapping_add(1).max(1);
    }

    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        let current_frame = lock.current_frame;
        if let Some(entry) = lock.tiles_by_key.get_mut(key) {
            entry.last_used_frame = current_frame;
            Ok(Some(entry.tile.clone()))
        } else {
            profiling::scope!("new tile");
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock
                .allocate(size, key.texture_kind())
                .context("failed to allocate")?;
            lock.upload_texture(tile.texture_id, tile.bounds, &bytes);
            lock.tiles_by_key.insert(
                key.clone(),
                AtlasEntry {
                    tile: tile.clone(),
                    last_used_frame: current_frame,
                },
            );
            Ok(Some(tile))
        }
    }

    fn end_frame(&self) {
        let mut lock = self.0.lock();
        let current_frame = lock.current_frame;
        let stale_keys = lock
            .tiles_by_key
            .iter()
            .filter_map(|(key, entry)| {
                atlas_entry_is_stale(entry.last_used_frame, current_frame).then_some(key.clone())
            })
            .collect::<Vec<_>>();
        for key in stale_keys {
            lock.remove_entry(&key);
        }
    }

    fn drain_removed_keys(&self, out: &mut Vec<AtlasKey>) {
        let mut lock = self.0.lock();
        out.append(&mut lock.removed_keys);
    }

    fn stats(&self) -> AtlasStats {
        let lock = self.0.lock();
        lock.stats()
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();
        lock.remove_entry(key);
    }
}

impl WgpuAtlasState {
    fn stats(&self) -> AtlasStats {
        AtlasStats {
            entry_count: self.tiles_by_key.len(),
            texture_count: self.storage.texture_count(),
            estimated_bytes: self.storage.estimated_bytes(),
        }
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel => wgpu::TextureFormat::Bgra8Unorm,
            AtlasTextureKind::Polychrome => wgpu::TextureFormat::Bgra8Unorm,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            size,
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        self.pending_uploads.push(PendingUpload {
            id,
            bounds,
            data: bytes.to_vec(),
        });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let texture = &self.storage[upload.id];
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn remove_entry(&mut self, key: &AtlasKey) {
        let Some(entry) = self.tiles_by_key.remove(key) else {
            return;
        };
        self.removed_keys.push(key.clone());
        let tile = entry.tile;
        self.pending_uploads
            .retain(|upload| upload.id != tile.texture_id || upload.bounds != tile.bounds);

        let textures = &mut self.storage[tile.texture_id.kind];
        let texture_index = tile.texture_id.index as usize;
        let Some(texture_slot) = textures.textures.get_mut(texture_index) else {
            return;
        };
        if let Some(texture) = texture_slot.as_mut() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                *texture_slot = None;
                textures.free_list.push(texture_index);
            }
        }
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize]
            .as_ref()
            .expect("texture must exist")
    }
}

impl WgpuAtlasStorage {
    fn texture_count(&self) -> usize {
        self.monochrome_textures.textures.iter().flatten().count()
            + self.subpixel_textures.textures.iter().flatten().count()
            + self.polychrome_textures.textures.iter().flatten().count()
    }

    fn estimated_bytes(&self) -> usize {
        self.monochrome_textures
            .textures
            .iter()
            .flatten()
            .map(WgpuAtlasTexture::estimated_bytes)
            .sum::<usize>()
            + self
                .subpixel_textures
                .textures
                .iter()
                .flatten()
                .map(WgpuAtlasTexture::estimated_bytes)
                .sum::<usize>()
            + self
                .polychrome_textures
                .textures
                .iter()
                .flatten()
                .map(WgpuAtlasTexture::estimated_bytes)
                .sum::<usize>()
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    size: Size<DevicePixels>,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn estimated_bytes(&self) -> usize {
        self.size.width.0.max(0) as usize
            * self.size.height.0.max(0) as usize
            * self.bytes_per_pixel() as usize
    }

    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}
