use std::{error::Error, fs, path::Path};

use raytrace_gpu::{RaytraceBuffers, RaytraceConfig, RaytraceStage};
use wgpu_compute_utils::{GpuContext, map_buffer, readback_buffer, storage_buffer};

const CONFIG: RaytraceConfig = RaytraceConfig {
    base_seed: 0x8f31_7a25,
    pixel_len: 64,
    rays_per_pixel: 4,
    batch_size: 16,
};
const OUTPUT_FILE: &str = "raytrace_preview.ppm";

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let gpu = GpuContext::create("Raytrace Preview Device").await?;
    let samples = storage_buffer(&gpu.device, "Preview Samples", CONFIG.sample_bytes(), false);
    let expects = storage_buffer(
        &gpu.device,
        "Preview Expected Distances",
        CONFIG.batch_bytes(),
        false,
    );
    let images = storage_buffer(
        &gpu.device,
        "Preview Float Images",
        CONFIG.image_bytes(),
        false,
    );
    let debug_texture = create_preview_texture(&gpu.device);
    let debug_view = debug_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Raytrace Preview Texture Array View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let raytrace = RaytraceStage::new(
        &gpu.device,
        CONFIG,
        RaytraceBuffers {
            samples: &samples,
            expects: &expects,
            images: &images,
        },
        Some(&debug_view),
    );

    let padded_bytes_per_row = (CONFIG.pixel_len * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback_size =
        u64::from(padded_bytes_per_row) * u64::from(CONFIG.pixel_len * CONFIG.batch_size);
    let readback = readback_buffer(&gpu.device, "Raytrace Preview Readback", readback_size);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Raytrace Preview Encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Raytrace Preview Pass"),
            timestamp_writes: None,
        });
        raytrace.encode(&mut pass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &debug_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(CONFIG.pixel_len),
            },
        },
        wgpu::Extent3d {
            width: CONFIG.pixel_len,
            height: CONFIG.pixel_len,
            depth_or_array_layers: CONFIG.batch_size,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let bytes = map_buffer(&gpu.device, &readback)?;
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(OUTPUT_FILE);
    write_contact_sheet_ppm(&output_path, &bytes, padded_bytes_per_row)?;
    readback.unmap();
    println!("saved {}", output_path.display());
    Ok(())
}

fn create_preview_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Raytrace Preview Texture Array"),
        size: wgpu::Extent3d {
            width: CONFIG.pixel_len,
            height: CONFIG.pixel_len,
            depth_or_array_layers: CONFIG.batch_size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn write_contact_sheet_ppm(
    path: &Path,
    texture_bytes: &[u8],
    padded_bytes_per_row: u32,
) -> Result<(), Box<dyn Error>> {
    let columns = (f64::from(CONFIG.batch_size).sqrt().ceil()) as u32;
    let rows = CONFIG.batch_size.div_ceil(columns);
    let width = columns * CONFIG.pixel_len;
    let height = rows * CONFIG.pixel_len;
    let header = format!("P6\n{width} {height}\n255\n");
    let mut ppm = Vec::with_capacity(header.len() + (width * height * 3) as usize);
    ppm.extend_from_slice(header.as_bytes());

    let layer_stride = u64::from(padded_bytes_per_row) * u64::from(CONFIG.pixel_len);
    for atlas_y in 0..height {
        let tile_y = atlas_y / CONFIG.pixel_len;
        let source_y = atlas_y % CONFIG.pixel_len;
        for atlas_x in 0..width {
            let tile_x = atlas_x / CONFIG.pixel_len;
            let source_x = atlas_x % CONFIG.pixel_len;
            let layer = tile_y * columns + tile_x;
            if layer < CONFIG.batch_size {
                let offset = u64::from(layer) * layer_stride
                    + u64::from(source_y) * u64::from(padded_bytes_per_row)
                    + u64::from(source_x) * 4;
                let offset = offset as usize;
                ppm.extend_from_slice(&texture_bytes[offset..offset + 3]);
            } else {
                ppm.extend_from_slice(&[0, 0, 0]);
            }
        }
    }
    fs::write(path, ppm)?;
    Ok(())
}
