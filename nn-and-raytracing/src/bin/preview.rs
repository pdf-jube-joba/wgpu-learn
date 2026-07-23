use std::{error::Error, fs, path::Path, sync::mpsc};

use wgpu::util::DeviceExt;

const PIXEL_LEN: u32 = 64;
const RAYS_PER_PIXEL: u32 = 4;
const BATCH_SIZE: u32 = 16;
const HIDDEN_SIZE: u32 = 64;
const LEARNING_RATE: f32 = 0.001;
const OUTPUT_FILE: &str = "raytrace_preview.bmp";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
    seed: u32,
    pixel_len: u32,
    sample_ray: u32,
    middle_num: u32,
    batch_size: u32,
    rate: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Raytrace Preview Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;

    let constants = Constants {
        seed: 0x8f31_7a25,
        pixel_len: PIXEL_LEN,
        sample_ray: RAYS_PER_PIXEL,
        middle_num: HIDDEN_SIZE,
        batch_size: BATCH_SIZE,
        rate: LEARNING_RATE,
    };
    let constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Preview Constants"),
        contents: bytemuck::bytes_of(&constants),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let samples_buffer = storage_buffer(&device, "Preview Samples", u64::from(BATCH_SIZE) * 3 * 4);
    let expects_buffer = storage_buffer(
        &device,
        "Preview Expected Distances",
        u64::from(BATCH_SIZE) * 4,
    );
    let images_buffer = storage_buffer(
        &device,
        "Preview Float Images",
        u64::from(BATCH_SIZE) * u64::from(PIXEL_LEN) * u64::from(PIXEL_LEN) * 4,
    );

    let debug_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Raytrace Preview Texture Array"),
        size: wgpu::Extent3d {
            width: PIXEL_LEN,
            height: PIXEL_LEN,
            depth_or_array_layers: BATCH_SIZE,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let debug_texture_view = debug_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Raytrace Preview Texture Array View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let constants_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Preview Constants Layout"),
        entries: &[buffer_layout_entry(0, wgpu::BufferBindingType::Uniform)],
    });
    let generate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Generate Samples Layout"),
        entries: &[
            buffer_layout_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let raytrace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Raytrace Layout"),
        entries: &[
            buffer_layout_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
        ],
    });

    let constants_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preview Constants Bind Group"),
        layout: &constants_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: constants_buffer.as_entire_binding(),
        }],
    });
    let generate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Generate Samples Bind Group"),
        layout: &generate_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: samples_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: expects_buffer.as_entire_binding(),
            },
        ],
    });
    let raytrace_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raytrace Bind Group"),
        layout: &raytrace_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: samples_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: images_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&debug_texture_view),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Raytrace Preview Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../generate_image.wgsl").into()),
    });
    let generate_pipeline = create_pipeline(
        &device,
        &shader,
        "Generate Samples Pipeline",
        "generate_samples",
        &constants_layout,
        &generate_layout,
    );
    let raytrace_pipeline = create_pipeline(
        &device,
        &shader,
        "Raytrace Pipeline",
        "raytrace",
        &constants_layout,
        &raytrace_layout,
    );

    let unpadded_bytes_per_row = PIXEL_LEN * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let layer_size = u64::from(padded_bytes_per_row) * u64::from(PIXEL_LEN);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Raytrace Preview Readback"),
        size: layer_size * u64::from(BATCH_SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Raytrace Preview Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Generate Samples Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&generate_pipeline);
        pass.set_bind_group(0, &constants_bind_group, &[]);
        pass.set_bind_group(1, &generate_bind_group, &[]);
        pass.dispatch_workgroups(BATCH_SIZE.div_ceil(64), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Raytrace Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&raytrace_pipeline);
        pass.set_bind_group(0, &constants_bind_group, &[]);
        pass.set_bind_group(1, &raytrace_bind_group, &[]);
        pass.dispatch_workgroups(PIXEL_LEN.div_ceil(8), PIXEL_LEN.div_ceil(8), BATCH_SIZE);
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
                rows_per_image: Some(PIXEL_LEN),
            },
        },
        wgpu::Extent3d {
            width: PIXEL_LEN,
            height: PIXEL_LEN,
            depth_or_array_layers: BATCH_SIZE,
        },
    );
    queue.submit([encoder.finish()]);

    let bytes = map_buffer(&device, &readback)?;
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(OUTPUT_FILE);
    write_contact_sheet_bmp(
        &output_path,
        &bytes,
        PIXEL_LEN,
        BATCH_SIZE,
        padded_bytes_per_row,
    )?;
    readback.unmap();

    println!("saved {}", output_path.display());
    Ok(())
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn buffer_layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry_point: &str,
    constants_layout: &wgpu::BindGroupLayout,
    data_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(constants_layout), Some(data_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn map_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, Box<dyn Error>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("readback receiver should exist");
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    Ok(bytes)
}

fn write_contact_sheet_bmp(
    path: &Path,
    texture_bytes: &[u8],
    image_size: u32,
    layers: u32,
    padded_bytes_per_row: u32,
) -> Result<(), Box<dyn Error>> {
    let columns = (f64::from(layers).sqrt().ceil()) as u32;
    let rows = layers.div_ceil(columns);
    let width = columns * image_size;
    let height = rows * image_size;
    let bmp_row_size = (width * 3).div_ceil(4) * 4;
    let pixel_data_size = bmp_row_size * height;
    let file_size = 54 + pixel_data_size;

    let mut bmp = Vec::with_capacity(file_size as usize);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&file_size.to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(-(height as i32)).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&pixel_data_size.to_le_bytes());
    bmp.extend_from_slice(&2835u32.to_le_bytes());
    bmp.extend_from_slice(&2835u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());

    let layer_stride = u64::from(padded_bytes_per_row) * u64::from(image_size);
    for atlas_y in 0..height {
        let tile_y = atlas_y / image_size;
        let source_y = atlas_y % image_size;
        for atlas_x in 0..width {
            let tile_x = atlas_x / image_size;
            let source_x = atlas_x % image_size;
            let layer = tile_y * columns + tile_x;
            if layer < layers {
                let offset = u64::from(layer) * layer_stride
                    + u64::from(source_y) * u64::from(padded_bytes_per_row)
                    + u64::from(source_x) * 4;
                let offset = offset as usize;
                let red = texture_bytes[offset];
                let green = texture_bytes[offset + 1];
                let blue = texture_bytes[offset + 2];
                bmp.extend_from_slice(&[blue, green, red]);
            } else {
                bmp.extend_from_slice(&[0, 0, 0]);
            }
        }
        bmp.resize(bmp.len() + (bmp_row_size - width * 3) as usize, 0);
    }

    fs::write(path, bmp)?;
    Ok(())
}
