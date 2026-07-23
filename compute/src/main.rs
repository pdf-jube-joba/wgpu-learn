use std::{env, error::Error, sync::mpsc, time::Instant};

const DEFAULT_WIDTH: u32 = 4_096;
const DEFAULT_HEIGHT: u32 = 4_096;
const DEFAULT_STEPS: u32 = 1000;
const WORKGROUP_SIZE: u32 = 16;
const ALPHA: f32 = 0.20;

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let config = Config::from_args()?;
    let buffer_size = config.buffer_size()?;
    let cells = config.cell_count();

    println!(
        "grid: {} x {} ({cells} cells, {:.1} MiB per buffer)",
        config.width,
        config.height,
        buffer_size as f64 / 1024.0 / 1024.0
    );
    println!(
        "allocated buffers: current + next + readback = {:.1} MiB",
        (buffer_size * 3) as f64 / 1024.0 / 1024.0
    );

    let (device, queue) = create_gpu(buffer_size).await?;
    let simulation = Simulation::new(&device, &config).await?;

    let started_at = Instant::now();
    dispatch_simulation(&device, &queue, &simulation, &config)?;
    let summary = summarize_readback(&device, &simulation.readback, &config, buffer_size)?;

    println!("steps: {}", config.steps);
    println!("elapsed: {:.2?}", started_at.elapsed());
    println!("min: {:.4}, max: {:.4}", summary.min, summary.max);
    println!(
        "center: {:.4}, quarter: {:.4}, corner: {:.4}",
        summary.center, summary.quarter, summary.corner
    );

    Ok(())
}

async fn create_gpu(buffer_size: u64) -> Result<(wgpu::Device, wgpu::Queue), Box<dyn Error>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("failed to find a GPU adapter");

    let adapter_limits = adapter.limits();
    if adapter_limits.max_buffer_size < buffer_size
        || adapter_limits.max_storage_buffer_binding_size < buffer_size
    {
        return Err(format!(
            "adapter limits are too small: need {buffer_size} bytes, \
             max_buffer_size={}, max_storage_buffer_binding_size={}",
            adapter_limits.max_buffer_size, adapter_limits.max_storage_buffer_binding_size
        )
        .into());
    }

    let device_and_queue = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_buffer_size: buffer_size,
                max_storage_buffer_binding_size: buffer_size,
                ..wgpu::Limits::default()
            },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;

    Ok(device_and_queue)
}

struct Simulation {
    current: wgpu::Buffer,
    next: wgpu::Buffer,
    readback: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    current_to_next: wgpu::BindGroup,
    next_to_current: wgpu::BindGroup,
}

impl Simulation {
    async fn new(device: &wgpu::Device, config: &Config) -> Result<Self, Box<dyn Error>> {
        let buffer_size = config.buffer_size()?;

        // buffers
        let current = create_initial_grid_buffer(device, config, buffer_size).await?;
        let next = create_buffer_checked(
            device,
            wgpu::BufferDescriptor {
                label: Some("Next Grid"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .await?;
        let params = create_params_buffer(device, config).await?;
        let readback = create_buffer_checked(
            device,
            wgpu::BufferDescriptor {
                label: Some("Readback"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .await?;

        // pipeline, bindgroups
        let (pipeline, bind_group_layout) = create_compute_pipeline(device);
        let current_to_next =
            create_bind_group(device, &bind_group_layout, &current, &next, &params);
        let next_to_current =
            create_bind_group(device, &bind_group_layout, &next, &current, &params);

        Ok(Self {
            current,
            next,
            readback,
            pipeline,
            current_to_next,
            next_to_current,
        })
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heat Diffusion Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Heat Diffusion Bind Group Layout"),
        entries: &[
            // 意味もなく番号を入れ替えているのは、これでも動くことを確かめるため。
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Heat Diffusion Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Heat Diffusion Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn dispatch_simulation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    simulation: &Simulation,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    let buffer_size = config.buffer_size()?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Heat Diffusion Encoder"),
    });
    let groups_x = config.width.div_ceil(WORKGROUP_SIZE);
    let groups_y = config.height.div_ceil(WORKGROUP_SIZE);

    for step in 0..config.steps {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Heat Diffusion Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&simulation.pipeline);
        pass.set_bind_group(
            0,
            if step % 2 == 0 {
                &simulation.current_to_next
            } else {
                &simulation.next_to_current
            },
            &[],
        );
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    let final_buffer = if config.steps % 2 == 0 {
        &simulation.current
    } else {
        &simulation.next
    };
    encoder.copy_buffer_to_buffer(final_buffer, 0, &simulation.readback, 0, buffer_size);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely())?;

    Ok(())
}

struct Config {
    width: u32,
    height: u32,
    steps: u32,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn Error>> {
        let mut args = env::args().skip(1);
        let width = parse_arg(args.next(), DEFAULT_WIDTH, "width")?;
        let height = parse_arg(args.next(), DEFAULT_HEIGHT, "height")?;
        let steps = parse_arg(args.next(), DEFAULT_STEPS, "steps")?;

        if args.next().is_some() {
            return Err("usage: cargo run -p compute -- [width] [height] [steps]".into());
        }

        if width == 0 || height == 0 {
            return Err("width and height must be greater than zero".into());
        }

        Ok(Self {
            width,
            height,
            steps,
        })
    }

    fn cell_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    fn buffer_size(&self) -> Result<u64, Box<dyn Error>> {
        let cells = self.cell_count();
        if cells > u64::from(u32::MAX) {
            return Err("grid is too large for this shader's u32 indexing".into());
        }

        Ok(cells * size_of::<f32>() as u64)
    }
}

fn parse_arg(value: Option<String>, default: u32, name: &str) -> Result<u32, Box<dyn Error>> {
    match value {
        Some(value) => value
            .parse()
            .map_err(|error| format!("invalid {name} `{value}`: {error}").into()),
        None => Ok(default),
    }
}

async fn create_initial_grid_buffer(
    device: &wgpu::Device,
    config: &Config,
    buffer_size: u64,
) -> Result<wgpu::Buffer, Box<dyn Error>> {
    let buffer = create_buffer_checked(
        device,
        wgpu::BufferDescriptor {
            label: Some("Current Grid"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        },
    )
    .await?;

    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        let hot_radius = config.width.min(config.height) / 20;
        let center_x = config.width / 2;
        let center_y = config.height / 2;

        for y in 0..config.height {
            for x in 0..config.width {
                let dx = x.abs_diff(center_x);
                let dy = y.abs_diff(center_y);
                let value = if dx * dx + dy * dy <= hot_radius * hot_radius {
                    100.0_f32
                } else {
                    0.0_f32
                };
                let offset = ((u64::from(y) * u64::from(config.width) + u64::from(x)) * 4) as usize;
                mapped
                    .slice(offset..offset + 4)
                    .copy_from_slice(&value.to_ne_bytes());
            }
        }
    }

    buffer.unmap();
    Ok(buffer)
}

async fn create_params_buffer(
    device: &wgpu::Device,
    config: &Config,
) -> Result<wgpu::Buffer, Box<dyn Error>> {
    let buffer = create_buffer_checked(
        device,
        wgpu::BufferDescriptor {
            label: Some("Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        },
    )
    .await?;

    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped
            .slice(0..4)
            .copy_from_slice(&config.width.to_ne_bytes());
        mapped
            .slice(4..8)
            .copy_from_slice(&config.height.to_ne_bytes());
        mapped.slice(8..12).copy_from_slice(&ALPHA.to_ne_bytes());
        mapped.slice(12..16).copy_from_slice(&0_u32.to_ne_bytes());
    }

    buffer.unmap();
    Ok(buffer)
}

async fn create_buffer_checked(
    device: &wgpu::Device,
    descriptor: wgpu::BufferDescriptor<'_>,
) -> Result<wgpu::Buffer, Box<dyn Error>> {
    let label = descriptor.label.unwrap_or("unnamed buffer").to_owned();
    let size = descriptor.size;
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = device.create_buffer(&descriptor);

    if let Some(error) = scope.pop().await {
        return Err(format!(
            "failed to allocate {label} ({:.1} MiB): {error}",
            size as f64 / 1024.0 / 1024.0
        )
        .into());
    }

    Ok(buffer)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Heat Diffusion Bind Group"),
        layout,
        entries: &[
            // 意味もなく入れ替えているのは、それでも動くことを確かめるため。
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: destination.as_entire_binding(),
            },
        ],
    })
}

struct Summary {
    min: f32,
    max: f32,
    center: f32,
    quarter: f32,
    corner: f32,
}

fn summarize_readback(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    config: &Config,
    buffer_size: u64,
) -> Result<Summary, Box<dyn Error>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("receiver should still exist");
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let mapped = slice.get_mapped_range();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for value in mapped.chunks_exact(4).map(f32_from_bytes) {
        min = min.min(value);
        max = max.max(value);
    }

    let center = read_cell(&mapped, config, config.width / 2, config.height / 2);
    let quarter = read_cell(&mapped, config, config.width / 4, config.height / 4);
    let corner = read_cell(&mapped, config, 0, 0);
    drop(mapped);
    buffer.unmap();

    assert_eq!(
        buffer_size,
        u64::from(config.width) * u64::from(config.height) * 4
    );

    Ok(Summary {
        min,
        max,
        center,
        quarter,
        corner,
    })
}

fn read_cell(mapped: &[u8], config: &Config, x: u32, y: u32) -> f32 {
    let offset = ((u64::from(y) * u64::from(config.width) + u64::from(x)) * 4) as usize;
    f32_from_bytes(&mapped[offset..offset + 4])
}

fn f32_from_bytes(bytes: &[u8]) -> f32 {
    f32::from_ne_bytes(bytes.try_into().expect("f32 is four bytes"))
}
