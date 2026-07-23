use std::{error::Error, fs, path::Path, sync::mpsc};

use wgpu::util::DeviceExt;

const BASE_SEED: u32 = 0x6ac6_8e9b;
const PIXEL_LEN: u32 = 32;
const RAYS_PER_PIXEL: u32 = 1;
const HIDDEN_SIZE: u32 = 64;
const BATCH_SIZE: u32 = 32;
const LEARNING_RATE: f32 = 0.001;
const TRAIN_STEPS: u32 = 2_000;
const REPORT_INTERVAL: u32 = 200;

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

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct Buffers {
    constants: wgpu::Buffer,
    samples: wgpu::Buffer,
    images: wgpu::Buffer,
    expects: wgpu::Buffer,
    middle: wgpu::Buffer,
    predicts: wgpu::Buffer,
    weights1: wgpu::Buffer,
    weights2: wgpu::Buffer,
    hidden_delta: wgpu::Buffer,
    loss: wgpu::Buffer,
    loss_readback: wgpu::Buffer,
}

struct Pipelines {
    generate_samples: wgpu::ComputePipeline,
    raytrace: wgpu::ComputePipeline,
    forward_middle: wgpu::ComputePipeline,
    forward_output: wgpu::ComputePipeline,
    compute_loss: wgpu::ComputePipeline,
    backward_hidden: wgpu::ComputePipeline,
    update_weights: wgpu::ComputePipeline,
}

struct BindGroups {
    constants: wgpu::BindGroup,
    generate_samples: wgpu::BindGroup,
    raytrace: wgpu::BindGroup,
    forward_middle: wgpu::BindGroup,
    forward_output: wgpu::BindGroup,
    compute_loss: wgpu::BindGroup,
    backward_hidden: wgpu::BindGroup,
    update_weights: wgpu::BindGroup,
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let gpu = GpuContext::create().await?;
    let initial_constants = constants_for_step(0);
    let buffers = create_buffers(&gpu.device, &initial_constants);
    let constants_layout = create_constants_layout(&gpu.device);

    let generate_shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Generate Image Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("generate_image.wgsl").into()),
        });
    let forward_shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("forward.wgsl").into()),
        });
    let back_shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Back Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("back.wgsl").into()),
        });

    let (pipelines, bind_groups) = create_pipelines_and_bind_groups(
        &gpu.device,
        &constants_layout,
        &buffers,
        &generate_shader,
        &forward_shader,
        &back_shader,
    );

    println!(
        "training: image={}x{}, hidden={}, batch={}, steps={}, rate={}",
        PIXEL_LEN, PIXEL_LEN, HIDDEN_SIZE, BATCH_SIZE, TRAIN_STEPS, LEARNING_RATE
    );

    for step in 0..TRAIN_STEPS {
        let constants = constants_for_step(step);
        gpu.queue
            .write_buffer(&buffers.constants, 0, bytemuck::bytes_of(&constants));

        let report = step == 0 || (step + 1) % REPORT_INTERVAL == 0;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NN Training Step Encoder"),
            });
        encode_training_step(&mut encoder, &pipelines, &bind_groups, report);
        if report {
            encoder.copy_buffer_to_buffer(&buffers.loss, 0, &buffers.loss_readback, 0, 4);
        }
        gpu.queue.submit([encoder.finish()]);

        if report {
            let loss = read_single_f32(&gpu.device, &buffers.loss_readback)?;
            if !loss.is_finite() {
                return Err(format!("loss became non-finite at step {}", step + 1).into());
            }
            println!("step {:>5}: mse = {:.6}", step + 1, loss);
        }
    }

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    save_buffer(
        &gpu,
        &buffers.weights1,
        weight1_bytes(),
        &output_dir.join("weights1.bin"),
    )?;
    save_buffer(
        &gpu,
        &buffers.weights2,
        weight2_bytes(),
        &output_dir.join("weights2.bin"),
    )?;
    println!("saved weights to {}", output_dir.display());
    Ok(())
}

impl GpuContext {
    async fn create() -> Result<Self, Box<dyn Error>> {
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
                label: Some("NN Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        Ok(Self { device, queue })
    }
}

fn constants_for_step(step: u32) -> Constants {
    Constants {
        seed: BASE_SEED.wrapping_add(step.wrapping_mul(0x9e37_79b9)),
        pixel_len: PIXEL_LEN,
        sample_ray: RAYS_PER_PIXEL,
        middle_num: HIDDEN_SIZE,
        batch_size: BATCH_SIZE,
        rate: LEARNING_RATE,
    }
}

fn create_buffers(device: &wgpu::Device, constants: &Constants) -> Buffers {
    let constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NN Constants"),
        contents: bytemuck::bytes_of(constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let (weights1, weights2) = initial_weights();

    Buffers {
        constants: constants_buffer,
        samples: storage_buffer(device, "Samples", u64::from(BATCH_SIZE) * 3 * 4, false),
        images: storage_buffer(device, "Images", image_bytes(), false),
        expects: storage_buffer(device, "Expected Distances", batch_bytes(), false),
        middle: storage_buffer(device, "Middle Activations", middle_bytes(), false),
        predicts: storage_buffer(device, "Predictions", batch_bytes(), false),
        weights1: initialized_storage_buffer(device, "Layer 1 Weights", &weights1),
        weights2: initialized_storage_buffer(device, "Layer 2 Weights", &weights2),
        hidden_delta: storage_buffer(device, "Hidden Delta", middle_bytes(), false),
        loss: storage_buffer(device, "Loss", 4, true),
        loss_readback: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loss Readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }),
    }
}

fn initial_weights() -> (Vec<f32>, Vec<f32>) {
    let inputs = (PIXEL_LEN * PIXEL_LEN) as usize;
    let hidden = HIDDEN_SIZE as usize;
    let mut seed = BASE_SEED ^ 0xa5a5_5a5a;
    let layer1_limit = (6.0 / (inputs + hidden) as f32).sqrt();
    let layer2_limit = (6.0 / (hidden + 1) as f32).sqrt();

    let mut weights1 = vec![0.0; inputs * hidden + hidden];
    for value in &mut weights1[..inputs * hidden] {
        *value = random_range(&mut seed, -layer1_limit, layer1_limit);
    }
    let mut weights2 = vec![0.0; hidden + 1];
    for value in &mut weights2[..hidden] {
        *value = random_range(&mut seed, -layer2_limit, layer2_limit);
    }
    (weights1, weights2)
}

fn random_range(seed: &mut u32, minimum: f32, maximum: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*seed >> 8) as f32 * (1.0 / 16_777_216.0);
    minimum + (maximum - minimum) * unit
}

fn create_constants_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NN Constants Layout"),
        entries: &[layout_buffer_entry(0, wgpu::BufferBindingType::Uniform)],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_pipelines_and_bind_groups(
    device: &wgpu::Device,
    constants_layout: &wgpu::BindGroupLayout,
    buffers: &Buffers,
    generate_shader: &wgpu::ShaderModule,
    forward_shader: &wgpu::ShaderModule,
    back_shader: &wgpu::ShaderModule,
) -> (Pipelines, BindGroups) {
    let generate_layout = data_layout(
        device,
        "Generate Samples Data Layout",
        &[(0, false), (1, false)],
    );
    let raytrace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Raytrace Data Layout"),
        entries: &[
            layout_buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            layout_buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
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
    let forward_middle_layout = data_layout(
        device,
        "Forward Middle Data Layout",
        &[(0, true), (1, true), (2, false)],
    );
    let forward_output_layout = data_layout(
        device,
        "Forward Output Data Layout",
        &[(2, false), (3, true), (4, false)],
    );
    let compute_loss_layout = data_layout(
        device,
        "Compute Loss Data Layout",
        &[(1, true), (3, true), (7, false)],
    );
    let backward_hidden_layout = data_layout(
        device,
        "Backward Hidden Data Layout",
        &[(1, true), (2, true), (3, true), (5, false), (6, false)],
    );
    let update_weights_layout = data_layout(
        device,
        "Update Weights Data Layout",
        &[
            (0, true),
            (1, true),
            (2, true),
            (3, true),
            (4, false),
            (5, false),
            (6, false),
        ],
    );

    let debug_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Training Debug Texture"),
        size: wgpu::Extent3d {
            width: PIXEL_LEN,
            height: PIXEL_LEN,
            depth_or_array_layers: BATCH_SIZE,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let debug_texture_view = debug_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Training Debug Texture View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let pipelines = Pipelines {
        generate_samples: pipeline(
            device,
            "Generate Samples Pipeline",
            generate_shader,
            "generate_samples",
            constants_layout,
            &generate_layout,
        ),
        raytrace: pipeline(
            device,
            "Raytrace Pipeline",
            generate_shader,
            "raytrace",
            constants_layout,
            &raytrace_layout,
        ),
        forward_middle: pipeline(
            device,
            "Forward Middle Pipeline",
            forward_shader,
            "forward_middle",
            constants_layout,
            &forward_middle_layout,
        ),
        forward_output: pipeline(
            device,
            "Forward Output Pipeline",
            forward_shader,
            "forward_output",
            constants_layout,
            &forward_output_layout,
        ),
        compute_loss: pipeline(
            device,
            "Compute Loss Pipeline",
            back_shader,
            "compute_loss",
            constants_layout,
            &compute_loss_layout,
        ),
        backward_hidden: pipeline(
            device,
            "Backward Hidden Pipeline",
            back_shader,
            "backward_hidden",
            constants_layout,
            &backward_hidden_layout,
        ),
        update_weights: pipeline(
            device,
            "Update Weights Pipeline",
            back_shader,
            "update_weights",
            constants_layout,
            &update_weights_layout,
        ),
    };

    let constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NN Constants Bind Group"),
        layout: constants_layout,
        entries: &[buffer_entry(0, &buffers.constants)],
    });
    let bind_groups = BindGroups {
        constants,
        generate_samples: bind_group(
            device,
            "Generate Samples Bind Group",
            &generate_layout,
            &[
                buffer_entry(0, &buffers.samples),
                buffer_entry(1, &buffers.expects),
            ],
        ),
        raytrace: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raytrace Bind Group"),
            layout: &raytrace_layout,
            entries: &[
                buffer_entry(0, &buffers.samples),
                buffer_entry(2, &buffers.images),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&debug_texture_view),
                },
            ],
        }),
        forward_middle: bind_group(
            device,
            "Forward Middle Bind Group",
            &forward_middle_layout,
            &[
                buffer_entry(0, &buffers.images),
                buffer_entry(1, &buffers.weights1),
                buffer_entry(2, &buffers.middle),
            ],
        ),
        forward_output: bind_group(
            device,
            "Forward Output Bind Group",
            &forward_output_layout,
            &[
                buffer_entry(2, &buffers.middle),
                buffer_entry(3, &buffers.weights2),
                buffer_entry(4, &buffers.predicts),
            ],
        ),
        compute_loss: bind_group(
            device,
            "Compute Loss Bind Group",
            &compute_loss_layout,
            &[
                buffer_entry(1, &buffers.expects),
                buffer_entry(3, &buffers.predicts),
                buffer_entry(7, &buffers.loss),
            ],
        ),
        backward_hidden: bind_group(
            device,
            "Backward Hidden Bind Group",
            &backward_hidden_layout,
            &[
                buffer_entry(1, &buffers.expects),
                buffer_entry(2, &buffers.middle),
                buffer_entry(3, &buffers.predicts),
                buffer_entry(5, &buffers.weights2),
                buffer_entry(6, &buffers.hidden_delta),
            ],
        ),
        update_weights: bind_group(
            device,
            "Update Weights Bind Group",
            &update_weights_layout,
            &[
                buffer_entry(0, &buffers.images),
                buffer_entry(1, &buffers.expects),
                buffer_entry(2, &buffers.middle),
                buffer_entry(3, &buffers.predicts),
                buffer_entry(4, &buffers.weights1),
                buffer_entry(5, &buffers.weights2),
                buffer_entry(6, &buffers.hidden_delta),
            ],
        ),
    };

    (pipelines, bind_groups)
}

fn encode_training_step(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &Pipelines,
    bind_groups: &BindGroups,
    compute_loss: bool,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("NN Training Step"),
        timestamp_writes: None,
    });

    dispatch(
        &mut pass,
        &pipelines.generate_samples,
        &bind_groups.constants,
        &bind_groups.generate_samples,
        (BATCH_SIZE.div_ceil(64), 1, 1),
    );
    dispatch(
        &mut pass,
        &pipelines.raytrace,
        &bind_groups.constants,
        &bind_groups.raytrace,
        (PIXEL_LEN.div_ceil(8), PIXEL_LEN.div_ceil(8), BATCH_SIZE),
    );
    dispatch(
        &mut pass,
        &pipelines.forward_middle,
        &bind_groups.constants,
        &bind_groups.forward_middle,
        (HIDDEN_SIZE.div_ceil(8), BATCH_SIZE.div_ceil(8), 1),
    );
    dispatch(
        &mut pass,
        &pipelines.forward_output,
        &bind_groups.constants,
        &bind_groups.forward_output,
        (BATCH_SIZE.div_ceil(64), 1, 1),
    );
    if compute_loss {
        dispatch(
            &mut pass,
            &pipelines.compute_loss,
            &bind_groups.constants,
            &bind_groups.compute_loss,
            (1, 1, 1),
        );
    }
    dispatch(
        &mut pass,
        &pipelines.backward_hidden,
        &bind_groups.constants,
        &bind_groups.backward_hidden,
        (HIDDEN_SIZE.div_ceil(8), BATCH_SIZE.div_ceil(8), 1),
    );
    let parameter_count = HIDDEN_SIZE * PIXEL_LEN * PIXEL_LEN + HIDDEN_SIZE + HIDDEN_SIZE + 1;
    dispatch(
        &mut pass,
        &pipelines.update_weights,
        &bind_groups.constants,
        &bind_groups.update_weights,
        (parameter_count.div_ceil(64), 1, 1),
    );
}

fn dispatch<'a>(
    pass: &mut wgpu::ComputePass<'a>,
    pipeline: &'a wgpu::ComputePipeline,
    constants: &'a wgpu::BindGroup,
    data: &'a wgpu::BindGroup,
    workgroups: (u32, u32, u32),
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, constants, &[]);
    pass.set_bind_group(1, data, &[]);
    pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
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

fn data_layout(
    device: &wgpu::Device,
    label: &str,
    bindings: &[(u32, bool)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = bindings
        .iter()
        .map(|&(binding, read_only)| {
            layout_buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only })
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn layout_buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    copy_source: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_source {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn initialized_storage_buffer(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read_single_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<f32, Box<dyn Error>> {
    let bytes = map_buffer(device, buffer)?;
    let value = f32::from_le_bytes(bytes[..4].try_into()?);
    buffer.unmap();
    Ok(value)
}

fn save_buffer(
    gpu: &GpuContext,
    source: &wgpu::Buffer,
    size: u64,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Weight Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Weight Readback Encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    gpu.queue.submit([encoder.finish()]);
    let bytes = map_buffer(&gpu.device, &readback)?;
    fs::write(path, bytes)?;
    readback.unmap();
    Ok(())
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

const fn image_bytes() -> u64 {
    BATCH_SIZE as u64 * PIXEL_LEN as u64 * PIXEL_LEN as u64 * 4
}

const fn batch_bytes() -> u64 {
    BATCH_SIZE as u64 * 4
}

const fn middle_bytes() -> u64 {
    BATCH_SIZE as u64 * HIDDEN_SIZE as u64 * 4
}

const fn weight1_bytes() -> u64 {
    (HIDDEN_SIZE as u64 * PIXEL_LEN as u64 * PIXEL_LEN as u64 + HIDDEN_SIZE as u64) * 4
}

const fn weight2_bytes() -> u64 {
    (HIDDEN_SIZE as u64 + 1) * 4
}
