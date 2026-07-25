use std::{error::Error, fs, path::Path};

use nn_and_raytracing::{
    AnyError, BackwardBuffers, BackwardStage, ForwardBuffers, ForwardStage, GpuContext,
    ModelConfig, RaytraceBuffers, RaytraceStage, initialized_storage_buffer, map_buffer,
    readback_buffer, storage_buffer,
};

const CONFIG: ModelConfig = ModelConfig {
    base_seed: 0x6ac6_8e9b,
    pixel_len: 32,
    rays_per_pixel: 1,
    hidden_size: 64,
    batch_size: 32,
    learning_rate: 0.001,
};
const TRAIN_STEPS: u32 = 2_000;
const REPORT_INTERVAL: u32 = 200;

struct TrainingBuffers {
    samples: wgpu::Buffer,
    images: wgpu::Buffer,
    expects: wgpu::Buffer,
    hidden: wgpu::Buffer,
    predicts: wgpu::Buffer,
    weights1: wgpu::Buffer,
    weights2: wgpu::Buffer,
    hidden_delta: wgpu::Buffer,
    loss: wgpu::Buffer,
    loss_readback: wgpu::Buffer,
}

impl TrainingBuffers {
    fn new(device: &wgpu::Device) -> Self {
        let (weights1, weights2) = initial_weights(CONFIG);
        Self {
            samples: storage_buffer(device, "Samples", CONFIG.sample_bytes(), false),
            images: storage_buffer(device, "Images", CONFIG.image_bytes(), false),
            expects: storage_buffer(device, "Expected Distances", CONFIG.batch_bytes(), false),
            hidden: storage_buffer(device, "Hidden Activations", CONFIG.hidden_bytes(), false),
            predicts: storage_buffer(device, "Predictions", CONFIG.batch_bytes(), false),
            weights1: initialized_storage_buffer(device, "Layer 1 Weights", &weights1),
            weights2: initialized_storage_buffer(device, "Layer 2 Weights", &weights2),
            hidden_delta: storage_buffer(device, "Hidden Delta", CONFIG.hidden_bytes(), false),
            loss: storage_buffer(device, "Loss", size_of::<f32>() as u64, true),
            loss_readback: readback_buffer(device, "Loss Readback", size_of::<f32>() as u64),
        }
    }
}

struct Trainer {
    buffers: TrainingBuffers,
    raytrace: RaytraceStage,
    forward: ForwardStage,
    backward: BackwardStage,
}

impl Trainer {
    fn new(device: &wgpu::Device) -> Self {
        let buffers = TrainingBuffers::new(device);
        let raytrace = RaytraceStage::new(
            device,
            CONFIG,
            RaytraceBuffers {
                samples: &buffers.samples,
                expects: &buffers.expects,
                images: &buffers.images,
            },
            None,
        );
        let forward = ForwardStage::new(
            device,
            CONFIG,
            ForwardBuffers {
                images: &buffers.images,
                weights1: &buffers.weights1,
                hidden: &buffers.hidden,
                weights2: &buffers.weights2,
                predicts: &buffers.predicts,
            },
        );
        let backward = BackwardStage::new(
            device,
            CONFIG,
            BackwardBuffers {
                images: &buffers.images,
                expects: &buffers.expects,
                hidden: &buffers.hidden,
                predicts: &buffers.predicts,
                weights1: &buffers.weights1,
                weights2: &buffers.weights2,
                hidden_delta: &buffers.hidden_delta,
                loss: &buffers.loss,
            },
        );
        Self {
            buffers,
            raytrace,
            forward,
            backward,
        }
    }

    fn encode_step(&self, encoder: &mut wgpu::CommandEncoder, compute_loss: bool) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("NN Training Step"),
            timestamp_writes: None,
        });
        self.raytrace.encode(&mut pass);
        self.forward.encode(&mut pass);
        self.backward.encode(&mut pass, compute_loss);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let gpu = GpuContext::create("NN Device").await?;
    let trainer = Trainer::new(&gpu.device);
    println!(
        "training: image={}x{}, hidden={}, batch={}, steps={}, rate={}",
        CONFIG.pixel_len,
        CONFIG.pixel_len,
        CONFIG.hidden_size,
        CONFIG.batch_size,
        TRAIN_STEPS,
        CONFIG.learning_rate
    );

    for step in 0..TRAIN_STEPS {
        trainer.raytrace.set_step(&gpu.queue, step);
        let report = step == 0 || (step + 1) % REPORT_INTERVAL == 0;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NN Training Step Encoder"),
            });
        trainer.encode_step(&mut encoder, report);
        if report {
            encoder.copy_buffer_to_buffer(
                &trainer.buffers.loss,
                0,
                &trainer.buffers.loss_readback,
                0,
                size_of::<f32>() as u64,
            );
        }
        gpu.queue.submit([encoder.finish()]);
        if report {
            let loss = read_single_f32(&gpu.device, &trainer.buffers.loss_readback)?;
            if !loss.is_finite() {
                return Err(format!("loss became non-finite at step {}", step + 1).into());
            }
            println!("step {:>5}: mse = {:.6}", step + 1, loss);
        }
    }

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    save_buffer(
        &gpu,
        &trainer.buffers.weights1,
        CONFIG.weight1_bytes(),
        &output_dir.join("weights1.bin"),
    )?;
    save_buffer(
        &gpu,
        &trainer.buffers.weights2,
        CONFIG.weight2_bytes(),
        &output_dir.join("weights2.bin"),
    )?;
    println!("saved weights to {}", output_dir.display());
    Ok(())
}

fn initial_weights(config: ModelConfig) -> (Vec<f32>, Vec<f32>) {
    let inputs = config.input_count() as usize;
    let hidden = config.hidden_size as usize;
    let mut seed = config.base_seed ^ 0xa5a5_5a5a;
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

fn read_single_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<f32, AnyError> {
    let bytes = map_buffer(device, buffer)?;
    let value = f32::from_le_bytes(bytes[..size_of::<f32>()].try_into()?);
    buffer.unmap();
    Ok(value)
}

fn save_buffer(
    gpu: &GpuContext,
    source: &wgpu::Buffer,
    size: u64,
    path: &Path,
) -> Result<(), AnyError> {
    let readback = readback_buffer(&gpu.device, "Weight Readback", size);
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
