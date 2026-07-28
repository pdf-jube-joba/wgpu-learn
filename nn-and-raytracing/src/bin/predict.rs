use std::{error::Error, fs, path::Path};

use nn_and_raytracing::{
    AnyError, ForwardBuffers, ForwardStage, GpuContext, ModelConfig, RaytraceBuffers,
    RaytraceStage, initialized_storage_bytes, map_buffer, readback_buffer, storage_buffer,
};

const CONFIG: ModelConfig = ModelConfig {
    base_seed: 0x31d8_c4a7,
    pixel_len: 32,
    rays_per_pixel: 1,
    hidden_size: 2048,
    batch_size: 16,
    learning_rate: 0.001,
};
const WEIGHTS1_FILE: &str = "weights1.bin";
const WEIGHTS2_FILE: &str = "weights2.bin";

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let weights1 = read_weights(&model_dir.join(WEIGHTS1_FILE), CONFIG.weight1_bytes())?;
    let weights2 = read_weights(&model_dir.join(WEIGHTS2_FILE), CONFIG.weight2_bytes())?;

    let gpu = GpuContext::create("NN Prediction Device").await?;
    let samples = storage_buffer(
        &gpu.device,
        "Prediction Samples",
        CONFIG.sample_bytes(),
        false,
    );
    let expects = storage_buffer(
        &gpu.device,
        "Prediction Expected Distances",
        CONFIG.batch_bytes(),
        true,
    );
    let images = storage_buffer(
        &gpu.device,
        "Prediction Images",
        CONFIG.image_bytes(),
        false,
    );
    let hidden = storage_buffer(
        &gpu.device,
        "Prediction Hidden Activations",
        CONFIG.hidden_bytes(),
        false,
    );
    let predicts = storage_buffer(
        &gpu.device,
        "Prediction Results",
        CONFIG.batch_bytes(),
        true,
    );
    let weights1 = initialized_storage_bytes(&gpu.device, "Prediction Layer 1 Weights", &weights1);
    let weights2 = initialized_storage_bytes(&gpu.device, "Prediction Layer 2 Weights", &weights2);

    let raytrace = RaytraceStage::new(
        &gpu.device,
        CONFIG,
        RaytraceBuffers {
            samples: &samples,
            expects: &expects,
            images: &images,
        },
        None,
    );
    let forward = ForwardStage::new(
        &gpu.device,
        CONFIG,
        ForwardBuffers {
            images: &images,
            weights1: &weights1,
            hidden: &hidden,
            weights2: &weights2,
            predicts: &predicts,
        },
    );

    let readback = readback_buffer(&gpu.device, "Prediction Readback", CONFIG.batch_bytes() * 2);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Prediction Encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prediction Pass"),
            timestamp_writes: None,
        });
        raytrace.encode(&mut pass);
        forward.encode(&mut pass);
    }
    encoder.copy_buffer_to_buffer(&expects, 0, &readback, 0, CONFIG.batch_bytes());
    encoder.copy_buffer_to_buffer(
        &predicts,
        0,
        &readback,
        CONFIG.batch_bytes(),
        CONFIG.batch_bytes(),
    );
    gpu.queue.submit([encoder.finish()]);

    let bytes = map_buffer(&gpu.device, &readback)?;
    print_predictions(&bytes)?;
    readback.unmap();
    Ok(())
}

fn read_weights(path: &Path, expected_size: u64) -> Result<Vec<u8>, AnyError> {
    let bytes = fs::read(path).map_err(|error| {
        format!(
            "failed to read {}: {error}; run the training binary first",
            path.display()
        )
    })?;
    if bytes.len() as u64 != expected_size {
        return Err(format!(
            "{} has {} bytes, expected {expected_size}; retrain with the current model config",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

fn print_predictions(bytes: &[u8]) -> Result<(), AnyError> {
    let split = CONFIG.batch_bytes() as usize;
    let expects = bytes[..split].chunks_exact(size_of::<f32>());
    let predicts = bytes[split..].chunks_exact(size_of::<f32>());
    println!(" sample | expected | predicted | abs error");
    println!("--------+----------+-----------+----------");
    for (index, (expected, predicted)) in expects.zip(predicts).enumerate() {
        let expected = f32::from_le_bytes(expected.try_into()?);
        let predicted = f32::from_le_bytes(predicted.try_into()?);
        println!(
            "{index:>7} | {expected:>8.4} | {predicted:>9.4} | {:>8.4}",
            (predicted - expected).abs()
        );
    }
    Ok(())
}
