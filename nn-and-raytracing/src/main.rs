use std::{error::Error, fs, path::Path};

use mlp_gpu::{MiniBatch, MlpRegressor, ModelWeights, NeuralNetworkConfig};
use raytrace_gpu::{RaytraceConfig, RaytraceGenerator};
use wgpu_compute_utils::{AnyError, GpuContext};

const RAYTRACE_CONFIG: RaytraceConfig = RaytraceConfig {
    base_seed: 0x6ac6_8e9b,
    pixel_len: 32,
    rays_per_pixel: 1,
    batch_size: 32,
};
const NN_CONFIG: NeuralNetworkConfig = NeuralNetworkConfig {
    input_size: (RAYTRACE_CONFIG.pixel_len * RAYTRACE_CONFIG.pixel_len),
    hidden1_size: 2048,
    hidden2_size: 1024,
    batch_size: RAYTRACE_CONFIG.batch_size,
    learning_rate: 0.001,
};
const TRAIN_STEPS: u32 = 100_000;
const REPORT_INTERVAL: u32 = 500;

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        NN_CONFIG.input_size as u64,
        RAYTRACE_CONFIG.image_value_count()
    );
    let gpu = GpuContext::create("NN Device").await?;
    let raytrace = RaytraceGenerator::new(&gpu.device, RAYTRACE_CONFIG);
    let model = MlpRegressor::new(&gpu.device, NN_CONFIG, RAYTRACE_CONFIG.base_seed);
    println!(
        "training: image={}x{}, hidden1={}, hidden2={}, batch={}, steps={}, rate={}",
        RAYTRACE_CONFIG.pixel_len,
        RAYTRACE_CONFIG.pixel_len,
        NN_CONFIG.hidden1_size,
        NN_CONFIG.hidden2_size,
        NN_CONFIG.batch_size,
        TRAIN_STEPS,
        NN_CONFIG.learning_rate
    );

    for step in 0..TRAIN_STEPS {
        let seed = RAYTRACE_CONFIG
            .base_seed
            .wrapping_add(step.wrapping_mul(0x9e37_79b9));
        raytrace.set_seed(&gpu.queue, seed);
        let batch = raytrace.generate_batch(&gpu)?;
        let report = step == 0 || (step + 1) % REPORT_INTERVAL == 0;
        let loss = model.train_minibatch(
            &gpu,
            MiniBatch {
                inputs: &batch.inputs,
                targets: &batch.expects,
                batch_size: batch.batch_size,
            },
            report,
        )?;
        if let Some(loss) = loss {
            if !loss.is_finite() {
                return Err(format!("loss became non-finite at step {}", step + 1).into());
            }
            println!("step {:>5}: mse = {:.6}", step + 1, loss);
        }
    }

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    save_weights(&output_dir, model.read_weights(&gpu)?)?;
    println!("saved weights to {}", output_dir.display());
    Ok(())
}

fn save_weights(output_dir: &Path, weights: ModelWeights) -> Result<(), AnyError> {
    fs::write(
        output_dir.join("weights1.bin"),
        bytemuck::cast_slice(&weights.weights1),
    )?;
    fs::write(
        output_dir.join("weights2.bin"),
        bytemuck::cast_slice(&weights.weights2),
    )?;
    fs::write(
        output_dir.join("weights3.bin"),
        bytemuck::cast_slice(&weights.weights3),
    )?;
    Ok(())
}
