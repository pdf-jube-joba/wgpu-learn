use std::{error::Error, fs, path::Path};

use mlp_gpu::{MlpRegressor, ModelWeights, NeuralNetworkConfig};
use raytrace_gpu::{RaytraceConfig, RaytraceGenerator};
use wgpu_compute_utils::{AnyError, GpuContext};

const RAYTRACE_CONFIG: RaytraceConfig = RaytraceConfig {
    base_seed: 0x31d8_c4a7,
    pixel_len: 32,
    rays_per_pixel: 1,
    batch_size: 16,
};
const NN_CONFIG: NeuralNetworkConfig = NeuralNetworkConfig {
    input_size: (RAYTRACE_CONFIG.pixel_len * RAYTRACE_CONFIG.pixel_len),
    hidden1_size: 2048,
    hidden2_size: 1024,
    batch_size: RAYTRACE_CONFIG.batch_size,
    learning_rate: 0.001,
};
const WEIGHTS1_FILE: &str = "weights1.bin";
const WEIGHTS2_FILE: &str = "weights2.bin";
const WEIGHTS3_FILE: &str = "weights3.bin";

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        NN_CONFIG.input_size as u64,
        RAYTRACE_CONFIG.image_value_count()
    );
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let weights = ModelWeights {
        weights1: read_weights(&model_dir.join(WEIGHTS1_FILE), NN_CONFIG.weight1_len())?,
        weights2: read_weights(&model_dir.join(WEIGHTS2_FILE), NN_CONFIG.weight2_len())?,
        weights3: read_weights(&model_dir.join(WEIGHTS3_FILE), NN_CONFIG.weight3_len())?,
    };

    let gpu = GpuContext::create("NN Prediction Device").await?;
    let raytrace = RaytraceGenerator::new(&gpu.device, RAYTRACE_CONFIG);
    let model = MlpRegressor::from_weights(&gpu.device, NN_CONFIG, weights)?;
    let batch = raytrace.generate_batch(&gpu)?;
    let predictions = model.predict_minibatch(&gpu, &batch.inputs, batch.batch_size)?;
    print_predictions(&batch.expects, &predictions);
    Ok(())
}

fn read_weights(path: &Path, expected_len: usize) -> Result<Vec<f32>, AnyError> {
    let bytes = fs::read(path).map_err(|error| {
        format!(
            "failed to read {}: {error}; run the training binary first",
            path.display()
        )
    })?;
    let expected_size = expected_len * size_of::<f32>();
    if bytes.len() != expected_size {
        return Err(format!(
            "{} has {} bytes, expected {expected_size}; retrain with the current model config",
            path.display(),
            bytes.len()
        )
        .into());
    }
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| Ok(f32::from_le_bytes(chunk.try_into()?)))
        .collect()
}

fn print_predictions(expects: &[f32], predictions: &[f32]) {
    println!(" sample | expected | predicted | abs error");
    println!("--------+----------+-----------+----------");
    for (index, (&expected, &predicted)) in expects.iter().zip(predictions).enumerate() {
        println!(
            "{index:>7} | {expected:>8.4} | {predicted:>9.4} | {:>8.4}",
            (predicted - expected).abs()
        );
    }
}
