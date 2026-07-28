mod backward_stage;
mod forward_stage;

pub use backward_stage::{BackwardBuffers, BackwardStage, BackwardStageConstants};
pub use forward_stage::{ForwardBuffers, ForwardStage, ForwardStageConstants};
use wgpu_compute_utils::{
    AnyError, GpuContext, initialized_storage_buffer, map_buffer, readback_buffer, storage_buffer,
    upload_storage_buffer,
};

const FLOAT_BYTES: u64 = size_of::<f32>() as u64;

#[derive(Clone, Copy)]
pub struct NeuralNetworkConfig {
    pub input_size: u32,
    pub hidden1_size: u32,
    pub hidden2_size: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
}

impl NeuralNetworkConfig {
    pub const fn batch_bytes(self) -> u64 {
        self.batch_size as u64 * FLOAT_BYTES
    }

    pub const fn input_bytes(self) -> u64 {
        self.batch_size as u64 * self.input_size as u64 * FLOAT_BYTES
    }

    pub const fn hidden1_bytes(self) -> u64 {
        self.batch_size as u64 * self.hidden1_size as u64 * FLOAT_BYTES
    }

    pub const fn hidden2_bytes(self) -> u64 {
        self.batch_size as u64 * self.hidden2_size as u64 * FLOAT_BYTES
    }

    pub const fn weight1_len(self) -> usize {
        (self.hidden1_size * self.input_size + self.hidden1_size) as usize
    }

    pub const fn weight2_len(self) -> usize {
        (self.hidden2_size * self.hidden1_size + self.hidden2_size) as usize
    }

    pub const fn weight3_len(self) -> usize {
        (self.hidden2_size + 1) as usize
    }

    pub const fn weight1_bytes(self) -> u64 {
        self.weight1_len() as u64 * FLOAT_BYTES
    }

    pub const fn weight2_bytes(self) -> u64 {
        self.weight2_len() as u64 * FLOAT_BYTES
    }

    pub const fn weight3_bytes(self) -> u64 {
        self.weight3_len() as u64 * FLOAT_BYTES
    }
}

pub struct MiniBatch<'a> {
    pub inputs: &'a [f32],
    pub targets: &'a [f32],
    pub batch_size: u32,
}

pub struct ModelWeights {
    pub weights1: Vec<f32>,
    pub weights2: Vec<f32>,
    pub weights3: Vec<f32>,
}

struct MlpBuffers {
    inputs: wgpu::Buffer,
    targets: wgpu::Buffer,
    hidden1: wgpu::Buffer,
    hidden2: wgpu::Buffer,
    predicts: wgpu::Buffer,
    weights1: wgpu::Buffer,
    weights2: wgpu::Buffer,
    weights3: wgpu::Buffer,
    hidden1_delta: wgpu::Buffer,
    hidden2_delta: wgpu::Buffer,
    loss: wgpu::Buffer,
    loss_readback: wgpu::Buffer,
}

pub struct MlpRegressor {
    config: NeuralNetworkConfig,
    buffers: MlpBuffers,
    forward: ForwardStage,
    backward: BackwardStage,
}

impl MlpRegressor {
    pub fn new(device: &wgpu::Device, config: NeuralNetworkConfig, seed: u32) -> Self {
        let weights = initial_weights(config, seed);
        Self::from_weights(device, config, weights).expect("initial weights should match config")
    }

    pub fn from_weights(
        device: &wgpu::Device,
        config: NeuralNetworkConfig,
        weights: ModelWeights,
    ) -> Result<Self, AnyError> {
        validate_weights(config, &weights)?;
        let buffers = MlpBuffers {
            inputs: upload_storage_buffer(device, "MLP Inputs", config.input_bytes()),
            targets: upload_storage_buffer(device, "MLP Targets", config.batch_bytes()),
            hidden1: storage_buffer(
                device,
                "MLP Hidden 1 Activations",
                config.hidden1_bytes(),
                false,
            ),
            hidden2: storage_buffer(
                device,
                "MLP Hidden 2 Activations",
                config.hidden2_bytes(),
                false,
            ),
            predicts: storage_buffer(device, "MLP Predictions", config.batch_bytes(), true),
            weights1: initialized_storage_buffer(device, "MLP Layer 1 Weights", &weights.weights1),
            weights2: initialized_storage_buffer(device, "MLP Layer 2 Weights", &weights.weights2),
            weights3: initialized_storage_buffer(device, "MLP Layer 3 Weights", &weights.weights3),
            hidden1_delta: storage_buffer(
                device,
                "MLP Hidden 1 Delta",
                config.hidden1_bytes(),
                false,
            ),
            hidden2_delta: storage_buffer(
                device,
                "MLP Hidden 2 Delta",
                config.hidden2_bytes(),
                false,
            ),
            loss: storage_buffer(device, "MLP Loss", size_of::<f32>() as u64, true),
            loss_readback: readback_buffer(device, "MLP Loss Readback", size_of::<f32>() as u64),
        };
        let forward = ForwardStage::new(
            device,
            config,
            ForwardBuffers {
                images: &buffers.inputs,
                weights1: &buffers.weights1,
                hidden1: &buffers.hidden1,
                weights2: &buffers.weights2,
                hidden2: &buffers.hidden2,
                weights3: &buffers.weights3,
                predicts: &buffers.predicts,
            },
        );
        let backward = BackwardStage::new(
            device,
            config,
            BackwardBuffers {
                images: &buffers.inputs,
                expects: &buffers.targets,
                hidden1: &buffers.hidden1,
                hidden2: &buffers.hidden2,
                predicts: &buffers.predicts,
                weights1: &buffers.weights1,
                weights2: &buffers.weights2,
                weights3: &buffers.weights3,
                hidden1_delta: &buffers.hidden1_delta,
                hidden2_delta: &buffers.hidden2_delta,
                loss: &buffers.loss,
            },
        );
        Ok(Self {
            config,
            buffers,
            forward,
            backward,
        })
    }

    pub fn train_minibatch(
        &self,
        gpu: &GpuContext,
        batch: MiniBatch<'_>,
        compute_loss: bool,
    ) -> Result<Option<f32>, AnyError> {
        self.validate_batch(&batch)?;
        gpu.queue
            .write_buffer(&self.buffers.inputs, 0, bytemuck::cast_slice(batch.inputs));
        gpu.queue.write_buffer(
            &self.buffers.targets,
            0,
            bytemuck::cast_slice(batch.targets),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MLP Training Minibatch Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MLP Training Minibatch Pass"),
                timestamp_writes: None,
            });
            self.forward.encode(&mut pass);
            self.backward.encode(&mut pass, compute_loss);
        }
        if compute_loss {
            encoder.copy_buffer_to_buffer(
                &self.buffers.loss,
                0,
                &self.buffers.loss_readback,
                0,
                size_of::<f32>() as u64,
            );
        }
        gpu.queue.submit([encoder.finish()]);

        if compute_loss {
            let bytes = map_buffer(&gpu.device, &self.buffers.loss_readback)?;
            let loss = f32::from_le_bytes(bytes[..size_of::<f32>()].try_into()?);
            self.buffers.loss_readback.unmap();
            Ok(Some(loss))
        } else {
            Ok(None)
        }
    }

    pub fn predict_minibatch(
        &self,
        gpu: &GpuContext,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<Vec<f32>, AnyError> {
        self.validate_inputs(inputs, batch_size)?;
        gpu.queue
            .write_buffer(&self.buffers.inputs, 0, bytemuck::cast_slice(inputs));

        let readback = readback_buffer(
            &gpu.device,
            "MLP Prediction Readback",
            self.config.batch_bytes(),
        );
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MLP Prediction Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MLP Prediction Pass"),
                timestamp_writes: None,
            });
            self.forward.encode(&mut pass);
        }
        encoder.copy_buffer_to_buffer(
            &self.buffers.predicts,
            0,
            &readback,
            0,
            self.config.batch_bytes(),
        );
        gpu.queue.submit([encoder.finish()]);

        let bytes = map_buffer(&gpu.device, &readback)?;
        let predictions = f32_vec_from_bytes(&bytes)?;
        readback.unmap();
        Ok(predictions)
    }

    pub fn read_weights(&self, gpu: &GpuContext) -> Result<ModelWeights, AnyError> {
        Ok(ModelWeights {
            weights1: read_f32_buffer(gpu, &self.buffers.weights1, self.config.weight1_bytes())?,
            weights2: read_f32_buffer(gpu, &self.buffers.weights2, self.config.weight2_bytes())?,
            weights3: read_f32_buffer(gpu, &self.buffers.weights3, self.config.weight3_bytes())?,
        })
    }

    fn validate_batch(&self, batch: &MiniBatch<'_>) -> Result<(), AnyError> {
        validate_batch_shape(self.config, batch)
    }

    fn validate_inputs(&self, inputs: &[f32], batch_size: u32) -> Result<(), AnyError> {
        validate_input_shape(self.config, inputs, batch_size)
    }
}

fn validate_batch_shape(
    config: NeuralNetworkConfig,
    batch: &MiniBatch<'_>,
) -> Result<(), AnyError> {
    validate_input_shape(config, batch.inputs, batch.batch_size)?;
    if batch.targets.len() != batch.batch_size as usize {
        return Err(format!(
            "target length is {}, expected {}",
            batch.targets.len(),
            batch.batch_size
        )
        .into());
    }
    Ok(())
}

fn validate_input_shape(
    config: NeuralNetworkConfig,
    inputs: &[f32],
    batch_size: u32,
) -> Result<(), AnyError> {
    if batch_size != config.batch_size {
        return Err(format!("batch size is {batch_size}, expected {}", config.batch_size).into());
    }
    let expected = config.input_size as usize * batch_size as usize;
    if inputs.len() != expected {
        return Err(format!("input length is {}, expected {expected}", inputs.len()).into());
    }
    Ok(())
}

fn initial_weights(config: NeuralNetworkConfig, seed: u32) -> ModelWeights {
    let input = config.input_size as usize;
    let hidden1 = config.hidden1_size as usize;
    let hidden2 = config.hidden2_size as usize;
    let mut seed = seed;
    let layer1_limit = (6.0 / (input + hidden1) as f32).sqrt();
    let layer2_limit = (6.0 / (hidden1 + hidden2) as f32).sqrt();
    let layer3_limit = (6.0 / (hidden2 + 1) as f32).sqrt();

    let mut weights1 = vec![0.0; input * hidden1 + hidden1];
    for value in &mut weights1[..input * hidden1] {
        *value = random_range(&mut seed, -layer1_limit, layer1_limit);
    }

    let mut weights2 = vec![0.0; hidden1 * hidden2 + hidden2];
    for value in &mut weights2[..hidden1 * hidden2] {
        *value = random_range(&mut seed, -layer2_limit, layer2_limit);
    }

    let mut weights3 = vec![0.0; hidden2 + 1];
    for value in &mut weights3[..hidden2] {
        *value = random_range(&mut seed, -layer3_limit, layer3_limit);
    }

    ModelWeights {
        weights1,
        weights2,
        weights3,
    }
}

fn validate_weights(config: NeuralNetworkConfig, weights: &ModelWeights) -> Result<(), AnyError> {
    if weights.weights1.len() != config.weight1_len() {
        return Err(format!(
            "weights1 length is {}, expected {}",
            weights.weights1.len(),
            config.weight1_len()
        )
        .into());
    }
    if weights.weights2.len() != config.weight2_len() {
        return Err(format!(
            "weights2 length is {}, expected {}",
            weights.weights2.len(),
            config.weight2_len()
        )
        .into());
    }
    if weights.weights3.len() != config.weight3_len() {
        return Err(format!(
            "weights3 length is {}, expected {}",
            weights.weights3.len(),
            config.weight3_len()
        )
        .into());
    }
    Ok(())
}

fn random_range(seed: &mut u32, minimum: f32, maximum: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*seed >> 8) as f32 * (1.0 / 16_777_216.0);
    minimum + (maximum - minimum) * unit
}

fn read_f32_buffer(
    gpu: &GpuContext,
    source: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<f32>, AnyError> {
    let readback = readback_buffer(&gpu.device, "MLP Weight Readback", size);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MLP Weight Readback Encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    gpu.queue.submit([encoder.finish()]);
    let bytes = map_buffer(&gpu.device, &readback)?;
    let values = f32_vec_from_bytes(&bytes)?;
    readback.unmap();
    Ok(values)
}

fn f32_vec_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, AnyError> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| Ok(f32::from_le_bytes(chunk.try_into()?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: NeuralNetworkConfig = NeuralNetworkConfig {
        input_size: 4,
        hidden1_size: 3,
        hidden2_size: 2,
        batch_size: 2,
        learning_rate: 0.01,
    };

    #[test]
    fn config_reports_weight_lengths() {
        assert_eq!(CONFIG.weight1_len(), 15);
        assert_eq!(CONFIG.weight2_len(), 8);
        assert_eq!(CONFIG.weight3_len(), 3);
    }

    #[test]
    fn accepts_matching_minibatch_shape() {
        let inputs = [0.0; 8];
        let targets = [0.0; 2];
        let batch = MiniBatch {
            inputs: &inputs,
            targets: &targets,
            batch_size: 2,
        };
        validate_batch_shape(CONFIG, &batch).unwrap();
    }

    #[test]
    fn rejects_mismatched_minibatch_shape() {
        let inputs = [0.0; 7];
        let targets = [0.0; 2];
        let batch = MiniBatch {
            inputs: &inputs,
            targets: &targets,
            batch_size: 2,
        };
        assert!(validate_batch_shape(CONFIG, &batch).is_err());
    }
}
