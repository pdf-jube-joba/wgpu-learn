mod raytrace_stage;

pub use raytrace_stage::{RaytraceBuffers, RaytraceStage, RaytraceStageConstants};
use wgpu_compute_utils::{AnyError, GpuContext, map_buffer, readback_buffer, storage_buffer};

const SAMPLE_STRIDE: u64 = 3;
const FLOAT_BYTES: u64 = size_of::<f32>() as u64;

#[derive(Clone, Copy)]
pub struct RaytraceConfig {
    pub base_seed: u32,
    pub pixel_len: u32,
    pub rays_per_pixel: u32,
    pub batch_size: u32,
}

impl RaytraceConfig {
    pub const fn image_value_count(self) -> u64 {
        self.pixel_len as u64 * self.pixel_len as u64
    }

    pub const fn sample_bytes(self) -> u64 {
        self.batch_size as u64 * SAMPLE_STRIDE * FLOAT_BYTES
    }

    pub const fn batch_bytes(self) -> u64 {
        self.batch_size as u64 * FLOAT_BYTES
    }

    pub const fn image_bytes(self) -> u64 {
        self.batch_size as u64 * self.image_value_count() * FLOAT_BYTES
    }
}

pub struct RaytraceBatch {
    pub inputs: Vec<f32>,
    pub expects: Vec<f32>,
    pub input_size: u32,
    pub batch_size: u32,
}

struct RaytraceGeneratorBuffers {
    samples: wgpu::Buffer,
    expects: wgpu::Buffer,
    images: wgpu::Buffer,
    readback: wgpu::Buffer,
}

pub struct RaytraceGenerator {
    config: RaytraceConfig,
    buffers: RaytraceGeneratorBuffers,
    stage: RaytraceStage,
}

impl RaytraceGenerator {
    pub fn new(device: &wgpu::Device, config: RaytraceConfig) -> Self {
        let buffers = RaytraceGeneratorBuffers {
            samples: storage_buffer(device, "Raytrace Samples", config.sample_bytes(), false),
            expects: storage_buffer(
                device,
                "Raytrace Expected Distances",
                config.batch_bytes(),
                true,
            ),
            images: storage_buffer(device, "Raytrace Images", config.image_bytes(), true),
            readback: readback_buffer(
                device,
                "Raytrace Batch Readback",
                config.batch_bytes() + config.image_bytes(),
            ),
        };
        let stage = RaytraceStage::new(
            device,
            config,
            RaytraceBuffers {
                samples: &buffers.samples,
                expects: &buffers.expects,
                images: &buffers.images,
            },
            None,
        );
        Self {
            config,
            buffers,
            stage,
        }
    }

    pub fn set_seed(&self, queue: &wgpu::Queue, seed: u32) {
        self.stage.set_seed(queue, seed);
    }

    pub fn generate_batch(&self, gpu: &GpuContext) -> Result<RaytraceBatch, AnyError> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Raytrace Batch Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Raytrace Batch Pass"),
                timestamp_writes: None,
            });
            self.stage.encode(&mut pass);
        }
        encoder.copy_buffer_to_buffer(
            &self.buffers.expects,
            0,
            &self.buffers.readback,
            0,
            self.config.batch_bytes(),
        );
        encoder.copy_buffer_to_buffer(
            &self.buffers.images,
            0,
            &self.buffers.readback,
            self.config.batch_bytes(),
            self.config.image_bytes(),
        );
        gpu.queue.submit([encoder.finish()]);

        let bytes = map_buffer(&gpu.device, &self.buffers.readback)?;
        let split = self.config.batch_bytes() as usize;
        let expects = f32_vec_from_bytes(&bytes[..split])?;
        let inputs = f32_vec_from_bytes(&bytes[split..])?;
        self.buffers.readback.unmap();

        Ok(RaytraceBatch {
            inputs,
            expects,
            input_size: self.config.image_value_count() as u32,
            batch_size: self.config.batch_size,
        })
    }
}

fn f32_vec_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, AnyError> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| Ok(f32::from_le_bytes(chunk.try_into()?)))
        .collect()
}
