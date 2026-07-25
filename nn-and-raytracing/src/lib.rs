mod backward_stage;
mod forward_stage;
mod raytrace_stage;
pub mod utils;

pub use backward_stage::{BackwardBuffers, BackwardStage, BackwardStageConstants};
pub use forward_stage::{ForwardBuffers, ForwardStage, ForwardStageConstants};
pub use raytrace_stage::{RaytraceBuffers, RaytraceStage, RaytraceStageConstants};
pub use utils::*;

pub const SAMPLE_STRIDE: u64 = 3;
pub const FLOAT_BYTES: u64 = size_of::<f32>() as u64;

#[derive(Clone, Copy)]
pub struct ModelConfig {
    pub base_seed: u32,
    pub pixel_len: u32,
    pub rays_per_pixel: u32,
    pub hidden_size: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
}

impl ModelConfig {
    pub const fn input_count(self) -> u64 {
        self.pixel_len as u64 * self.pixel_len as u64
    }

    pub const fn sample_bytes(self) -> u64 {
        self.batch_size as u64 * SAMPLE_STRIDE * FLOAT_BYTES
    }

    pub const fn batch_bytes(self) -> u64 {
        self.batch_size as u64 * FLOAT_BYTES
    }

    pub const fn image_bytes(self) -> u64 {
        self.batch_size as u64 * self.input_count() * FLOAT_BYTES
    }

    pub const fn hidden_bytes(self) -> u64 {
        self.batch_size as u64 * self.hidden_size as u64 * FLOAT_BYTES
    }

    pub const fn weight1_bytes(self) -> u64 {
        (self.hidden_size as u64 * self.input_count() + self.hidden_size as u64) * FLOAT_BYTES
    }

    pub const fn weight2_bytes(self) -> u64 {
        (self.hidden_size as u64 + 1) * FLOAT_BYTES
    }

    pub const fn parameter_count(self) -> u32 {
        self.hidden_size * self.pixel_len * self.pixel_len + self.hidden_size + self.hidden_size + 1
    }
}
