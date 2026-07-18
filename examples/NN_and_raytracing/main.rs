use std::error::Error;

use wgpu::wgc::device;

fn main() {
    pollster::block_on(run());
}

async fn run() {}

const GLOBAL_SEED: usize = 28_669_662_883;
const PIXEL_LEN: usize = 64;
const MIDDLE_NUM: usize = 1024;
const BATCH_SIZE: usize = 32;

static RATE: f32 = 0.001;

struct GPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GPU {
    async fn create_gpu() -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("failed");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        Ok(GPU { device, queue })
    }
}

struct Buffers {
    // generated iamges
    // f32 * PIXEL * PIXEL * BATCH
    images: wgpu::Buffer,
    // true values of distance
    // f32 * BATCH
    expects: wgpu::Buffer,
    // for store middle layer outputs
    // f32 * MID
    middle: wgpu::Buffer,
    // result of computation
    // f32 * 1
    predicts: wgpu::Buffer,
    // weights of inputs -> middle
    // f32 * (PIXEL * PIXEL) * MID
    weights1: wgpu::Buffer,
    // weights of middle -> outputs
    // f32 * MID * 1
    weights2: wgpu::Buffer,
    // constant for rate mutiplier of gradient
    rate: wgpu::Buffer,
    // seed for rand
    seed: wgpu::Buffer,
}

struct Pipelines {
    // 各 1 invocation ごと
    // seed をもとに id を使って raytracing 画像を1枚生成して
    // images の対応する部分と expect に書き込む
    generate_images: wgpu::ComputePipeline,
    //
}

async fn generate_images_pipeline() -> wgpu::ComputePipeline {
    todo!()
}
