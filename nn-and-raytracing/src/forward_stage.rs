use crate::{
    ModelConfig, bind_group, buffer_entry, data_layout, dispatch, pipeline, uniform_bind_group,
    uniform_buffer, uniform_layout,
};

mod binding {
    pub const IMAGES: u32 = 0;
    pub const WEIGHTS1: u32 = 1;
    pub const HIDDEN: u32 = 2;
    pub const WEIGHTS2: u32 = 3;
    pub const PREDICTS: u32 = 4;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForwardStageConstants {
    pixel_len: u32,
    hidden_size: u32,
    batch_size: u32,
}

impl From<ModelConfig> for ForwardStageConstants {
    fn from(config: ModelConfig) -> Self {
        Self {
            pixel_len: config.pixel_len,
            hidden_size: config.hidden_size,
            batch_size: config.batch_size,
        }
    }
}

pub struct ForwardBuffers<'a> {
    pub images: &'a wgpu::Buffer,
    pub weights1: &'a wgpu::Buffer,
    pub hidden: &'a wgpu::Buffer,
    pub weights2: &'a wgpu::Buffer,
    pub predicts: &'a wgpu::Buffer,
}

pub struct ForwardStage {
    constants_bind_group: wgpu::BindGroup,
    hidden_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    hidden_bind_group: wgpu::BindGroup,
    output_bind_group: wgpu::BindGroup,
    config: ModelConfig,
}

impl ForwardStage {
    pub fn new(device: &wgpu::Device, config: ModelConfig, buffers: ForwardBuffers<'_>) -> Self {
        let constants_value = ForwardStageConstants::from(config);
        let constants = uniform_buffer(device, "Forward Constants", &constants_value, false);
        let constants_layout = uniform_layout(device, "Forward Constants Layout");
        let constants_bind_group = uniform_bind_group(
            device,
            "Forward Constants Bind Group",
            &constants_layout,
            &constants,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("forward_stage.wgsl").into()),
        });
        let hidden_layout = data_layout(
            device,
            "Forward Hidden Data Layout",
            &[
                (binding::IMAGES, true),
                (binding::WEIGHTS1, true),
                (binding::HIDDEN, false),
            ],
        );
        let output_layout = data_layout(
            device,
            "Forward Output Data Layout",
            &[
                (binding::HIDDEN, false),
                (binding::WEIGHTS2, true),
                (binding::PREDICTS, false),
            ],
        );
        Self {
            constants_bind_group,
            hidden_pipeline: pipeline(
                device,
                "Forward Hidden Pipeline",
                &shader,
                "forward_hidden",
                &constants_layout,
                &hidden_layout,
            ),
            output_pipeline: pipeline(
                device,
                "Forward Output Pipeline",
                &shader,
                "forward_output",
                &constants_layout,
                &output_layout,
            ),
            hidden_bind_group: bind_group(
                device,
                "Forward Hidden Bind Group",
                &hidden_layout,
                &[
                    buffer_entry(binding::IMAGES, buffers.images),
                    buffer_entry(binding::WEIGHTS1, buffers.weights1),
                    buffer_entry(binding::HIDDEN, buffers.hidden),
                ],
            ),
            output_bind_group: bind_group(
                device,
                "Forward Output Bind Group",
                &output_layout,
                &[
                    buffer_entry(binding::HIDDEN, buffers.hidden),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                ],
            ),
            config,
        }
    }

    pub fn encode<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        dispatch(
            pass,
            &self.hidden_pipeline,
            &self.constants_bind_group,
            &self.hidden_bind_group,
            (
                self.config.hidden_size.div_ceil(8),
                self.config.batch_size.div_ceil(8),
                1,
            ),
        );
        dispatch(
            pass,
            &self.output_pipeline,
            &self.constants_bind_group,
            &self.output_bind_group,
            (self.config.batch_size.div_ceil(64), 1, 1),
        );
    }
}
