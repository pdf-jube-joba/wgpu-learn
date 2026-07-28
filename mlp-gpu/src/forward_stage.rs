use crate::NeuralNetworkConfig;
use wgpu_compute_utils::{
    bind_group, buffer_entry, data_layout, dispatch, pipeline, uniform_bind_group, uniform_buffer,
    uniform_layout,
};

mod binding {
    pub const IMAGES: u32 = 0;
    pub const WEIGHTS1: u32 = 1;
    pub const HIDDEN1: u32 = 2;
    pub const WEIGHTS2: u32 = 3;
    pub const HIDDEN2: u32 = 4;
    pub const WEIGHTS3: u32 = 5;
    pub const PREDICTS: u32 = 6;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForwardStageConstants {
    input_size: u32,
    hidden1_size: u32,
    hidden2_size: u32,
    batch_size: u32,
}

impl From<NeuralNetworkConfig> for ForwardStageConstants {
    fn from(config: NeuralNetworkConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden1_size: config.hidden1_size,
            hidden2_size: config.hidden2_size,
            batch_size: config.batch_size,
        }
    }
}

pub struct ForwardBuffers<'a> {
    pub images: &'a wgpu::Buffer,
    pub weights1: &'a wgpu::Buffer,
    pub hidden1: &'a wgpu::Buffer,
    pub weights2: &'a wgpu::Buffer,
    pub hidden2: &'a wgpu::Buffer,
    pub weights3: &'a wgpu::Buffer,
    pub predicts: &'a wgpu::Buffer,
}

pub struct ForwardStage {
    constants_bind_group: wgpu::BindGroup,
    hidden1_pipeline: wgpu::ComputePipeline,
    hidden2_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    hidden1_bind_group: wgpu::BindGroup,
    hidden2_bind_group: wgpu::BindGroup,
    output_bind_group: wgpu::BindGroup,
    config: NeuralNetworkConfig,
}

impl ForwardStage {
    pub fn new(
        device: &wgpu::Device,
        config: NeuralNetworkConfig,
        buffers: ForwardBuffers<'_>,
    ) -> Self {
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
        let hidden1_layout = data_layout(
            device,
            "Forward Hidden 1 Data Layout",
            &[
                (binding::IMAGES, true),
                (binding::WEIGHTS1, true),
                (binding::HIDDEN1, false),
            ],
        );
        let hidden2_layout = data_layout(
            device,
            "Forward Hidden 2 Data Layout",
            &[
                (binding::HIDDEN1, false),
                (binding::WEIGHTS2, true),
                (binding::HIDDEN2, false),
            ],
        );
        let output_layout = data_layout(
            device,
            "Forward Output Data Layout",
            &[
                (binding::HIDDEN2, false),
                (binding::WEIGHTS3, true),
                (binding::PREDICTS, false),
            ],
        );
        Self {
            constants_bind_group,
            hidden1_pipeline: pipeline(
                device,
                "Forward Hidden 1 Pipeline",
                &shader,
                "forward_hidden1",
                &constants_layout,
                &hidden1_layout,
            ),
            hidden2_pipeline: pipeline(
                device,
                "Forward Hidden 2 Pipeline",
                &shader,
                "forward_hidden2",
                &constants_layout,
                &hidden2_layout,
            ),
            output_pipeline: pipeline(
                device,
                "Forward Output Pipeline",
                &shader,
                "forward_output",
                &constants_layout,
                &output_layout,
            ),
            hidden1_bind_group: bind_group(
                device,
                "Forward Hidden 1 Bind Group",
                &hidden1_layout,
                &[
                    buffer_entry(binding::IMAGES, buffers.images),
                    buffer_entry(binding::WEIGHTS1, buffers.weights1),
                    buffer_entry(binding::HIDDEN1, buffers.hidden1),
                ],
            ),
            hidden2_bind_group: bind_group(
                device,
                "Forward Hidden 2 Bind Group",
                &hidden2_layout,
                &[
                    buffer_entry(binding::HIDDEN1, buffers.hidden1),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::HIDDEN2, buffers.hidden2),
                ],
            ),
            output_bind_group: bind_group(
                device,
                "Forward Output Bind Group",
                &output_layout,
                &[
                    buffer_entry(binding::HIDDEN2, buffers.hidden2),
                    buffer_entry(binding::WEIGHTS3, buffers.weights3),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                ],
            ),
            config,
        }
    }

    pub fn encode<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        dispatch(
            pass,
            &self.hidden1_pipeline,
            &self.constants_bind_group,
            &self.hidden1_bind_group,
            (
                self.config.hidden1_size.div_ceil(8),
                self.config.batch_size.div_ceil(8),
                1,
            ),
        );
        dispatch(
            pass,
            &self.hidden2_pipeline,
            &self.constants_bind_group,
            &self.hidden2_bind_group,
            (
                self.config.hidden2_size.div_ceil(8),
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
