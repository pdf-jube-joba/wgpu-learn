use crate::{
    ModelConfig, bind_group, buffer_entry, data_layout, dispatch, pipeline, uniform_bind_group,
    uniform_buffer, uniform_layout,
};

mod binding {
    pub const IMAGES: u32 = 0;
    pub const EXPECTS: u32 = 1;
    pub const HIDDEN: u32 = 2;
    pub const PREDICTS: u32 = 3;
    pub const WEIGHTS1: u32 = 4;
    pub const WEIGHTS2: u32 = 5;
    pub const HIDDEN_DELTA: u32 = 6;
    pub const LOSS: u32 = 7;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BackwardStageConstants {
    pixel_len: u32,
    hidden_size: u32,
    batch_size: u32,
    rate: f32,
}

impl From<ModelConfig> for BackwardStageConstants {
    fn from(config: ModelConfig) -> Self {
        Self {
            pixel_len: config.pixel_len,
            hidden_size: config.hidden_size,
            batch_size: config.batch_size,
            rate: config.learning_rate,
        }
    }
}

pub struct BackwardBuffers<'a> {
    pub images: &'a wgpu::Buffer,
    pub expects: &'a wgpu::Buffer,
    pub hidden: &'a wgpu::Buffer,
    pub predicts: &'a wgpu::Buffer,
    pub weights1: &'a wgpu::Buffer,
    pub weights2: &'a wgpu::Buffer,
    pub hidden_delta: &'a wgpu::Buffer,
    pub loss: &'a wgpu::Buffer,
}

pub struct BackwardStage {
    constants_bind_group: wgpu::BindGroup,
    loss_pipeline: wgpu::ComputePipeline,
    hidden_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    loss_bind_group: wgpu::BindGroup,
    hidden_bind_group: wgpu::BindGroup,
    update_bind_group: wgpu::BindGroup,
    config: ModelConfig,
}

impl BackwardStage {
    pub fn new(device: &wgpu::Device, config: ModelConfig, buffers: BackwardBuffers<'_>) -> Self {
        let constants_value = BackwardStageConstants::from(config);
        let constants = uniform_buffer(device, "Backward Constants", &constants_value, false);
        let constants_layout = uniform_layout(device, "Backward Constants Layout");
        let constants_bind_group = uniform_bind_group(
            device,
            "Backward Constants Bind Group",
            &constants_layout,
            &constants,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("backward_stage.wgsl").into()),
        });
        let loss_layout = data_layout(
            device,
            "Compute Loss Data Layout",
            &[
                (binding::EXPECTS, true),
                (binding::PREDICTS, true),
                (binding::LOSS, false),
            ],
        );
        let hidden_layout = data_layout(
            device,
            "Backward Hidden Data Layout",
            &[
                (binding::EXPECTS, true),
                (binding::HIDDEN, true),
                (binding::PREDICTS, true),
                (binding::WEIGHTS2, false),
                (binding::HIDDEN_DELTA, false),
            ],
        );
        let update_layout = data_layout(
            device,
            "Update Weights Data Layout",
            &[
                (binding::IMAGES, true),
                (binding::EXPECTS, true),
                (binding::HIDDEN, true),
                (binding::PREDICTS, true),
                (binding::WEIGHTS1, false),
                (binding::WEIGHTS2, false),
                (binding::HIDDEN_DELTA, false),
            ],
        );
        Self {
            constants_bind_group,
            loss_pipeline: pipeline(
                device,
                "Compute Loss Pipeline",
                &shader,
                "compute_loss",
                &constants_layout,
                &loss_layout,
            ),
            hidden_pipeline: pipeline(
                device,
                "Backward Hidden Pipeline",
                &shader,
                "backward_hidden",
                &constants_layout,
                &hidden_layout,
            ),
            update_pipeline: pipeline(
                device,
                "Update Weights Pipeline",
                &shader,
                "update_weights",
                &constants_layout,
                &update_layout,
            ),
            loss_bind_group: bind_group(
                device,
                "Compute Loss Bind Group",
                &loss_layout,
                &[
                    buffer_entry(binding::EXPECTS, buffers.expects),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                    buffer_entry(binding::LOSS, buffers.loss),
                ],
            ),
            hidden_bind_group: bind_group(
                device,
                "Backward Hidden Bind Group",
                &hidden_layout,
                &[
                    buffer_entry(binding::EXPECTS, buffers.expects),
                    buffer_entry(binding::HIDDEN, buffers.hidden),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::HIDDEN_DELTA, buffers.hidden_delta),
                ],
            ),
            update_bind_group: bind_group(
                device,
                "Update Weights Bind Group",
                &update_layout,
                &[
                    buffer_entry(binding::IMAGES, buffers.images),
                    buffer_entry(binding::EXPECTS, buffers.expects),
                    buffer_entry(binding::HIDDEN, buffers.hidden),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                    buffer_entry(binding::WEIGHTS1, buffers.weights1),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::HIDDEN_DELTA, buffers.hidden_delta),
                ],
            ),
            config,
        }
    }

    pub fn encode<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, compute_loss: bool) {
        if compute_loss {
            dispatch(
                pass,
                &self.loss_pipeline,
                &self.constants_bind_group,
                &self.loss_bind_group,
                (1, 1, 1),
            );
        }
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
            &self.update_pipeline,
            &self.constants_bind_group,
            &self.update_bind_group,
            (self.config.parameter_count().div_ceil(64), 1, 1),
        );
    }
}
