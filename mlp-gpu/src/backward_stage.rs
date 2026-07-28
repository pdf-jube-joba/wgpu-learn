use crate::NeuralNetworkConfig;
use wgpu_compute_utils::{
    bind_group, buffer_entry, data_layout, dispatch, pipeline, uniform_bind_group, uniform_buffer,
    uniform_layout,
};

mod binding {
    pub const IMAGES: u32 = 0;
    pub const EXPECTS: u32 = 1;
    pub const HIDDEN1: u32 = 2;
    pub const PREDICTS: u32 = 3;
    pub const WEIGHTS1: u32 = 4;
    pub const WEIGHTS2: u32 = 5;
    pub const HIDDEN1_DELTA: u32 = 6;
    pub const LOSS: u32 = 7;
    pub const HIDDEN2: u32 = 8;
    pub const WEIGHTS3: u32 = 9;
    pub const HIDDEN2_DELTA: u32 = 10;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BackwardStageConstants {
    input_size: u32,
    hidden1_size: u32,
    hidden2_size: u32,
    batch_size: u32,
    rate: f32,
    _padding: [u32; 3],
}

impl From<NeuralNetworkConfig> for BackwardStageConstants {
    fn from(config: NeuralNetworkConfig) -> Self {
        Self {
            input_size: config.input_size,
            hidden1_size: config.hidden1_size,
            hidden2_size: config.hidden2_size,
            batch_size: config.batch_size,
            rate: config.learning_rate,
            _padding: [0; 3],
        }
    }
}

pub struct BackwardBuffers<'a> {
    pub images: &'a wgpu::Buffer,
    pub expects: &'a wgpu::Buffer,
    pub hidden1: &'a wgpu::Buffer,
    pub hidden2: &'a wgpu::Buffer,
    pub predicts: &'a wgpu::Buffer,
    pub weights1: &'a wgpu::Buffer,
    pub weights2: &'a wgpu::Buffer,
    pub weights3: &'a wgpu::Buffer,
    pub hidden1_delta: &'a wgpu::Buffer,
    pub hidden2_delta: &'a wgpu::Buffer,
    pub loss: &'a wgpu::Buffer,
}

pub struct BackwardStage {
    constants_bind_group: wgpu::BindGroup,
    loss_pipeline: wgpu::ComputePipeline,
    hidden2_pipeline: wgpu::ComputePipeline,
    hidden1_pipeline: wgpu::ComputePipeline,
    update1_pipeline: wgpu::ComputePipeline,
    update2_pipeline: wgpu::ComputePipeline,
    update3_pipeline: wgpu::ComputePipeline,
    loss_bind_group: wgpu::BindGroup,
    hidden2_bind_group: wgpu::BindGroup,
    hidden1_bind_group: wgpu::BindGroup,
    update1_bind_group: wgpu::BindGroup,
    update2_bind_group: wgpu::BindGroup,
    update3_bind_group: wgpu::BindGroup,
    config: NeuralNetworkConfig,
}

impl BackwardStage {
    pub fn new(
        device: &wgpu::Device,
        config: NeuralNetworkConfig,
        buffers: BackwardBuffers<'_>,
    ) -> Self {
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
        let hidden2_layout = data_layout(
            device,
            "Backward Hidden 2 Data Layout",
            &[
                (binding::EXPECTS, true),
                (binding::HIDDEN2, true),
                (binding::PREDICTS, true),
                (binding::WEIGHTS3, false),
                (binding::HIDDEN2_DELTA, false),
            ],
        );
        let hidden1_layout = data_layout(
            device,
            "Backward Hidden 1 Data Layout",
            &[
                (binding::HIDDEN1, true),
                (binding::WEIGHTS2, false),
                (binding::HIDDEN2_DELTA, false),
                (binding::HIDDEN1_DELTA, false),
            ],
        );
        let update1_layout = data_layout(
            device,
            "Update Layer 1 Weights Data Layout",
            &[
                (binding::IMAGES, true),
                (binding::WEIGHTS1, false),
                (binding::HIDDEN1_DELTA, false),
            ],
        );
        let update2_layout = data_layout(
            device,
            "Update Layer 2 Weights Data Layout",
            &[
                (binding::HIDDEN1, true),
                (binding::WEIGHTS2, false),
                (binding::HIDDEN2_DELTA, false),
            ],
        );
        let update3_layout = data_layout(
            device,
            "Update Layer 3 Weights Data Layout",
            &[
                (binding::EXPECTS, true),
                (binding::HIDDEN2, true),
                (binding::PREDICTS, true),
                (binding::WEIGHTS3, false),
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
            hidden2_pipeline: pipeline(
                device,
                "Backward Hidden 2 Pipeline",
                &shader,
                "backward_hidden2",
                &constants_layout,
                &hidden2_layout,
            ),
            hidden1_pipeline: pipeline(
                device,
                "Backward Hidden 1 Pipeline",
                &shader,
                "backward_hidden1",
                &constants_layout,
                &hidden1_layout,
            ),
            update1_pipeline: pipeline(
                device,
                "Update Layer 1 Weights Pipeline",
                &shader,
                "update_weights1",
                &constants_layout,
                &update1_layout,
            ),
            update2_pipeline: pipeline(
                device,
                "Update Layer 2 Weights Pipeline",
                &shader,
                "update_weights2",
                &constants_layout,
                &update2_layout,
            ),
            update3_pipeline: pipeline(
                device,
                "Update Layer 3 Weights Pipeline",
                &shader,
                "update_weights3",
                &constants_layout,
                &update3_layout,
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
            hidden2_bind_group: bind_group(
                device,
                "Backward Hidden 2 Bind Group",
                &hidden2_layout,
                &[
                    buffer_entry(binding::EXPECTS, buffers.expects),
                    buffer_entry(binding::HIDDEN2, buffers.hidden2),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                    buffer_entry(binding::WEIGHTS3, buffers.weights3),
                    buffer_entry(binding::HIDDEN2_DELTA, buffers.hidden2_delta),
                ],
            ),
            hidden1_bind_group: bind_group(
                device,
                "Backward Hidden 1 Bind Group",
                &hidden1_layout,
                &[
                    buffer_entry(binding::HIDDEN1, buffers.hidden1),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::HIDDEN2_DELTA, buffers.hidden2_delta),
                    buffer_entry(binding::HIDDEN1_DELTA, buffers.hidden1_delta),
                ],
            ),
            update1_bind_group: bind_group(
                device,
                "Update Layer 1 Weights Bind Group",
                &update1_layout,
                &[
                    buffer_entry(binding::IMAGES, buffers.images),
                    buffer_entry(binding::WEIGHTS1, buffers.weights1),
                    buffer_entry(binding::HIDDEN1_DELTA, buffers.hidden1_delta),
                ],
            ),
            update2_bind_group: bind_group(
                device,
                "Update Layer 2 Weights Bind Group",
                &update2_layout,
                &[
                    buffer_entry(binding::HIDDEN1, buffers.hidden1),
                    buffer_entry(binding::WEIGHTS2, buffers.weights2),
                    buffer_entry(binding::HIDDEN2_DELTA, buffers.hidden2_delta),
                ],
            ),
            update3_bind_group: bind_group(
                device,
                "Update Layer 3 Weights Bind Group",
                &update3_layout,
                &[
                    buffer_entry(binding::EXPECTS, buffers.expects),
                    buffer_entry(binding::HIDDEN2, buffers.hidden2),
                    buffer_entry(binding::PREDICTS, buffers.predicts),
                    buffer_entry(binding::WEIGHTS3, buffers.weights3),
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
            &self.hidden1_pipeline,
            &self.constants_bind_group,
            &self.hidden1_bind_group,
            (
                self.config.hidden1_size.div_ceil(8),
                self.config.batch_size.div_ceil(8),
                1,
            ),
        );
        let layer1_parameters =
            self.config.hidden1_size * self.config.input_size + self.config.hidden1_size;
        let layer2_parameters =
            self.config.hidden2_size * self.config.hidden1_size + self.config.hidden2_size;
        let layer3_parameters = self.config.hidden2_size + 1;

        dispatch(
            pass,
            &self.update1_pipeline,
            &self.constants_bind_group,
            &self.update1_bind_group,
            (layer1_parameters.div_ceil(64), 1, 1),
        );
        dispatch(
            pass,
            &self.update2_pipeline,
            &self.constants_bind_group,
            &self.update2_bind_group,
            (layer2_parameters.div_ceil(64), 1, 1),
        );
        dispatch(
            pass,
            &self.update3_pipeline,
            &self.constants_bind_group,
            &self.update3_bind_group,
            (layer3_parameters.div_ceil(64), 1, 1),
        );
    }
}
