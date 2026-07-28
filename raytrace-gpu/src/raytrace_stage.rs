use crate::RaytraceConfig;
use wgpu_compute_utils::{
    bind_group, buffer_entry, buffer_layout_entry, data_layout, dispatch, pipeline,
    uniform_bind_group, uniform_buffer, uniform_layout,
};

mod binding {
    pub const SAMPLES: u32 = 0;
    pub const EXPECTS: u32 = 1;
    pub const IMAGES: u32 = 2;
    pub const DEBUG_IMAGES: u32 = 3;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RaytraceStageConstants {
    seed: u32,
    pixel_len: u32,
    rays_per_pixel: u32,
    batch_size: u32,
}

impl From<RaytraceConfig> for RaytraceStageConstants {
    fn from(config: RaytraceConfig) -> Self {
        Self {
            seed: config.base_seed,
            pixel_len: config.pixel_len,
            rays_per_pixel: config.rays_per_pixel,
            batch_size: config.batch_size,
        }
    }
}

pub struct RaytraceBuffers<'a> {
    pub samples: &'a wgpu::Buffer,
    pub expects: &'a wgpu::Buffer,
    pub images: &'a wgpu::Buffer,
}

pub struct RaytraceStage {
    constants: wgpu::Buffer,
    constants_bind_group: wgpu::BindGroup,
    generate_samples_pipeline: wgpu::ComputePipeline,
    raytrace_pipeline: wgpu::ComputePipeline,
    generate_samples_bind_group: wgpu::BindGroup,
    raytrace_bind_group: wgpu::BindGroup,
    config: RaytraceConfig,
}

impl RaytraceStage {
    pub fn new(
        device: &wgpu::Device,
        config: RaytraceConfig,
        buffers: RaytraceBuffers<'_>,
        debug_images: Option<&wgpu::TextureView>,
    ) -> Self {
        // constants
        let constants = uniform_buffer(
            device,
            "Raytrace Constants",
            &RaytraceStageConstants::from(config),
            true,
        );
        let constants_layout = uniform_layout(device, "Raytrace Constants Layout");
        let constants_bind_group = uniform_bind_group(
            device,
            "Raytrace Constants Bind Group",
            &constants_layout,
            &constants,
        );

        // generate
        let generate_layout = data_layout(
            device,
            "Generate Samples Data Layout",
            &[(binding::SAMPLES, false), (binding::EXPECTS, false)],
        );
        let mut raytrace_entries = vec![
            buffer_layout_entry(
                binding::SAMPLES,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            buffer_layout_entry(
                binding::IMAGES,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
        ];
        if debug_images.is_some() {
            raytrace_entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding::DEBUG_IMAGES,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            });
        }
        let raytrace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Raytrace Data Layout"),
            entries: &raytrace_entries,
        });

        // pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Generate Image Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raytrace_stage.wgsl").into()),
        });
        let generate_samples_pipeline = pipeline(
            device,
            "Generate Samples Pipeline",
            &shader,
            "generate_samples",
            &constants_layout,
            &generate_layout,
        );
        let raytrace_pipeline = pipeline(
            device,
            "Raytrace Pipeline",
            &shader,
            if debug_images.is_some() {
                "raytrace_preview"
            } else {
                "raytrace"
            },
            &constants_layout,
            &raytrace_layout,
        );

        // bindgroup
        let generate_samples_bind_group = bind_group(
            device,
            "Generate Samples Bind Group",
            &generate_layout,
            &[
                buffer_entry(binding::SAMPLES, buffers.samples),
                buffer_entry(binding::EXPECTS, buffers.expects),
            ],
        );
        let mut raytrace_bindings = vec![
            buffer_entry(binding::SAMPLES, buffers.samples),
            buffer_entry(binding::IMAGES, buffers.images),
        ];
        if let Some(view) = debug_images {
            raytrace_bindings.push(wgpu::BindGroupEntry {
                binding: binding::DEBUG_IMAGES,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let raytrace_bind_group = bind_group(
            device,
            "Raytrace Bind Group",
            &raytrace_layout,
            &raytrace_bindings,
        );
        Self {
            constants,
            constants_bind_group,
            generate_samples_pipeline,
            raytrace_pipeline,
            generate_samples_bind_group,
            raytrace_bind_group,
            config,
        }
    }

    pub fn set_seed(&self, queue: &wgpu::Queue, seed: u32) {
        let mut constants = RaytraceStageConstants::from(self.config);
        constants.seed = seed;
        queue.write_buffer(&self.constants, 0, bytemuck::bytes_of(&constants));
    }

    pub fn encode<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        dispatch(
            pass,
            &self.generate_samples_pipeline,
            &self.constants_bind_group,
            &self.generate_samples_bind_group,
            (self.config.batch_size.div_ceil(64), 1, 1),
        );
        dispatch(
            pass,
            &self.raytrace_pipeline,
            &self.constants_bind_group,
            &self.raytrace_bind_group,
            (
                self.config.pixel_len.div_ceil(8),
                (self.config.pixel_len * self.config.batch_size).div_ceil(8),
                1,
            ),
        );
    }
}
