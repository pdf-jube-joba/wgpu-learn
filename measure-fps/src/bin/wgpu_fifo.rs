use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

const RUN_FOR: Duration = Duration::from_secs(1);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();

    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    started_at: Option<Instant>,
    redraw_events: u64,
    request_redraw_calls: u64,
    frames: u64,
    acquire_time: Duration,
    render_time: Duration,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("measure_fps_wgpu_fifo")
                    .with_resizable(false),
            )
            .expect("failed to create a window");
        let window = Arc::new(window);

        self.gpu = Some(pollster::block_on(Gpu::new(
            window.clone(),
            wgpu::PresentMode::Fifo,
        )));
        self.started_at = Some(Instant::now());
        window.request_redraw();
        self.request_redraw_calls += 1;
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        if id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.redraw_events += 1;

                if let Some(gpu) = self.gpu.as_ref() {
                    if let Some(sample) = gpu.render() {
                        self.frames += 1;
                        self.acquire_time += sample.acquire_time;
                        self.render_time += sample.render_time;
                    }
                }

                let elapsed = self.started_at.expect("timer started").elapsed();
                if elapsed >= RUN_FOR {
                    print_report(
                        "wgpu fifo",
                        self.redraw_events,
                        self.request_redraw_calls,
                        self.frames,
                        elapsed,
                        self.acquire_time,
                        self.render_time,
                    );
                    event_loop.exit();
                    return;
                }

                window.request_redraw();
                self.request_redraw_calls += 1;
            }
            _ => {}
        }
    }
}

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct RenderSample {
    acquire_time: Duration,
    render_time: Duration,
}

impl Gpu {
    async fn new(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .expect("failed to create a surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("failed to find a GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create a device");

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by the adapter");
        config.present_mode = present_mode;

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
        }
    }

    fn render(&self) -> Option<RenderSample> {
        let render_started_at = Instant::now();
        let acquire_started_at = Instant::now();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return None,
        };
        let acquire_time = acquire_started_at.elapsed();

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.12,
                            b: 0.18,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        self.queue.submit([encoder.finish()]);
        frame.present();

        Some(RenderSample {
            acquire_time,
            render_time: render_started_at.elapsed(),
        })
    }
}

fn print_report(
    name: &str,
    redraw_events: u64,
    request_redraw_calls: u64,
    frames: u64,
    elapsed: Duration,
    acquire_time: Duration,
    render_time: Duration,
) {
    let seconds = elapsed.as_secs_f64();
    let frames = frames.max(1);

    println!(
        "{}: redraw_events={}, request_redraw_calls={}, frames={}, seconds={:.3}, fps={:.1}, avg_acquire_ms={:.3}, avg_render_cpu_ms={:.3}",
        name,
        redraw_events,
        request_redraw_calls,
        frames,
        seconds,
        frames as f64 / seconds,
        acquire_time.as_secs_f64() * 1000.0 / frames as f64,
        render_time.as_secs_f64() * 1000.0 / frames as f64,
    );
}
