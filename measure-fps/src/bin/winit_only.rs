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
    started_at: Option<Instant>,
    redraw_events: u64,
    request_redraw_calls: u64,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("measure_fps_winit_only")
                    .with_resizable(false),
            )
            .expect("failed to create a window");
        let window = Arc::new(window);

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

                let elapsed = self.started_at.expect("timer started").elapsed();
                if elapsed >= RUN_FOR {
                    println!(
                        "winit only: redraw_events={}, request_redraw_calls={}, seconds={:.3}, redraw_events/sec={:.1}",
                        self.redraw_events,
                        self.request_redraw_calls,
                        elapsed.as_secs_f64(),
                        self.redraw_events as f64 / elapsed.as_secs_f64()
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
