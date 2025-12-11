use std::sync::{Arc, Mutex};
use std::time::Instant;

slint::slint! {
    import { Button, VerticalBox, HorizontalBox, GroupBox } from "std-widgets.slint";

    export component EventViewer3DWindow inherits Window {
        title: "Faery 3D Event Viewer";
        width: 1000px;
        height: 800px;

        in property <image> current-frame;
        in property <string> event-info: "";
        in property <bool> is-playing: false;

        callback play-pause();
        callback reset();
        callback close-window();

        FocusScope {
            key-pressed(event) => {
                if (event.text == Key.Escape) {
                    close-window();
                    return accept;
                }
                return reject;
            }

            VerticalBox {
                padding: 10px;
                spacing: 10px;

                GroupBox {
                    title: "3D Event Stream Visualization";
                    Rectangle {
                        background: #000000;
                        border-radius: 4px;

                        Image {
                            source: current-frame;
                            image-fit: contain;
                            width: 100%;
                            height: 100%;
                        }
                    }
                }

                Text {
                    text: event-info;
                    font-size: 12px;
                    color: #999;
                }

                HorizontalBox {
                    spacing: 10px;
                    alignment: center;

                    Button {
                        text: is-playing ? "Pause" : "Play";
                        clicked => { play-pause(); }
                    }

                    Button {
                        text: "Reset";
                        clicked => { reset(); }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct EventStreamer3D {
    shutdown: Arc<Mutex<bool>>,
    is_paused: Arc<Mutex<bool>>,
    event_count: Arc<Mutex<usize>>,
    event_rate: Arc<Mutex<f64>>,
    last_event_time: Arc<Mutex<Instant>>,
}

impl EventStreamer3D {
    pub fn new() -> Self {
        Self {
            shutdown: Arc::new(Mutex::new(false)),
            is_paused: Arc::new(Mutex::new(false)),
            event_count: Arc::new(Mutex::new(0)),
            event_rate: Arc::new(Mutex::new(0.0)),
            last_event_time: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub fn is_shutdown(&self) -> bool {
        *self.shutdown.lock().unwrap()
    }

    pub fn set_shutdown(&self, shutdown: bool) {
        *self.shutdown.lock().unwrap() = shutdown;
    }

    pub fn is_paused(&self) -> bool {
        *self.is_paused.lock().unwrap()
    }

    pub fn toggle_pause(&self) -> bool {
        let mut is_paused = self.is_paused.lock().unwrap();
        *is_paused = !*is_paused;
        *is_paused
    }

    pub fn reset(&self) {
        *self.event_count.lock().unwrap() = 0;
        *self.event_rate.lock().unwrap() = 0.0;
        *self.is_paused.lock().unwrap() = false;
    }

    pub fn add_events(&self, count: usize) {
        let mut total = self.event_count.lock().unwrap();
        *total += count;

        let now = Instant::now();
        let mut last_time = self.last_event_time.lock().unwrap();
        let time_diff = now.duration_since(*last_time);

        if time_diff.as_millis() > 0 && time_diff.as_millis() < 500 {
            let new_rate = (count as f64 * 1000.0) / time_diff.as_millis() as f64;
            let mut event_rate = self.event_rate.lock().unwrap();
            *event_rate = *event_rate * 0.8 + new_rate * 0.2;
        }
        *last_time = now;
    }

    pub fn get_event_count(&self) -> usize {
        *self.event_count.lock().unwrap()
    }

    pub fn get_event_rate(&self) -> f64 {
        *self.event_rate.lock().unwrap()
    }
}

pub struct EventViewer3D {
    ui: EventViewer3DWindow,
    streamer: EventStreamer3D,
    _timer: slint::Timer,
}

impl EventViewer3D {
    pub fn new(ui: EventViewer3DWindow, streamer: EventStreamer3D) -> Self {
        let timer = slint::Timer::default();
        let viewer = Self {
            ui,
            streamer,
            _timer: timer,
        };

        viewer.ui.set_is_playing(true);
        viewer.setup_callbacks();
        viewer
    }

    fn setup_callbacks(&self) {
        let streamer_clone = self.streamer.clone();
        self.ui.window().on_close_requested(move || {
            streamer_clone.set_shutdown(true);
            slint::CloseRequestResponse::HideWindow
        });

        self.ui.on_close_window({
            let streamer = self.streamer.clone();
            move || {
                streamer.set_shutdown(true);
            }
        });

        self.ui.on_play_pause({
            let streamer = self.streamer.clone();
            let ui_weak = self.ui.as_weak();
            move || {
                let is_paused = streamer.toggle_pause();
                ui_weak
                    .upgrade_in_event_loop(move |ui| {
                        ui.set_is_playing(!is_paused);
                    })
                    .unwrap_or_else(|_| {
                        println!("Failed to update play/pause button state");
                    });
            }
        });

        self.ui.on_reset({
            let streamer = self.streamer.clone();
            let ui_weak = self.ui.as_weak();
            move || {
                streamer.reset();
                ui_weak
                    .upgrade_in_event_loop(move |ui| {
                        ui.set_is_playing(true);
                    })
                    .unwrap_or_else(|_| {
                        println!("Failed to update play state after reset");
                    });
            }
        });
    }

    pub fn get_streamer(&self) -> EventStreamer3D {
        self.streamer.clone()
    }

    pub fn show(&self) -> Result<(), slint::PlatformError> {
        self.ui.show()
    }

    pub fn update_frame(&self, frame_data: Vec<u8>, width: u32, height: u32) {
        let ui_weak = self.ui.as_weak();
        ui_weak
            .upgrade_in_event_loop(move |ui| {
                let image = slint::Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(
                    &frame_data, width, height,
                ));
                ui.set_current_frame(image);
            })
            .unwrap_or_else(|_| {
                println!("Failed to update frame");
            });
    }

    pub fn update_info(&self, info: String) {
        let ui_weak = self.ui.as_weak();
        ui_weak
            .upgrade_in_event_loop(move |ui| {
                ui.set_event_info(info.into());
            })
            .unwrap_or_else(|_| {
                println!("Failed to update info");
            });
    }
}
