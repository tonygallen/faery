use numpy::PyArrayMethods;
use pyo3::prelude::*;
use std::thread;

pub mod render_3d;
pub mod viewer;
pub mod viewer_3d;

pub use viewer::{FrameStreamer, FrameViewer, FrameViewerWindow};
pub use viewer_3d::{EventStreamer3D, EventViewer3D, EventViewer3DWindow};

#[pyfunction]
#[pyo3(signature = (frame_stream, frame_rate=None))]
pub fn run_frame_viewer_from_iterator(
    py: Python,
    frame_stream: PyObject,
    frame_rate: Option<f64>,
) -> PyResult<()> {
    // Create window first to catch any errors directly
    let ui = FrameViewerWindow::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create window: {}", e))
    })?;

    // Create streamer for thread-safe communication
    let streamer = FrameStreamer::new(frame_rate);

    // Create viewer with window and streamer
    let viewer = FrameViewer::new(ui, streamer.clone());

    // Show the window
    viewer.show().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to show window: {}", e))
    })?;

    // Background thread to collect frames from Python iterator
    let iter = frame_stream.call_method0(py, "__iter__")?;
    let streamer_clone = streamer.clone();
    let _handle = thread::spawn(move || {
        loop {
            // Check for shutdown signal
            if streamer_clone.is_shutdown() {
                break;
            }

            // If paused, don't consume frames from the iterator
            if streamer_clone.is_paused() {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let result = Python::with_gil(|py| iter.call_method0(py, "__next__"));

            match result {
                Ok(frame_obj) => {
                    if let Ok(frame_data) =
                        Python::with_gil(|py| -> PyResult<numpy::ndarray::Array3<u8>> {
                            let pixels = frame_obj.getattr(py, "pixels")?;
                            let frame_array = pixels.bind(py).downcast::<numpy::PyArray3<u8>>()?;
                            let readonly_frame = frame_array.readonly();
                            let array = readonly_frame.as_array();
                            Ok(array.to_owned())
                        })
                    {
                        streamer_clone.add_frame(frame_data);
                    }
                }
                Err(_) => {
                    // Iterator ended, mark stream as ended but don't shutdown the GUI
                    streamer_clone.set_stream_ended(true);
                    break;
                }
            }
        }
    });

    // Timer to check for shutdown and quit event loop
    let streamer_check = streamer.clone();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(100),
        move || {
            if streamer_check.is_shutdown() {
                println!("Shutdown detected, quitting event loop");
                let _ = slint::quit_event_loop();
            }
        },
    );

    // Start the GUI in a thread so we can return and release the GIL
    py.allow_threads(|| {
        // Run event loop on main thread
        slint::run_event_loop()
    })
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to run event loop: {}", e))
    })?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (event_stream, time_window_us=1000000, time_slices=50, colormap_rgba=None))]
pub fn run_event_viewer_3d_from_iterator(
    py: Python,
    event_stream: PyObject,
    time_window_us: u64,
    time_slices: usize,
    colormap_rgba: Option<&pyo3::Bound<'_, numpy::PyArray2<f64>>>,
) -> PyResult<()> {
    let ui = EventViewer3DWindow::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create window: {}", e))
    })?;

    let streamer = EventStreamer3D::new();
    let viewer = std::sync::Arc::new(std::sync::Mutex::new(
        EventViewer3D::new(ui, streamer.clone())
    ));

    {
        let viewer_guard = viewer.lock().unwrap();
        viewer_guard.show().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to show window: {}", e))
        })?;
    }

    let (colormap_off, colormap_on) = if let Some(colormap_array) = colormap_rgba {
        let readonly_colormap = colormap_array.readonly();
        let array = readonly_colormap.as_array();
        parse_diverging_colormap_for_3d(array)?
    } else {
        (
            vec![0xFF0000FFu32; 256],
            vec![0x00FF00FFu32; 256],
        )
    };

    let background_color = 0x000000FFu32;

    let iter = event_stream.call_method0(py, "__iter__")?;
    let streamer_clone = streamer.clone();
    let viewer_thread = viewer.clone();

    let renderer = std::sync::Arc::new(std::sync::Mutex::new(None::<render_3d::Renderer3D>));
    let renderer_clone = renderer.clone();

    let _handle = thread::spawn(move || {
        let mut current_t = 0u64;
        let mut initialized = false;

        loop {
            if streamer_clone.is_shutdown() {
                break;
            }

            if streamer_clone.is_paused() {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let result = Python::with_gil(|py| iter.call_method0(py, "__next__"));

            match result {
                Ok(events_obj) => {
                    if let Ok((events, width, height)) =
                        Python::with_gil(|py| -> PyResult<(Vec<render_3d::Event3D>, u16, u16)> {
                            let events_array =
                                events_obj.bind(py).downcast::<numpy::PyArray1<u8>>()?;
                            let readonly = events_array.readonly();
                            let raw_data = readonly.as_slice()?;

                            let event_size = std::mem::size_of::<neuromorphic_types::PolarityEvent<u64, u16, u16>>();
                            let num_events = raw_data.len() / event_size;

                            let mut parsed_events = Vec::new();
                            for i in 0..num_events {
                                let offset = i * event_size;
                                let event_bytes = &raw_data[offset..offset + event_size];

                                unsafe {
                                    let event: &neuromorphic_types::PolarityEvent<u64, u16, u16> =
                                        &*(event_bytes.as_ptr()
                                            as *const neuromorphic_types::PolarityEvent<
                                                u64,
                                                u16,
                                                u16,
                                            >);
                                    parsed_events.push(render_3d::Event3D {
                                        t: event.t,
                                        x: event.x,
                                        y: event.y,
                                        polarity: event.polarity,
                                    });
                                }
                            }

                            let dimensions_obj = events_obj.getattr(py, "dimensions")?;
                            let dimensions_tuple = dimensions_obj.extract::<(u16, u16)>(py)?;

                            Ok((parsed_events, dimensions_tuple.0, dimensions_tuple.1))
                        })
                    {
                        if !initialized {
                            let mut r = renderer_clone.lock().unwrap();
                            *r = Some(render_3d::Renderer3D::new(
                                width,
                                height,
                                time_window_us,
                                time_slices,
                                colormap_off.clone(),
                                colormap_on.clone(),
                                background_color,
                            ));
                            initialized = true;
                        }

                        if !events.is_empty() {
                            current_t = events.iter().map(|e| e.t).max().unwrap_or(current_t);
                            streamer_clone.add_events(events.len());
                        }

                        if let Some(ref mut r) = *renderer_clone.lock().unwrap() {
                            r.add_events(&events, current_t);
                        }
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }
    });

    let renderer_ui = renderer.clone();
    let streamer_ui = streamer.clone();
    let viewer_ui = viewer.clone();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(33),
        move || {
            if streamer_ui.is_shutdown() {
                let _ = slint::quit_event_loop();
                return;
            }

            if let Some(ref r) = *renderer_ui.lock().unwrap() {
                let current_t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;

                let (frame_data, width, height) = r.render(current_t);

                let event_count = r.get_event_count();
                let event_rate = streamer_ui.get_event_rate();

                if let Ok(viewer_guard) = viewer_ui.lock() {
                    viewer_guard.update_frame(frame_data, width, height);
                    viewer_guard.update_info(format!(
                        "Events: {} | Rate: {:.1} kHz",
                        event_count,
                        event_rate / 1000.0
                    ));
                }
            }
        },
    );

    py.allow_threads(|| slint::run_event_loop())
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to run event loop: {}", e))
        })?;

    Ok(())
}

fn parse_diverging_colormap_for_3d(
    array: numpy::ndarray::ArrayBase<
        numpy::ndarray::ViewRepr<&f64>,
        numpy::ndarray::Dim<[usize; 2]>,
    >,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let dimensions = array.dim();
    let (mut off_colormap, mut on_colormap) = if dimensions.0 % 2 == 0 {
        (
            Vec::with_capacity(dimensions.0 / 2 + 1),
            Vec::with_capacity(dimensions.0 / 2),
        )
    } else {
        (
            Vec::with_capacity(dimensions.0 / 2 + 1),
            Vec::with_capacity(dimensions.0 / 2 + 1),
        )
    };

    fn component_f64_to_u8(component: f64) -> u8 {
        let value = (component * 255.0).round();
        if value <= 0.0 {
            0
        } else if value >= 255.0 {
            255
        } else {
            value as u8
        }
    }

    for index in (0..=dimensions.0 / 2).rev() {
        off_colormap.push(u32::from_ne_bytes([
            component_f64_to_u8(*array.get((index, 0)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 1)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 2)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 3)).expect("array index valid")),
        ]));
    }
    for index in dimensions.0 / 2..dimensions.0 {
        on_colormap.push(u32::from_ne_bytes([
            component_f64_to_u8(*array.get((index, 0)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 1)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 2)).expect("array index valid")),
            component_f64_to_u8(*array.get((index, 3)).expect("array index valid")),
        ]));
    }
    Ok((off_colormap, on_colormap))
}
