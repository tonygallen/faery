use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Event3D {
    pub t: u64,
    pub x: u16,
    pub y: u16,
    pub polarity: neuromorphic_types::Polarity,
}

pub struct Renderer3D {
    width: u16,
    height: u16,
    time_window_us: u64,
    time_slices: usize,
    events: VecDeque<Event3D>,
    colormap_off: Vec<u32>,
    colormap_on: Vec<u32>,
    background_color: u32,
    render_width: u32,
    render_height: u32,
}

impl Renderer3D {
    pub fn new(
        width: u16,
        height: u16,
        time_window_us: u64,
        time_slices: usize,
        colormap_off: Vec<u32>,
        colormap_on: Vec<u32>,
        background_color: u32,
    ) -> Self {
        let render_width = (width as f32 * 1.5 + time_slices as f32 * 0.5) as u32;
        let render_height = (height as f32 * 1.5 + time_slices as f32 * 0.3) as u32;

        Self {
            width,
            height,
            time_window_us,
            time_slices,
            events: VecDeque::new(),
            colormap_off,
            colormap_on,
            background_color,
            render_width,
            render_height,
        }
    }

    pub fn add_events(&mut self, new_events: &[Event3D], current_t: u64) {
        for event in new_events {
            self.events.push_back(event.clone());
        }

        let cutoff_t = if current_t > self.time_window_us {
            current_t - self.time_window_us
        } else {
            0
        };

        while let Some(front) = self.events.front() {
            if front.t < cutoff_t {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn render(&self, current_t: u64) -> (Vec<u8>, u32, u32) {
        let mut buffer = vec![0u32; (self.render_width * self.render_height) as usize];

        for pixel in buffer.iter_mut() {
            *pixel = self.background_color;
        }

        let cutoff_t = if current_t > self.time_window_us {
            current_t - self.time_window_us
        } else {
            0
        };

        for event in &self.events {
            if event.t < cutoff_t {
                continue;
            }

            let age_us = current_t.saturating_sub(event.t);
            let age_normalized = age_us as f32 / self.time_window_us as f32;
            let time_depth = age_normalized * self.time_slices as f32;

            let (screen_x, screen_y) = self.project_3d_to_2d(
                event.x as f32,
                event.y as f32,
                time_depth,
            );

            if screen_x >= 0.0
                && screen_x < self.render_width as f32
                && screen_y >= 0.0
                && screen_y < self.render_height as f32
            {
                let px = screen_x as usize;
                let py = screen_y as usize;
                let idx = py * self.render_width as usize + px;

                if idx < buffer.len() {
                    let colormap = match event.polarity {
                        neuromorphic_types::Polarity::Off => &self.colormap_off,
                        neuromorphic_types::Polarity::On => &self.colormap_on,
                    };

                    let alpha = 1.0 - age_normalized * 0.7;
                    let color_idx =
                        ((1.0 - age_normalized) * (colormap.len() - 1) as f32) as usize;
                    let color = colormap[color_idx.min(colormap.len() - 1)];

                    buffer[idx] = Self::blend_colors(buffer[idx], color, alpha);
                }
            }
        }

        let mut rgba_buffer = vec![0u8; (self.render_width * self.render_height * 4) as usize];
        for (i, &color) in buffer.iter().enumerate() {
            let bytes = color.to_ne_bytes();
            rgba_buffer[i * 4] = bytes[0];
            rgba_buffer[i * 4 + 1] = bytes[1];
            rgba_buffer[i * 4 + 2] = bytes[2];
            rgba_buffer[i * 4 + 3] = bytes[3];
        }

        (rgba_buffer, self.render_width, self.render_height)
    }

    fn project_3d_to_2d(&self, x: f32, y: f32, time_depth: f32) -> (f32, f32) {
        let time_offset_x = time_depth * 0.5;
        let time_offset_y = time_depth * 0.3;

        let screen_x = x + time_offset_x;
        let screen_y = y + time_offset_y;

        (screen_x, screen_y)
    }

    fn blend_colors(bg: u32, fg: u32, alpha: f32) -> u32 {
        let bg_bytes = bg.to_ne_bytes();
        let fg_bytes = fg.to_ne_bytes();

        let r = (fg_bytes[0] as f32 * alpha + bg_bytes[0] as f32 * (1.0 - alpha)) as u8;
        let g = (fg_bytes[1] as f32 * alpha + bg_bytes[1] as f32 * (1.0 - alpha)) as u8;
        let b = (fg_bytes[2] as f32 * alpha + bg_bytes[2] as f32 * (1.0 - alpha)) as u8;
        let a = 255u8;

        u32::from_ne_bytes([r, g, b, a])
    }

    pub fn get_event_count(&self) -> usize {
        self.events.len()
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }
}
