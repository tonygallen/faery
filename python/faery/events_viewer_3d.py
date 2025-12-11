import collections.abc
import typing
import numpy as np

from . import events_stream, frame_stream, stream, timestamp, color as color_module


class EventsViewer3D:
    def __init__(
        self,
        parent: stream.Stream[np.ndarray],
        time_window_us: int = 1000000,
        time_slices: int = 50,
        colormap: typing.Optional[color_module.Colormap] = None,
    ):
        self.parent = parent
        self.time_window_us = time_window_us
        self.time_slices = time_slices
        
        if colormap is None:
            from . import colormaps
            self.colormap = colormaps.managua
        else:
            self.colormap = colormap
        
        self.dimensions = None
        
    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
        from collections import deque
        import time as time_module
        
        event_buffer = deque()
        
        for events in self.parent:
            if len(events) == 0:
                continue
                
            if self.dimensions is None:
                self.dimensions = self.parent.dimensions()
            
            current_t = int(events['t'][-1])
            
            for event in events:
                event_buffer.append({
                    't': int(event['t']),
                    'x': int(event['x']),
                    'y': int(event['y']),
                    'p': bool(event['on']),
                })
            
            cutoff_t = current_t - self.time_window_us
            while event_buffer and event_buffer[0]['t'] < cutoff_t:
                event_buffer.popleft()
            
            frame = self._render_3d_frame(event_buffer, current_t)
            
            yield frame_stream.Frame(
                t=timestamp.Time(microseconds=current_t),
                pixels=frame
            )
    
    def _render_3d_frame(self, event_buffer, current_t):
        width, height = self.dimensions
        
        render_width = int(width * 1.5 + self.time_slices * 0.5)
        render_height = int(height * 1.5 + self.time_slices * 0.3)
        
        frame = np.zeros((render_height, render_width, 4), dtype=np.uint8)
        
        colormap_rgba = self.colormap.rgba
        
        if self.colormap.type in ["diverging", "cyclic"]:
            half_idx = len(colormap_rgba) // 2
            colormap_off = colormap_rgba[:half_idx]
            colormap_on = colormap_rgba[half_idx:]
        else:
            colormap_off = colormap_rgba
            colormap_on = colormap_rgba
        
        for event in event_buffer:
            age_us = current_t - event['t']
            if age_us < 0:
                continue
                
            age_normalized = min(1.0, age_us / self.time_window_us)
            time_depth = age_normalized * self.time_slices
            
            screen_x = int(event['x'] + time_depth * 0.5)
            screen_y = int(event['y'] + time_depth * 0.3)
            
            if 0 <= screen_x < render_width and 0 <= screen_y < render_height:
                colormap = colormap_on if event['p'] else colormap_off
                color_idx = int((1.0 - age_normalized) * (len(colormap) - 1))
                color_idx = min(color_idx, len(colormap) - 1)
                
                alpha = 1.0 - age_normalized * 0.7
                color = colormap[color_idx]
                
                frame[screen_y, screen_x] = (
                    min(255, int(color[0] * 255 * alpha + frame[screen_y, screen_x, 0] * (1 - alpha))),
                    min(255, int(color[1] * 255 * alpha + frame[screen_y, screen_x, 1] * (1 - alpha))),
                    min(255, int(color[2] * 255 * alpha + frame[screen_y, screen_x, 2] * (1 - alpha))),
                    min(255, int(color[3] * 255)),
                )
        
        return frame
    
    def view(self):
        from .extension import gui
        gui.run_frame_viewer_from_iterator(self, frame_rate=30)
