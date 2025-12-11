# 3D Event Stream Viewer

The 3D Event Stream Viewer provides a visualization of event streams where events are displayed in a 3D space with time as one of the axes.

## Overview

The viewer creates an isometric-style 3D visualization where:
- **X-axis** (horizontal): Spatial X coordinate of events
- **Y-axis** (vertical): Spatial Y coordinate of events  
- **Time-axis** (depth): Time, with older events appearing further back

As new events arrive, older events fade and are eventually removed from the display based on a configurable time window.

## Usage

### Basic Example

```python
import faery

# Load an event stream and view it in 3D
(
    faery.events_stream_from_file("events.es")
    .view_3d(
        time_window_us=200000,  # 200ms time window
        time_slices=40,          # Number of time slices for depth
        colormap=faery.colormaps.managua,
    )
    .view()
)
```

### Parameters

- **time_window_us** (int, default: 1000000): Time window in microseconds. Events older than this are removed from the display.
- **time_slices** (int, default: 50): Number of time slices used to represent the depth axis. Higher values create deeper 3D effect.
- **colormap** (Colormap, optional): Colormap for visualizing event polarity. If not specified, uses `managua` colormap.

### Integration with Event Streams

The 3D viewer integrates seamlessly with the Faery event stream pipeline:

```python
import faery

(
    faery.events_stream_from_file("recording.aedat4")
    .crop(left=50, right=250, top=30, bottom=200)  # Crop to region of interest
    .filter_hot_pixels(maximum_relative_event_count=0.01)  # Remove noise
    .view_3d(time_window_us=150000, time_slices=30)
    .view()
)
```

## How It Works

1. **Event Collection**: Events are collected in a time-windowed buffer (using `deque`)
2. **3D Projection**: Each event is projected to screen coordinates using:
   - `screen_x = x + (age_normalized * time_slices * 0.5)`
   - `screen_y = y + (age_normalized * time_slices * 0.3)`
3. **Coloring**: Events are colored based on:
   - **Polarity**: ON events use one half of the colormap, OFF events use the other
   - **Age**: Newer events are brighter, older events fade (alpha = 1.0 - age * 0.7)
4. **Rendering**: Events are rendered as pixels with alpha blending

## Colormap Support

The 3D viewer supports all Faery colormaps:

- **Sequential**: Single-color scale (same color for both polarities)
- **Diverging**: Two-color scale (different colors for ON/OFF events)
- **Cyclic**: Circular color scale

Example with different colormaps:

```python
# Use a diverging colormap to distinguish ON/OFF events
stream.view_3d(colormap=faery.colormaps.managua)

# Use a sequential colormap for single-color visualization
stream.view_3d(colormap=faery.colormaps.viridis)
```

## Performance Notes

- The current implementation is pure Python for simplicity and compatibility
- For high-throughput event streams, consider:
  - Reducing `time_window_us` to keep fewer events in memory
  - Reducing `time_slices` for faster rendering
  - Using `.regularize()` before viewing to control frame rate

## Future Enhancements

Planned improvements include:
- Native Rust/Slint implementation for better performance
- Mouse-based camera rotation
- Adjustable viewing angle
- Real-time parameter adjustment
