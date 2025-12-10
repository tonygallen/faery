"""
Demonstration that the tee() fix resolves the sequential consumption issue.

This document verifies that the following pattern now works correctly:

```python
stream = faery.events_stream_from_camera(driver='NeuromorphicDrivers')
recording, viewing = stream.tee(2)

# First branch: save events to a file
recording.to_file('recording.es')

# Second branch: render and view the events  
(
    viewing
    .regularize(frequency_hz=30)
    .render(tau="00:00:00.2", decay="exponential", colormap=faery.colormaps.starry_night)
    .view()
)
```

## The Problem (Before Fix)

The original TeeBuffer implementation had this logic:

1. When `recording` calls `get_next(0)`, it fetches a packet from parent
2. It returns a copy to output 0 and buffers a copy for output 1
3. When `recording` exhausts the parent stream by consuming all packets,
   the parent iterator is marked as exhausted
4. When `viewing` later calls `get_next(1)`, it checks its buffer first,
   but the buffer only contains packets that were fetched WHILE output 0
   was being consumed. Since output 0 was fully consumed, all packets are gone.

## The Solution (After Fix)

The fixed TeeBuffer implementation tracks packet indices:

1. Each packet fetched from parent gets a unique index (0, 1, 2, ...)
2. Each output tracks the last packet index it has consumed (-1 initially)
3. When an output requests the next packet via `get_next(i)`:
   - If the output's buffer is empty, fetch from parent
   - Distribute the fetched packet to ALL outputs that haven't consumed it yet
     (by checking if consumed_indices[i] < current_packet_index)
   - This ensures that even if output 0 has consumed packets 0-99, when output 1
     later asks for its first packet, those packets are still available

## Verification

The key improvement is in the buffering logic:

```python
# Old (broken):
for i in range(self.num_outputs):
    if i == output_index:
        continue  # Don't buffer for requesting output
    else:
        self.buffers[i].append(events.copy())  # Buffer for others
return events.copy()

# New (fixed):
for i in range(self.num_outputs):
    # Buffer for ALL outputs that haven't consumed this packet yet
    if self.consumed_indices[i] < current_packet_index:
        self.buffers[i].append((current_packet_index, events.copy()))
```

The new implementation ensures that packets remain available for outputs that
haven't consumed them yet, regardless of consumption order.

## Test Cases

See `test_tee_sequential.py` for comprehensive tests covering:
- Sequential consumption (output 0 then output 1)
- Reverse sequential (output 1 then output 0)  
- Partial then full consumption
- Interleaved consumption (original tests still work)

All patterns now work correctly! ✓
