use std::io::Read;
use std::io::Seek;

use crate::evt::common;
use crate::utilities;

enum State {
    Evt2 {
        t: u64,
        t_high: u64,
        t_offset: u64,
        initialized: bool,
        t0: u64,
    },
    Evt21 {},
    Evt3 {
        t: u64,
        overflows: u64,
        previous_msb_t: u16,
        previous_lsb_t: u16,
        initialized: bool,
        x: u16,
        y: u16,
        t0: u64,
    },
}

pub struct Decoder {
    pub dimensions: (u16, u16),
    path: std::path::PathBuf,
    data_start: u64,
    t0: u64,
    file: std::fs::File,
    raw_buffer: Vec<u8>,
    event_buffer: Vec<neuromorphic_types::PolarityEvent<u64, u16, u16>>,
    trigger_buffer: Vec<neuromorphic_types::TriggerEvent<u64, u8>>,
    state: State,
    polarity: neuromorphic_types::Polarity,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Version(#[from] common::Error),

    #[error(
        "the header has no size information (width and height) and no size fallback was provided"
    )]
    MissingSize,

    #[error("the header has no version information and no versions fallback was provided")]
    MissingVersion,

    #[error("unknown version \"{0}\" (supports \"evt2\", \"evt2.1\", and \"evt3\")")]
    UnknownVersion(String),

    #[error("truncated EVT word (expected a multiple of {word_size} bytes, got {trailing_bytes} trailing bytes)")]
    TruncatedWord {
        word_size: usize,
        trailing_bytes: usize,
    },

    #[error("timestamp overflow")]
    TimestampOverflow,
}

impl Decoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        dimensions_fallback: Option<(u16, u16)>,
        version_fallback: Option<common::Version>,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        let header = utilities::read_prophesee_header(
            &mut std::io::BufReader::new(std::fs::File::open(path)?),
            '%',
        )?;
        let dimensions = match header.dimensions {
            Some(dimensions) => dimensions,
            None => match dimensions_fallback {
                Some(size) => size,
                None => return Err(Error::MissingSize),
            },
        };
        let mut file = std::fs::File::open(path)?;
        file.seek(std::io::SeekFrom::Start(header.length))?;
        let version = match header.version {
            Some(version) => match version.as_str() {
                "2" => common::Version::Evt2,
                "2.1" => common::Version::Evt21,
                "3" => common::Version::Evt3,
                _ => return Err(Error::UnknownVersion(version)),
            },
            None => match version_fallback {
                Some(version) => version,
                None => return Err(Error::MissingVersion),
            },
        };
        Ok(Decoder {
            dimensions,
            path: path.to_path_buf(),
            data_start: header.length,
            t0: header.t0,
            file,
            raw_buffer: vec![0u8; utilities::BUFFER_SIZE],
            event_buffer: Vec::new(),
            trigger_buffer: Vec::new(),
            state: match version {
                common::Version::Evt2 => State::Evt2 {
                    t: 0,
                    t_high: 0,
                    t_offset: 0,
                    initialized: false,
                    t0: header.t0,
                },
                common::Version::Evt21 => State::Evt21 {},
                common::Version::Evt3 => State::Evt3 {
                    t: 0,
                    overflows: 0,
                    previous_msb_t: 0,
                    previous_lsb_t: 0,
                    initialized: false,
                    x: 0,
                    y: 0,
                    t0: header.t0,
                },
            },
            polarity: neuromorphic_types::Polarity::Off,
        })
    }

    pub fn version(&self) -> common::Version {
        match self.state {
            State::Evt2 { .. } => common::Version::Evt2,
            State::Evt21 { .. } => common::Version::Evt21,
            State::Evt3 { .. } => common::Version::Evt3,
        }
    }

    /// Return the minimum and one-past-maximum polarity-event timestamps.
    ///
    /// This scans only timestamp and event-marker words. It deliberately does
    /// not allocate decoded events or validate coordinates, making it much
    /// cheaper than iterating through the regular decoder. A scan is still
    /// required because EVT timestamps wrap and the format has no duration
    /// footer.
    pub fn time_range(&self) -> Result<Option<(u64, u64)>, Error> {
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(std::io::SeekFrom::Start(self.data_start))?;
        let mut bounds = TimeBounds::default();

        match self.version() {
            common::Version::Evt2 => {
                let mut t_high = 0u64;
                let mut t_offset = 0u64;
                let mut initialized = false;
                for_each_word::<4>(&mut file, |bytes| {
                    let word = u32::from_le_bytes(*bytes);
                    match word >> 28 {
                        0b0000 | 0b0001 if initialized => {
                            let t =
                                t_high + ((word & 0b1111110000000000000000000000_u32) >> 22) as u64;
                            bounds.note_with_offset(t, self.t0)?;
                        }
                        0b1000 => update_evt2_time_high(
                            ((word & 0xFFFFFFF_u32) as u64) << 6,
                            &mut t_high,
                            &mut t_offset,
                            &mut initialized,
                        ),
                        _ => (),
                    }
                    Ok(())
                })?;
            }
            common::Version::Evt21 => {
                let mut layout = None;
                let mut t_high = 0u64;
                let mut t_offset = 0u64;
                let mut initialized = false;
                for_each_word::<8>(&mut file, |bytes| {
                    let Some(word) = decode_evt21_word(bytes, &mut layout) else {
                        return Ok(());
                    };
                    match word >> 60 {
                        0b0000 | 0b0001 if initialized => {
                            let x = ((word >> 43) & 0x7FF) as u16;
                            let width = self.dimensions.0.saturating_sub(x).min(32);
                            let mask = match width {
                                0 => 0,
                                32 => u32::MAX,
                                width => (1u32 << width) - 1,
                            };
                            if (word as u32) & mask != 0 {
                                let t = t_high + ((word >> 54) & 0x3F);
                                bounds.note_with_offset(t, self.t0)?;
                            }
                        }
                        0b1000 => update_evt2_time_high(
                            ((word >> 32) & 0xFFFFFFF) << 6,
                            &mut t_high,
                            &mut t_offset,
                            &mut initialized,
                        ),
                        _ => (),
                    }
                    Ok(())
                })?;
            }
            common::Version::Evt3 => {
                let mut t = 0u64;
                let mut overflows = 0u64;
                let mut previous_msb_t = 0u16;
                let mut previous_lsb_t = 0u16;
                let mut initialized = false;
                let mut x = 0u16;
                for_each_word::<2>(&mut file, |bytes| {
                    let word = u16::from_le_bytes(*bytes);
                    let word_type = word >> 12;
                    if !initialized && word_type != 0b1000 {
                        return Ok(());
                    }
                    match word_type {
                        0b0010 => bounds.note_with_offset(t, self.t0)?,
                        0b0011 => x = word & 0b11111111111,
                        0b0100 => {
                            let width = self.dimensions.0.saturating_sub(x).min(12);
                            let mask = if width == 0 { 0 } else { (1u16 << width) - 1 };
                            if word & mask != 0 {
                                bounds.note_with_offset(t, self.t0)?;
                            }
                            x = x
                                .saturating_add(12)
                                .min(self.dimensions.0.saturating_sub(1));
                        }
                        0b0101 => {
                            let width = self.dimensions.0.saturating_sub(x).min(8);
                            let mask = if width == 0 { 0 } else { (1u16 << width) - 1 };
                            if word & mask != 0 {
                                bounds.note_with_offset(t, self.t0)?;
                            }
                            x = x.saturating_add(8).min(self.dimensions.0.saturating_sub(1));
                        }
                        0b0110 => update_evt3_time_low(
                            word & 0xFFF,
                            &mut t,
                            overflows,
                            previous_msb_t,
                            &mut previous_lsb_t,
                        ),
                        0b1000 => update_evt3_time_high(
                            word & 0xFFF,
                            &mut t,
                            &mut overflows,
                            &mut previous_msb_t,
                            &mut previous_lsb_t,
                            &mut initialized,
                        ),
                        _ => (),
                    }
                    Ok(())
                })?;
            }
        }

        bounds.into_range()
    }

    #[allow(clippy::type_complexity)]
    pub fn next(
        &mut self,
    ) -> Result<
        Option<(
            &Vec<neuromorphic_types::PolarityEvent<u64, u16, u16>>,
            &Vec<neuromorphic_types::TriggerEvent<u64, u8>>,
        )>,
        utilities::ReadError,
    > {
        let read = self.file.read(&mut self.raw_buffer)?;
        if read == 0 {
            return Ok(None);
        }
        self.event_buffer.clear();
        self.trigger_buffer.clear();
        match self.state {
            State::Evt2 {
                ref mut t,
                ref mut t_high,
                ref mut t_offset,
                ref mut initialized,
                t0,
            } => {
                for index in 0..read / 4 {
                    let word = u32::from_le_bytes(
                        self.raw_buffer[index * 4..(index + 1) * 4]
                            .try_into()
                            .expect("4 bytes"),
                    );
                    let word_type = word >> 28;
                    if !*initialized && word_type != 0b1000 {
                        continue;
                    }
                    match word_type {
                        0b0000 | 0b0001 => {
                            *t = *t_high
                                + ((word & 0b1111110000000000000000000000_u32) >> 22) as u64;
                            let x = ((word & 0b1111111111100000000000_u32) >> 11) as u16;
                            if x >= self.dimensions.0 {
                                return Err(utilities::ReadError::XOverflow {
                                    x,
                                    width: self.dimensions.0,
                                });
                            }
                            let y = (word & 0b0000000000011111111111_u32) as u16;
                            if y >= self.dimensions.1 {
                                return Err(utilities::ReadError::YOverflow {
                                    y,
                                    height: self.dimensions.1,
                                });
                            }
                            self.event_buffer.push(neuromorphic_types::PolarityEvent {
                                t: *t + t0,
                                x,
                                y,
                                polarity: if (word >> 28) & 0b1 > 0 {
                                    neuromorphic_types::Polarity::On
                                } else {
                                    neuromorphic_types::Polarity::Off
                                },
                            });
                        }
                        0b1000 => update_evt2_time_high(
                            ((word & 0xFFFFFFF_u32) as u64) << 6,
                            t_high,
                            t_offset,
                            initialized,
                        ),
                        0b1010 => {
                            *t = *t_high
                                + ((word & 0b1111110000000000000000000000_u32) >> 22) as u64;
                            self.trigger_buffer.push(neuromorphic_types::TriggerEvent {
                                t: *t + t0,
                                id: ((word & 0b1111100000000) >> 8) as u8,
                                polarity: if (word & 1) > 0 {
                                    neuromorphic_types::TriggerPolarity::Rising
                                } else {
                                    neuromorphic_types::TriggerPolarity::Falling
                                },
                            })
                        }
                        #[allow(clippy::manual_range_patterns)]
                        0b1110 | 0b1111 => (),
                        _ => (),
                    }
                }
            }
            State::Evt21 {} => {
                todo!("implement EVT2.1 (example / test data needed)");
            }
            State::Evt3 {
                ref mut t,
                ref mut overflows,
                ref mut previous_msb_t,
                ref mut previous_lsb_t,
                ref mut initialized,
                ref mut x,
                ref mut y,
                t0,
            } => {
                for index in 0..read / 2 {
                    let word = u16::from_le_bytes([
                        self.raw_buffer[index * 2],
                        self.raw_buffer[index * 2 + 1],
                    ]);
                    let word_type = word >> 12;
                    if !*initialized && word_type != 0b1000 {
                        continue;
                    }
                    match word_type {
                        0b0000 => {
                            let candidate_y = word & 0b11111111111;
                            if candidate_y < self.dimensions.1 {
                                *y = candidate_y;
                            } else {
                                return Err(utilities::ReadError::YOverflow {
                                    y: candidate_y,
                                    height: self.dimensions.1,
                                });
                            }
                        }
                        0b0001 => (),
                        0b0010 => {
                            let candidate_x = word & 0b11111111111;
                            if candidate_x < self.dimensions.0 {
                                *x = candidate_x;
                            } else {
                                return Err(utilities::ReadError::XOverflow {
                                    x: candidate_x,
                                    width: self.dimensions.0,
                                });
                            }
                            self.polarity = if (word & (1 << 11)) > 0 {
                                neuromorphic_types::Polarity::On
                            } else {
                                neuromorphic_types::Polarity::Off
                            };
                            self.event_buffer.push(neuromorphic_types::PolarityEvent {
                                t: *t + t0,
                                x: *x,
                                y: *y,
                                polarity: self.polarity,
                            });
                        }
                        0b0011 => {
                            let candidate_x = word & 0b11111111111;
                            if candidate_x < self.dimensions.0 {
                                *x = candidate_x;
                            } else {
                                return Err(utilities::ReadError::XOverflow {
                                    x: candidate_x,
                                    width: self.dimensions.0,
                                });
                            }
                            self.polarity = if (word & (1 << 11)) > 0 {
                                neuromorphic_types::Polarity::On
                            } else {
                                neuromorphic_types::Polarity::Off
                            };
                        }
                        0b0100 => {
                            let set = word & ((1 << std::cmp::min(12, self.dimensions.0 - *x)) - 1);
                            for bit in 0..12 {
                                if (set & (1 << bit)) > 0 {
                                    self.event_buffer.push(neuromorphic_types::PolarityEvent {
                                        t: *t + t0,
                                        x: *x + bit,
                                        y: *y,
                                        polarity: self.polarity,
                                    });
                                }
                            }
                            *x = (*x + 12).min(self.dimensions.0 - 1);
                        }
                        0b0101 => {
                            let set = word & ((1 << std::cmp::min(8, self.dimensions.0 - *x)) - 1);
                            for bit in 0..8 {
                                if (set & (1 << bit)) > 0 {
                                    self.event_buffer.push(neuromorphic_types::PolarityEvent {
                                        t: *t + t0,
                                        x: *x + bit,
                                        y: *y,
                                        polarity: self.polarity,
                                    });
                                }
                            }
                            *x = (*x + 8).min(self.dimensions.0 - 1);
                        }
                        0b0110 => {
                            update_evt3_time_low(
                                word & 0xFFF,
                                t,
                                *overflows,
                                *previous_msb_t,
                                previous_lsb_t,
                            );
                        }
                        0b0111 => (),
                        0b1000 => update_evt3_time_high(
                            word & 0xFFF,
                            t,
                            overflows,
                            previous_msb_t,
                            previous_lsb_t,
                            initialized,
                        ),
                        0b1001 => (),
                        0b1010 => self.trigger_buffer.push(neuromorphic_types::TriggerEvent {
                            t: *t + t0,
                            id: ((word >> 8) & 0b1111) as u8,
                            polarity: if (word & 1) > 0 {
                                neuromorphic_types::TriggerPolarity::Rising
                            } else {
                                neuromorphic_types::TriggerPolarity::Falling
                            },
                        }),
                        #[allow(clippy::manual_range_patterns)]
                        0b1011 | 0b1100 | 0b1101 | 0b1110 | 0b1111 => (),
                        _ => (),
                    }
                }
            }
        }
        Ok(Some((&self.event_buffer, &self.trigger_buffer)))
    }
}

const EVT2_TIME_LOOP: u64 = 1 << 34;
const EVT2_MAXIMUM_TIME_BASE: u64 = EVT2_TIME_LOOP - (1 << 6);
const EVT2_LOOP_THRESHOLD: u64 = 10 << 6;
const EVT3_TIME_LOOP: u64 = 1 << 24;
const EVT3_MAXIMUM_TIME_BASE: u64 = EVT3_TIME_LOOP - (1 << 12);
const EVT3_LOOP_THRESHOLD: u64 = 10 << 12;

#[derive(Default)]
struct TimeBounds {
    minimum: Option<u64>,
    maximum: Option<u64>,
}

impl TimeBounds {
    fn note_with_offset(&mut self, t: u64, t0: u64) -> Result<(), Error> {
        let t = t.checked_add(t0).ok_or(Error::TimestampOverflow)?;
        self.minimum = Some(self.minimum.map_or(t, |minimum| minimum.min(t)));
        self.maximum = Some(self.maximum.map_or(t, |maximum| maximum.max(t)));
        Ok(())
    }

    fn into_range(self) -> Result<Option<(u64, u64)>, Error> {
        match (self.minimum, self.maximum) {
            (Some(minimum), Some(maximum)) => Ok(Some((
                minimum,
                maximum.checked_add(1).ok_or(Error::TimestampOverflow)?,
            ))),
            _ => Ok(None),
        }
    }
}

fn for_each_word<const WORD_SIZE: usize>(
    file: &mut std::fs::File,
    mut callback: impl FnMut(&[u8; WORD_SIZE]) -> Result<(), Error>,
) -> Result<(), Error> {
    assert!(WORD_SIZE > 0);
    let mut buffer = vec![0u8; utilities::BUFFER_SIZE + WORD_SIZE - 1];
    let mut pending = 0usize;
    loop {
        let read = file.read(&mut buffer[pending..pending + utilities::BUFFER_SIZE])?;
        if read == 0 {
            break;
        }
        let length = pending + read;
        let complete_length = length / WORD_SIZE * WORD_SIZE;
        for bytes in buffer[..complete_length].chunks_exact(WORD_SIZE) {
            callback(bytes.try_into().expect("a complete EVT word"))?;
        }
        pending = length - complete_length;
        if pending > 0 {
            buffer.copy_within(complete_length..length, 0);
        }
    }
    if pending > 0 {
        return Err(Error::TruncatedWord {
            word_size: WORD_SIZE,
            trailing_bytes: pending,
        });
    }
    Ok(())
}

fn update_evt2_time_high(
    raw_time_base: u64,
    current_time_base: &mut u64,
    offset: &mut u64,
    initialized: &mut bool,
) {
    if !*initialized {
        *current_time_base = raw_time_base;
        *initialized = true;
        return;
    }
    let mut candidate = raw_time_base.saturating_add(*offset);
    if *current_time_base > candidate
        && *current_time_base - candidate >= EVT2_MAXIMUM_TIME_BASE - EVT2_LOOP_THRESHOLD
    {
        *offset = (*offset).saturating_add(EVT2_TIME_LOOP);
        candidate = raw_time_base.saturating_add(*offset);
    }
    *current_time_base = candidate;
}

fn update_evt3_time_low(
    lsb_t: u16,
    t: &mut u64,
    overflows: u64,
    previous_msb_t: u16,
    previous_lsb_t: &mut u16,
) {
    *previous_lsb_t = lsb_t;
    *t = (overflows << 24) | ((previous_msb_t as u64) << 12) | lsb_t as u64;
}

fn update_evt3_time_high(
    msb_t: u16,
    t: &mut u64,
    overflows: &mut u64,
    previous_msb_t: &mut u16,
    previous_lsb_t: &mut u16,
    initialized: &mut bool,
) {
    if *initialized {
        let current_time_base = (*overflows << 24) | ((*previous_msb_t as u64) << 12);
        let candidate_time_base = (*overflows << 24) | ((msb_t as u64) << 12);
        if current_time_base > candidate_time_base
            && current_time_base - candidate_time_base
                >= EVT3_MAXIMUM_TIME_BASE - EVT3_LOOP_THRESHOLD
        {
            *overflows = (*overflows).saturating_add(1);
        }
    } else {
        *initialized = true;
    }
    *previous_msb_t = msb_t;
    *previous_lsb_t = 0;
    *t = (*overflows << 24) | ((msb_t as u64) << 12);
}

#[derive(Clone, Copy)]
enum Evt21Layout {
    LittleEndian64,
    LittleEndian32,
    BigEndian64,
}

impl Evt21Layout {
    fn decode(self, bytes: &[u8; 8]) -> u64 {
        match self {
            Self::LittleEndian64 => u64::from_le_bytes(*bytes),
            Self::LittleEndian32 => u64::from_le_bytes(*bytes).rotate_left(32),
            Self::BigEndian64 => u64::from_be_bytes(*bytes),
        }
    }
}

fn decode_evt21_word(bytes: &[u8; 8], layout: &mut Option<Evt21Layout>) -> Option<u64> {
    if let Some(layout) = *layout {
        return Some(layout.decode(bytes));
    }
    for candidate in [
        Evt21Layout::LittleEndian64,
        Evt21Layout::LittleEndian32,
        Evt21Layout::BigEndian64,
    ] {
        let word = candidate.decode(bytes);
        if word >> 60 == 0b1000 && word & u32::MAX as u64 == 0 {
            *layout = Some(candidate);
            return Some(word);
        }
    }
    None
}
