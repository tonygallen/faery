use numpy::Element;
use numpy::PyArrayMethods;
use pyo3::prelude::*;

use crate::types;
use crate::utilities;

#[derive(Clone, Copy)]
enum Decay {
    None,
    Linear { slope: f32 },
    Exponential { tau: f64 },
    Step,
}

struct Inner {
    dimensions: (u16, u16),
    decay: Decay,
    min_potential: f32,
    max_potential: f32,
    neutral_potential: f32,
    event_contribution: f32,
    ignore_polarity: bool,
    synchronous_decay: bool,
    colormap: Vec<u32>,
    potentials: Vec<f32>,
    last_update_ts: Vec<Option<u64>>,
    origin_t: Option<u64>,
    current_t: Option<u64>,
}

impl Inner {
    fn decay_potential(&self, potential: f32, delta_t: u64) -> f32 {
        match self.decay {
            Decay::None | Decay::Step => potential,
            Decay::Linear { slope } => {
                let amount = delta_t as f32 * slope;
                if potential > self.neutral_potential {
                    (potential - amount).max(self.neutral_potential)
                } else if potential < self.neutral_potential {
                    (potential + amount).min(self.neutral_potential)
                } else {
                    potential
                }
            }
            // dv-processing's exponential accumulator decays toward zero; its
            // neutral potential is intentionally ignored for this decay mode.
            Decay::Exponential { tau } => potential * (-(delta_t as f64) / tau).exp() as f32,
        }
    }

    fn decay_pixel_to(&mut self, index: usize, t: u64) {
        let previous_t = self.last_update_ts[index].unwrap_or_else(|| {
            self.origin_t
                .expect("origin_t is initialized before pixels are decayed")
        });
        self.potentials[index] = self
            .decay_potential(self.potentials[index], t - previous_t)
            .clamp(self.min_potential, self.max_potential);
        self.last_update_ts[index] = Some(t);
    }

    fn accept(
        &mut self,
        python: Python<'_>,
        events: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let (array, length) = types::check_array(python, types::ArrayType::Dvs, events)?;
        for event_index in 0..length {
            let (t, x, y, polarity) = unsafe {
                let event_cell: *mut neuromorphic_types::PolarityEvent<u64, u16, u16> =
                    types::array_at(python, array, event_index);
                (
                    (*event_cell).t,
                    (*event_cell).x,
                    (*event_cell).y,
                    (*event_cell).polarity,
                )
            };
            if let Some(previous_t) = self.current_t {
                if t < previous_t {
                    return Err(utilities::WriteError::NonMonotonic { previous_t, t }.into());
                }
            }
            if x >= self.dimensions.0 {
                return Err(utilities::WriteError::XOverflow {
                    x,
                    width: self.dimensions.0,
                }
                .into());
            }
            if y >= self.dimensions.1 {
                return Err(utilities::WriteError::YOverflow {
                    y,
                    height: self.dimensions.1,
                }
                .into());
            }
            if self.origin_t.is_none() {
                self.origin_t = Some(t);
            }
            let pixel_index = y as usize * self.dimensions.0 as usize + x as usize;
            self.decay_pixel_to(pixel_index, t);
            let contribution =
                if self.ignore_polarity || matches!(polarity, neuromorphic_types::Polarity::On) {
                    self.event_contribution
                } else {
                    -self.event_contribution
                };
            self.potentials[pixel_index] = (self.potentials[pixel_index] + contribution)
                .clamp(self.min_potential, self.max_potential);
            self.current_t = Some(t);
        }
        Ok(())
    }

    fn generate_frame(&mut self, python: Python<'_>, frame_t: u64) -> PyResult<PyObject> {
        if let Some(previous_t) = self.current_t {
            if frame_t < previous_t {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "frame_t cannot be smaller than the accumulator's current timestamp (frame_t is {frame_t}, current timestamp is {previous_t})"
                )));
            }
        }
        if self.origin_t.is_none() {
            self.origin_t = Some(frame_t);
        }
        if self.synchronous_decay
            && matches!(self.decay, Decay::Linear { .. } | Decay::Exponential { .. })
        {
            for pixel_index in 0..self.potentials.len() {
                self.decay_pixel_to(pixel_index, frame_t);
            }
        }
        self.current_t = Some(frame_t);

        let array = unsafe {
            let mut dimensions = [
                self.dimensions.1 as numpy::npyffi::npy_intp,
                self.dimensions.0 as numpy::npyffi::npy_intp,
                4,
            ];
            numpy::PY_ARRAY_API.PyArray_Empty(
                python,
                3,
                dimensions.as_mut_ptr(),
                u8::get_dtype(python).into_ptr() as *mut numpy::npyffi::PyArray_Descr,
                0,
            )
        } as *mut numpy::npyffi::PyArrayObject;
        let output = {
            let mut index = [
                0 as numpy::npyffi::npy_intp,
                0 as numpy::npyffi::npy_intp,
                0 as numpy::npyffi::npy_intp,
            ];
            unsafe {
                std::slice::from_raw_parts_mut(
                    numpy::PY_ARRAY_API.PyArray_GetPtr(python, array, index.as_mut_ptr())
                        as *mut u32,
                    self.potentials.len(),
                )
            }
        };
        let scale = (self.colormap.len() - 1) as f32 / (self.max_potential - self.min_potential);
        for (potential, color) in self.potentials.iter().zip(output.iter_mut()) {
            let colormap_index = ((*potential - self.min_potential) * scale)
                .round()
                .clamp(0.0, (self.colormap.len() - 1) as f32)
                as usize;
            *color = self.colormap[colormap_index];
        }

        if matches!(self.decay, Decay::Step) {
            self.potentials.fill(self.neutral_potential);
            self.last_update_ts.fill(Some(frame_t));
        }

        Ok(unsafe { PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject) })
    }
}

#[pyclass]
pub struct Accumulator {
    inner: Option<Inner>,
}

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

fn parse_colormap(
    colormap_rgba: Option<&pyo3::Bound<'_, numpy::PyArray2<f64>>>,
) -> PyResult<Vec<u32>> {
    let Some(colormap_rgba) = colormap_rgba else {
        return Ok((0..=u8::MAX)
            .map(|intensity| u32::from_ne_bytes([intensity, intensity, intensity, 255]))
            .collect());
    };
    let readonly_colormap = colormap_rgba.readonly();
    let colormap_array = readonly_colormap.as_array();
    let dimensions = colormap_array.dim();
    if dimensions.0 == 0 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "colormap_rgba must be an array with at least one row (got a {} x {} array)",
            dimensions.0, dimensions.1,
        )));
    }
    if dimensions.1 != 4 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "colormap_rgba must be an array whose last dimension is 4 (RGBA) (got a {} x {} array)",
            dimensions.0, dimensions.1,
        )));
    }
    Ok((0..dimensions.0)
        .map(|index| {
            u32::from_ne_bytes([
                component_f64_to_u8(colormap_array[(index, 0)]),
                component_f64_to_u8(colormap_array[(index, 1)]),
                component_f64_to_u8(colormap_array[(index, 2)]),
                component_f64_to_u8(colormap_array[(index, 3)]),
            ])
        })
        .collect())
}

#[pymethods]
impl Accumulator {
    #[new]
    #[pyo3(signature = (
        dimensions,
        decay="exponential",
        decay_param=1_000_000.0,
        min_potential=0.0,
        max_potential=1.0,
        neutral_potential=0.5,
        event_contribution=0.15,
        ignore_polarity=false,
        synchronous_decay=false,
        colormap_rgba=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dimensions: (u16, u16),
        decay: &str,
        decay_param: f64,
        min_potential: f32,
        max_potential: f32,
        neutral_potential: f32,
        event_contribution: f32,
        ignore_polarity: bool,
        synchronous_decay: bool,
        colormap_rgba: Option<&pyo3::Bound<'_, numpy::PyArray2<f64>>>,
    ) -> PyResult<Self> {
        if dimensions.0 == 0 || dimensions.1 == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimensions must both be positive (got {} x {})",
                dimensions.0, dimensions.1
            )));
        }
        if !min_potential.is_finite()
            || !max_potential.is_finite()
            || min_potential >= max_potential
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "min_potential must be finite and smaller than max_potential (got {min_potential} and {max_potential})"
            )));
        }
        if !neutral_potential.is_finite()
            || neutral_potential < min_potential
            || neutral_potential > max_potential
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "neutral_potential must be within [min_potential, max_potential] (got {neutral_potential})"
            )));
        }
        if !event_contribution.is_finite() || event_contribution < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "event_contribution must be finite and non-negative (got {event_contribution})"
            )));
        }
        let decay = match decay {
            "none" => Decay::None,
            "linear" => {
                if !decay_param.is_finite() || decay_param < 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "decay_param must be finite and non-negative for linear decay (got {decay_param})"
                    )));
                }
                Decay::Linear {
                    slope: decay_param as f32,
                }
            }
            "exponential" => {
                if !decay_param.is_finite() || decay_param <= 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "decay_param must be finite and positive for exponential decay (got {decay_param})"
                    )));
                }
                Decay::Exponential { tau: decay_param }
            }
            "step" => Decay::Step,
            value => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown decay \"{value}\" (expected \"none\", \"linear\", \"exponential\", or \"step\")"
                )))
            }
        };
        let pixel_count = dimensions.0 as usize * dimensions.1 as usize;
        Ok(Self {
            inner: Some(Inner {
                dimensions,
                decay,
                min_potential,
                max_potential,
                neutral_potential,
                event_contribution,
                ignore_polarity,
                synchronous_decay,
                colormap: parse_colormap(colormap_rgba)?,
                potentials: vec![neutral_potential; pixel_count],
                last_update_ts: vec![None; pixel_count],
                origin_t: None,
                current_t: None,
            }),
        })
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exception_type, _value, _traceback))]
    fn __exit__(
        &mut self,
        _exception_type: Option<PyObject>,
        _value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        if self.inner.is_none() {
            return Err(pyo3::exceptions::PyException::new_err(
                "multiple calls to __exit__",
            ));
        }
        let _ = self.inner.take();
        Ok(false)
    }

    fn accept(&mut self, events: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        Python::with_gil(|python| match self.inner.as_mut() {
            Some(accumulator) => accumulator.accept(python, events),
            None => Err(pyo3::exceptions::PyException::new_err(
                "accept called after __exit__",
            )),
        })
    }

    #[pyo3(signature = (frame_t=None))]
    fn generate_frame(&mut self, frame_t: Option<u64>) -> PyResult<PyObject> {
        Python::with_gil(|python| match self.inner.as_mut() {
            Some(accumulator) => {
                accumulator.generate_frame(python, frame_t.or(accumulator.current_t).unwrap_or(0))
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "generate_frame called after __exit__",
            )),
        })
    }

    fn accumulate(
        &mut self,
        events: &pyo3::Bound<'_, pyo3::types::PyAny>,
        frame_t: u64,
    ) -> PyResult<PyObject> {
        Python::with_gil(|python| match self.inner.as_mut() {
            Some(accumulator) => {
                accumulator.accept(python, events)?;
                accumulator.generate_frame(python, frame_t)
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "accumulate called after __exit__",
            )),
        })
    }

    fn clear(&mut self) -> PyResult<()> {
        match self.inner.as_mut() {
            Some(accumulator) => {
                accumulator.potentials.fill(accumulator.neutral_potential);
                Ok(())
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "clear called after __exit__",
            )),
        }
    }
}
