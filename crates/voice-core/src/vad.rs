//! Voice activity: Silero VAD v5 on the CPU, 32 ms frames at 16 kHz.

use crate::onnx;
use anyhow::{Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// Samples per frame (32 ms).
pub const FRAME: usize = 512;
/// The model wants the end of the previous frame in front of each frame.
const CONTEXT: usize = 64;
const STATE: usize = 2 * 128;

pub struct Vad {
    session: Session,
    state: Vec<f32>,
    context: [f32; CONTEXT],
}

impl Vad {
    pub fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            session: onnx::session(path, 1)?,
            state: vec![0.0; STATE],
            context: [0.0; CONTEXT],
        })
    }

    pub fn reset(&mut self) {
        self.state = vec![0.0; STATE];
        self.context = [0.0; CONTEXT];
    }

    /// The chance (0..1) that `frame` ([`FRAME`] samples) is speech.
    pub fn speech(&mut self, frame: &[i16]) -> Result<f32> {
        if frame.len() != FRAME {
            return Err(anyhow!("a VAD frame is {FRAME} samples, not {}", frame.len()));
        }
        let mut input = Vec::with_capacity(CONTEXT + FRAME);
        input.extend_from_slice(&self.context);
        input.extend(frame.iter().map(|&s| s as f32 / 32768.0));
        self.context.copy_from_slice(&input[input.len() - CONTEXT..]);
        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array(([1usize, CONTEXT + FRAME], input))?,
            "state" => Tensor::from_array(([2usize, 1, 128], self.state.clone()))?,
            "sr" => Tensor::from_array(([1usize], vec![16000i64]))?,
        ])?;
        let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
        let (_, state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(state);
        prob.first().copied().ok_or_else(|| anyhow!("the VAD gave no value"))
    }
}
