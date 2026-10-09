//! The wake word, "Telamon": openWakeWord's streaming pipeline, ported. Each
//! 80 ms chunk of 16 kHz audio becomes mel frames (melspectrogram.onnx), the
//! last 76 frames an embedding (embedding_model.onnx), and the last 16
//! embeddings a score (our own telamon.onnx). `models/README.md` has the
//! algorithm this follows.

use crate::onnx;
use anyhow::{Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;

/// Samples per chunk (80 ms at 16 kHz).
pub const CHUNK: usize = 1280;
/// Extra samples fed to the mel model before each chunk (3 hops of 160).
const OVERLAP: usize = 480;
const MELS: usize = 32;
const WINDOW: usize = 76;
const EMBEDDING: usize = 96;
const HEAD: usize = 16;
/// Silent chunks pushed through after a reset, so the buffers hold what
/// training saw (3.2 s); their scores are dropped.
const PRIMING: usize = 40;

pub struct WakeWord {
    melspec: Session,
    embedding: Session,
    head: Session,
    raw: VecDeque<i16>,
    mel: VecDeque<[f32; MELS]>,
    embeddings: VecDeque<[f32; EMBEDDING]>,
}

impl WakeWord {
    pub fn new(models: &Path, head: &Path) -> Result<Self> {
        let mut wake = Self {
            melspec: onnx::session(&models.join("melspectrogram.onnx"), 1)?,
            embedding: onnx::session(&models.join("embedding_model.onnx"), 1)?,
            head: onnx::session(head, 1)?,
            raw: VecDeque::new(),
            mel: VecDeque::new(),
            embeddings: VecDeque::new(),
        };
        wake.reset()?;
        Ok(wake)
    }

    /// Forgets what was heard, as after a detection.
    pub fn reset(&mut self) -> Result<()> {
        self.raw.clear();
        self.mel.clear();
        self.embeddings.clear();
        let silence = [0i16; CHUNK];
        for _ in 0..PRIMING {
            self.process(&silence)?;
        }
        Ok(())
    }

    /// The wake score (0..1) after one chunk of [`CHUNK`] samples.
    pub fn process(&mut self, chunk: &[i16]) -> Result<f32> {
        if chunk.len() != CHUNK {
            return Err(anyhow!(
                "a wake word chunk is {CHUNK} samples, not {}",
                chunk.len()
            ));
        }
        self.raw.extend(chunk.iter().copied());
        while self.raw.len() > CHUNK + OVERLAP {
            self.raw.pop_front();
        }
        let frames = self.mel_frames()?;
        self.mel.extend(frames);
        while self.mel.len() > WINDOW {
            self.mel.pop_front();
        }
        if self.mel.len() < WINDOW {
            return Ok(0.0);
        }
        let embedding = self.embed()?;
        self.embeddings.push_back(embedding);
        while self.embeddings.len() > HEAD {
            self.embeddings.pop_front();
        }
        if self.mel.len() < WINDOW || self.embeddings.len() < HEAD {
            return Ok(0.0);
        }
        self.score()
    }

    fn mel_frames(&mut self) -> Result<Vec<[f32; MELS]>> {
        let input: Vec<f32> = self.raw.iter().map(|&s| s as f32).collect();
        let n = input.len();
        let outputs = self
            .melspec
            .run(ort::inputs![Tensor::from_array(([1usize, n], input))?])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data
            .as_chunks::<MELS>()
            .0
            .iter()
            .map(|row| row.map(|v| v / 10.0 + 2.0))
            .collect())
    }

    fn embed(&mut self) -> Result<[f32; EMBEDDING]> {
        let input: Vec<f32> = self.mel.iter().flatten().copied().collect();
        let outputs = self.embedding.run(ort::inputs![Tensor::from_array((
            [1usize, WINDOW, MELS, 1],
            input
        ))?])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        let mut embedding = [0.0; EMBEDDING];
        if data.len() != EMBEDDING {
            return Err(anyhow!("the embedding model gave {} values", data.len()));
        }
        embedding.copy_from_slice(data);
        Ok(embedding)
    }

    fn score(&mut self) -> Result<f32> {
        let input: Vec<f32> = self.embeddings.iter().flatten().copied().collect();
        let outputs = self.head.run(ort::inputs![Tensor::from_array((
            [1usize, HEAD, EMBEDDING],
            input
        ))?])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        data.first()
            .copied()
            .ok_or_else(|| anyhow!("the wake model gave no score"))
    }
}
