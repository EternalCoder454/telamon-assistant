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
/// Scores right after a reset mean nothing.
const WARMUP: usize = 5;

pub struct WakeWord {
    melspec: Session,
    embedding: Session,
    head: Session,
    raw: VecDeque<i16>,
    mel: VecDeque<[f32; MELS]>,
    embeddings: VecDeque<[f32; EMBEDDING]>,
    /// The embeddings of noise the buffer starts with after a reset.
    primed: Vec<[f32; EMBEDDING]>,
    chunks: usize,
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
            primed: Vec::new(),
            chunks: 0,
        };
        wake.primed = wake.noise_embeddings()?;
        wake.reset();
        Ok(wake)
    }

    /// Forgets what was heard, as after a detection.
    pub fn reset(&mut self) {
        self.raw.clear();
        self.mel.clear();
        self.mel.extend(std::iter::repeat_n([1.0; MELS], WINDOW));
        self.embeddings.clear();
        self.embeddings.extend(self.primed.iter().copied());
        self.chunks = 0;
    }

    /// The wake score (0..1) after one chunk of [`CHUNK`] samples.
    pub fn process(&mut self, chunk: &[i16]) -> Result<f32> {
        if chunk.len() != CHUNK {
            return Err(anyhow!("a wake word chunk is {CHUNK} samples, not {}", chunk.len()));
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
        let embedding = self.embed()?;
        self.embeddings.push_back(embedding);
        while self.embeddings.len() > HEAD {
            self.embeddings.pop_front();
        }
        self.chunks += 1;
        let score = self.score()?;
        Ok(if self.chunks <= WARMUP { 0.0 } else { score })
    }

    fn mel_frames(&mut self) -> Result<Vec<[f32; MELS]>> {
        let input: Vec<f32> = self.raw.iter().map(|&s| s as f32).collect();
        let n = input.len();
        let outputs = self
            .melspec
            .run(ort::inputs![Tensor::from_array(([1usize, n], input))?])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data
            .chunks_exact(MELS)
            .map(|row| {
                let mut frame = [0.0; MELS];
                for (f, v) in frame.iter_mut().zip(row) {
                    *f = v / 10.0 + 2.0;
                }
                frame
            })
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
        let outputs = self
            .head
            .run(ort::inputs![Tensor::from_array(([1usize, HEAD, EMBEDDING], input))?])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        data.first().copied().ok_or_else(|| anyhow!("the wake model gave no score"))
    }

    /// openWakeWord starts its embedding buffer from 4 s of quiet noise; so
    /// does this, with a fixed seed.
    fn noise_embeddings(&mut self) -> Result<Vec<[f32; EMBEDDING]>> {
        let mut seed: u32 = 0x5eed;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 16) as i32 % 2000 - 1000) as i16
        };
        self.raw.clear();
        self.mel.clear();
        self.mel.extend(std::iter::repeat_n([1.0; MELS], WINDOW));
        let mut out = Vec::new();
        for _ in 0..(16000 * 4 / CHUNK) {
            let chunk: Vec<i16> = (0..CHUNK).map(|_| noise()).collect();
            self.raw.extend(chunk);
            while self.raw.len() > CHUNK + OVERLAP {
                self.raw.pop_front();
            }
            let frames = self.mel_frames()?;
            self.mel.extend(frames);
            while self.mel.len() > WINDOW {
                self.mel.pop_front();
            }
            out.push(self.embed()?);
        }
        Ok(out.split_off(out.len().saturating_sub(HEAD)))
    }
}
