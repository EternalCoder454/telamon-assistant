//! Where the models are and how the pipeline behaves.

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    /// melspectrogram.onnx, embedding_model.onnx, telamon.onnx,
    /// silero_vad.onnx, ggml-base.en.bin, kokoro-v1.0.onnx, voices-v1.0.bin,
    /// kokoro-config.json (`scripts/fetch-models.sh`).
    pub models: PathBuf,
    /// The wake word head; `models/telamon.onnx` unless set.
    pub wake_model: Option<PathBuf>,
    /// The wake score (0..1) that wakes Telamon.
    pub wake_threshold: f32,
    /// Silence after speech that ends the question, in ms.
    pub end_silence_ms: u32,
    /// How long to wait for the question to start after the wake word, in ms.
    pub start_timeout_ms: u32,
    /// The longest question, in ms.
    pub max_question_ms: u32,
    /// The OpenAI-compatible llama-server, e.g. `http://127.0.0.1:8091`.
    pub llama_url: String,
    /// The Kokoro voice, e.g. `bm_george`.
    pub voice: String,
    /// Speech speed, 1.0 is normal.
    pub speed: f32,
    /// whisper.cpp on the GPU (Vulkan) when there is one.
    pub stt_gpu: bool,
    /// The place the weather is for, e.g. "London"; empty: from the time zone.
    pub location: String,
}

impl Config {
    pub fn new(models: PathBuf) -> Self {
        Self {
            models,
            wake_model: None,
            // models/telamon.json
            wake_threshold: 0.9,
            end_silence_ms: 700,
            start_timeout_ms: 4000,
            max_question_ms: 15000,
            llama_url: "http://127.0.0.1:8091".to_string(),
            voice: "bm_george".to_string(),
            speed: 1.0,
            stt_gpu: true,
            location: String::new(),
        }
    }

    pub fn model(&self, name: &str) -> PathBuf {
        self.models.join(name)
    }
}
