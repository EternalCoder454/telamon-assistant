//! Telamon's voice pipeline, with nothing of Qt in it: the microphone (or a
//! WAV file) goes through the wake word, voice detection, speech to text, the
//! model with its read-only tools, and speech, and the app hears about each
//! step as an [`Event`]. See `docs/DESIGN.md`.

pub mod audio;
pub mod config;
pub mod llm;
pub mod onnx;
pub mod pipeline;
pub mod server;
pub mod stt;
pub mod tools;
pub mod tts;
pub mod vad;
pub mod wake;

pub use config::Config;
pub use pipeline::{Assistant, Event, Phase};
