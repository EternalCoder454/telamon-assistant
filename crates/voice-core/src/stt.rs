//! Speech to text: whisper.cpp (Vulkan when built with it and a GPU is
//! there, else the CPU), the model loaded once.

use anyhow::{Context, Result, anyhow};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Stt {
    context: WhisperContext,
    threads: i32,
}

impl Stt {
    pub fn new(model: &Path, gpu: bool) -> Result<Self> {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(gpu);
        let path = model.to_str().ok_or_else(|| anyhow!("the model path is not UTF-8"))?;
        let context = WhisperContext::new_with_params(path, params)
            .with_context(|| format!("cannot load the speech model {}", model.display()))?;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as i32;
        Ok(Self { context, threads })
    }

    /// The text of `audio` (16 kHz mono).
    pub fn transcribe(&self, audio: &[i16]) -> Result<String> {
        let samples: Vec<f32> = audio.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut state = self.context.create_state().context("whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_n_threads(self.threads);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        // A hint, so "Telamon" and the tools' words come out spelled right.
        params.set_initial_prompt("Hey Telamon, what's my CPU usage?");
        state.full(params, &samples).context("whisper")?;
        let mut text = String::new();
        for segment in state.as_iter() {
            text.push_str(&segment.to_string());
        }
        Ok(clean(&text))
    }
}

/// Whisper's text without the wake word and the bracketed noise notes.
pub fn clean(text: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in text.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = (depth - 1).max(0),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let mut text = out.trim();
    loop {
        let lower = text.to_lowercase();
        let rest = ["hey", "hi", "okay", "ok", "telamon", "mon"]
            .iter()
            .find(|w| {
                lower.starts_with(*w)
                    && !lower[w.len()..].starts_with(|c: char| c.is_alphanumeric())
            });
        match rest {
            Some(w) => text = text[w.len()..].trim_start_matches([',', '.', '!', ' ']),
            None => break,
        }
    }
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::clean;

    #[test]
    fn strips_the_wake_word_and_notes() {
        assert_eq!(clean(" Hey Telamon, what time is it?"), "what time is it?");
        assert_eq!(clean("[BLANK_AUDIO]"), "");
        assert_eq!(clean("mon. How busy is my CPU? (wind)"), "How busy is my CPU?");
        assert_eq!(clean("Monday plans"), "Monday plans");
    }
}
