//! The loop: listen for the wake word, record the question until the speaker
//! stops, transcribe it, ask the model (with tools), and speak the answer
//! sentence by sentence, the next sentence synthesized while one plays.

use crate::audio::{self, Sink, Source};
use crate::config::Config;
use crate::llm::{self, Llm};
use crate::stt::Stt;
use crate::tools;
use crate::tts::Tts;
use crate::vad::{self, Vad};
use crate::wake::{self, WakeWord};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for "Telamon".
    Listening,
    /// Woken, recording the question.
    Awake,
    /// Transcribing and asking the model.
    Thinking,
    /// Saying the answer.
    Speaking,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Listening => "listening",
            Phase::Awake => "awake",
            Phase::Thinking => "thinking",
            Phase::Speaking => "speaking",
        }
    }
}

/// Milliseconds each step took for one question.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    /// The wake word to the first sound of the answer.
    pub wake_to_audio_ms: u64,
    /// The end of the question (as the VAD saw it) to the first sound.
    pub end_to_audio_ms: u64,
    pub stt_ms: u64,
    pub llm_ms: u64,
    pub tts_first_ms: u64,
}

#[derive(Clone, Debug)]
pub enum Event {
    Phase(Phase),
    /// The loudness (0..1) of whoever speaks, for the glow.
    Level(f32),
    Woke {
        score: f32,
    },
    Heard(String),
    Tool(String),
    Reply(String),
    Error(String),
    Timing(Timing),
}

pub struct Assistant {
    cfg: Config,
    wake: WakeWord,
    vad: Vad,
    stt: Stt,
    llm: Llm,
    tts: Tts,
}

impl Assistant {
    /// Loads every model (a few seconds, and a few hundred MB).
    pub fn load(cfg: &Config, home: PathBuf) -> Result<Self> {
        let head = cfg
            .wake_model
            .clone()
            .unwrap_or_else(|| cfg.model("telamon.onnx"));
        Ok(Self {
            wake: WakeWord::new(&cfg.models, &head)?,
            vad: Vad::new(&cfg.model("silero_vad.onnx"))?,
            stt: Stt::new(&cfg.model("ggml-base.en.bin"), cfg.stt_gpu)?,
            llm: Llm::new(
                &cfg.llama_url,
                tools::Context {
                    home,
                    location: cfg.location.clone(),
                },
            ),
            tts: Tts::new(&cfg.models, &cfg.voice, cfg.speed)?,
            cfg: cfg.clone(),
        })
    }

    pub fn llm(&self) -> &Llm {
        &self.llm
    }

    /// Runs until `source` ends or `stop` is set.
    pub fn run(
        &mut self,
        source: &mut dyn Source,
        sink: &mut dyn Sink,
        stop: &AtomicBool,
        on: &mut dyn FnMut(Event),
    ) -> Result<()> {
        on(Event::Phase(Phase::Listening));
        let mut chunk = vec![0i16; wake::CHUNK];
        while !stop.load(Ordering::Relaxed) && source.read(&mut chunk)? {
            let score = self.wake.process(&chunk)?;
            if score < self.cfg.wake_threshold {
                continue;
            }
            let woke = Instant::now();
            on(Event::Woke { score });
            on(Event::Phase(Phase::Awake));
            if let Some((question, ended)) = self.record(source, stop, on)? {
                self.answer(&question, woke, ended, sink, on);
            }
            self.wake.reset()?;
            self.vad.reset();
            source.flush();
            on(Event::Level(0.0));
            on(Event::Phase(Phase::Listening));
        }
        Ok(())
    }

    /// The question, from the wake word until the speaker stops; None when
    /// nobody speaks.
    fn record(
        &mut self,
        source: &mut dyn Source,
        stop: &AtomicBool,
        on: &mut dyn FnMut(Event),
    ) -> Result<Option<(Vec<i16>, Instant)>> {
        let ms = |samples: usize| (samples * 1000 / audio::MIC_RATE as usize) as u32;
        let mut audio = Vec::new();
        let mut pending: Vec<i16> = Vec::new();
        let mut chunk = vec![0i16; wake::CHUNK];
        let (mut spoke, mut silent) = (false, 0usize);
        while !stop.load(Ordering::Relaxed) && source.read(&mut chunk)? {
            on(Event::Level(audio::level_i16(&chunk)));
            audio.extend_from_slice(&chunk);
            pending.extend_from_slice(&chunk);
            while pending.len() >= vad::FRAME {
                let frame: Vec<i16> = pending.drain(..vad::FRAME).collect();
                if self.vad.speech(&frame)? >= 0.5 {
                    spoke = true;
                    silent = 0;
                } else {
                    silent += vad::FRAME;
                }
            }
            if spoke && ms(silent) >= self.cfg.end_silence_ms {
                // Whisper needs no trailing silence.
                audio.truncate(audio.len().saturating_sub(silent.saturating_sub(4000)));
                return Ok(Some((audio, Instant::now())));
            }
            if !spoke && ms(audio.len()) >= self.cfg.start_timeout_ms {
                return Ok(None);
            }
            if ms(audio.len()) >= self.cfg.max_question_ms {
                return Ok(Some((audio, Instant::now())));
            }
        }
        Ok(spoke.then(|| (audio, Instant::now())))
    }

    fn answer(
        &mut self,
        question: &[i16],
        woke: Instant,
        ended: Instant,
        sink: &mut dyn Sink,
        on: &mut dyn FnMut(Event),
    ) {
        on(Event::Phase(Phase::Thinking));
        on(Event::Level(0.0));
        let mut timing = Timing::default();
        let t = Instant::now();
        let text = match self.stt.transcribe(question) {
            Ok(t) => t,
            Err(e) => {
                on(Event::Error(format!("speech to text: {e:#}")));
                return;
            }
        };
        timing.stt_ms = t.elapsed().as_millis() as u64;
        on(Event::Heard(text.clone()));
        if text.trim().is_empty() {
            return;
        }
        let t = Instant::now();
        let reply = match self
            .llm
            .ask(&text, &mut |name| on(Event::Tool(name.to_string())))
        {
            Ok(a) => a.text,
            Err(e) => {
                on(Event::Error(format!("the model: {e:#}")));
                "Sorry, I can't reach my language model right now.".to_string()
            }
        };
        timing.llm_ms = t.elapsed().as_millis() as u64;
        on(Event::Reply(reply.clone()));

        // One thread synthesizes, this one plays, so sentence two is ready
        // when sentence one ends.
        let sentences = llm::sentences(&reply);
        let tts = &mut self.tts;
        let t = Instant::now();
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::sync_channel::<Result<Vec<f32>>>(2);
            scope.spawn(move || {
                for s in &sentences {
                    if tx.send(tts.speak(s)).is_err() {
                        break;
                    }
                }
            });
            let mut first = true;
            for audio in rx {
                let audio = match audio {
                    Ok(a) => a,
                    Err(e) => {
                        on(Event::Error(format!("speech: {e:#}")));
                        break;
                    }
                };
                if first {
                    first = false;
                    timing.tts_first_ms = t.elapsed().as_millis() as u64;
                    timing.wake_to_audio_ms = woke.elapsed().as_millis() as u64;
                    timing.end_to_audio_ms = ended.elapsed().as_millis() as u64;
                    on(Event::Phase(Phase::Speaking));
                }
                if let Err(e) = sink.play(&audio, &mut |l| on(Event::Level(l))) {
                    on(Event::Error(format!("playing: {e:#}")));
                    break;
                }
            }
        });
        on(Event::Timing(timing));
    }
}
