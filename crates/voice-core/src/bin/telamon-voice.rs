//! The pipeline without the app, for tests and benchmarks:
//!   telamon-voice --models DIR --wav FILE [--realtime] [--out reply.wav]
//!                 [--llama URL] [--threshold X] [--voice V] [--cpu]
//!   telamon-voice --models DIR --wav FILE --scores    the wake score per chunk
//!   telamon-voice --models DIR --say TEXT --out FILE   speech only
//!   telamon-voice --models DIR --mic                    the real microphone
//!   telamon-voice --models DIR --transcribe FILE       speech to text of any mono WAV
//! Events go to stdout as JSON lines, logs to stderr.

use anyhow::{Context, Result, anyhow};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use voice_core::audio::{self, Source};
use voice_core::{Assistant, Config, Event};

fn rss_mb(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .map_or(0.0, |kb| (kb / 1024.0 * 10.0).round() / 10.0)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut args = std::env::args().skip(1);
    let mut models = None;
    let mut wav = None;
    let mut out = None;
    let mut say = None;
    let mut transcribe = None;
    let (mut realtime, mut scores, mut mic) = (false, false, false);
    let mut cfg = Config::new(PathBuf::new());
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or_else(|| anyhow!("{a} needs a value"));
        match a.as_str() {
            "--models" => models = Some(PathBuf::from(value()?)),
            "--wav" => wav = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--say" => say = Some(value()?),
            "--transcribe" => transcribe = Some(PathBuf::from(value()?)),
            "--llama" => cfg.llama_url = value()?,
            "--threshold" => cfg.wake_threshold = value()?.parse()?,
            "--voice" => cfg.voice = value()?,
            "--wake-model" => cfg.wake_model = Some(PathBuf::from(value()?)),
            "--location" => cfg.location = value()?,
            "--cpu" => cfg.stt_gpu = false,
            "--realtime" => realtime = true,
            "--scores" => scores = true,
            "--mic" => mic = true,
            _ => return Err(anyhow!("unknown argument {a}; see the source's header")),
        }
    }
    cfg.models = models.context("--models is required")?;
    let emit = |v: serde_json::Value| println!("{v}");

    if scores {
        let head = cfg
            .wake_model
            .clone()
            .unwrap_or_else(|| cfg.model("telamon.onnx"));
        let mut wake = voice_core::wake::WakeWord::new(&cfg.models, &head)?;
        let mut src = audio::WavSource::open(&wav.context("--scores needs --wav")?, false)?;
        let mut chunk = vec![0i16; voice_core::wake::CHUNK];
        let mut i = 0;
        while src.read(&mut chunk)? {
            println!("{i}\t{:.4}", wake.process(&chunk)?);
            i += 1;
        }
        return Ok(());
    }

    if let Some(path) = transcribe {
        let mut reader = hound::WavReader::open(&path)?;
        let rate = reader.spec().sample_rate as f64;
        let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
        // Linear resampling to 16 kHz is plenty for checking what was said.
        let n = (samples.len() as f64 * 16000.0 / rate) as usize;
        let resampled: Vec<i16> = (0..n)
            .map(|i| {
                let x = i as f64 * rate / 16000.0;
                let (j, f) = (x as usize, x.fract());
                let a = samples[j.min(samples.len() - 1)] as f64;
                let b = samples[(j + 1).min(samples.len() - 1)] as f64;
                (a + (b - a) * f) as i16
            })
            .collect();
        let stt = voice_core::stt::Stt::new(&cfg.model("ggml-base.en.bin"), cfg.stt_gpu)?;
        emit(json!({"event": "transcribed", "text": stt.transcribe(&resampled)?}));
        return Ok(());
    }

    if let Some(text) = say {
        let t = Instant::now();
        let mut tts = voice_core::tts::Tts::new(&cfg.models, &cfg.voice, cfg.speed)?;
        let loaded = t.elapsed().as_millis();
        let t = Instant::now();
        let audio = tts.speak(&text)?;
        let ms = t.elapsed().as_millis();
        let mut sink = audio::WavSink::new(false);
        audio::Sink::play(&mut sink, &audio, &mut |_| {})?;
        sink.save(&out.context("--say needs --out")?)?;
        emit(json!({"event": "spoken", "load_ms": loaded, "synth_ms": ms,
            "audio_ms": audio.len() as u64 * 1000 / voice_core::tts::RATE as u64}));
        return Ok(());
    }

    let t = Instant::now();
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/".into());
    let mut assistant = Assistant::load(&cfg, home)?;
    emit(
        json!({"event": "loaded", "ms": t.elapsed().as_millis(), "rss_mb": rss_mb("VmRSS:"),
        "llama_ready": assistant.llm().ready()}),
    );

    let stop = Arc::new(AtomicBool::new(false));
    let mut source: Box<dyn Source> = if mic {
        Box::new(audio::PipeWireSource::open(
            stop.clone(),
            Default::default(),
        )?)
    } else {
        Box::new(audio::WavSource::open(
            &wav.clone().context("--wav or --mic")?,
            realtime,
        )?)
    };
    let mut wav_sink = audio::WavSink::new(realtime);
    let mut pw_sink = audio::PipeWireSink { stop: stop.clone() };
    let sink: &mut dyn audio::Sink = if mic { &mut pw_sink } else { &mut wav_sink };
    let start = Instant::now();
    let mut peak_level = 0.0f32;
    assistant.run(source.as_mut(), sink, &stop, &mut |e| {
        let at = start.elapsed().as_millis();
        match e {
            Event::Level(l) => peak_level = peak_level.max(l),
            Event::Phase(p) => emit(json!({"event": "phase", "phase": p.name(), "at_ms": at})),
            Event::Woke { score } => emit(json!({"event": "woke", "score": score, "at_ms": at})),
            Event::Heard(t) => emit(json!({"event": "heard", "text": t, "at_ms": at})),
            Event::Tool(t) => emit(json!({"event": "tool", "name": t, "at_ms": at})),
            Event::Reply(t) => emit(json!({"event": "reply", "text": t, "at_ms": at})),
            Event::Error(t) => emit(json!({"event": "error", "text": t, "at_ms": at})),
            Event::Timing(t) => emit(json!({"event": "timing",
                "wake_to_audio_ms": t.wake_to_audio_ms, "end_to_audio_ms": t.end_to_audio_ms,
                "stt_ms": t.stt_ms, "llm_ms": t.llm_ms, "tts_first_ms": t.tts_first_ms})),
        }
    })?;
    if let Some(out) = out {
        wav_sink.save(&out)?;
    }
    emit(
        json!({"event": "done", "spoken_ms": wav_sink.audio.len() as u64 * 1000 / voice_core::tts::RATE as u64,
        "peak_level": peak_level, "rss_mb": rss_mb("VmRSS:"), "peak_rss_mb": rss_mb("VmHWM:")}),
    );
    Ok(())
}
