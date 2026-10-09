//! Sound in and out. The microphone is PipeWire's `pw-record`, the speaker
//! `pw-play`: nothing is opened until the user turns Telamon on, and a test
//! reads a WAV file instead.

use anyhow::{Context, Result, anyhow};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The microphone's rate: what the wake word, VAD and whisper take.
pub const MIC_RATE: u32 = 16000;

/// Where 16 kHz mono audio comes from, chunk by chunk.
pub trait Source: Send {
    /// Fills `buf` (blocking); false at the end.
    fn read(&mut self, buf: &mut [i16]) -> Result<bool>;
    /// Drops what was recorded while Telamon spoke or thought.
    fn flush(&mut self) {}
}

/// Where Telamon's voice (24 kHz mono, -1..1) goes.
pub trait Sink: Send {
    /// Plays `audio`, calling `level` (0..1) about every 30 ms as it goes.
    fn play(&mut self, audio: &[f32], level: &mut dyn FnMut(f32)) -> Result<()>;
}

/// The loudness of `samples` (0..1, scaled so speech reaches about 1).
pub fn level_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
    ((sum / samples.len() as f64).sqrt() as f32 * 6.0).min(1.0)
}

pub fn level_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    ((sum / samples.len() as f64).sqrt() as f32 * 6.0).min(1.0)
}

/// A 16 kHz mono 16-bit WAV file, optionally at the speed of real time.
pub struct WavSource {
    samples: Vec<i16>,
    pos: usize,
    realtime: Option<Instant>,
}

impl WavSource {
    pub fn open(path: &std::path::Path, realtime: bool) -> Result<Self> {
        let mut reader =
            hound::WavReader::open(path).with_context(|| format!("{}", path.display()))?;
        let spec = reader.spec();
        if spec.sample_rate != MIC_RATE || spec.channels != 1 || spec.bits_per_sample != 16 {
            return Err(anyhow!("{} must be 16 kHz mono 16-bit", path.display()));
        }
        let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            samples,
            pos: 0,
            realtime: realtime.then(Instant::now),
        })
    }

    /// Where the file is, in seconds.
    pub fn position(&self) -> f64 {
        self.pos as f64 / MIC_RATE as f64
    }
}

impl Source for WavSource {
    fn read(&mut self, buf: &mut [i16]) -> Result<bool> {
        if self.pos >= self.samples.len() {
            return Ok(false);
        }
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self.samples.get(self.pos + i).copied().unwrap_or(0);
        }
        self.pos += buf.len();
        if let Some(start) = self.realtime {
            let due = Duration::from_secs_f64(self.pos as f64 / MIC_RATE as f64);
            if let Some(wait) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        Ok(true)
    }

    // A file goes on where it was, as a person would keep talking.
}

/// The running `pw-record`, shared so the GUI thread can close the
/// microphone at once ([`close_mic`]) while the worker is busy elsewhere.
pub type MicSlot = Arc<Mutex<Option<Child>>>;

/// Kills the microphone's `pw-record`, if one runs.
pub fn close_mic(slot: &MicSlot) {
    let child = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(mut child) = child {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// The microphone through `pw-record`. Once `stop` is set no new
/// `pw-record` starts (checked under the slot's lock, so a racing
/// [`close_mic`] always wins).
pub struct PipeWireSource {
    out: ChildStdout,
    slot: MicSlot,
    stop: Arc<AtomicBool>,
    program: std::path::PathBuf,
}

impl PipeWireSource {
    pub fn open(stop: Arc<AtomicBool>, slot: MicSlot) -> Result<Self> {
        Self::open_with("pw-record".into(), stop, slot)
    }

    /// `program` instead of pw-record (the tests' fake).
    fn open_with(
        program: std::path::PathBuf,
        stop: Arc<AtomicBool>,
        slot: MicSlot,
    ) -> Result<Self> {
        let out = spawn_mic(&program, &stop, &slot)?;
        Ok(Self {
            out,
            slot,
            stop,
            program,
        })
    }
}

/// Starts the recorder into `slot`, unless `stop` is set (checked under the
/// slot's lock, so a racing [`close_mic`] always wins).
fn spawn_mic(program: &Path, stop: &AtomicBool, slot: &MicSlot) -> Result<ChildStdout> {
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    if stop.load(Ordering::Relaxed) {
        return Err(anyhow!("Telamon was turned off"));
    }
    let mut child = Command::new(program)
        .args([
            "--raw",
            "--rate",
            "16000",
            "--channels",
            "1",
            "--format",
            "s16",
            "--media-role",
            "Communication",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("cannot run pw-record (install pipewire-utils)")?;
    let out = child.stdout.take().context("pw-record")?;
    *guard = Some(child);
    Ok(out)
}

impl Source for PipeWireSource {
    fn read(&mut self, buf: &mut [i16]) -> Result<bool> {
        if self.stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let mut bytes = vec![0u8; buf.len() * 2];
        // Ends when pw-record is killed (turned off) or PipeWire goes.
        if self.out.read_exact(&mut bytes).is_err() {
            return Ok(false);
        }
        for (s, b) in buf.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *s = i16::from_le_bytes(*b);
        }
        Ok(!self.stop.load(Ordering::Relaxed))
    }

    fn flush(&mut self) {
        // pw-record's pipe holds what was said meanwhile: a new one, in
        // place (replacing `self` would run Drop, which kills the new one).
        close_mic(&self.slot);
        match spawn_mic(&self.program, &self.stop, &self.slot) {
            Ok(out) => self.out = out,
            // The old pipe is at its end, so the next read ends the loop.
            Err(e) => log::warn!("cannot restart the microphone: {e:#}"),
        }
    }
}

impl Drop for PipeWireSource {
    fn drop(&mut self) {
        close_mic(&self.slot);
    }
}

/// Plays through `pw-play`, paced in real time so `level` follows the voice.
pub struct PipeWireSink {
    /// Set: stop speaking at once.
    pub stop: Arc<AtomicBool>,
}

impl Sink for PipeWireSink {
    fn play(&mut self, audio: &[f32], level: &mut dyn FnMut(f32)) -> Result<()> {
        let mut child = Command::new("pw-play")
            .args([
                "--raw",
                "--rate",
                "24000",
                "--channels",
                "1",
                "--format",
                "f32",
                "--media-role",
                "Assistant",
                "-",
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("cannot run pw-play (install pipewire-utils)")?;
        let stdin = child.stdin.take().context("pw-play")?;
        let result = paced(stdin, audio, &self.stop, level);
        if self.stop.load(Ordering::Relaxed) {
            let _ = child.kill();
        }
        let _ = child.wait();
        result
    }
}

fn paced(
    mut out: ChildStdin,
    audio: &[f32],
    stop: &AtomicBool,
    level: &mut dyn FnMut(f32),
) -> Result<()> {
    let step = crate::tts::RATE as usize * 30 / 1000;
    let start = Instant::now();
    for (i, chunk) in audio.chunks(step).enumerate() {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        out.write_all(&bytes)?;
        let due = Duration::from_millis(30 * i as u64);
        if let Some(wait) = due.checked_sub(start.elapsed()) {
            std::thread::sleep(wait);
        }
        level(level_f32(chunk));
    }
    drop(out);
    level(0.0);
    Ok(())
}

/// Collects the voice into a WAV file (the tests' speaker). `realtime`
/// waits as long as playing would.
pub struct WavSink {
    pub audio: Vec<f32>,
    pub first: Option<Instant>,
    pub realtime: bool,
}

impl WavSink {
    pub fn new(realtime: bool) -> Self {
        Self {
            audio: Vec::new(),
            first: None,
            realtime,
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: crate::tts::RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)?;
        for &s in &self.audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

impl Sink for WavSink {
    fn play(&mut self, audio: &[f32], level: &mut dyn FnMut(f32)) -> Result<()> {
        self.first.get_or_insert_with(Instant::now);
        self.audio.extend_from_slice(audio);
        let step = crate::tts::RATE as usize * 30 / 1000;
        for chunk in audio.chunks(step) {
            level(level_f32(chunk));
            if self.realtime {
                std::thread::sleep(Duration::from_millis(30));
            }
        }
        level(0.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn alive(slot: &MicSlot) -> bool {
        let mut guard = slot.lock().unwrap();
        guard
            .as_mut()
            .is_some_and(|c| c.try_wait().unwrap().is_none())
    }

    #[test]
    fn the_mic_lives_on_after_a_flush_and_dies_with_the_source() {
        let dir = std::env::temp_dir().join(format!("telamon-mic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("pw-record");
        std::fs::write(&fake, "#!/bin/sh\nexec cat /dev/zero\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();

        let slot = MicSlot::default();
        let stop = Arc::new(AtomicBool::new(false));
        let mut mic = PipeWireSource::open_with(fake.clone(), stop.clone(), slot.clone()).unwrap();
        let mut buf = [1i16; 1280];
        assert!(mic.read(&mut buf).unwrap());
        assert_eq!(buf[0], 0);
        mic.flush();
        assert!(alive(&slot), "flush left no pw-record running");
        assert!(mic.read(&mut buf).unwrap(), "the mic ended after a flush");

        // Turned off: closed at once, and a flush starts nothing new.
        stop.store(true, Ordering::Relaxed);
        close_mic(&slot);
        mic.flush();
        assert!(slot.lock().unwrap().is_none());
        assert!(!mic.read(&mut buf).unwrap());

        stop.store(false, Ordering::Relaxed);
        let mic = PipeWireSource::open_with(fake, stop, slot.clone()).unwrap();
        drop(mic);
        assert!(slot.lock().unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
