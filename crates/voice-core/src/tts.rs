//! Speech: Kokoro-82M (ONNX) on the CPU. Text becomes IPA through the
//! `espeak-ng` program (run, not linked: it is GPL), the IPA becomes Kokoro's
//! tokens through the vocab in Kokoro's config.json, and the voice's style
//! row is picked by the token count, as kokoro-onnx does.

use crate::onnx;
use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// Kokoro's sample rate.
pub const RATE: u32 = 24000;
/// The most tokens Kokoro takes at once.
const MAX_TOKENS: usize = 510;
const STYLE: usize = 256;

pub struct Tts {
    session: Session,
    tokens_input: String,
    vocab: HashMap<char, i64>,
    /// [rows][256], one row per token count.
    style: Vec<[f32; STYLE]>,
    language: &'static str,
    speed: f32,
}

impl Tts {
    /// `models` holds kokoro-v1.0(.int8).onnx, voices-v1.0.bin and
    /// kokoro-config.json.
    pub fn new(models: &Path, voice: &str, speed: f32) -> Result<Self> {
        // fp32 first: int8 is only ~15 % faster (models/README.md).
        let model = ["kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx"]
            .iter()
            .map(|m| models.join(m))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow!("no Kokoro model in {}", models.display()))?;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
        let session = onnx::session(&model, threads)?;
        let tokens_input = session
            .inputs
            .iter()
            .map(|i| i.name.clone())
            .find(|n| n == "tokens" || n == "input_ids")
            .ok_or_else(|| anyhow!("the Kokoro model has no tokens input"))?;
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(models.join("kokoro-config.json")).context("kokoro-config.json")?,
        )?;
        let vocab = config["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("kokoro-config.json has no vocab"))?
            .iter()
            .filter_map(|(k, v)| Some((k.chars().next()?, v.as_i64()?)))
            .collect();
        let style = load_voice(&models.join("voices-v1.0.bin"), voice)?;
        // Kokoro's voice names start with the accent: b for British.
        let language = if voice.starts_with('b') {
            "en-gb"
        } else {
            "en-us"
        };
        Ok(Self {
            session,
            tokens_input,
            vocab,
            style,
            language,
            speed,
        })
    }

    /// The audio (24 kHz mono, -1..1) of one sentence.
    pub fn speak(&mut self, text: &str) -> Result<Vec<f32>> {
        let ipa = phonemize(&for_speech(text), self.language)?;
        let mut tokens: Vec<i64> = ipa
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect();
        tokens.truncate(MAX_TOKENS);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Row n-1 for n tokens, as Kokoro's own pipeline does.
        let style = self.style[(tokens.len() - 1).min(self.style.len() - 1)].to_vec();
        let n = tokens.len();
        let mut padded = Vec::with_capacity(n + 2);
        padded.push(0);
        padded.extend(tokens);
        padded.push(0);
        let outputs = self.session.run(ort::inputs![
            self.tokens_input.as_str() => Tensor::from_array(([1usize, n + 2], padded))?,
            "style" => Tensor::from_array(([1usize, STYLE], style))?,
            "speed" => Tensor::from_array(([1usize], vec![self.speed]))?,
        ])?;
        let (_, audio) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(audio.to_vec())
    }
}

/// Words for symbols espeak would read oddly or skip.
fn for_speech(text: &str) -> String {
    // "8:19" is a time, not a pause: "8 19" reads as "eight nineteen", and
    // "8:00" as "eight o'clock".
    let chars: Vec<char> = text.chars().collect();
    let mut times = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        let digit = |j: usize| chars.get(j).is_some_and(|c| c.is_ascii_digit());
        if c == ':' && i > 0 && digit(i - 1) && digit(i + 1) && digit(i + 2) {
            if chars[i + 1] == '0' && chars[i + 2] == '0' {
                times.push_str(" o'clock");
            } else {
                times.push(' ');
            }
        } else if times.ends_with(" o'clock") && c == '0' {
            // The "00" after "o'clock".
        } else {
            times.push(c);
        }
    }
    times
        .replace("°C", " degrees Celsius")
        .replace("°F", " degrees Fahrenheit")
        .replace('°', " degrees")
        .replace('%', " percent")
        .replace(" GiB", " gigabytes")
        .replace(" GB", " gigabytes")
        .replace(" MiB", " megabytes")
        .replace(" MB", " megabytes")
        .replace('&', " and ")
}

/// IPA with stress marks, keeping the punctuation (it shapes Kokoro's
/// pauses and intonation), as kokoro-onnx's phonemizer call does.
pub fn phonemize(text: &str, language: &str) -> Result<String> {
    let mut out = String::new();
    let mut segment = String::new();
    let flush = |segment: &mut String, out: &mut String| -> Result<()> {
        if segment.trim().is_empty() {
            segment.clear();
            return Ok(());
        }
        if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push_str(&espeak(segment.trim(), language)?);
        segment.clear();
        Ok(())
    };
    for c in text.chars() {
        if matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '—' | '…') {
            flush(&mut segment, &mut out)?;
            out.push(c);
        } else {
            segment.push(c);
        }
    }
    flush(&mut segment, &mut out)?;
    Ok(out
        .replace('ʲ', "j")
        .replace('r', "ɹ")
        .replace('x', "k")
        .replace('ɬ', "l"))
}

fn espeak(text: &str, language: &str) -> Result<String> {
    let mut child = Command::new("espeak-ng")
        .args(["-q", "--ipa", "-v", language, "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("cannot run espeak-ng (install espeak-ng)")?;
    child
        .stdin
        .take()
        .context("espeak-ng stdin")?
        .write_all(text.as_bytes())?;
    let mut ipa = String::new();
    child
        .stdout
        .take()
        .context("espeak-ng stdout")?
        .read_to_string(&mut ipa)?;
    child.wait()?;
    Ok(ipa.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// One voice's style rows from voices-v1.0.bin, a NumPy .npz of float32
/// arrays shaped [510, 1, 256].
fn load_voice(path: &Path, voice: &str) -> Result<Vec<[f32; STYLE]>> {
    let file = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut entry = zip
        .by_name(&format!("{voice}.npy"))
        .map_err(|_| anyhow!("there is no voice {voice}"))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    let data = npy_f32(&bytes)?;
    if data.is_empty() || data.len() % STYLE != 0 {
        return Err(anyhow!("the voice {voice} has {} values", data.len()));
    }
    Ok(data.as_chunks::<STYLE>().0.to_vec())
}

/// The data of a little-endian float32 .npy file.
fn npy_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(anyhow!("not a .npy file"));
    }
    let (len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => return Err(anyhow!("a short .npy file")),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + len)
            .ok_or_else(|| anyhow!("a short .npy header"))?,
    )?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(anyhow!("the voice is not little-endian float32: {header}"));
    }
    Ok(bytes[start + len..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_npy() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((header.len() as u16).to_le_bytes());
        b.extend(header.as_bytes());
        b.extend(1.5f32.to_le_bytes());
        b.extend((-2.0f32).to_le_bytes());
        assert_eq!(npy_f32(&b).unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn says_symbols() {
        assert_eq!(
            for_speech("45% at 60°C"),
            "45 percent at 60 degrees Celsius"
        );
        assert_eq!(
            for_speech("It's 8:19 PM, not 9:00."),
            "It's 8 19 PM, not 9 o'clock."
        );
    }
}
