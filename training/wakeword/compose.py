"""Compose and augment clips from raw Kokoro speech.

Augmentations: random gain, white/pink/brown noise at random SNR, synthetic
reverb (exponentially decaying noise as RIR), random silence padding, light
dither always on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from .common import SR, to_int16
from .texts import NEAR_HOMOPHONES
from .features import CHUNK, CHUNK_S


# ---------------------------------------------------------------------------
# raw speech bank
# ---------------------------------------------------------------------------
class Bank:
    """All raw TTS clips of a set of voices, indexed by kind and voice."""

    def __init__(self, tts_dir: Path, voices: list[str]):
        self.voices = list(voices)
        self.by: dict[str, dict[str, list[tuple[np.ndarray, dict]]]] = {
            k: {v: [] for v in voices} for k in ("pos", "conf", "sent")
        }
        for v in voices:
            for fn in (f"{v}.npz", f"{v}.extra.npz"):
                if not (tts_dir / fn).exists():
                    continue
                z = np.load(tts_dir / fn)
                audio, offs, meta = z["audio"], z["offsets"], json.loads(str(z["meta"]))
                for i, m in enumerate(meta):
                    if m["kind"] == "conf" and m["text"] in NEAR_HOMOPHONES:
                        continue
                    self.by[m["kind"]][v].append((audio[offs[i] : offs[i + 1]], m))

    def pick(self, rng: np.random.Generator, kind: str, voice: str):
        pool = self.by[kind][voice]
        return pool[int(rng.integers(len(pool)))]

    def voice(self, rng: np.random.Generator) -> str:
        return self.voices[int(rng.integers(len(self.voices)))]


# ---------------------------------------------------------------------------
# noise / reverb / gain
# ---------------------------------------------------------------------------
def white(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n)


def pink(n: int, rng) -> np.ndarray:
    f = np.fft.rfft(rng.standard_normal(n))
    k = np.arange(len(f), dtype=np.float64)
    k[0] = 1.0
    x = np.fft.irfft(f / np.sqrt(k), n)
    return x / (x.std() + 1e-9)


def brown(n: int, rng) -> np.ndarray:
    f = np.fft.rfft(rng.standard_normal(n))
    k = np.arange(len(f), dtype=np.float64)
    k[0] = 1.0
    x = np.fft.irfft(f / k, n)
    return x / (x.std() + 1e-9)


def hum(n: int, rng) -> np.ndarray:
    t = np.arange(n) / SR
    f0 = rng.choice([50.0, 60.0])
    x = sum(rng.uniform(0.2, 1.0) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) for h in range(1, 5))
    return x / (x.std() + 1e-9) + 0.05 * white(n, rng)


NOISES = {"white": white, "pink": pink, "brown": brown, "hum": hum}


def make_noise(n: int, rng, kind: str | None = None) -> np.ndarray:
    kind = kind or str(rng.choice(["white", "pink", "pink", "brown", "hum"]))
    return NOISES[kind](n, rng)


def reverb(x: np.ndarray, rng, t60: float | None = None) -> np.ndarray:
    t60 = t60 or float(rng.uniform(0.15, 0.7))
    n = int(t60 * SR * 1.2)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9078 * t / t60)  # -60 dB at t60
    rir[0] = 1.0 + abs(rir[0])  # direct path first so timing is preserved
    rir /= np.sqrt((rir**2).sum())
    return fftconvolve(x, rir)[: len(x)]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    audio: np.ndarray  # int16
    wake_end: int | None  # sample index of the end of the wake phrase
    kind: str  # pos | neg | conf | noise
    weight: float = 1.0
    voice: str = ""
    cond: str = ""
    extra: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)


def _gap(rng, lo: float, hi: float) -> np.ndarray:
    return np.zeros(int(rng.uniform(lo, hi) * SR))


def compose(
    rng: np.random.Generator,
    segments: list[np.ndarray],
    wake_index: int | None,
    lead=(0.3, 2.0),
    gap=(0.15, 1.0),
    tail=(0.4, 1.5),
    snr_db: float | None | str = "random",
    do_reverb: bool | str = "random",
    peak_db: tuple[float, float] = (-24.0, -3.0),
    noise_kind: str | None = None,
    names: list[str] | None = None,
) -> tuple[np.ndarray, int | None, list[tuple[str, int, int]]]:
    """Lay segments (int16 arrays) out with random silence; mix noise. Returns (int16, wake_end)."""
    parts = [_gap(rng, *lead)]
    wake_end = None
    spans: list[tuple[str, int, int]] = []
    pos = len(parts[0])
    for i, seg in enumerate(segments):
        s = seg.astype(np.float64)
        parts.append(s)
        spans.append((names[i] if names else "", pos, pos + len(s)))
        pos += len(s)
        if i == wake_index:
            wake_end = pos
        if i < len(segments) - 1:
            g = _gap(rng, *gap)
            parts.append(g)
            pos += len(g)
    parts.append(_gap(rng, *tail))
    x = np.concatenate(parts)
    speech_mask = np.abs(x) > 0

    if do_reverb == "random":
        do_reverb = rng.random() < 0.3
    if do_reverb:
        x = reverb(x, rng)

    # gain: speech peak to a random level below full scale
    peak = np.abs(x).max() + 1e-9
    x = x * (32767.0 * 10 ** (rng.uniform(*peak_db) / 20) / peak)

    # noise bed
    if snr_db == "random":
        snr_db = None if rng.random() < 0.2 else float(rng.uniform(0, 30))
    if snr_db is not None:
        sp = x[speech_mask]
        sp_rms = rms(sp) if len(sp) else 1000.0
        nz = make_noise(len(x), rng, noise_kind)
        x = x + nz * (sp_rms / 10 ** (snr_db / 20)) / (nz.std() + 1e-9)
    # always-on light dither/noise floor
    x = x + rng.standard_normal(len(x)) * rng.uniform(1.0, 6.0)
    return to_int16(x), wake_end, spans


def noise_clip(rng: np.random.Generator, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    kind = str(rng.choice(list(NOISES)))
    lvl = 10 ** (rng.uniform(-60, -15) / 20) * 32767
    x = make_noise(n, rng, kind)
    x = x / (x.std() + 1e-9) * lvl
    if rng.random() < 0.3:  # a few clicks/bursts
        for _ in range(int(rng.integers(1, 6))):
            a = int(rng.integers(0, max(1, n - 800)))
            x[a : a + 800] += rng.standard_normal(min(800, n - a)) * lvl * 4 * np.hanning(min(800, n - a))
    return to_int16(x)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
POS_AFTER = 0.40  # window end up to this long after the word end: positive
IGN_BEFORE = 0.30  # window ends this much before the word end: ignored
IGN_AFTER = 1.00  # ... until this long after: ignored


def chunk_labels(n_chunks: int, wake_end: int | None) -> np.ndarray:
    """1 positive, 0 negative, -1 ignore, per chunk (window ends at end of the chunk)."""
    lab = np.zeros(n_chunks, np.int8)
    if wake_end is None:
        return lab
    d = ((np.arange(n_chunks) + 1) * CHUNK - wake_end) / SR
    lab[(d >= -IGN_BEFORE) & (d < 0.0)] = -1
    lab[(d >= 0.0) & (d <= POS_AFTER)] = 1
    lab[(d > POS_AFTER) & (d <= IGN_AFTER)] = -1
    return lab
