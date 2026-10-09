#!/usr/bin/env python3
"""Print the per-80 ms wake score of a WAV using the exact streaming pipeline.

    python training/wakeword/score_wav.py tests/audio/hey-telamon-time.wav

This is the reference the Rust implementation must match (see models/README.md):
1280-sample chunks of raw int16 values (as float32), melspectrogram.onnx on
the last 1760 samples, x/10 + 2, 76-frame embedding windows with stride 8,
telamon.onnx on the last 16 embeddings. The stream starts from the state left
by 40 chunks of digital silence (so scores are meaningful from the first chunk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "training.wakeword"

import numpy as np
import soundfile as sf

from .common import HEAD_PATH, MODELS, SR, load_threshold
from .features import CHUNK, CHUNK_S, Frontend, Head, Streamer


def read_wav(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="int16", always_2d=True)
    x = x[:, 0] if x.shape[1] == 1 else (x.astype(np.int32).mean(axis=1)).astype(np.int16)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, SR)
        x = np.clip(np.round(resample_poly(x.astype(np.float64), SR // g, sr // g)), -32768, 32767).astype(np.int16)
        print(f"# resampled {sr} -> {SR} Hz", file=sys.stderr)
    return x


def score_stream(fe: Frontend, head: Head, audio: np.ndarray) -> np.ndarray:
    """Literal chunk-by-chunk scoring. Last partial chunk is zero-padded."""
    s = Streamer(fe)
    n_chunks = -(-len(audio) // CHUNK)
    padded = np.zeros(n_chunks * CHUNK, np.int16)
    padded[: len(audio)] = audio
    scores = np.empty(n_chunks, np.float32)
    for c in range(n_chunks):
        s.push_chunk(padded[c * CHUNK : (c + 1) * CHUNK])
        scores[c] = head.score(s.emb_buf)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--models", default=str(MODELS), help="dir with melspectrogram.onnx and embedding_model.onnx")
    ap.add_argument("--head", default=str(HEAD_PATH))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--summary", action="store_true", help="only print the summary line")
    ap.add_argument("--check", action="store_true", help="also verify batch features == streaming features")
    a = ap.parse_args()

    thr = a.threshold if a.threshold is not None else load_threshold()
    fe = Frontend(a.models, threads=1)
    head = Head(a.head)
    audio = read_wav(a.wav)
    sc = score_stream(fe, head, audio)

    if a.check:
        from .features import windows_from_embeddings

        s = Streamer(fe)
        padded = np.zeros(len(sc) * CHUNK, np.int16)
        padded[: len(audio)] = audio
        emb = np.stack([s.push_chunk(padded[c * CHUNK : (c + 1) * CHUNK]) for c in range(len(sc))])
        emb_b = fe.embed_clip(audio)
        print(f"# batch-vs-streaming embedding max abs diff: {np.abs(emb - emb_b).max():.2e}", file=sys.stderr)

    trig, last = [], -10**9
    for c in np.flatnonzero(sc >= thr):
        if c - last >= 25:
            trig.append(int(c))
            last = int(c)

    if not a.summary:
        print(f"# {a.wav}: {len(audio) / SR:.2f} s, {len(sc)} chunks, threshold {thr:.2f}")
        print("chunk  t_end(s)  score")
        for c, v in enumerate(sc):
            bar = "#" * int(round(v * 40))
            mark = "  <-- WAKE" if c in trig else ""
            print(f"{c:5d}  {(c + 1) * CHUNK_S:7.2f}  {v:6.3f}  {bar}{mark}")
    mx = int(np.argmax(sc))
    verdict = f"WAKE x{len(trig)} (first at {(trig[0] + 1) * CHUNK_S:.2f} s)" if trig else "no wake"
    print(f"{Path(a.wav).name}: max {sc[mx]:.3f} at {(mx + 1) * CHUNK_S:.2f} s  -> {verdict}")


if __name__ == "__main__":
    main()
