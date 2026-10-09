"""openWakeWord streaming feature pipeline (the reference for the Rust port).

Exactly what models/README.md specifies:

    raw 16 kHz int16 audio, 1280-sample chunks (80 ms)
      -> melspectrogram.onnx on the last 1760 samples (480 of overlap + 1280 new)
         -> 8 frames x 32 mels, post-transform x/10 + 2
      -> embedding_model.onnx on the last 76 mel frames -> 96 floats (one per chunk)
      -> telamon.onnx on the last 16 embeddings -> score in 0..1 (one per chunk)

`Streamer` is the literal chunk-by-chunk implementation (used by score_wav.py).
`embed_clip` is an equivalent that runs the same per-window melspectrogram calls
but batches the embedding model; used for training and evaluation.
`Streamer` vs `embed_clip` equivalence is checked by score_wav.py --check.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

SR = 16000
CHUNK = 1280  # samples per step (80 ms)
OVERLAP = 480  # previous samples re-fed to the melspectrogram
MEL_WIN = CHUNK + OVERLAP  # 1760 samples per melspectrogram call
MEL_BINS = 32
FRAMES_PER_CHUNK = 8
EMB_FRAMES = 76  # mel frames per embedding
EMB_DIM = 96
HEAD_EMB = 16  # embeddings per head call
PRIME_CHUNKS = 40  # chunks of digital silence fed before real audio (3.2 s)
CHUNK_S = CHUNK / SR


def _session(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class Frontend:
    """melspectrogram.onnx + embedding_model.onnx."""

    def __init__(self, models_dir: str, threads: int = 2):
        self.mel = _session(f"{models_dir}/melspectrogram.onnx", threads)
        self.emb = _session(f"{models_dir}/embedding_model.onnx", threads)
        self.mel_in = self.mel.get_inputs()[0].name
        self.emb_in = self.emb.get_inputs()[0].name
        self._prime = None

    # -- single model calls -------------------------------------------------
    def melspec(self, samples: np.ndarray) -> np.ndarray:
        """float32 samples (int16 values, unscaled) [N] -> [T, 32], x/10 + 2."""
        x = np.asarray(samples, dtype=np.float32)[None, :]
        out = self.mel.run(None, {self.mel_in: x})[0]
        return np.squeeze(out, axis=(0, 1)) / 10.0 + 2.0

    def embed_windows(self, windows: np.ndarray, batch: int = 256) -> np.ndarray:
        """[B, 76, 32] -> [B, 96]."""
        outs = []
        for i in range(0, len(windows), batch):
            w = np.ascontiguousarray(windows[i : i + batch, :, :, None], dtype=np.float32)
            o = self.emb.run(None, {self.emb_in: w})[0]
            outs.append(o.reshape(len(w), EMB_DIM))
        return np.concatenate(outs) if outs else np.zeros((0, EMB_DIM), np.float32)

    # -- primed (silence) initial state ---------------------------------------
    def primed_state(self):
        """(mel_buffer [76,32], emb_buffer [16,96]) after PRIME_CHUNKS of zeros."""
        if self._prime is None:
            s = Streamer(self, primed=False)
            for _ in range(PRIME_CHUNKS):
                s.push_chunk(np.zeros(CHUNK, np.int16))
            self._prime = (s.mel_buf.copy(), s.emb_buf.copy())
        return self._prime

    # -- batched clip embedding ------------------------------------------------
    def embed_clip(self, audio: np.ndarray) -> np.ndarray:
        """int16 audio -> [n_chunks, 96] embeddings, starting from the primed state.

        Audio is zero-padded at the end to a whole number of chunks. Returns the
        embedding emitted by each chunk. Prepend `primed_state()[1]` to get the
        sequence the head sees.
        """
        n_chunks = max(1, -(-len(audio) // CHUNK))
        x = np.zeros(OVERLAP + n_chunks * CHUNK, np.float32)
        x[OVERLAP : OVERLAP + len(audio)] = audio
        # One melspectrogram call per 1760-sample window, exactly like the stream:
        # the model clamps its output to (max of the call - 80 dB), so a single
        # call over the whole clip or a batch of windows gives different values.
        mel = np.concatenate([self.melspec(x[k * CHUNK : k * CHUNK + MEL_WIN]) for k in range(n_chunks)])
        assert mel.shape[0] == FRAMES_PER_CHUNK * n_chunks, mel.shape
        mel_buf0, _ = self.primed_state()
        buf = np.concatenate([mel_buf0, mel], axis=0)
        idx = (np.arange(n_chunks)[:, None] * FRAMES_PER_CHUNK + FRAMES_PER_CHUNK) + np.arange(EMB_FRAMES)[None, :]
        return self.embed_windows(buf[idx])


class Streamer:
    """Literal chunk-by-chunk pipeline. push_chunk() takes exactly 1280 int16 samples."""

    def __init__(self, fe: Frontend, primed: bool = True):
        self.fe = fe
        self.raw_tail = np.zeros(OVERLAP, np.float32)
        self.mel_buf = np.ones((EMB_FRAMES, MEL_BINS), np.float32)
        self.emb_buf = np.zeros((HEAD_EMB, EMB_DIM), np.float32)
        if primed:
            self.mel_buf, self.emb_buf = (a.copy() for a in fe.primed_state())

    def push_chunk(self, chunk: np.ndarray) -> np.ndarray:
        assert len(chunk) == CHUNK
        x = np.concatenate([self.raw_tail, chunk.astype(np.float32)])  # 1760
        self.raw_tail = x[-OVERLAP:]
        mel = self.fe.melspec(x)  # [8, 32]
        self.mel_buf = np.concatenate([self.mel_buf, mel])[-EMB_FRAMES:]
        e = self.fe.embed_windows(self.mel_buf[None])[0]
        self.emb_buf = np.concatenate([self.emb_buf[1:], e[None]])
        return e


class Head:
    """telamon.onnx: input [1,16,96] float32 -> output [1,1]."""

    def __init__(self, path: str, threads: int = 1):
        self.sess = _session(path, threads)
        self.inp = self.sess.get_inputs()[0].name

    def score(self, emb16: np.ndarray) -> float:
        out = self.sess.run(None, {self.inp: emb16[None].astype(np.float32)})[0]
        return float(out.reshape(-1)[0])


def windows_from_embeddings(frontend: Frontend, emb: np.ndarray) -> np.ndarray:
    """[n_chunks, 96] -> [n_chunks, 16, 96], one head input per chunk (primed history)."""
    _, e0 = frontend.primed_state()
    full = np.concatenate([e0, emb], axis=0)
    idx = np.arange(len(emb))[:, None] + 1 + np.arange(HEAD_EMB)[None, :]
    return full[idx]
