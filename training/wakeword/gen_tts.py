"""Generate raw 16 kHz speech with Kokoro for every English voice.

One .npz per voice under WORK/tts/. Each holds concatenated int16 audio, an
offsets array and a JSON list of {kind, text, speed} records.

kinds: pos (wake phrase), conf (confusable negative), sent (ordinary sentence).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time

import numpy as np
from scipy.signal import resample_poly

from . import texts
from .common import KOKORO_SR, MODELS, SR, WORK, lang_for, roles, to_int16

SPEEDS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]


def trim_silence(x: np.ndarray, sr: int = SR, floor_db: float = -45.0, margin_s: float = 0.03) -> np.ndarray:
    """Cut leading/trailing silence (10 ms RMS frames below peak+floor_db)."""
    if len(x) == 0:
        return x
    hop = sr // 100
    n = len(x) // hop
    if n == 0:
        return x
    fr = x[: n * hop].reshape(n, hop).astype(np.float32)
    rms = np.sqrt((fr**2).mean(axis=1) + 1e-12)
    thr = rms.max() * 10 ** (floor_db / 20)
    act = np.where(rms > thr)[0]
    if len(act) == 0:
        return x
    a = max(0, act[0] * hop - int(margin_s * sr))
    b = min(len(x), (act[-1] + 1) * hop + int(margin_s * sr))
    return x[a:b]


def plan(role: str, voice: str, seed: int) -> list[dict]:
    """The list of clips to synthesize for one voice."""
    rng = random.Random(f"{seed}/{role}/{voice}")
    sents = texts.split_sentences()[role]
    phrases = texts.wake_phrases()
    core = phrases[:8]
    rest = phrases[8:]
    jobs: list[dict] = []

    def spd() -> float:
        return rng.choice(SPEEDS)

    if role == "train":
        for p in core:
            for _ in range(2):
                jobs.append({"kind": "pos", "text": p[0], "ph": p[1], "speed": spd()})
        for p in rng.sample(rest, min(44, len(rest))):
            jobs.append({"kind": "pos", "text": p[0], "ph": p[1], "speed": spd()})
        for t in rng.sample(texts.CONFUSABLES, 40):
            jobs.append({"kind": "conf", "text": t, "ph": None, "speed": rng.choice(SPEEDS[1:-1])})
        for t in rng.sample(sents, min(50, len(sents))):
            jobs.append({"kind": "sent", "text": t, "ph": None, "speed": rng.choice(SPEEDS[2:-2])})
    else:
        for p in core:
            for s in (0.85, 1.0, 1.15):
                jobs.append({"kind": "pos", "text": p[0], "ph": p[1], "speed": s})
        for p in rng.sample(rest, min(16, len(rest))):
            jobs.append({"kind": "pos", "text": p[0], "ph": p[1], "speed": spd()})
        confs = texts.CONFUSABLES if role == "test" else rng.sample(texts.CONFUSABLES, 40)
        for t in confs:
            jobs.append({"kind": "conf", "text": t, "ph": None, "speed": 1.0})
        for t in sents:
            jobs.append({"kind": "sent", "text": t, "ph": None, "speed": rng.choice(SPEEDS[2:-2])})
    return jobs


def plan_extra(role: str, voice: str, seed: int) -> list[dict]:
    """Extra negative sentences (word salad) for more phonetic variety."""
    rng = random.Random(f"{seed}/extra/{role}/{voice}")
    pool = texts.split_salad()[role]
    n = {"train": 130, "val": 70, "test": 120}[role]
    return [
        {"kind": "sent", "text": t, "ph": None, "speed": rng.choice(SPEEDS[2:-2])}
        for t in rng.sample(pool, min(n, len(pool)))
    ]


def synth_one(kokoro, voice: str, job: dict) -> np.ndarray:
    lang = lang_for(voice)
    if job["ph"] is not None:
        audio, sr = kokoro.create(job["ph"], voice=voice, speed=job["speed"], lang=lang, is_phonemes=True)
    else:
        audio, sr = kokoro.create(job["text"], voice=voice, speed=job["speed"], lang=lang)
    assert sr == KOKORO_SR
    x = resample_poly(audio.astype(np.float64), SR, KOKORO_SR)
    x = trim_silence(to_int16(x * 32767.0))
    return x


def work(args) -> tuple[str, int, float]:
    role, voice, model, voices, threads, seed, out_dir, part = args
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    t0 = time.time()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(sess, voices)
    jobs = plan(role, voice, seed) if part == "base" else plan_extra(role, voice, seed)
    chunks, meta, offs = [], [], [0]
    for job in jobs:
        try:
            x = synth_one(kokoro, voice, job)
        except Exception as e:  # keep going, record nothing for this clip
            print(f"[{voice}] skip {job['text']!r}: {e}", file=sys.stderr)
            continue
        chunks.append(x)
        offs.append(offs[-1] + len(x))
        meta.append({k: job[k] for k in ("kind", "text", "speed")} | {"voice": voice, "role": role})
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    out = out_dir / (f"{voice}.npz" if part == "base" else f"{voice}.extra.npz")
    np.savez(out, audio=audio, offsets=np.array(offs, np.int64), meta=json.dumps(meta))
    return f"{voice}/{part}", len(chunks), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "kokoro-v1.0.onnx"))
    ap.add_argument("--voices", default=str(MODELS / "voices-v1.0.bin"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*", help="restrict to these voices (debug)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    names = list(np.load(a.voices).keys())
    r = roles(names)
    out_dir = WORK / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    # Biggest jobs (test voices) first so the pool tail is short.
    for role in ("test", "val", "train"):
        for v in r[role]:
            if a.only and v not in a.only:
                continue
            for part, fn in (("base", f"{v}.npz"), ("extra", f"{v}.extra.npz")):
                if (out_dir / fn).exists() and not a.force:
                    continue
                tasks.append((role, v, a.model, a.voices, a.threads, a.seed, out_dir, part))
    print(f"voices: train={len(r['train'])} val={r['val']} test={r['test']}; {len(tasks)} to generate", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for voice, n, dt in pool.imap_unordered(work, tasks):
            print(f"  {voice}: {n} clips in {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    print(f"tts done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
