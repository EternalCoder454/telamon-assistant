"""Compose augmented clips, run the frozen front end, store embeddings.

Each clip is stored as a block: 16 primed-silence embeddings, then one
embedding per 80 ms chunk. The head input for chunk c is the 16 rows ending at
row (block_start + 16 + c). Stored float16 to keep the files small.

Modes
  train     augmented training clips (positives, negatives, confusables, noise)
  evalpos   wake phrases of held-out voices under fixed conditions
  evalneg   long negative streams (sentences + confusables) of held-out voices
  evalnoise noise-only streams
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from . import compose as C
from .common import MODELS, SR, WORK, roles
from .features import CHUNK, HEAD_EMB, Frontend

COND_NAMES = ["clean", "snr30", "snr20", "snr10", "snr5", "reverb"]
KIND_ID = {"pos": 0, "neg": 1, "conf": 2, "noise": 3}


def _train_clips(bank: C.Bank, rng, n_pos, n_neg, n_conf, n_noise):
    for _ in range(n_pos):
        v = bank.voice(rng)
        wake, m = bank.pick(rng, "pos", v)
        r = rng.random()
        if r < 0.45:
            s2 = bank.pick(rng, "sent", v)
            segs, wi, nm = [wake, s2[0]], 0, [m["text"], s2[1]["text"]]
        elif r < 0.65:
            s2 = bank.pick(rng, "sent", v)
            segs, wi, nm = [s2[0], wake], 1, [s2[1]["text"], m["text"]]
        else:
            segs, wi, nm = [wake], 0, [m["text"]]
        a, we, sp = C.compose(rng, segs, wi, names=nm)
        yield C.Clip(a, we, "pos", 1.0, v, spans=sp)
    for _ in range(n_neg):
        v = bank.voice(rng)
        its = [bank.pick(rng, "sent", v) for _ in range(int(rng.integers(1, 4)))]
        a, _, sp = C.compose(rng, [i[0] for i in its], None, names=[i[1]["text"] for i in its])
        yield C.Clip(a, None, "neg", 1.0, v, spans=sp)
    for _ in range(n_conf):
        v = bank.voice(rng)
        c = bank.pick(rng, "conf", v)
        s = bank.pick(rng, "sent", v)
        order = rng.random()
        its = [c] if order < 0.5 else ([s, c] if order < 0.75 else [c, s])
        a, _, sp = C.compose(rng, [i[0] for i in its], None, names=[i[1]["text"] for i in its])
        yield C.Clip(a, None, "conf", 2.0, v, spans=sp)
    for _ in range(n_noise):
        a = C.noise_clip(rng, float(rng.uniform(3, 10)))
        yield C.Clip(a, None, "noise", 1.0, "")


def _evalpos_clips(bank: C.Bank, rng, repeats: int):
    for v in bank.voices:
        for wake, m in bank.by["pos"][v]:
            for _ in range(repeats):
                for ci, cond in enumerate(COND_NAMES):
                    segs, wi = [wake], 0
                    if rng.random() < 0.4:
                        segs = [wake, bank.pick(rng, "sent", v)[0]]
                    snr = {"clean": None, "snr30": 30.0, "snr20": 20.0, "snr10": 10.0, "snr5": 5.0, "reverb": 20.0}[cond]
                    a, we, sp = C.compose(rng, segs, wi, snr_db=snr, do_reverb=(cond == "reverb"), peak_db=(-20.0, -6.0),
                                          names=[m["text"], "<cmd>"][: len(segs)])
                    yield C.Clip(a, we, "pos", 1.0, v, cond, {"text": m["text"], "speed": m["speed"]}, spans=sp)


def _evalneg_clips(bank: C.Bank, rng, passes: int, seg_per_stream: int = 14, conf_p: float = 0.0):
    for _ in range(passes):
        for v in bank.voices:
            sents = list(bank.by["sent"][v])
            order = rng.permutation(len(sents))
            confs = bank.by["conf"][v]
            for i in range(0, len(order), seg_per_stream):
                segs, nm = [], []
                for j in order[i : i + seg_per_stream]:
                    segs.append(sents[j][0])
                    nm.append(sents[j][1]["text"])
                    if confs and rng.random() < conf_p:
                        c = confs[int(rng.integers(len(confs)))]
                        segs.append(c[0])
                        nm.append("[conf] " + c[1]["text"])
                a, _, sp = C.compose(rng, segs, None, lead=(0.5, 1.5), gap=(0.3, 1.5), peak_db=(-20.0, -4.0), names=nm)
                yield C.Clip(a, None, "neg", 1.0, v, spans=sp)
            # one stream of just the confusables, back to back
            if confs:
                segs = [c[0] for c in confs]
                a, _, sp = C.compose(rng, segs, None, lead=(0.5, 1.0), gap=(0.4, 1.2), peak_db=(-20.0, -4.0),
                                     names=["[conf] " + c[1]["text"] for c in confs])
                yield C.Clip(a, None, "conf", 1.0, v, spans=sp)


def _evalnoise_clips(rng, minutes: float):
    total = 0.0
    while total < minutes * 60:
        sec = 30.0
        a = C.noise_clip(rng, sec)
        total += sec
        yield C.Clip(a, None, "noise", 1.0, "")


def build(task: dict) -> dict:
    mode = task["mode"]
    rng = np.random.default_rng(task["seed"])
    fe = Frontend(str(MODELS), threads=task.get("threads", 1))
    if mode == "evalnoise":
        gen = _evalnoise_clips(rng, task["minutes"])
    else:
        bank = C.Bank(WORK / "tts", task["voices"])
        if mode == "train":
            gen = _train_clips(bank, rng, task["n_pos"], task["n_neg"], task["n_conf"], task["n_noise"])
        elif mode == "evalpos":
            gen = _evalpos_clips(bank, rng, task.get("repeats", 1))
        elif mode == "evalneg":
            gen = _evalneg_clips(bank, rng, task.get("passes", 1))
        else:
            raise ValueError(mode)

    rows, starts, ns, labels, weights, kinds, conds, voices_, wake_ends = [], [], [], [], [], [], [], [], []
    audio_len, spans_ = [], []
    row = 0
    _, e0 = fe.primed_state()
    for clip in gen:
        emb = fe.embed_clip(clip.audio)
        n = len(emb)
        rows.append(np.concatenate([e0, emb]).astype(np.float16))
        starts.append(row)
        ns.append(n)
        row += HEAD_EMB + n
        lab = C.chunk_labels(n, clip.wake_end)
        labels.append(lab)
        weights.append(np.full(n, clip.weight, np.float32))
        kinds.append(KIND_ID[clip.kind])
        conds.append(COND_NAMES.index(clip.cond) if clip.cond else -1)
        voices_.append(clip.voice)
        wake_ends.append(-1 if clip.wake_end is None else clip.wake_end)
        audio_len.append(len(clip.audio))
        spans_.append(json.dumps(clip.spans))
    return {
        "emb": np.concatenate(rows),
        "starts": np.array(starts, np.int64),
        "n": np.array(ns, np.int32),
        "labels": np.concatenate(labels),
        "weights": np.concatenate(weights),
        "kind": np.array(kinds, np.int8),
        "cond": np.array(conds, np.int8),
        "voice": np.array(voices_),
        "wake_end": np.array(wake_ends, np.int64),
        "audio_len": np.array(audio_len, np.int64),
        "spans": np.array(spans_),
    }


def merge(parts: list[dict]) -> dict:
    out = {}
    base = 0
    starts, bases = [], []
    for p in parts:
        starts.append(p["starts"] + base)
        base += len(p["emb"])
    for k in parts[0]:
        if k == "starts":
            out[k] = np.concatenate(starts)
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def run(name: str, tasks: list[dict], procs: int) -> None:
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(procs, len(tasks))) as pool:
        parts = pool.map(build, tasks)
    d = merge(parts)
    out = WORK / "feat"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", **d)
    nchunks = int(d["n"].sum())
    print(f"{name}: {len(d['n'])} clips, {nchunks} chunks ({nchunks * CHUNK / SR / 3600:.2f} h), {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the training clip counts")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--which", nargs="*", default=["train", "val", "test"])
    a = ap.parse_args()
    import numpy as _np

    from .common import MODELS as M

    names = list(_np.load(M / "voices-v1.0.bin").keys())
    r = roles(names)

    if "train" in a.which:
        shards = a.procs
        tasks = []
        for s in range(shards):
            tasks.append(
                dict(mode="train", voices=r["train"], seed=a.seed * 1000 + s,
                     n_pos=int(420 * a.scale), n_neg=int(450 * a.scale), n_conf=int(200 * a.scale), n_noise=int(40 * a.scale))
            )
        run("train", tasks, a.procs)

    for role in ("val", "test"):
        if role not in a.which:
            continue
        tasks = []
        # one shard per voice for the long streams, and for positives
        for i, v in enumerate(r[role]):
            tasks.append(dict(mode="evalpos", voices=[v], seed=a.seed * 77 + i, repeats=1))
        run(f"{role}_pos", tasks, a.procs)
        tasks = []
        passes = 2 if role == "test" else 3
        for i, v in enumerate(r[role]):
            tasks.append(dict(mode="evalneg", voices=[v], seed=a.seed * 91 + i, passes=passes))
        tasks.append(dict(mode="evalnoise", voices=[], seed=a.seed * 5 + (role == "test"), minutes=10 if role == "test" else 4))
        run(f"{role}_neg", tasks, a.procs)


if __name__ == "__main__":
    main()
