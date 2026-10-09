"""Evaluate models/telamon.onnx on the held-out voices and pick the threshold.

Threshold is chosen on the VAL voices (no false accepts there, highest
sensitivity), then recall and false accepts per hour are reported on the TEST
voices, which were never used for training or for choosing anything.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort

from .common import HEAD_META, HEAD_PATH, WORK
from .compose import POS_AFTER
from .dataset import COND_NAMES, KIND_ID
from .features import CHUNK, CHUNK_S, HEAD_EMB, SR

MIN_THRESHOLD = 0.90
REFRACTORY = 25  # chunks (2 s) after a trigger during which new ones are ignored
HIT_BEFORE, HIT_AFTER = 0.0, 1.0  # seconds around the word end where a score counts as a detection


def head_scores(sess, d: dict) -> list[np.ndarray]:
    inp = sess.get_inputs()[0].name
    out = []
    emb = d["emb"].astype(np.float32)
    for s, n in zip(d["starts"], d["n"]):
        sc = np.empty(n, np.float32)
        for c in range(n):
            w = emb[s + 1 + c : s + 1 + c + HEAD_EMB][None]
            sc[c] = sess.run(None, {inp: w})[0].reshape(-1)[0]
        out.append(sc)
    return out


def triggers(sc: np.ndarray, thr: float) -> int:
    n, last = 0, -10**9
    for c in np.flatnonzero(sc >= thr):
        if c - last >= REFRACTORY:
            n += 1
            last = c
    return n


def pos_hits(d: dict, scores: list[np.ndarray], thr: float):
    rows = []
    for i, sc in enumerate(scores):
        if d["kind"][i] != KIND_ID["pos"]:
            continue
        we = d["wake_end"][i]
        lo = max(0, int((we + HIT_BEFORE * SR) // CHUNK) - 1)
        hi = min(len(sc), int((we + HIT_AFTER * SR) // CHUNK) + 1)
        rows.append((str(d["voice"][i]), COND_NAMES[d["cond"][i]], bool(sc[lo:hi].max() >= thr)))
    return rows


def neg_stats(d: dict, scores: list[np.ndarray], thr: float):
    """Triggers and hours per negative category. 'confusable' also counts utterances in [2]."""
    out = {"speech": [0, 0.0], "confusable": [0, 0.0, 0], "noise": [0, 0.0]}
    for i, sc in enumerate(scores):
        k = int(d["kind"][i])
        name = {KIND_ID["neg"]: "speech", KIND_ID["conf"]: "confusable", KIND_ID["noise"]: "noise"}.get(k)
        if name is None:
            continue
        out[name][0] += triggers(sc, thr)
        out[name][1] += len(sc) * CHUNK_S / 3600
        if name == "confusable":
            out[name][2] += len(json.loads(str(d["spans"][i])))
    return out


def total_fa(stats) -> tuple[int, float]:
    """The headline false-accept figure: ordinary speech and noise only.

    Confusable phrases are reported separately (they are packed back to back
    in the test streams, far denser than in real life).
    """
    keys = ("speech", "noise")
    return sum(stats[k][0] for k in keys), sum(stats[k][1] for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-meta", action="store_true")
    ap.add_argument("--head", default=str(HEAD_PATH))
    a = ap.parse_args()
    sess = ort.InferenceSession(a.head, providers=["CPUExecutionProvider"])

    data = {}
    for name in ("val_pos", "val_neg", "test_pos", "test_neg"):
        d = np.load(WORK / "feat" / f"{name}.npz")
        d = {k: d[k] for k in d.files}
        data[name] = (d, head_scores(sess, d))

    grid = np.round(np.arange(0.30, 0.995, 0.01), 2)

    def table(role: str):
        dp, sp = data[f"{role}_pos"]
        dn, sn = data[f"{role}_neg"]
        res = {}
        for t in grid:
            hits = pos_hits(dp, sp, float(t))
            st = neg_stats(dn, sn, float(t))
            fa, hours = total_fa(st)
            res[float(t)] = dict(recall=float(np.mean([h[2] for h in hits])), fa=fa, hours=hours, fah=fa / hours, stats=st)
        return res

    val, test = table("val"), table("test")

    # threshold: smallest with zero speech/noise false accepts on VAL (never below 0.5)
    cands = [t for t in val if val[t]["fa"] == 0 and t >= 0.5]
    thr = min(cands) if cands else max(val)
    # VAL is only ~1.5 h and two voices, so zero false accepts there is weak
    # evidence; keep a safety margin and never go below MIN_THRESHOLD.
    thr = max(thr, MIN_THRESHOLD)
    print(f"\nchosen threshold: max(smallest with zero VAL false accepts = {min(cands) if cands else float('nan'):.2f}, {MIN_THRESHOLD}) = {thr:.2f}")

    print("\nthr   | VAL recall  FA/h  | TEST recall  FA/h  (FAs / hours)")
    for t in grid:
        t = float(t)
        if round(t * 100) % 5 and t != thr:
            continue
        v, s = val[t], test[t]
        print(f"{t:.2f}  | {v['recall']:.3f}   {v['fah']:6.2f} | {s['recall']:.3f}   {s['fah']:6.2f}  ({s['fa']} / {s['hours']:.2f} h)")

    dp, sp = data["test_pos"]
    hits = pos_hits(dp, sp, thr)
    print(f"\nTEST recall at {thr:.2f}: {np.mean([h[2] for h in hits]):.3f} over {len(hits)} held-out positive clips")
    print("  by condition:", {c: round(float(np.mean([h[2] for h in hits if h[1] == c])), 3) for c in COND_NAMES})
    voices = sorted({h[0] for h in hits})
    print("  by voice:    ", {v: round(float(np.mean([h[2] for h in hits if h[0] == v])), 3) for v in voices})
    dn, sn = data["test_neg"]
    st = neg_stats(dn, sn, thr)
    fa, hours = total_fa(st)
    print(f"TEST false accepts at {thr:.2f} (ordinary speech + noise): {fa} in {hours:.2f} h = {fa / hours:.2f}/h")
    for k, v in st.items():
        print(f"  {k:10s} {v[0]} in {v[1]:.2f} h" + (f"  ({v[0]}/{v[2]} confusable utterances triggered)" if k == "confusable" else ""))
    # also the maximum scores seen on negatives, as a margin indicator
    mx = max(float(s.max()) for i, s in enumerate(sn) if dn["kind"][i] != KIND_ID["conf"])
    print(f"  max negative score seen on TEST: {mx:.3f}")

    res = dict(
        threshold=thr,
        test_recall=float(np.mean([h[2] for h in hits])),
        test_fa_per_hour=fa / hours,
        test_neg_hours=hours,
        test_pos_clips=len(hits),
        test_by_condition={c: float(np.mean([h[2] for h in hits if h[1] == c])) for c in COND_NAMES},
        test_fa_breakdown={k: {"count": v[0], "hours": v[1]} for k, v in st.items()},
        test_confusable={"triggers": st["confusable"][0], "utterances": st["confusable"][2]},
        refractory_chunks=REFRACTORY,
    )
    (WORK / "eval.json").write_text(json.dumps(res, indent=2))
    if a.write_meta:
        meta = {
            "threshold": thr,
            "refractory_chunks": REFRACTORY,
            "chunk_samples": CHUNK,
            "input": {"name": "input", "shape": [1, 16, 96], "dtype": "float32"},
            "output": {"name": "output", "shape": [1, 1], "dtype": "float32"},
            "eval_synthetic": {
                "held_out_test_voices": ["af_nova", "am_michael", "bf_emma", "bm_lewis"],
                "recall": res["test_recall"],
                "false_accepts_per_hour": res["test_fa_per_hour"],
                "negative_hours": hours,
                "confusable_utterances_triggered": f"{st['confusable'][0]}/{st['confusable'][2]}",
                "positive_clips": len(hits),
            },
        }
        HEAD_META.write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", HEAD_META)


if __name__ == "__main__":
    main()
