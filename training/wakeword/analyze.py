"""Debug helper: which negatives trigger and which positives are missed."""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import onnxruntime as ort

from .common import HEAD_PATH, WORK
from .evaluate import REFRACTORY, head_scores, pos_hits
from .features import CHUNK, SR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--role", default="test")
    a = ap.parse_args()
    sess = ort.InferenceSession(str(HEAD_PATH), providers=["CPUExecutionProvider"])

    d = np.load(WORK / "feat" / f"{a.role}_neg.npz")
    d = {k: d[k] for k in d.files}
    sc = head_scores(sess, d)
    hits = collections.Counter()
    for i, s in enumerate(sc):
        spans = json.loads(str(d["spans"][i])) if "spans" in d else []
        last = -10**9
        for c in np.flatnonzero(s >= a.thr):
            if c - last < REFRACTORY:
                continue
            last = c
            t = (c + 1) * CHUNK  # window end (samples)
            near = [sp for sp in spans if sp[1] - SR // 2 <= t <= sp[2] + SR // 2]
            txt = " | ".join(sp[0] for sp in near) or "(none/noise)"
            print(f"neg clip {i} voice={d['voice'][i]} kind={d['kind'][i]} t={t / SR:.2f}s score={s[c]:.3f}: {txt}")
            hits[txt] += 1
    print(f"{sum(hits.values())} triggers")

    dp = np.load(WORK / "feat" / f"{a.role}_pos.npz")
    dp = {k: dp[k] for k in dp.files}
    sp_ = head_scores(sess, dp)
    miss = collections.Counter()
    tot = collections.Counter()
    for i, s in enumerate(sp_):
        info = json.loads(str(dp["spans"][i]))
        name = info[0][0] if info else "?"
        we = dp["wake_end"][i]
        lo = max(0, int(we // CHUNK) - 1)
        hi = min(len(s), int((we + SR) // CHUNK) + 1)
        ok = s[lo:hi].max() >= a.thr
        tot[name] += 1
        if not ok:
            miss[name] += 1
    print("\nmisses by wake text (miss/total):")
    for k, v in sorted(miss.items(), key=lambda kv: -kv[1] / tot[kv[0]])[:25]:
        print(f"  {v}/{tot[k]}  {k}")


if __name__ == "__main__":
    main()
