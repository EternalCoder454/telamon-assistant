"""Train the Telamon head and export it as models/telamon.onnx.

Architecture follows openWakeWord's training head: Flatten -> Linear -> LayerNorm
-> ReLU (x2) -> Linear -> Sigmoid, on [1, 16, 96] embeddings, output [1, 1].
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from .common import HEAD_PATH, WORK
from .dataset import KIND_ID
from .evaluate import pos_hits, triggers
from .features import CHUNK_S, EMB_DIM, HEAD_EMB


class Head(nn.Module):
    def __init__(self, dim: int = 64, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(HEAD_EMB * EMB_DIM, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load(name: str):
    d = np.load(WORK / "feat" / f"{name}.npz")
    return {k: d[k] for k in d.files}


def window_index(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-chunk first row of its 16-row window, label, weight (chunks flattened over clips)."""
    first = np.concatenate([s + 1 + np.arange(n) for s, n in zip(d["starts"], d["n"])])
    return first, d["labels"], d["weights"]


def gather(emb: np.ndarray, first: np.ndarray) -> torch.Tensor:
    idx = first[:, None] + np.arange(HEAD_EMB)[None, :]
    return torch.from_numpy(emb[idx].astype(np.float32))


def clip_scores(model, d: dict) -> list[np.ndarray]:
    """Head scores for every chunk of every clip of a feature set (torch)."""
    first = np.concatenate([s + 1 + np.arange(n) for s, n in zip(d["starts"], d["n"])])
    out = []
    with torch.no_grad():
        for i in range(0, len(first), 8192):
            out.append(model(gather(d["emb"], first[i : i + 8192])).squeeze(1).numpy())
    flat = np.concatenate(out)
    bounds = np.cumsum(d["n"])[:-1]
    return np.split(flat, bounds)


def select_metric(model, vp: dict, vn: dict, thr: float = 0.9):
    """(recall, false accepts, hours) on the held-out VAL voices at `thr`."""
    sp, sn = clip_scores(model, vp), clip_scores(model, vn)
    recall = float(np.mean([h[2] for h in pos_hits(vp, sp, thr)]))
    fa, hours = 0, 0.0
    for i, sc in enumerate(sn):
        if vn["kind"][i] in (KIND_ID["neg"], KIND_ID["noise"]):
            fa += triggers(sc, thr)
            hours += len(sc) * CHUNK_S / 3600
    return recall, fa, hours


def mine(model, emb, first, lab, w, w0, cap: float = 30.0):
    """Hard-negative mining: boost the weight of negative windows the model scores high."""
    model.eval()
    scores = np.empty(len(lab), np.float32)
    with torch.no_grad():
        for i in range(0, len(lab), 16384):
            scores[i : i + 16384] = model(gather(emb, first[i : i + 16384])).squeeze(1).numpy()
    hard = (lab == 0) & (scores > 0.2)
    w[:] = np.where(hard, np.minimum(w * (1.0 + 8.0 * scores), w0 * cap), w)
    print(f"  mined {int(hard.sum())} hard negatives (score>0.2); max neg score {scores[lab == 0].max():.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--mine-at", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)

    tr = load("train")
    first, lab, w = window_index(tr)
    keep = lab >= 0  # drop ignored chunks
    first, lab, w = first[keep], lab[keep], w[keep].copy()
    w0 = w.copy()
    emb = tr["emb"]
    print(f"train windows: {len(lab)}  pos={int((lab == 1).sum())} neg={int((lab == 0).sum())}", flush=True)
    labf = lab.astype(np.float32)

    vp, vn = load("val_pos"), load("val_neg")

    def val_set(d):
        f, l, ww = window_index(d)
        k = l >= 0
        return gather(d["emb"], f[k]), torch.from_numpy(l[k].astype(np.float32)), torch.from_numpy(ww[k])

    vxp, vyp, vwp = val_set(vp)
    vxn, vyn, vwn = val_set(vn)
    vx, vy, vw = torch.cat([vxp, vxn]), torch.cat([vyp, vyn]), torch.cat([vwp, vwn])

    model = Head(a.dim, a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    bs = 2048
    steps_per_epoch = -(-len(lab) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    bce = nn.BCELoss(reduction="none")
    best_key, best_state = None, None
    t0 = time.time()
    for ep in range(a.epochs):
        if ep in a.mine_at:
            mine(model, emb, first, lab, w, w0)
        model.train()
        perm = rng.permutation(len(lab))
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            x = gather(emb, first[b])
            y = torch.from_numpy(labf[b])
            ww = torch.from_numpy(w[b]) * torch.where(y > 0.5, a.pos_weight, 1.0)
            x = x + 0.02 * torch.randn_like(x) * x.std()  # light feature noise as a regularizer
            loss = (bce(model(x).squeeze(1), y) * ww).sum() / ww.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            p = model(vx).squeeze(1)
            vl = float(((bce(p, vy) * vw).sum() / vw.sum()))
        rec, fa, hours = select_metric(model, vp, vn)
        # prefer: recall >= 0.95, then fewest val false accepts, then higher recall
        key = (0 if rec >= 0.95 else 1, fa, -rec, vl)
        print(f"epoch {ep + 1:2d}  train {tot / len(perm):.4f}  val loss {vl:.4f}  val@0.9: recall {rec:.3f} FA {fa} in {hours:.2f} h  ({time.time() - t0:.0f}s)", flush=True)
        if best_key is None or key < best_key:
            best_key, best_state = key, {k: v.clone() for k, v in model.state_dict().items()}
            print("  ^ best so far", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, HEAD_EMB, EMB_DIM)
    torch.onnx.export(
        model, dummy, str(HEAD_PATH), input_names=["input"], output_names=["output"],
        opset_version=17, dynamo=False,
    )
    # check export against torch
    import onnxruntime as ort

    sess = ort.InferenceSession(str(HEAD_PATH), providers=["CPUExecutionProvider"])
    x = vx[:64]
    ref = model(x).detach().numpy().reshape(-1)
    got = np.array([sess.run(None, {"input": x[i : i + 1].numpy()})[0].reshape(-1)[0] for i in range(len(x))])
    print("onnx vs torch max abs diff:", float(np.abs(ref - got).max()))
    print("exported", HEAD_PATH, HEAD_PATH.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
