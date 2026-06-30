"""Exp #3: the encoder is the lever. Training-free recall == the encoder's recall,
so swapping the encoder (on the SAME identities) should move recall monotonically
with encoder quality. We compute the cross-condition recall@1 protocol (1 reg/id,
cross-condition queries, cosine-NN over N registered keys) for several face encoders
on matched identity sets. Since AttMem reproduces the encoder exactly, these are
also AttMem's numbers under each encoder.

Matched sets:
  LFW   : ArcFace-R100 (strong) vs AntelopeV2 (strong, different net)
  AgeDB : ArcFace (strong) vs Qwen-VL native vision tokens (general VLM, weak key)
"""
import numpy as np
from collections import defaultdict
from pathlib import Path
import json

EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def recall_at_N(emb, pid, N, n_q=3, draws=range(90, 110)):
    by = defaultdict(list)
    for i, p in enumerate(pid):
        by[str(p)].append(i)
    ids = sorted(by.keys())
    ids = [p for p in ids if len(by[p]) >= 2][:N]
    if len(ids) < N:
        return None
    accs = []
    for s in draws:
        rng = np.random.default_rng(s)
        reg, lab, qs = [], [], []
        for p in ids:
            ix = list(by[p]); rng.shuffle(ix)
            reg.append(emb[ix[0]]); lab.append(p)
            for qi in ix[1:1 + n_q]:
                qs.append((emb[qi], p))
        R = l2(np.stack(reg)); Q = l2(np.stack([q[0] for q in qs]))
        pred = (Q @ R.T).argmax(1)
        accs.append(np.mean([lab[pred[k]] == qs[k][1] for k in range(len(qs))]))
    return float(np.mean(accs)), float(np.std(accs))


def load(name):
    d = np.load(EMB / f"{name}.npz")
    key = "emb" if "emb" in d else "keys"
    emb = d[key].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    return emb, pid


def main():
    sets = {
        "LFW": [("ArcFace-R100", "arcface_lfw_xxl"), ("AntelopeV2", "antelope_lfw_xxl")],
        "AgeDB": [("ArcFace", "arcface_agedb"), ("Qwen-VL native tokens", "qwenvl_agedb_keys")],
    }
    Ns = [10, 50, 100]
    out = {}
    for setname, encoders in sets.items():
        print(f"\n=== {setname}: recall@1 by encoder (training-free AttMem == these) ===")
        print(f"{'encoder':24} " + " ".join(f"N={N:>4}" for N in Ns))
        for label, fname in encoders:
            try:
                emb, pid = load(fname)
            except FileNotFoundError:
                print(f"{label:24} (missing {fname})"); continue
            row = []
            for N in Ns:
                r = recall_at_N(emb, pid, N)
                row.append(f"{r[0]:.3f}" if r else "  -- ")
                out.setdefault(setname, {}).setdefault(label, {})[N] = r[0] if r else None
            n_ids = len([p for p in set(pid.tolist())])
            print(f"{label:24} " + " ".join(f"{v:>6}" for v in row) + f"   ({n_ids} ids, d={emb.shape[1]})")
    Path("/home/ubuntu/multimodal-user-memory/results/encoder_lever.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/encoder_lever.json")
    print("Read: recall moves with encoder quality on the SAME identities; the general")
    print("VLM-native key is far weaker than a purpose-built face encoder. AttMem inherits")
    print("each encoder's recall exactly, so the encoder is the lever.")


if __name__ == "__main__":
    main()
