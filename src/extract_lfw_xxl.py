"""Extract LFW XXL — min_faces_per_person=3 for ~5000 identities."""
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.datasets import fetch_lfw_people

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def main():
    print("Loading LFW (min_faces=3) ...")
    lfw = fetch_lfw_people(min_faces_per_person=3, color=True, resize=1.0)
    print(f"  {lfw.images.shape[0]} photos, {len(lfw.target_names)} people")

    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = sorted([(p, idxs) for p, idxs in by_person.items() if len(idxs) >= 3])
    print(f"  {len(eligible)} eligible (>= 3 photos)")

    # Take up to 2000 identities
    TARGET = 2000
    chosen = random.sample(eligible, k=min(TARGET, len(eligible)))
    print(f"  sampled {len(chosen)} identities")

    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=['CPUExecutionProvider'],
    )
    inp_name = sess.get_inputs()[0].name

    embs, pids = [], []
    for k, (pid, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=min(3, len(idxs)))
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp_name: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(str(pid))
        if (k + 1) % 200 == 0:
            print(f"  processed {k+1}/{len(chosen)} ids")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    out = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xxl.npz")
    np.savez(out, emb=emb, pid=pid)
    print(f"\n[done] {emb.shape}, {len(set(pid))} identities -> {out}")


if __name__ == "__main__":
    sys.exit(main())
