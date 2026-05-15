"""Extract LFW at scale — min_faces_per_person=5 instead of 10.

sklearn fetch_lfw_people(min_faces_per_person=5) returns ~3700 identities
with at least 5 photos each. We cap at 1000 to make eval tractable.
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SEED = 42
random.seed(SEED); np.random.seed(SEED)
TARGET_IDS = 1000
PHOTOS_PER_ID = 5


def main():
    print("Extracting LFW at scale (min_faces=5, target 1000 IDs) ...")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=5, color=True, resize=1.0)
    print(f"  LFW: {lfw.images.shape[0]} photos, {len(lfw.target_names)} people")

    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = sorted([(p, idxs) for p, idxs in by_person.items() if len(idxs) >= PHOTOS_PER_ID])
    print(f"  {len(eligible)} people have >= {PHOTOS_PER_ID} photos")
    chosen = random.sample(eligible, k=min(TARGET_IDS, len(eligible)))
    print(f"  sampled {len(chosen)} identities")

    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=['CPUExecutionProvider'],
    )
    inp_name = sess.get_inputs()[0].name

    embs, pids = [], []
    for k, (pid, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=PHOTOS_PER_ID)
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp_name: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(str(pid))
        if (k + 1) % 100 == 0:
            print(f"  processed {k+1}/{len(chosen)} identities, {len(embs)} embeddings")
    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    out = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xl.npz")
    np.savez(out, emb=emb, pid=pid)
    print(f"\n[done] {emb.shape}, {len(set(pid))} identities -> {out}")


if __name__ == "__main__":
    sys.exit(main())
