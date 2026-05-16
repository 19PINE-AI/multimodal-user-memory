"""Expand LFW training pool: min_faces=2 → all IDs with ≥2 photos.

Then combine with AgeDB (already extracted; identity-disjoint by prefix)
to form arcface_face_xxxl.npz for the continual-pretraining experiment.
"""
import os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.datasets import fetch_lfw_people

SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT_LFW = "/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw_xxxl.npz"
OUT_COMBINED = "/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_face_xxxl.npz"


def main():
    print("Loading LFW (min_faces=2) ...")
    lfw = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
    print(f"  {lfw.images.shape[0]} photos, {len(lfw.target_names)} people with ≥2 photos")

    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = [(p, idxs) for p, idxs in by_person.items() if len(idxs) >= 2]
    print(f"  {len(eligible)} eligible identities")

    print("\nLoading ArcFace R50 (buffalo_l)...")
    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=["CPUExecutionProvider"],
    )
    inp_name = sess.get_inputs()[0].name

    embs, pids = [], []
    n_processed = 0
    import time
    t0 = time.time()
    for pid, idxs in eligible:
        # Take up to 4 samples per identity to keep dataset size moderate
        sel = random.sample(idxs, k=min(4, len(idxs)))
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp_name: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(f"L{pid}")
        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed}/{len(eligible)} IDs ({time.time()-t0:.0f}s)")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    np.savez(OUT_LFW, emb=emb, pid=pid)
    print(f"\n[saved] {OUT_LFW}: {emb.shape}, {len(set(pid))} IDs")

    # Now combine with AgeDB (already extracted with 'A' prefix)
    print("\nCombining with AgeDB ...")
    agedb = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_agedb.npz")
    agedb_pid = np.array([f"A{p}" for p in agedb["pid"]])
    combined_emb = np.concatenate([emb, agedb["emb"].astype(np.float32)], axis=0)
    combined_pid = np.concatenate([pid, agedb_pid])
    np.savez(OUT_COMBINED, emb=combined_emb, pid=combined_pid)
    print(f"[saved] {OUT_COMBINED}: {combined_emb.shape}, "
          f"{len(set(combined_pid.tolist()))} IDs (combined)")


if __name__ == "__main__":
    import sys
    sys.exit(main())
