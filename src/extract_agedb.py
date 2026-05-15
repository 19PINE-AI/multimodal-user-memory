"""Extract ArcFace embeddings on AgeDB for cross-age benchmark.

AgeDB has 16,488 images of celebrities with explicit identity + age
labels. We extract embeddings AND age metadata, then partition
later into cross-age pairs (e.g., young photo as registration vs
old photo as query for the same identity).
"""
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from datasets import load_dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TARGET_IDS = 500       # cap
PHOTOS_PER_ID = 6       # min per identity to keep


def main():
    print("Loading AgeDB ...")
    ds = load_dataset("ljnlonoljpiljm/agedb", split="train")
    print(f"  {len(ds)} images")

    # Inspect age distribution
    ages = ds["age"]
    identities = ds["identity"]
    print(f"  age range: {min(ages)} - {max(ages)}")
    print(f"  unique identities: {len(set(identities))}")

    # Group by identity, keep only those with multiple ages and >= PHOTOS_PER_ID
    by_id = defaultdict(list)
    for i, (idn, age) in enumerate(zip(identities, ages)):
        by_id[idn].append((i, int(age)))

    eligible = []
    for idn, items in by_id.items():
        if len(items) >= PHOTOS_PER_ID:
            ages_set = {age for _, age in items}
            if len(ages_set) >= 2:  # at least two distinct ages
                age_span = max(ages_set) - min(ages_set)
                if age_span >= 10:  # at least 10-year span
                    eligible.append((idn, items, age_span))
    eligible.sort(key=lambda x: -x[2])  # prefer larger age spans
    print(f"  {len(eligible)} identities with >= {PHOTOS_PER_ID} photos AND >= 10yr age span")

    chosen = eligible[:TARGET_IDS]
    print(f"  taking top {len(chosen)} identities by age span")

    sess = ort.InferenceSession(
        "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx",
        providers=['CPUExecutionProvider'],
    )
    inp_name = sess.get_inputs()[0].name

    embs, pids, age_list = [], [], []
    for k, (idn, items, span) in enumerate(chosen):
        sel = random.sample(items, k=min(PHOTOS_PER_ID, len(items)))
        for i, age in sel:
            img = ds[i]["image"]
            try:
                img = img.convert("RGB")
                arr_img = np.array(img)
                if arr_img.shape[-1] == 3:
                    arr_img = arr_img[..., ::-1]  # RGB -> BGR
                arr_img = cv2.resize(arr_img, (112, 112), interpolation=cv2.INTER_LINEAR)
                arr = ((arr_img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
                e = sess.run(None, {inp_name: arr})[0][0]
                e = e / (np.linalg.norm(e) + 1e-9)
                embs.append(e); pids.append(str(idn)); age_list.append(age)
            except Exception as ex:
                pass
        if (k + 1) % 100 == 0:
            print(f"  processed {k+1}/{len(chosen)} ids, {len(embs)} embeddings")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    age_arr = np.array(age_list, dtype=np.int32)
    out = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_agedb.npz")
    np.savez(out, emb=emb, pid=pid, age=age_arr)
    print(f"\n[done] {emb.shape}, {len(set(pid))} identities -> {out}")
    print(f"  age summary: min={age_arr.min()} max={age_arr.max()} mean={age_arr.mean():.1f}")


if __name__ == "__main__":
    sys.exit(main())
