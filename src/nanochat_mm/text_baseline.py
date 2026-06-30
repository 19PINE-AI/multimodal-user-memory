"""Text-only memory baseline (caption-and-search) vs parametric encoder, paired.

A text-only memory writes a natural-language note about each perception and later
retrieves by text similarity. We give it its best shot: a VLM describes each face for
re-identification, the note is embedded with a sentence encoder, and recognition is
cosine-NN over the stored notes. We score it against the ArcFace encoder and chance on
IDENTICAL registrations/queries (same images, same draws), so the only thing that varies
is text-note vs perceptual-embedding.

Captions and ArcFace embeddings are computed once per (identity, view) and cached, then
many draws sample N identities from the pool -> tight CIs cheaply.

Usage: ATTMEM_VLM=Qwen/Qwen2.5-VL-7B-Instruct python3 text_baseline.py [POOL] [N] [draws]
"""
import sys, os, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from face_encoder import ArcFaceEncoder

DEV = "cuda"; VLM = os.environ.get("ATTMEM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
CAP_PROMPT = ("Describe this person's face for later identification. Note age, gender, "
              "skin tone, hair, face shape, and any distinctive features. 2-3 sentences.")


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()
    arc = ArcFaceEncoder()
    sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)

    ds = load_dataset("ljnlonoljpiljm/agedb", split="train")
    by = defaultdict(list)
    for i, ident in enumerate(ds["identity"]):
        by[str(ident)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:POOL]
    img = ds["image"]

    def caption(pil):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil.convert("RGB")},
                 {"type": "text", "text": CAP_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil.convert("RGB")], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=96, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    # cache caption + arcface for two fixed views per identity
    caps = {}; arcs = {}
    for k, p in enumerate(ids):
        ix = list(by[p]); r = np.random.default_rng(100 + k); r.shuffle(ix)
        v0, v1 = ix[0], ix[1]
        caps[p] = [caption(img[v0]), caption(img[v1])]
        arcs[p] = [arc.encode_pil(img[v0]), arc.encode_pil(img[v1])]
        if k % 10 == 0: print(f"  caption/encode {k}/{POOL}", flush=True)

    txt_emb = {p: sent.encode(caps[p], normalize_embeddings=True) for p in ids}

    def draw(seed):
        r = np.random.default_rng(seed); pick = list(ids); r.shuffle(pick); pick = pick[:N]
        # register view0, query view1
        Rt = np.stack([txt_emb[p][0] for p in pick]); Qt = np.stack([txt_emb[p][1] for p in pick])
        Ra = np.stack([arcs[p][0] for p in pick]);    Qa = np.stack([arcs[p][1] for p in pick])
        Ra /= np.linalg.norm(Ra, axis=1, keepdims=True) + 1e-9
        Qa /= np.linalg.norm(Qa, axis=1, keepdims=True) + 1e-9
        pt = (Qt @ Rt.T).argmax(1); pa = (Qa @ Ra.T).argmax(1)
        gt = np.arange(N)
        return float((pt == gt).mean()), float((pa == gt).mean())

    res = [draw(s) for s in range(2000, 2000 + draws)]
    tt = np.array([x[0] for x in res]); aa = np.array([x[1] for x in res])
    def ci(x): return 1.96 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
    out = {"VLM": VLM, "POOL": POOL, "N": N, "draws": draws,
           "text_caption": {"recall": float(tt.mean()), "ci95": ci(tt)},
           "arcface": {"recall": float(aa.mean()), "ci95": ci(aa)},
           "chance": 1.0 / N, "example_caption": caps[ids[0]][0]}
    print(f"\n=== Text-only vs parametric, AgeDB faces (N={N}, {draws} draws) ===")
    print(f"  text caption-and-search : {tt.mean():.3f} +/- {ci(tt):.3f}")
    print(f"  ArcFace (parametric)    : {aa.mean():.3f} +/- {ci(aa):.3f}")
    print(f"  chance                  : {1.0/N:.3f}")
    print(f"  example caption: {caps[ids[0]][0][:160]}")
    Path("/home/ubuntu/multimodal-user-memory/results/text_baseline.json").write_text(json.dumps(out, indent=2))
    print("wrote results/text_baseline.json")


if __name__ == "__main__":
    main()
