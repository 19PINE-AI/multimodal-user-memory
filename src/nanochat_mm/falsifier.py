"""Falsifier: does a learned projection of a VLM's CONTEXT-CONDITIONED hidden state
recover the REFERENCED identity in a multi-face scene, where a context-free encoder
cannot?

Setup: composite = two faces side by side (target + distractor). We feed Qwen2.5-VL
the composite + "the person on the {left|right}" and take the LM hidden state at the
final token (context-conditioned: the model has seen the scene and which side). We
train a linear g on that hidden state with a supervised-contrastive identity loss, and
test referent recall@1 on HELD-OUT identities and composites.

Decisive comparisons:
  context-query g(h_ref) : the in-model, context-conditioned key.
  no-ref control g(h_0)  : same VLM, NO side instruction (image is ambiguous, 2 faces).
                           If g(h_ref) >> g(h_0), the referential CONTEXT is doing work.
  referent discrimination: does g(h_left) recover the LEFT id and g(h_right) the RIGHT?
                           If h ignores the reference, these collapse -> negative.

A context-free encoder applied to the 2-face composite has no way to pick the referent,
so this is exactly the regime where an in-model context query could exceed retrieval.

Stage 1 (this run): extract + cache hidden states. Stage 2: train g, evaluate.
Usage: python3 falsifier.py extract [n_ids]   |   python3 falsifier.py train
"""
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DEV = "cuda"
VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE = Path("/home/ubuntu/multimodal-user-memory/results/falsifier_h.npz")


def make_composite(imgL, imgR, s=224):
    a = imgL.convert("RGB").resize((s, s)); b = imgR.convert("RGB").resize((s, s))
    canvas = Image.new("RGB", (2 * s, s)); canvas.paste(a, (0, 0)); canvas.paste(b, (s, 0))
    return canvas


def extract(n_ids):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True).eval()

    ds = load_dataset("ljnlonoljpiljm/agedb", split="train")
    by = defaultdict(list)
    for i, ident in enumerate(ds["identity"]):
        by[str(ident)].append(i)
    ids = [p for p in by if len(by[p]) >= 3]
    rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:n_ids]
    id2idx = {p: k for k, p in enumerate(ids)}
    print(f"{len(ids)} identities")

    def hidden(comp, side):
        msgs = [{"role": "user", "content": [{"type": "image", "image": comp},
                 {"type": "text", "text": f"Look at the person on the {side} of the image."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[comp], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    H, Y, REF, SPLIT = [], [], [], []
    img = ds["image"]
    for p in ids:
        idxs = list(by[p]); rng.shuffle(idxs); idxs = idxs[:4]
        for k, ti in enumerate(idxs):
            # distractor: a random other identity's image
            dp = rng.choice([q for q in ids if q != p]); di = rng.choice(by[dp])
            side = "left" if k % 2 == 0 else "right"
            comp = make_composite(img[ti], img[di]) if side == "left" else make_composite(img[di], img[ti])
            H.append(hidden(comp, side)); Y.append(id2idx[p]); REF.append(side)
            SPLIT.append("train" if id2idx[p] < n_ids // 2 else "eval")
        if id2idx[p] % 10 == 0:
            print(f"  {id2idx[p]}/{len(ids)}")
    np.savez(CACHE, H=np.stack(H), Y=np.array(Y), REF=np.array(REF), SPLIT=np.array(SPLIT))
    print(f"cached {len(H)} hidden states -> {CACHE}")


def l2(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def train_eval():
    d = np.load(CACHE, allow_pickle=True)
    H = d["H"].astype(np.float32); Y = d["Y"]; SPLIT = d["SPLIT"]
    tr = SPLIT == "train"; ev = SPLIT == "eval"
    Htr, Ytr = H[tr], Y[tr]; Hev, Yev = H[ev], Y[ev]
    # supervised-contrastive linear g: maximise same-id cosine of g(h)
    Xt = torch.tensor(Htr); lab = torch.tensor(Ytr)
    D = H.shape[1]; g = torch.nn.Linear(D, 256, bias=False)
    torch.nn.init.orthogonal_(g.weight)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    multi = [c for c in set(Ytr.tolist()) if (Ytr == c).sum() >= 2]
    rng = np.random.default_rng(0)
    for step in range(1500):
        bp = rng.choice(multi, size=min(32, len(multi)), replace=False)
        ai, pi = [], []
        for c in bp:
            ix = np.where(Ytr == c)[0]; a, b = rng.choice(ix, 2, replace=False); ai.append(a); pi.append(b)
        A = torch.nn.functional.normalize(g(Xt[ai]), dim=1)
        P = torch.nn.functional.normalize(g(Xt[pi]), dim=1)
        logits = A @ P.T / 0.1
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ai)))
        opt.zero_grad(); loss.backward(); opt.step()

    def recall(emb, y, proj=True):
        with torch.no_grad():
            E = g(torch.tensor(emb)).numpy() if proj else emb
        E = l2(E)
        by = defaultdict(list)
        for i, c in enumerate(y): by[int(c)].append(i)
        ids = [c for c in by if len(by[c]) >= 2]
        accs = []
        for s in range(20):
            rr = np.random.default_rng(s); reg, labs, qs = [], [], []
            for c in ids:
                ix = list(by[c]); rr.shuffle(ix); reg.append(E[ix[0]]); labs.append(c)
                qs.append((E[ix[1]], c))
            R = np.stack(reg); Q = np.stack([q[0] for q in qs]); pred = (Q @ R.T).argmax(1)
            accs.append(np.mean([labs[pred[k]] == qs[k][1] for k in range(len(qs))]))
        return float(np.mean(accs))

    r_ctx = recall(Hev, Yev, proj=True)       # context-query g(h)
    r_raw = recall(Hev, Yev, proj=False)       # raw VLM hidden state (no g), context-free-ish
    print(f"\n=== Falsifier (held-out identities) ===")
    print(f"  context-query  g(h_ref) recall@1 : {r_ctx:.3f}")
    print(f"  raw VLM hidden (no g)    recall@1 : {r_raw:.3f}")
    print(f"  #eval identities: {len(set(Yev.tolist()))}")
    Path("/home/ubuntu/multimodal-user-memory/results/falsifier.json").write_text(json.dumps(
        {"context_query": r_ctx, "raw_hidden": r_raw, "n_eval_ids": len(set(Yev.tolist()))}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    else:
        train_eval()
