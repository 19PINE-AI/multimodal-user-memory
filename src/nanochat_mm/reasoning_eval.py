"""Exp #1+#2: in-model memory vs a retrieve-and-reprompt pipeline.

Both recognise with the SAME encoder cosine and reason with the SAME frozen LM, so
we expect accuracy parity; the question is whether the in-model memory (perception
as a token, one integrated model) matches the pipeline (external cosine retrieval +
name substitution + re-prompt) and how cost scales with the number of perceptions.

#1 single-perception composition: face -> bound fact.
#2 multi-perception relational: K faces in one prompt -> "how many distinct people?"
   (a task that forces every perception to be recognised before reasoning).

We report accuracy (both methods) and wall-clock latency per query vs K.

Usage: python3 reasoning_eval.py [M] [n_draws]
"""
import sys, time, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE

EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
NAMES_OK = None  # filled at runtime: single-token first names


def first_name(p):
    if not p.startswith("A"): return None
    raw = p[1:]
    if not raw or not raw[0].isupper(): return None
    f = ""
    for ch in raw:
        if ch.isupper() and f: break
        f += ch
    return f if 3 <= len(f) <= 12 else None


def fwd_last(bolt, ids, key=None):
    T = len(ids)
    mod = [MODALITY_TEXT] * T
    feeds = {}
    if key is not None:
        mod[-1] = int(MODALITY_VISION)
        feeds = {int(MODALITY_VISION): torch.from_numpy(key[None].astype(np.float32)).to(DEVICE)}
    with torch.no_grad():
        lg = bolt(torch.tensor([mod], device=DEVICE),
                  torch.tensor([ids], dtype=torch.long, device=DEVICE), feeds)
    return lg[0, -1, :]


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    import math
    inv_temp, gain = 200.0, 64.0

    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = defaultdict(list)
    for i, p in enumerate(pid):
        by[str(p)].append(i)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True).eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    with torch.no_grad():
        for b in bolt.attmem.banks.values():
            b.log_inv_temp.copy_(torch.tensor(math.log(inv_temp)))
            b.out_gain.copy_(torch.tensor(gain))

    # candidate identities with single-token first names
    cand, seen = [], set()
    for p in by:
        f = first_name(p)
        if f and f not in seen and len(by[p]) >= 2:
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1:
                seen.add(f); cand.append((p, f, t[0]))

    def timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn(); torch.cuda.synchronize()
        return r, time.perf_counter() - t0

    # ---- #1 single-perception composition: in-model vs pipeline ----
    comp_in, comp_pipe, lat_in, lat_pipe = [], [], [], []
    # ---- #2 multi-perception count: in-model vs pipeline, K=2,3,4 ----
    Ks = [2, 3, 4]
    cnt_in = {K: [] for K in Ks}; cnt_pipe = {K: [] for K in Ks}
    latK_in = {K: [] for K in Ks}; latK_pipe = {K: [] for K in Ks}

    for draw in range(n_draws):
        rng = np.random.default_rng(500 + draw)
        people = [cand[i] for i in rng.permutation(len(cand))[:M]]
        profs = ["teacher","doctor","pilot","chef","lawyer","artist","farmer","nurse",
                 "banker","writer","actor","judge","sailor","baker","tailor","guard",
                 "clerk","poet","monk","coach"]
        prof_of = {people[j][1]: profs[j] for j in range(M)}
        name_tid = {f: t for (_, f, t) in people}
        ptid = {f: tok.encode(" "+prof_of[f], add_special_tokens=False)[0] for (_, f, _) in people}
        bolt.reset_banks()
        reg = []
        for (p, f, t) in people:
            ix = list(by[p]); rng.shuffle(ix); reg.append((ix[0], p, f, t))
        keys = np.stack([emb[r[0]] for r in reg])
        regn = keys / (np.linalg.norm(keys, axis=1, keepdims=True)+1e-9)
        bolt.insert_batch(MODALITY_VISION, torch.from_numpy(keys.astype(np.float32)).to(DEVICE),
                          [r[3] for r in reg])
        ctx = "".join(f"{f} is a {prof_of[f]}. " for (_, f, _) in people)

        def retrieve(qemb):  # cosine -> nearest registered name
            qn = qemb/(np.linalg.norm(qemb)+1e-9)
            return reg[int((regn@qn).argmax())][2]

        # --- #1: pick one held-out query per person ---
        for (regix, p, f, t) in reg:
            ix = [i for i in by[p] if i != regix]; rng.shuffle(ix); q = ix[0]
            # in-model: <ctx>You see <FACE>. is a -> profession token argmax over names
            seq = tok.encode(ctx + "You see ", add_special_tokens=False) + [tok.pad_token_id or 0]
            (lg, _t) = timed(lambda: fwd_last(bolt, seq, key=emb[q]))
            nm = max(name_tid, key=lambda n: lg[name_tid[n]].item())  # recalled name
            seq2 = tok.encode(ctx + nm + " is a", add_special_tokens=False)
            lg2 = fwd_last(bolt, seq2)
            comp_in.append(prof_of[max(ptid, key=lambda n: lg2[ptid[n]].item())] == prof_of[f])
            lat_in.append(_t)
            # pipeline: cosine-retrieve name -> reprompt
            def pipe():
                rn = retrieve(emb[q])
                s = tok.encode(ctx + rn + " is a", add_special_tokens=False)
                l = fwd_last(bolt, s)
                return prof_of[max(ptid, key=lambda n: l[ptid[n]].item())], rn
            ((pp, rn), _tp) = timed(pipe)
            comp_pipe.append(pp == prof_of[f]); lat_pipe.append(_tp)

        # --- #2: K-perception distinct-count task ---
        for K in Ks:
            sel = rng.permutation(M)[:K]
            # choose K query images (with possible repeats to vary true count)
            true_people = [reg[s] for s in sel]
            # build held-out query images for each
            qfaces = []
            for (regix, p, f, t) in true_people:
                ix = [i for i in by[p] if i != regix]; rng.shuffle(ix); qfaces.append((emb[ix[0]], f))
            true_count = len(set(f for _, f in qfaces))
            # in-model: recognise each face token, count distinct recalled names
            def cin():
                names = []
                for (qe, _) in qfaces:
                    seq = tok.encode(ctx + "You see ", add_special_tokens=False) + [tok.pad_token_id or 0]
                    lg = fwd_last(bolt, seq, key=qe)
                    names.append(max(name_tid, key=lambda n: lg[name_tid[n]].item()))
                return len(set(names))
            (ci, tci) = timed(cin)
            cnt_in[K].append(ci == true_count); latK_in[K].append(tci)
            # pipeline: cosine-retrieve each -> count distinct names
            def cpipe():
                return len(set(retrieve(qe) for (qe, _) in qfaces))
            (cp, tcp) = timed(cpipe)
            cnt_pipe[K].append(cp == true_count); latK_pipe[K].append(tcp)

    def ms(x): return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x)>1 else 0.0)
    print(f"\n=== #1 single-perception composition (M={M}, {n_draws} draws) ===")
    print(f"  in-model compose acc : {ms(comp_in)[0]:.3f}   latency {np.mean(lat_in)*1000:.1f} ms")
    print(f"  pipeline compose acc : {ms(comp_pipe)[0]:.3f}   latency {np.mean(lat_pipe)*1000:.1f} ms")
    print(f"\n=== #2 multi-perception distinct-count (accuracy; latency ms) ===")
    print(f"  {'K':>2}  {'in-model':>10} {'pipeline':>10}   {'lat_in':>8} {'lat_pipe':>8}")
    for K in Ks:
        print(f"  {K:>2}  {ms(cnt_in[K])[0]:>10.3f} {ms(cnt_pipe[K])[0]:>10.3f}   "
              f"{np.mean(latK_in[K])*1000:>7.1f}m {np.mean(latK_pipe[K])*1000:>7.1f}m")
    import json
    Path("/home/ubuntu/multimodal-user-memory/results/reasoning_eval.json").write_text(json.dumps({
        "M": M, "n_draws": n_draws,
        "compose": {"in_model": ms(comp_in)[0], "pipeline": ms(comp_pipe)[0],
                    "lat_in_ms": float(np.mean(lat_in)*1000), "lat_pipe_ms": float(np.mean(lat_pipe)*1000)},
        "count": {str(K): {"in_model": ms(cnt_in[K])[0], "pipeline": ms(cnt_pipe[K])[0],
                           "lat_in_ms": float(np.mean(latK_in[K])*1000),
                           "lat_pipe_ms": float(np.mean(latK_pipe[K])*1000)} for K in Ks}}, indent=2))
    print("wrote results/reasoning_eval.json")


if __name__ == "__main__":
    main()
