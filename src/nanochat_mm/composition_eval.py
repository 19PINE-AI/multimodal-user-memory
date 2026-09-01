"""In-model composition: does the recalled marker behave as a token the model can
think with? We register M faces with their real first name as the marker, bind a
one-line fact to each name in the text context, then show a held-out face and ask
the model for the fact. Correct only if the face -> name (bank) -> fact (text)
chain fires inside a single model, with no retrieval re-prompt.

This is the experiment that justifies the paper once recall == encoder: the value
is that the perception becomes an in-context token that composes with text memory.

Metrics (paired over draws):
  recog    : face -> correct name token (bank).                 == encoder recall
  lookup   : given the TRUE name, model returns the right fact. == in-context text
  compose  : face -> recalled name -> fact, end to end.         ~ recog * lookup
  blind    : same fact question with the face withheld.         == chance (1/M)

Usage: python3 composition_eval.py [M] [n_draws]
"""
import sys
from collections import defaultdict
from pathlib import Path
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
from v2_retrieval import split_by_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB = REPO_ROOT / "runs" / "embeddings"
# Distinct single-token professions (verified at runtime; expanded for larger M).
PROFESSIONS = ["teacher", "doctor", "pilot", "chef", "lawyer", "artist", "farmer",
               "nurse", "banker", "writer", "dancer", "soldier", "painter", "singer",
               "actor", "judge", "sailor", "baker", "tailor", "guard", "clerk", "poet",
               "monk", "coach", "miner", "hunter", "priest", "guide", "cook", "maid",
               "agent", "pilot", "ranger", "scout", "mayor", "nun", "spy", "vet"]


def first_name(p):
    if not p.startswith("A"):
        return None
    raw = p[1:]
    if not raw or not raw[0].isupper():
        return None
    first = ""
    for ch in raw:
        if ch.isupper() and first:
            break
        first += ch
    return first if 3 <= len(first) <= 12 else None


def last_logits(bolt, tok, ctx_ids, modality_id=None, key=None):
    """Forward a single sequence; return last-position logits. If key is given, the
    final position is a perceptual (vision) token carrying that encoder embedding."""
    T = len(ctx_ids)
    text_ids_t = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
    mod = [MODALITY_TEXT] * T
    feeds = {}
    if key is not None:
        mod[-1] = int(MODALITY_VISION)
        feeds = {int(MODALITY_VISION): torch.from_numpy(key[None].astype(np.float32)).to(DEVICE)}
    mod_t = torch.tensor([mod], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = bolt(mod_t, text_ids_t, feeds)
    return logits[0, -1, :]


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    inv_temp = float(os.environ.get("ATTMEM_INV_TEMP", "100"))
    gain = float(os.environ.get("ATTMEM_OUT_GAIN", "16"))

    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True).eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    import math
    with torch.no_grad():
        for b in bolt.attmem.banks.values():
            b.log_inv_temp.copy_(torch.tensor(math.log(inv_temp)))
            b.out_gain.copy_(torch.tensor(gain))

    # professions that are single-token, de-duplicated, order-preserving
    profs, _seen_pr = [], set()
    for p in PROFESSIONS:
        if p in _seen_pr:
            continue
        if len(tok.encode(" " + p, add_special_tokens=False)) == 1:
            profs.append(p); _seen_pr.add(p)

    # candidate identities: single-token first name, >=2 samples, distinct names
    cand = []
    seen = set()
    for p in by_id:
        f = first_name(p)
        if f is None or f in seen or len(by_id[p]) < 2:
            continue
        t = tok.encode(f, add_special_tokens=False)
        if len(t) != 1:
            continue
        seen.add(f)
        cand.append((p, f, t[0]))
    print(f"{len(cand)} candidate single-token-name identities, {len(profs)} professions")

    recog_d, lookup_d, compose_d, blind_d = [], [], [], []
    for draw in range(n_draws):
        rng = np.random.default_rng(1000 + draw)
        idx = rng.permutation(len(cand))[:M]
        people = [cand[i] for i in idx]
        prof_of = {people[j][1]: profs[j % len(profs)] for j in range(M)}
        name_tid = {f: tid for (_, f, tid) in people}
        prof_tid = {f: tok.encode(" " + prof_of[f], add_special_tokens=False)[0] for (_, f, _) in people}

        # register one image per person -> name marker
        bolt.reset_banks()
        reg = []
        for (p, f, tid) in people:
            ix = list(by_id[p]); rng.shuffle(ix); reg.append((ix[0], p, f, tid))
        keys = np.stack([emb[r[0]] for r in reg])
        bolt.insert_batch(MODALITY_VISION, torch.from_numpy(keys.astype(np.float32)).to(DEVICE),
                          [r[3] for r in reg])

        # context: "Anna is a teacher. Mark is a doctor. ..."
        ctx = "".join(f"{f} is a {prof_of[f]}. " for (_, f, _) in people)
        name_ids = set(name_tid.values())

        nrec = nlook = ncomp = nblind = ntot = 0
        for (regix, p, f, tid) in reg:
            ix = [i for i in by_id[p] if i != regix]
            rng.shuffle(ix)
            q = ix[0]  # held-out cross-condition image
            # --- recog: "<ctx>You see <FACE>" -> next token among names ---
            pref = tok.encode(ctx + "You see ", add_special_tokens=False)
            seq = pref + [tok.pad_token_id or 0]  # last pos is the vision token
            lg = last_logits(bolt, tok, seq, key=emb[q])
            name_logits = {nm: lg[name_tid[nm]].item() for nm in name_tid}
            pred_name = max(name_logits, key=name_logits.get)
            recog_ok = (pred_name == f)
            # --- lookup: given TRUE name, "<ctx><name> is a" -> profession ---
            seq2 = tok.encode(ctx + f + " is a", add_special_tokens=False)
            lg2 = last_logits(bolt, tok, seq2)  # no key -> pure text
            prof_scores = {nm: lg2[prof_tid[nm]].item() for nm in prof_tid}
            lookup_ok = (max(prof_scores, key=prof_scores.get) == f)  # right prof = the one bound to f
            # --- compose: face -> recalled name -> profession of RECALLED name ---
            seq3 = tok.encode(ctx + pred_name + " is a", add_special_tokens=False)
            lg3 = last_logits(bolt, tok, seq3)
            prof_scores3 = {nm: lg3[prof_tid[nm]].item() for nm in prof_tid}
            pred_prof_name = max(prof_scores3, key=prof_scores3.get)
            compose_ok = (prof_of[pred_prof_name] == prof_of[f]) and recog_ok
            # --- blind: face withheld, "<ctx>You see a person" -> name (chance) ---
            seqb = tok.encode(ctx + "You see a person named", add_special_tokens=False)
            lgb = last_logits(bolt, tok, seqb)
            nb = {nm: lgb[name_tid[nm]].item() for nm in name_tid}
            blind_ok = (max(nb, key=nb.get) == f)
            nrec += recog_ok; nlook += lookup_ok; ncomp += compose_ok; nblind += blind_ok; ntot += 1
        recog_d.append(nrec/ntot); lookup_d.append(nlook/ntot)
        compose_d.append(ncomp/ntot); blind_d.append(nblind/ntot)
        print(f"  draw {draw}: recog {nrec/ntot:.2f}  lookup {nlook/ntot:.2f}  "
              f"compose {ncomp/ntot:.2f}  blind {nblind/ntot:.2f}")

    def ms(x): return f"{np.mean(x):.3f}±{np.std(x, ddof=1) if len(x)>1 else 0:.3f}"
    print(f"\n=== COMPOSITION (M={M}, {n_draws} draws, inv_temp={inv_temp}, gain={gain}) ===")
    print(f"  recog (face->name, bank)        : {ms(recog_d)}")
    print(f"  lookup (true name->fact, text)  : {ms(lookup_d)}")
    print(f"  compose (face->name->fact, e2e) : {ms(compose_d)}")
    print(f"  blind (no face, guess)          : {ms(blind_d)}   chance={1/M:.3f}")
    import json
    (REPO_ROOT / "results" / "composition.json").write_text(json.dumps(
        {"M": M, "n_draws": n_draws, "inv_temp": inv_temp, "gain": gain,
         "recog": float(np.mean(recog_d)), "recog_std": float(np.std(recog_d, ddof=1)),
         "lookup": float(np.mean(lookup_d)), "compose": float(np.mean(compose_d)),
         "compose_std": float(np.std(compose_d, ddof=1)),
         "blind": float(np.mean(blind_d)), "chance": 1/M}, indent=2))
    print("wrote results/composition.json")


if __name__ == "__main__":
    main()
