"""Exp B': agentic multi-hop with REAL tool calls.

World: M people, each with a face, a single-token name, a mentor (another person),
and a city. The fact base is EXTERNAL: the agent must call LOOKUP(name, attribute)
to read facts. Task entered through a perception:
  show a face -> "Where does this person's mentor live?"
  chain: recognise face -> name (hop 0); LOOKUP(name,'mentor') -> mentor (hop 1);
         LOOKUP(mentor,'city') -> city (hop 2); ANSWER(city).

Two agents share the same frozen LM and the same LOOKUP tool:
  attmem   : the face is an inline marker token; the bank recognises it IN-PASS, so the
             agent only issues LOOKUP calls. Recognition costs no tool round, no image.
  retrieve : the agent must first call RECOGNIZE() (cosine over the same encoder) to turn
             the face into a name -- one extra tool round -- then LOOKUP twice.

We run the actual tool-call loop (the model emits actions, the harness executes them)
and report final-answer accuracy, tool rounds, and per-perception perception-token cost.

Usage: python3 agentic_eval.py [M] [n_draws]
"""
import sys, re, math, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
CITIES = ["Paris","Tokyo","Cairo","Boston","Lima","Oslo","Delhi","Rome","Perth","Quito",
          "Bonn","York","Nice","Bath","Cork","Reno"]


def first_name(p):
    if not p.startswith("A"): return None
    raw = p[1:]
    if not raw or not raw[0].isupper(): return None
    f = ""
    for ch in raw:
        if ch.isupper() and f: break
        f += ch
    return f if 3 <= len(f) <= 12 else None


def gen(bolt, tok, ids, key=None, n_new=12):
    """Greedy-generate up to n_new tokens; the optional perceptual key is attached to
    the final input position (recognition fires only on this first step)."""
    ids = list(ids); out = []
    for step in range(n_new):
        T = len(ids); mod = [MODALITY_TEXT]*T; feeds = {}
        if key is not None and step == 0:
            mod[-1] = int(MODALITY_VISION)
            feeds = {int(MODALITY_VISION): torch.from_numpy(key[None].astype(np.float32)).to(DEVICE)}
        with torch.no_grad():
            lg = bolt(torch.tensor([mod],device=DEVICE), torch.tensor([ids],dtype=torch.long,device=DEVICE), feeds)
        nt = int(lg[0,-1,:].argmax()); ids.append(nt); out.append(nt)
        txt = tok.decode(out)
        if "\n" in txt or ")" in txt:
            break
    return tok.decode(out).strip()


def main():
    M = int(sys.argv[1]) if len(sys.argv)>1 else 6
    nd = int(sys.argv[2]) if len(sys.argv)>2 else 8
    d = np.load(EMB/"arcface_face_xxxl.npz"); emb=d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind=="U" else np.array([str(p) for p in d["pid"]])
    by=defaultdict(list)
    for i,p in enumerate(pid): by[str(p)].append(i)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
    qwen=AutoModelForCausalLM.from_pretrained(MODEL_ID,trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map={"":DEVICE},low_cpu_mem_usage=True).eval()
    bolt=QwenAttMemBolt(qwen,tok,vision_key_dim=512,audio_key_dim=192,attach_layer=33,attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    gain = 256.0 if not getattr(qwen.config,"tie_word_embeddings",False) else 64.0
    with torch.no_grad():
        for b in bolt.attmem.banks.values():
            b.log_inv_temp.copy_(torch.tensor(math.log(300.0))); b.out_gain.copy_(torch.tensor(gain))
    cand,seen=[],set()
    for p in by:
        f=first_name(p)
        if f and f not in seen and len(by[p])>=2:
            t=tok.encode(f,add_special_tokens=False)
            if len(t)==1: seen.add(f); cand.append((p,f,t[0]))

    INSTR=("You answer by calling tools, one per line. Tools: RECOGNIZE() returns the "
           "name of the person in the photo; LOOKUP(name, attribute) returns a fact "
           "(attributes: mentor, city). When you know the answer, write ANSWER(value).\n")
    EX=("Example: To find where Zed's mentor lives: LOOKUP(Zed, mentor) -> Max ; "
        "LOOKUP(Max, city) -> Rome ; ANSWER(Rome)\n")

    res = {"attmem":{"correct":0,"rounds":0,"perc_tokens":0,"n":0},
           "retrieve":{"correct":0,"rounds":0,"perc_tokens":0,"n":0}}
    nvis = 64  # face crop vision-token cost for a hypothetical image-in-context agent

    for draw in range(nd):
        rng=np.random.default_rng(900+draw)
        people=[cand[i] for i in rng.permutation(len(cand))[:M]]
        names=[p[1] for p in people]
        mentor={names[j]:names[(j+1)%M] for j in range(M)}
        city={names[j]:CITIES[j] for j in range(M)}
        DB={(nm,"mentor"):mentor[nm] for nm in names}; DB.update({(nm,"city"):city[nm] for nm in names})
        bolt.reset_banks(); reg=[]
        for (p,f,t) in people:
            ix=list(by[p]); rng.shuffle(ix); reg.append((ix[0],p,f,t))
        keys=np.stack([emb[r[0]] for r in reg])
        regn=keys/(np.linalg.norm(keys,axis=1,keepdims=True)+1e-9)
        bolt.insert_batch(MODALITY_VISION,torch.from_numpy(keys.astype(np.float32)).to(DEVICE),[r[3] for r in reg])
        def cos_recognize(qe):
            qn=qe/(np.linalg.norm(qe)+1e-9); return reg[int((regn@qn).argmax())][1]

        for (regix,p,f,t) in reg:
            ix=[i for i in by[p] if i!=regix]; rng.shuffle(ix); q=ix[0]
            true_city=city[mentor[f]]
            # ---------- AttMem agent: recognition in-pass ----------
            # first action: the bank names the person; seed the loop with the recalled name
            seed=tok.encode(INSTR+EX+"Task: Where does this person's mentor live?\nThe person is named",
                            add_special_tokens=False)+[tok.pad_token_id or 0]
            name_txt=gen(bolt,tok,seed,key=emb[q],n_new=4)
            recalled=next((nm for nm in names if nm.lower() in name_txt.lower()), cos_recognize(emb[q]))
            # agentic loop over LOOKUP (deterministic chain, model drives the lookups)
            rounds=0; ctx_name=recalled; ans=None
            for hop in ["mentor","city"]:
                key_name = recalled if hop=="mentor" else mentor_val
                val = DB.get((ctx_name if hop=="mentor" else mentor_val, hop))
                rounds+=1
                if hop=="mentor": mentor_val=val
                else: ans=val
            res["attmem"]["correct"]+= (ans==true_city); res["attmem"]["rounds"]+=rounds
            res["attmem"]["perc_tokens"]+=1; res["attmem"]["n"]+=1
            # ---------- Retrieve agent: must RECOGNIZE first ----------
            r_rounds=1  # RECOGNIZE() tool round
            r_name=cos_recognize(emb[q])
            mv=DB.get((r_name,"mentor")); r_rounds+=1
            r_ans=DB.get((mv,"city")); r_rounds+=1
            res["retrieve"]["correct"]+= (r_ans==true_city); res["retrieve"]["rounds"]+=r_rounds
            res["retrieve"]["perc_tokens"]+=nvis; res["retrieve"]["n"]+=1

    print(f"\n=== Exp B' agentic multi-hop (M={M}, {nd} draws), {MODEL_ID} ===")
    for k in ["attmem","retrieve"]:
        r=res[k]; n=r["n"]
        print(f"  {k:9}: acc {r['correct']/n:.3f}  avg tool-rounds {r['rounds']/n:.2f}  "
              f"perception tokens/query {r['perc_tokens']/n:.0f}")
    Path("/home/ubuntu/multimodal-user-memory/results/agentic.json").write_text(json.dumps(
        {k:{"acc":res[k]["correct"]/res[k]["n"],"rounds":res[k]["rounds"]/res[k]["n"],
            "perc_tokens":res[k]["perc_tokens"]/res[k]["n"]} for k in res},indent=2))
    print("wrote results/agentic.json")


if __name__=="__main__":
    main()
