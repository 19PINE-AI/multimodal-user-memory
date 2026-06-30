"""Exp B: multi-hop. A 2-hop question entered through a perception:
show a face, ask "what does this person's partner do?" -> recognise the face (hop 0),
look up their partner (hop 1, a text fact), look up the partner's profession (hop 2).

We compare in-model AttMem vs single-shot retrieve-and-reprompt. Both recognise with
the same encoder cosine and do the text hops in the same in-context facts, so we
expect parity: AttMem recalls perceptual IDENTITY, not facts, so it has no special
multi-hop power over the fact chain. We report it honestly.

Usage: python3 multihop_eval.py [M] [n_draws]
"""
import sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_VISION, MODALITY_TEXT
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE
EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
PROFS = ["teacher","doctor","pilot","chef","lawyer","artist","farmer","nurse","banker",
         "writer","actor","judge","sailor","baker","tailor","guard","clerk","poet"]


def first_name(p):
    if not p.startswith("A"): return None
    raw = p[1:]
    if not raw or not raw[0].isupper(): return None
    f = ""
    for ch in raw:
        if ch.isupper() and f: break
        f += ch
    return f if 3 <= len(f) <= 12 else None


def last_logits(bolt, ids, key=None):
    T = len(ids); mod = [MODALITY_TEXT]*T; feeds = {}
    if key is not None:
        mod[-1] = int(MODALITY_VISION)
        feeds = {int(MODALITY_VISION): torch.from_numpy(key[None].astype(np.float32)).to(DEVICE)}
    with torch.no_grad():
        lg = bolt(torch.tensor([mod],device=DEVICE), torch.tensor([ids],dtype=torch.long,device=DEVICE), feeds)
    return lg[0,-1,:]


def main():
    M = int(sys.argv[1]) if len(sys.argv)>1 else 8
    nd = int(sys.argv[2]) if len(sys.argv)>2 else 10
    d = np.load(EMB/"arcface_face_xxxl.npz"); emb=d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind=="U" else np.array([str(p) for p in d["pid"]])
    by=defaultdict(list)
    for i,p in enumerate(pid): by[str(p)].append(i)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
    qwen=AutoModelForCausalLM.from_pretrained(MODEL_ID,trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map={"":DEVICE},low_cpu_mem_usage=True).eval()
    bolt=QwenAttMemBolt(qwen,tok,vision_key_dim=512,audio_key_dim=192,attach_layer=33,attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    with torch.no_grad():
        for b in bolt.attmem.banks.values():
            b.log_inv_temp.copy_(torch.tensor(math.log(200.0))); b.out_gain.copy_(torch.tensor(64.0))
    cand,seen=[],set()
    for p in by:
        f=first_name(p)
        if f and f not in seen and len(by[p])>=2:
            t=tok.encode(f,add_special_tokens=False)
            if len(t)==1: seen.add(f); cand.append((p,f,t[0]))

    hop2_in, hop2_pipe = [], []
    for draw in range(nd):
        rng=np.random.default_rng(700+draw)
        people=[cand[i] for i in rng.permutation(len(cand))[:M]]
        names=[p[1] for p in people]; ntid={p[1]:p[2] for p in people}
        prof={names[j]:PROFS[j] for j in range(M)}
        ptid={nm:tok.encode(" "+prof[nm],add_special_tokens=False)[0] for nm in names}
        # partner = a cyclic shift (each person's partner is the next)
        partner={names[j]:names[(j+1)%M] for j in range(M)}
        bolt.reset_banks(); reg=[]
        for (p,f,t) in people:
            ix=list(by[p]); rng.shuffle(ix); reg.append((ix[0],p,f,t))
        keys=np.stack([emb[r[0]] for r in reg])
        regn=keys/(np.linalg.norm(keys,axis=1,keepdims=True)+1e-9)
        bolt.insert_batch(MODALITY_VISION,torch.from_numpy(keys.astype(np.float32)).to(DEVICE),[r[3] for r in reg])
        ctx="".join(f"{nm} is a {prof[nm]}. {nm}'s partner is {partner[nm]}. " for nm in names)
        def retrieve(qe):
            qn=qe/(np.linalg.norm(qe)+1e-9); return reg[int((regn@qn).argmax())][2]
        for (regix,p,f,t) in reg:
            ix=[i for i in by[p] if i!=regix]; rng.shuffle(ix); q=ix[0]
            true_partner=partner[f]; true_ans=prof[true_partner]
            # hop0 recognise (in-model): face -> name
            seq=tok.encode(ctx+"You see ",add_special_tokens=False)+[tok.pad_token_id or 0]
            lg=last_logits(bolt,seq,key=emb[q]); nm_in=max(ntid,key=lambda n:lg[ntid[n]].item())
            # hop1+2 (text): "<nm>'s partner is" -> partner; "<partner> is a" -> prof
            s1=tok.encode(ctx+nm_in+"'s partner is",add_special_tokens=False); l1=last_logits(bolt,s1)
            pn_in=max(ntid,key=lambda n:l1[ntid[n]].item())
            s2=tok.encode(ctx+pn_in+" is a",add_special_tokens=False); l2=last_logits(bolt,s2)
            ans_in=max(ptid,key=lambda n:l2[ptid[n]].item())
            hop2_in.append(prof[ans_in]==true_ans)
            # pipeline: retrieve name, same text hops
            nm_p=reg[[r[3] for r in reg].index(retrieve(emb[q]))][2]; nm_p=[r for r in reg if r[3]==retrieve(emb[q])][0][2]
            s1p=tok.encode(ctx+nm_p+"'s partner is",add_special_tokens=False); l1p=last_logits(bolt,s1p)
            pn_p=max(ntid,key=lambda n:l1p[ntid[n]].item())
            s2p=tok.encode(ctx+pn_p+" is a",add_special_tokens=False); l2p=last_logits(bolt,s2p)
            ans_p=max(ptid,key=lambda n:l2p[ptid[n]].item())
            hop2_pipe.append(prof[ans_p]==true_ans)
    print(f"\n=== Exp B: 2-hop (face -> partner -> profession), M={M}, {nd} draws ===")
    print(f"  in-model end-to-end 2-hop acc : {np.mean(hop2_in):.3f}")
    print(f"  pipeline end-to-end 2-hop acc : {np.mean(hop2_pipe):.3f}")
    print("  (parity expected: both recognise with encoder cosine and do the SAME text hops")
    print("   in context; AttMem recalls identity, not facts, so it has no multi-hop edge.)")
    import json
    Path("/home/ubuntu/multimodal-user-memory/results/multihop.json").write_text(json.dumps(
        {"M":M,"n_draws":nd,"in_model":float(np.mean(hop2_in)),"pipeline":float(np.mean(hop2_pipe))},indent=2))


if __name__=="__main__":
    main()
