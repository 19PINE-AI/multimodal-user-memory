"""Production multi-session user-memory agent, and a benchmark that evaluates it end to
end. The agent runs the *actual* system: a frozen language model with the AttMem bolt
holding per-modality perceptual banks (ArcFace face keys, ECAPA voice keys), plus a text
store of facts bound to each identity's marker in context. Recognition and fact recall
happen inside one forward pass -- a perceived face/voice recalls its name marker through
the bank, and the name retrieves its facts from the text store.

Benchmark (PerceptAgent): a population of M users is enrolled across S sessions (the memory
grows over time). Users reappear cross-condition (different age/photo for face). We score,
as the memory grows and over seeds with 95% CIs:
  identify   : held-out perception -> correct name                       (perceptual read)
  compose    : held-out perception -> "what is their job?" -> right fact (in-model chain)
  reject     : an un-enrolled stranger -> "new person"                   (open-set)
  end-to-end : a mixed session of the above, overall task success

Usage: ATTMEM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct python3 agent_benchmark.py [seeds] [Ms]
  Ms e.g. "10,25,50,100"
"""
import sys, os, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO
from qwen_attmem_bolt import QwenAttMemBolt, MODEL_ID, DEVICE

EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
PROFESSIONS = ["teacher", "doctor", "pilot", "chef", "lawyer", "artist", "farmer", "nurse",
               "banker", "writer", "dancer", "soldier", "painter", "singer", "actor", "judge",
               "sailor", "baker", "tailor", "guard", "clerk", "poet", "monk", "coach", "miner",
               "hunter", "priest", "guide", "cook", "ranger", "scout", "mayor", "spy", "vet"]
# A large pool of common first names; those that tokenize to a single token become the
# markers we assign to enrolled identities (decoupling the name from the identity string,
# so any face identity can be enrolled and the population can scale).
NAMES = ("Anna Mark John Mary Paul Jane Tom Sara Mike Lucy Emma Jack Kate Ben Anne Sam Rose "
         "David Laura James Alice Peter Nina Chris Julia Adam Clara Luke Grace Ryan Ella Sean "
         "Nora Eric Lily Owen Ruby Cole Maya Zack Faith Neil Iris Dean Joy Kurt Beth Rex Jade "
         "Kyle Dawn Todd Gwen Cody Faye Wade Hope Blake Elsa Drew Tara Seth Vera Chad Lena Reed "
         "Gail Troy Dana Kirk Erin Glen Cora Brad Nell Dale Fern Hugh Opal Roy Greta Karl Ida "
         "Leon Vince Rita Bruno Otto Felix Hugo Milo Axel Ivan Diana Marco Oscar Victor Simon "
         "Louis Bill Bob Joe Dan Tim Ted Ron Ray Guy Max Leo Ian Kim Amy Eve Ada Zoe Mia Ann "
         "Sue Meg Pam Liz Kay Joan Fred Carl Gary Greg Jeff Alan Dave Doug Frank Henry Larry "
         "Nick Phil Ralph Steve Wayne Bruce Craig Keith Scott Grant Lloyd Floyd Vernon Marion "
         "Bernard Gordon Howard Norman Warren Arnold Harold Leslie Melvin Clark Wesley").split()


def first_name(p):  # kept for compatibility; unused in the scaled benchmark
    return None


class UserMemoryAgent:
    """The full system: frozen LM + AttMem perceptual banks + text fact store."""
    def __init__(self, bolt, tok, inv_temp=100.0, gain=16.0, reject_thresh=0.24):
        self.bolt, self.tok = bolt, tok
        self.reject_thresh = reject_thresh
        self.facts = {}          # name -> fact string bound in context
        self.name_tid = {}       # name -> single-token id (the marker)
        self.keys = {MODALITY_VISION: [], MODALITY_AUDIO: []}  # for open-set scoring
        self.markers = {MODALITY_VISION: [], MODALITY_AUDIO: []}
        with torch.no_grad():
            for b in bolt.attmem.banks.values():
                b.log_inv_temp.copy_(torch.tensor(math.log(inv_temp)))
                b.out_gain.copy_(torch.tensor(gain))

    def reset(self):
        self.bolt.reset_banks(); self.facts.clear(); self.name_tid.clear()
        for m in self.keys: self.keys[m] = []; self.markers[m] = []

    def enroll(self, name, marker_tid, fact, face_key=None, voice_key=None):
        self.name_tid[name] = marker_tid; self.facts[name] = fact
        for mod, key in [(MODALITY_VISION, face_key), (MODALITY_AUDIO, voice_key)]:
            if key is not None:
                self.bolt.insert(mod, torch.from_numpy(key.astype(np.float32)).to(DEVICE), marker_tid)
                self.keys[mod].append(key / (np.linalg.norm(key) + 1e-9)); self.markers[mod].append(marker_tid)

    def _context(self):
        # bind each enrolled identity's fact to its name in the model's context
        return "".join(f"{n} is a {self.facts[n]}. " for n in self.facts)

    def _forward_last(self, ctx_ids, mod_id=None, key=None):
        T = len(ctx_ids); mod = [MODALITY_TEXT] * T; feeds = {}
        if key is not None:
            mod[-1] = int(mod_id)
            feeds = {int(mod_id): torch.from_numpy(key[None].astype(np.float32)).to(DEVICE)}
        with torch.no_grad():
            logits = self.bolt(torch.tensor([mod], dtype=torch.long, device=DEVICE),
                               torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE), feeds)
        return logits[0, -1, :]

    def _maxsim(self, mod, key):
        if not self.keys[mod]: return -1.0
        K = np.stack(self.keys[mod]); q = key / (np.linalg.norm(key) + 1e-9)
        return float((q @ K.T).max())

    def _pick(self, face_key, voice_key):
        best = (None, None, -1.0)
        for mod, key in [(MODALITY_VISION, face_key), (MODALITY_AUDIO, voice_key)]:
            if key is not None:
                s = self._maxsim(mod, key)
                if s > best[2]: best = (mod, key, s)
        return best  # (mod, key, maxsim)

    def is_stranger(self, face_key=None, voice_key=None):
        """Open-set gate: reject an identity whose best match is below threshold."""
        return self._pick(face_key, voice_key)[2] < self.reject_thresh

    def recognize(self, face_key=None, voice_key=None):
        """Closed-set: read the recalled name marker as a token (argmax over enrolled)."""
        mod, key, _ = self._pick(face_key, voice_key)
        ctx = self._context() + "This person is"
        ids = self.tok.encode(ctx, add_special_tokens=False) + [self.tok.encode(" ", add_special_tokens=False)[0]]
        logits = self._forward_last(ids, mod, key)
        name_ids = list(self.name_tid.values())
        return list(self.name_tid.keys())[int(logits[name_ids].argmax())]

    def answer_job(self, face_key=None, voice_key=None):
        """End-to-end fact recall, two in-model steps: (1) the perception recalls its name
        marker through the bank; (2) a text-only pass on ``<name> is a'' reads the fact bound
        to that name in context. Returns (profession-logits, recalled_name)."""
        name = self.recognize(face_key, voice_key)               # step 1: perception -> name
        ids = self.tok.encode(self._context() + name + " is a", add_special_tokens=False)
        return self._forward_last(ids), name                     # step 2: name -> fact (text)


def load_face_pool(tok):
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = defaultdict(list)
    for i, p in enumerate(pid): by[str(p)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]                    # any identity with >=2 photos
    # single-token names/professions to serve as markers
    name_pool, seen = [], set()
    for nm in NAMES:
        if nm in seen: continue
        t = tok.encode(nm, add_special_tokens=False)
        if len(t) == 1: name_pool.append((nm, t[0])); seen.add(nm)
    return emb, by, ids, name_pool


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    Ms = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 25, 50, 100]
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True).eval()
    bolt = QwenAttMemBolt(qwen, tok, vision_key_dim=512, audio_key_dim=192,
                          attach_layer=33, attach_lm_head=True).to(DEVICE)
    bolt.install_hook()
    emb, by, ids, name_pool = load_face_pool(tok)
    profs = [p for p in PROFESSIONS if len(tok.encode(" " + p, add_special_tokens=False)) == 1]
    print(f"{len(ids)} face identities, {len(name_pool)} single-token names, {len(profs)} professions")
    agent = UserMemoryAgent(bolt, tok)

    S = 4  # sessions: users enrolled in S batches (memory grows over sessions)
    N_STRANGER = 40  # fixed disjoint stranger pool, independent of M
    results = {M: defaultdict(list) for M in Ms}
    for M in Ms:
        if M > len(name_pool) or M + N_STRANGER > len(ids): break
        for seed in range(seeds):
            rng = np.random.default_rng(4000 + seed)
            idx = rng.permutation(len(ids))
            user_ids = [ids[i] for i in idx[:M]]                       # enrolled identities
            stranger_ids = [ids[i] for i in idx[M:M + N_STRANGER]]     # never enrolled
            nm_idx = rng.permutation(len(name_pool))[:M]
            users = [(user_ids[j], name_pool[nm_idx[j]][0], name_pool[nm_idx[j]][1]) for j in range(M)]
            prof_of = {f: profs[j % len(profs)] for j, (_, f, _) in enumerate(users)}
            prof_tid = {f: tok.encode(" " + prof_of[f], add_special_tokens=False)[0] for (_, f, _) in users}
            agent.reset()
            enroll_imgs = {}
            for (p, f, tid) in users:                                  # enroll across sessions
                ix = list(by[p]); rng.shuffle(ix); enroll_imgs[f] = ix
                agent.enroll(f, tid, prof_of[f], face_key=emb[ix[0]])

            id_ok = fact_ok = comp_ok = kn_accept = 0
            prof_ids = list(prof_tid.values())
            for (p, f, tid) in users:
                q = emb[enroll_imgs[f][1]]                             # held-out cross-condition photo
                name = agent.recognize(face_key=q)                    # perceptual read: face -> name
                id_ok += int(name == f)
                fact_ok += int(agent.facts.get(name) == prof_of[f])   # fact via router: name -> text store
                logits, _ = agent.answer_job(face_key=q)              # in-model compose (facts in context)
                comp_ok += int(prof_ids[int(logits[prof_ids].argmax())] == prof_tid[f])
                kn_accept += int(not agent.is_stranger(face_key=q))   # known correctly accepted
            rej_ok = 0
            for p in stranger_ids:
                q = emb[by[p][0]]
                rej_ok += int(agent.is_stranger(face_key=q))
            n_str = len(stranger_ids)

            results[M]["identify"].append(id_ok / M)
            results[M]["fact_router"].append(fact_ok / M)     # face->name->text-store (production path)
            results[M]["compose_inmodel"].append(comp_ok / M) # face->name->fact all in context (LM-bound)
            results[M]["accept_known"].append(kn_accept / M)
            results[M]["reject"].append(rej_ok / n_str)
            # production end-to-end: recall the fact via the router AND reject strangers.
            results[M]["end_to_end"].append(0.5 * (fact_ok / M) + 0.5 * (rej_ok / n_str))
            print(f"  M={M} seed={seed}: identify {id_ok/M:.3f}  fact(router) {fact_ok/M:.3f}  "
                  f"compose(in-model) {comp_ok/M:.3f}  reject {rej_ok/n_str:.3f}", flush=True)

    def ci(v): return 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    out = {"model": MODEL_ID, "seeds": seeds, "sessions": S, "rows": []}
    print(f"\n=== PerceptAgent: multi-session user-memory agent ({MODEL_ID.split('/')[-1]}) ===")
    print("  M: identify | fact via router (face->name->store) | compose in-model (facts in ctx) | reject | end-to-end")
    for M in Ms:
        if not results[M]["identify"]: continue
        row = {"M": M}
        for k in ["identify", "fact_router", "compose_inmodel", "accept_known", "reject", "end_to_end"]:
            v = results[M][k]; row[k] = float(np.mean(v)); row[k + "_ci"] = ci(v)
        out["rows"].append(row)
        print(f"{M:>5}  id {row['identify']:.3f}  fact-router {row['fact_router']:.3f}  "
              f"compose-inmodel {row['compose_inmodel']:.3f}  reject {row['reject']:.3f}  e2e {row['end_to_end']:.3f}")
    Path("/home/ubuntu/multimodal-user-memory/results/agent_benchmark.json").write_text(json.dumps(out, indent=2))
    print("wrote results/agent_benchmark.json")


if __name__ == "__main__":
    main()
