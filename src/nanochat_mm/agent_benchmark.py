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


def first_name(p):
    if not p.startswith("A"): return None
    raw = p[1:]
    if not raw or not raw[0].isupper(): return None
    first = ""
    for ch in raw:
        if ch.isupper() and first: break
        first += ch
    return first if 3 <= len(first) <= 12 else None


class UserMemoryAgent:
    """The full system: frozen LM + AttMem perceptual banks + text fact store."""
    def __init__(self, bolt, tok, inv_temp=100.0, gain=16.0, reject_thresh=0.28):
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
        """End-to-end: perceive -> recall name (bank) -> recall job (text), one pass."""
        mod, key, _ = self._pick(face_key, voice_key)
        ctx = self._context() + "This person works as a"
        ids = self.tok.encode(ctx, add_special_tokens=False)
        return self._forward_last(ids + [self.tok.encode(" ", add_special_tokens=False)[0]], mod, key)


def load_face_pool(tok):
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = defaultdict(list)
    for i, p in enumerate(pid): by[str(p)].append(i)
    cand, seen = [], set()
    for p in by:
        f = first_name(p)
        if f is None or f in seen or len(by[p]) < 2: continue
        t = tok.encode(f, add_special_tokens=False)
        if len(t) != 1: continue
        seen.add(f); cand.append((p, f, t[0]))
    return emb, by, cand


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
    emb, by, cand = load_face_pool(tok)
    profs = [p for p in PROFESSIONS if len(tok.encode(" " + p, add_special_tokens=False)) == 1]
    print(f"{len(cand)} single-token-name identities, {len(profs)} professions")
    agent = UserMemoryAgent(bolt, tok)

    S = 4  # sessions: users enrolled in S batches (memory grows over sessions)
    results = {M: defaultdict(list) for M in Ms}
    for M in Ms:
        if M > len(cand) or 2 * M > len(cand): break
        for seed in range(seeds):
            rng = np.random.default_rng(4000 + seed)
            idx = rng.permutation(len(cand))
            users = [cand[i] for i in idx[:M]]                 # enrolled
            strangers = [cand[i] for i in idx[M:2 * M]]        # never enrolled
            prof_of = {u[1]: profs[j % len(profs)] for j, u in enumerate(users)}
            prof_tid = {f: tok.encode(" " + prof_of[f], add_special_tokens=False)[0] for (_, f, _) in users}
            agent.reset()
            # enroll across S sessions (order does not change the bank, but models the flow)
            enroll_imgs = {}
            for (p, f, tid) in users:
                ix = list(by[p]); rng.shuffle(ix); enroll_imgs[f] = ix
                agent.enroll(f, tid, prof_of[f], face_key=emb[ix[0]])
            name_list = list(agent.name_tid.keys()); name_ids = list(agent.name_tid.values())

            # --- evaluate on held-out cross-condition perceptions ---
            id_ok = comp_ok = kn_accept = 0
            prof_ids = list(prof_tid.values())
            for (p, f, tid) in users:
                q = emb[enroll_imgs[f][1]]                     # a different, held-out photo
                id_ok += int(agent.recognize(face_key=q) == f)               # identify (closed-set)
                logits = agent.answer_job(face_key=q)                        # compose face->job
                comp_ok += int(prof_ids[int(logits[prof_ids].argmax())] == prof_tid[f])
                kn_accept += int(not agent.is_stranger(face_key=q))          # known correctly accepted
            # --- stranger rejection (open-set) ---
            rej_ok = 0
            for (p, f, tid) in strangers:
                ix = list(by[p]); q = emb[ix[0]]
                rej_ok += int(agent.is_stranger(face_key=q))

            results[M]["identify"].append(id_ok / M)
            results[M]["compose"].append(comp_ok / M)
            results[M]["accept_known"].append(kn_accept / M)
            results[M]["reject"].append(rej_ok / M)
            # end-to-end task success on a balanced mixed session: answer the fact for
            # known users AND reject strangers.
            results[M]["end_to_end"].append(0.5 * (comp_ok / M) + 0.5 * (rej_ok / M))
            print(f"  M={M} seed={seed}: identify {id_ok/M:.3f}  compose {comp_ok/M:.3f}  "
                  f"accept-known {kn_accept/M:.3f}  reject {rej_ok/M:.3f}", flush=True)

    def ci(v): return 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    out = {"model": MODEL_ID, "seeds": seeds, "sessions": S, "rows": []}
    print(f"\n=== PerceptAgent: multi-session user-memory agent ({MODEL_ID.split('/')[-1]}) ===")
    print(f"{'M':>5}  {'identify':>16} {'compose':>16} {'reject-stranger':>18} {'end-to-end':>16}")
    for M in Ms:
        if not results[M]["identify"]: continue
        row = {"M": M}
        for k in ["identify", "compose", "accept_known", "reject", "end_to_end"]:
            v = results[M][k]; row[k] = float(np.mean(v)); row[k + "_ci"] = ci(v)
        out["rows"].append(row)
        print(f"{M:>5}  {row['identify']:.3f}+/-{row['identify_ci']:.3f}   {row['compose']:.3f}+/-{row['compose_ci']:.3f}   "
              f"{row['reject']:.3f}+/-{row['reject_ci']:.3f}    {row['end_to_end']:.3f}+/-{row['end_to_end_ci']:.3f}")
    Path("/home/ubuntu/multimodal-user-memory/results/agent_benchmark.json").write_text(json.dumps(out, indent=2))
    print("wrote results/agent_benchmark.json")


if __name__ == "__main__":
    main()
