"""Binding follow-up (study "T3"): is the latent-binding failure of code_memory.py
(1 code 0.87, 2 codes 0.06 at 3,000 steps) a training-length artifact?

Subcommands
  train    --chars C --mmax M --seed S [--role]   train write head (+LoRA reader) and save trainable state
  analyze  --ckpt PATH                             LM eval (teacher-forced + greedy), probes, interference

Reuses CodeMemory from the paper repo unchanged; --role adds a structural variant
where latent query group j may only attend to doc line j (slot-per-item write).
"""
from __future__ import annotations

import argparse, json, math, sys, time, logging
from pathlib import Path

import os
import torch
# The study ran on a shared 96 GB GPU capped at 14 GB (fraction 14/96).
# Set BINDING_GPU_FRACTION to reproduce that cap; by default no cap is applied.
if os.environ.get("BINDING_GPU_FRACTION"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["BINDING_GPU_FRACTION"]))
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src/latentmem"))  # the paper's code_memory.py, imported unchanged
import code_memory as cm  # noqa: E402

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path(os.environ.get("BINDING_OUT", _REPO / "results/binding_followup/runs"))
CK = Path(os.environ.get("BINDING_CKPT", _REPO / "runs/binding_followup"))  # *.pt, gitignored
log = logging.getLogger("t3")
A = cm.ALPHANUM
A_IDX = {c: i for i, c in enumerate(A)}


# ---------------------------------------------------------------- data
def make_doc(rng, M, chars, tok):
    """Same as cm.make_doc but also returns line spans, names and code strings."""
    names = rng.choice(cm.NAMES, size=M, replace=False)
    nl = tok("\n", add_special_tokens=False).input_ids
    doc, items, spans, codes = [], [], [], []
    for n in names:
        code = "".join(rng.choice(list(A), size=chars))
        p = tok(f"{n}: ", add_special_tokens=False).input_ids
        c = tok(code, add_special_tokens=False).input_ids
        s = len(doc)
        doc += p + c + nl
        spans.append((s, len(doc)))
        items.append((p, c)); codes.append(code)
    return dict(doc=doc, items=items, spans=spans, names=list(names), codes=codes)


def doc_from(names, codes, tok):
    nl = tok("\n", add_special_tokens=False).input_ids
    doc, items, spans = [], [], []
    for n, code in zip(names, codes):
        p = tok(f"{n}: ", add_special_tokens=False).input_ids
        c = tok(code, add_special_tokens=False).input_ids
        s = len(doc); doc += p + c + nl; spans.append((s, len(doc))); items.append((p, c))
    return dict(doc=doc, items=items, spans=spans, names=list(names), codes=list(codes))


# ---------------------------------------------------------------- model
class Mem(cm.CodeMemory):
    def __init__(self, role=False, **kw):
        super().__init__(MODEL_ID, **kw)
        self.role = role

    def encode(self, docs):
        ids_l = [d["doc"] for d in docs]
        T = max(len(x) for x in ids_l); pad = self.tok.pad_token_id; B = len(docs)
        ids = torch.full((B, T), pad, dtype=torch.long, device=DEVICE)
        am = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
        for i, x in enumerate(ids_l):
            ids[i, :len(x)] = torch.tensor(x, device=DEVICE); am[i, :len(x)] = 1
        with torch.no_grad(), self._ctx():
            h = self.lm(input_ids=ids, attention_mask=am, output_hidden_states=True,
                        use_cache=False).hidden_states[self.cap].float()
        q = self.wq.unsqueeze(0).expand(B, -1, -1)
        kv = self.ln_kv(h)
        if self.role:
            # disallowed = True. query group j -> only tokens of line j
            mask = torch.ones((B, self.k, T), dtype=torch.bool, device=DEVICE)
            for i, d in enumerate(docs):
                groups = np.array_split(np.arange(self.k), len(d["spans"]))
                for g, (s, e) in zip(groups, d["spans"]):
                    mask[i, g[0]:g[-1] + 1, s:e] = False
            mask = mask.repeat_interleave(self.attn.num_heads, dim=0)
            a, _ = self.attn(q, kv, kv, attn_mask=mask, need_weights=False)
        else:
            a, _ = self.attn(q, kv, kv, key_padding_mask=~am.bool(), need_weights=False)
        return self.ln_out(self.wq.unsqueeze(0) + a) * self.out_scale


def build(role, k=16, seed=42, lora_rank=32):
    torch.manual_seed(seed); np.random.seed(seed)
    return Mem(role=role, k=k, n_decode=24, lora_rank=lora_rank)


def save(model, path, meta):
    st = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    blob = {"state": st, "meta": meta}
    if getattr(model, "_opt", None) is not None:
        blob["opt"] = model._opt.state_dict()
    torch.save(blob, path)


def load(path):
    blob = torch.load(path, map_location="cpu")
    meta = blob["meta"]
    model = build(meta["role"], seed=meta["seed"], lora_rank=meta.get("lora_rank", 16))
    missing = model.load_state_dict(blob["state"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    model.eval()
    return model, meta


# ---------------------------------------------------------------- train
def train(model, mmax, chars, steps, batch, lr, seed, conv=False):
    """conv=False: paper protocol (cosine over `steps`).
    conv=True : constant lr after warmup; when the 500-step window loss has not improved by >3%
    for 5000 steps (and step>=8000), anneal linearly to 0 over 1000 steps and stop. Cap = `steps`."""
    opt = torch.optim.AdamW(model.trainable(), lr=lr)
    warm = 150 if conv else max(10, steps // 20)
    rng = np.random.default_rng(seed); t0 = time.time()
    curve, win, best, best_step, anneal_from = [], [], float("inf"), 0, None
    stop_rule = "acc2" if (conv and mmax >= 2) else ("loss" if conv else "fixed")
    best_acc, best_acc_step = 0.0, 0
    stop_reason = "max_steps"
    step = 0
    while step < steps:
        if conv:
            if step < warm: f = step / warm
            elif anneal_from is not None: f = max(0.0, 1 - (step - anneal_from) / 1000)
            else: f = 1.0
        else:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups: g["lr"] = lr * f
        M = int(rng.integers(1, mmax + 1))
        docs = [make_doc(rng, M, chars, model.tok) for _ in range(batch)]
        qs = [int(rng.integers(M)) for _ in docs]
        prompts = [d["items"][q][0] for d, q in zip(docs, qs)]
        codes = [d["items"][q][1] for d, q in zip(docs, qs)]
        model.train()
        ids, am, cmask = model.pack(prompts, codes)
        logits = model.lm_logits(model.encode(docs), ids, am)
        tgt = ids.clone(); tgt[~cmask] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0); opt.step()
        win.append(loss.item()); step += 1
        if step % 500 == 0:
            wl = float(np.mean(win)); win = []
            rec = dict(step=step, loss=wl, lr=lr * f)
            rec["tf_M1"] = tf_quick(model, 1, chars, n=256)
            for Mq in range(2, mmax + 1): rec[f"tf_M{Mq}"] = tf_quick(model, Mq, chars, n=256)
            curve.append(rec)
            log.info("step %5d loss %.3f lr %.1e %s [%.0fs]", step, wl, lr * f,
                     {k: round(v, 3) for k, v in rec.items() if k.startswith("tf_")}, time.time() - t0)
            if stop_rule == "acc2" and anneal_from is None:
                acc = rec[f"tf_M{mmax}"]
                if acc >= best_acc + 0.01: best_acc, best_acc_step = acc, step
                # stop only when top-M teacher-forced exact has not improved by >=0.01 for 4000 steps
                # (never before step 8000, so a late escape from the initial loss plateau is not cut off)
                if wl < best * 0.97: best, best_step = wl, step
                # v8: also require the loss to have stalled (<3% gain in 4000 steps); otherwise slow
                # learners (role-keyed, M=3) are cut off while still improving
                if step >= 8000 and step - best_acc_step >= 4000 and step - best_step >= 4000:
                    anneal_from = step; stop_reason = "acc_plateau"
                    log.info("acc plateau at %d (best tf_M%d %.3f @ %d) -> anneal 1000 steps", step, mmax, best_acc, best_acc_step)
            if stop_rule == "loss" and anneal_from is None:
                if wl < best * 0.97: best, best_step = wl, step
                if step >= 8000 and step - best_step >= 5000:
                    anneal_from = step; stop_reason = "plateau"
                    log.info("plateau at %d (best %.3f @ %d) -> anneal 1000 steps", step, best, best_step)
            if conv and anneal_from is not None and step >= anneal_from + 1000:
                break
            if conv and anneal_from is None and step >= steps - 1000:
                anneal_from = step  # cap reached: anneal the last 1000 steps
    model._opt = opt
    return dict(final_step=step, stop_reason=stop_reason, curve=curve, stop_rule=stop_rule)


@torch.no_grad()
def tf_quick(model, M, chars, n=128, seed=7):
    model.eval(); rng = np.random.default_rng(seed)
    docs = [make_doc(rng, M, chars, model.tok) for _ in range(n)]
    qs = [int(rng.integers(M)) for _ in docs]
    ems = []
    for i in range(0, n, 32):
        d = docs[i:i + 32]; q = qs[i:i + 32]
        ids, am, cm_ = model.pack([x["items"][j][0] for x, j in zip(d, q)], [x["items"][j][1] for x, j in zip(d, q)])
        pred = model.lm_logits(model.encode(d), ids, am).argmax(-1)
        ems += ((pred == ids) | ~cm_).all(dim=1).float().tolist()
    model.train()
    return float(np.mean(ems))


# ---------------------------------------------------------------- LM eval
@torch.no_grad()
def greedy(model, Mlat, prompts, n_tok):
    B = len(prompts); out = [list(p) for p in prompts]
    for _ in range(n_tok):
        T = max(len(o) for o in out); pad = model.tok.pad_token_id
        ids = torch.full((B, T), pad, dtype=torch.long, device=DEVICE)
        am = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
        for i, o in enumerate(out):
            ids[i, :len(o)] = torch.tensor(o, device=DEVICE); am[i, :len(o)] = 1
        temb = model.embed(ids)
        inp = torch.cat([Mlat.to(model.dtype), temb.to(model.dtype)], 1)
        fm = torch.cat([torch.ones(B, model.k, dtype=torch.long, device=DEVICE), am], 1)
        lg = model.lm(inputs_embeds=inp, attention_mask=fm, use_cache=False).logits
        for i, o in enumerate(out):
            o.append(int(lg[i, model.k + len(o) - 1].argmax()))  # last real position -> next token
    return [o[len(p):] for o, p in zip(out, prompts)]


def chars_of(model, tok_ids, n):
    s = model.tok.decode(tok_ids).replace(" ", "").replace("\n", "")
    return s[:n]


@torch.no_grad()
def lm_eval(model, M, chars, n, seed=999, batch=32):
    model.eval(); rng = np.random.default_rng(seed)
    tf, g_exact, g_other_exact, g_any, cq, co = [], [], [], [], [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        docs = [make_doc(rng, M, chars, model.tok) for _ in range(b)]
        qs = [int(rng.integers(M)) for _ in docs]
        prompts = [d["items"][q][0] for d, q in zip(docs, qs)]
        codes = [d["items"][q][1] for d, q in zip(docs, qs)]
        Mlat = model.encode(docs)
        ids, am, cmask = model.pack(prompts, codes)
        pred = model.lm_logits(Mlat, ids, am).argmax(-1)
        # logits index t predicts token t (see cm.lm_logits), so compare directly
        tf += ((pred == ids) | ~cmask).all(dim=1).float().tolist()
        gen = greedy(model, Mlat, prompts, max(len(c) for c in codes) + 2)
        for d, q, gtok in zip(docs, qs, gen):
            s = chars_of(model, gtok, chars)
            want = d["codes"][q]
            others = [c for j, c in enumerate(d["codes"]) if j != q]
            g_exact.append(float(s == want))
            g_other_exact.append(float(any(s == o for o in others)))
            g_any.append(float(s in d["codes"]))
            cq.append(np.mean([a == b for a, b in zip(s.ljust(chars), want)]))
            if others:
                co.append(np.mean([np.mean([a == b for a, b in zip(s.ljust(chars), o)]) for o in others]))
        done += b
    r = dict(M=M, n=n, tf_exact=float(np.mean(tf)), greedy_exact=float(np.mean(g_exact)),
             greedy_other_code_exact=float(np.mean(g_other_exact)), greedy_any_stored_code=float(np.mean(g_any)),
             char_acc_queried=float(np.mean(cq)), char_acc_other=float(np.mean(co)) if co else None,
             char_chance=1 / len(A))
    return r


# ---------------------------------------------------------------- probes
@torch.no_grad()
def collect(model, M, chars, n, seed, batch=64, random_head=None):
    rng = np.random.default_rng(seed); X, slot_y, nq, ny = [], [], [], []
    enc = random_head if random_head is not None else model
    for s in range(0, n, batch):
        docs = [make_doc(rng, M, chars, model.tok) for _ in range(min(batch, n - s))]
        X.append(enc.encode(docs).flatten(1).float().cpu())
        for d in docs:
            slot_y.append([[A_IDX[c] for c in code] for code in d["codes"]])  # [M, chars]
            q = int(rng.integers(M))
            nq.append(cm.NAMES.index(d["names"][q])); ny.append([A_IDX[c] for c in d["codes"][q]])
    return (torch.cat(X), torch.tensor(slot_y), torch.tensor(nq), torch.tensor(ny))


def fit_probe(Xtr, Ytr, Xte, Yte, kind, extra_tr=None, extra_te=None, epochs=40, shuffle=False):
    """Y: [N, chars] class ids. Returns per-char acc and full-code exact on test."""
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    f = lambda X: ((X - mu) / sd)
    Xtr_, Xte_ = f(Xtr), f(Xte)
    if extra_tr is not None:
        oh = lambda i: F.one_hot(i, len(cm.NAMES)).float() * 5.0
        Xtr_ = torch.cat([Xtr_, oh(extra_tr)], 1); Xte_ = torch.cat([Xte_, oh(extra_te)], 1)
    if shuffle:
        Ytr = Ytr[torch.randperm(len(Ytr))]
    nv = len(Xtr_) // 10
    Xva, Yva, Xtr_, Ytr = Xtr_[:nv], Ytr[:nv], Xtr_[nv:], Ytr[nv:]
    D, C = Xtr_.shape[1], Ytr.shape[1]
    if kind == "linear":
        net = nn.Linear(D, C * len(A))
    else:
        net = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU(),
                            nn.Linear(1024, C * len(A)))
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if kind == "mlp" else 3e-4, weight_decay=0.05)
    Xtr_d, Ytr_d = Xtr_.to(DEVICE), Ytr.to(DEVICE)

    def score(X, Y):
        net.eval()
        with torch.no_grad():
            P = torch.cat([net(X[i:i + 512].to(DEVICE)).view(-1, C, len(A)).argmax(-1).cpu()
                           for i in range(0, len(X), 512)])
        net.train()
        return float((P == Y).float().mean()), float((P == Y).all(1).float().mean())

    best = (-1, None)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_d), device=DEVICE)
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]
            loss = F.cross_entropy(net(Xtr_d[j]).view(-1, len(A)), Ytr_d[j].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        va = score(Xva, Yva)
        if va[0] > best[0]:
            best = (va[0], score(Xte_, Yte))
    return dict(char_acc=best[1][0], code_exact=best[1][1])


def probe_suite(model, M, chars, n_train, n_test, full=True, rand_head=None):
    Xtr, Str, NQtr, NYtr = collect(model, M, chars, n_train, seed=101)
    Xte, Ste, NQte, NYte = collect(model, M, chars, n_test, seed=202)
    res = {}
    kinds = ["linear", "mlp"] if full else ["mlp"]
    for kind in kinds:
        for slot in range(M):
            res[f"slot{slot}_{kind}"] = fit_probe(Xtr, Str[:, slot], Xte, Ste[:, slot], kind)
        res[f"namecond_{kind}"] = fit_probe(Xtr, NYtr, Xte, NYte, kind, NQtr, NQte)
    if full:
        res["slot0_mlp_shuffled_labels"] = fit_probe(Xtr, Str[:, 0], Xte, Ste[:, 0], "mlp", shuffle=True)
        if rand_head is not None:
            Rtr, RStr, RNQtr, RNYtr = collect(model, M, chars, n_train, seed=101, random_head=rand_head)
            Rte, RSte, RNQte, RNYte = collect(model, M, chars, n_test, seed=202, random_head=rand_head)
            for slot in range(M):
                res[f"slot{slot}_mlp_untrained_head"] = fit_probe(Rtr, RStr[:, slot], Rte, RSte[:, slot], "mlp")
            res["namecond_mlp_untrained_head"] = fit_probe(Rtr, RNYtr, Rte, RNYte, "mlp", RNQtr, RNQte)
    return res


# ---------------------------------------------------------------- interference
@torch.no_grad()
def interference(model, chars, n=400, seed=303):
    rng = np.random.default_rng(seed)
    def lat(docs):
        return torch.cat([model.encode(docs[i:i + 64]).flatten(1).float() for i in range(0, len(docs), 64)])
    pool1 = lat([make_doc(rng, 1, chars, model.tok) for _ in range(1000)]).mean(0)
    pool2 = lat([make_doc(rng, 2, chars, model.tok) for _ in range(1000)]).mean(0)
    dA, dB, dAB, dBA, dX, dY = [], [], [], [], [], []
    for _ in range(n):
        na, nb = rng.choice(cm.NAMES, 2, replace=False)
        ca, cb = ["".join(rng.choice(list(A), chars)) for _ in range(2)]
        dA.append(doc_from([na], [ca], model.tok)); dB.append(doc_from([nb], [cb], model.tok))
        dAB.append(doc_from([na, nb], [ca, cb], model.tok)); dBA.append(doc_from([nb, na], [cb, ca], model.tok))
        nx, ny = rng.choice(cm.NAMES, 2, replace=False)  # unrelated control pair
        dX.append(doc_from([nx], ["".join(rng.choice(list(A), chars))], model.tok))
        dY.append(doc_from([ny], ["".join(rng.choice(list(A), chars))], model.tok))
    LA, LB, LX, LY = (lat(d) - pool1 for d in (dA, dB, dX, dY))
    LAB, LBA = lat(dAB) - pool2, lat(dBA) - pool2

    def fit(T, P, Q):  # per-sample least squares T ≈ a P + b Q ; return R2, a, b
        r2s, ab = [], []
        for t, p, q in zip(T, P, Q):
            Xm = torch.stack([p, q], 1)
            sol = torch.linalg.lstsq(Xm, t.unsqueeze(1)).solution.squeeze(1)
            res = t - Xm @ sol
            r2s.append(1 - res.pow(2).sum() / t.pow(2).sum()); ab.append(sol)
        ab = torch.stack(ab)
        return float(torch.stack(r2s).mean()), float(ab[:, 0].mean()), float(ab[:, 1].mean())
    cos = lambda u, v: float(F.cosine_similarity(u, v, dim=1).mean())
    r2, a, b = fit(LAB, LA, LB)
    r2c, _, _ = fit(LAB, LX, LY)
    # per-token-group view: does latent token t of the 2-code doc look like A's or B's?
    k = model.k; H = model.H
    tA = F.cosine_similarity(LAB.view(n, k, H), LA.view(n, k, H), dim=2).mean(0).tolist()
    tB = F.cosine_similarity(LAB.view(n, k, H), LB.view(n, k, H), dim=2).mean(0).tolist()
    return dict(cos_AB_A=cos(LAB, LA), cos_AB_B=cos(LAB, LB), cos_AB_sum=cos(LAB, LA + LB),
                cos_AB_unrelated=cos(LAB, LX), cos_AB_BA=cos(LAB, LBA), cos_A_B=cos(LA, LB),
                lin_R2=r2, lin_alpha=a, lin_beta=b, lin_R2_unrelated_control=r2c,
                norm_ratio_AB_over_A=float(LAB.norm(dim=1).mean() / LA.norm(dim=1).mean()),
                per_token_cos_with_A=tA, per_token_cos_with_B=tB)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--chars", type=int, default=6); t.add_argument("--mmax", type=int, default=2)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--role", action="store_true")
    t.add_argument("--steps", type=int, default=3000); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--suffix", default="")
    t.add_argument("--conv", action="store_true"); t.add_argument("--lora_rank", type=int, default=32)
    a = sp.add_parser("analyze")
    a.add_argument("--ckpt", required=True); a.add_argument("--full", action="store_true")
    a.add_argument("--n_train", type=int, default=12000); a.add_argument("--n_test", type=int, default=1500)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    CK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "train":
        tag = f"c{args.chars}_m{args.mmax}_s{args.seed}" + ("_role" if args.role else "") + args.suffix
        model = build(args.role, seed=args.seed, lora_rank=args.lora_rank)
        tr = train(model, args.mmax, args.chars, args.steps, args.batch, 2e-4, args.seed, conv=args.conv)
        meta = dict(tag=tag, chars=args.chars, mmax=args.mmax, seed=args.seed, role=args.role, steps=args.steps,
                    conv=args.conv, lora_rank=args.lora_rank, **tr)
        save(model, CK / f"{tag}.pt", meta)
        log.info("saved %s", tag)
        return
    model, meta = load(args.ckpt)
    tag = meta["tag"]; chars = meta["chars"]; mmax = meta["mmax"]
    out = dict(meta=meta, lm={}, probes={}, interference=None)
    for M in range(1, max(mmax, 2) + 1 if mmax > 1 else 2):
        out["lm"][M] = lm_eval(model, M, chars, 512)
        log.info("%s lm M=%d %s", tag, M, out["lm"][M])
    rand = None
    if args.full:
        rand = build(meta["role"], seed=meta["seed"], lora_rank=meta.get("lora_rank", 16))  # untrained write head, same init
        rand.eval()
    for M in sorted({1, mmax}):
        out["probes"][M] = probe_suite(model, M, chars, args.n_train, args.n_test, full=args.full,
                                       rand_head=rand if M == mmax else None)
        log.info("%s probes M=%d %s", tag, M, json.dumps(out["probes"][M]))
    if mmax >= 2:
        out["interference"] = interference(model, chars)
        log.info("%s interference %s", tag, {k: v for k, v in out["interference"].items() if not k.startswith("per_token")})
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", OUT / f"{tag}.json")


if __name__ == "__main__":
    main()
