"""Multimodal user-memory benchmark + system: the regime where HYBRID wins.

Real scenario (a personal assistant): recognize a user by their FACE (perceptual,
non-captionable) and recall an EXACT private fact about them (a booking code, a
name, an address -- high-entropy, multi-token). Recalling it needs BOTH legs:
  * perceptual identity  -> latent wins (faces cannot be captioned apart)
  * exact fact content   -> text wins  (a single latent cannot hold exact strings)

Three architectures, one frozen LM, identical registered population:
  text_only    caption(face) -> exact fact via dict     (PERCEPTUAL leg fails)
  latent_only  face -> retrieve a latent code -> DECODE the fact string from it
               (a faithful ICAE-style latent codec; FACT leg fails for exact content)
  hybrid       face -> identity (latent) -> exact fact via text dict (both legs OK)

The latent codec is leak-free: the fact string is encoded into k soft tokens M,
then decoded by L content-free learned query tokens that attend only to M (no
teacher-forced ground-truth tokens in context), so M must actually carry the
content. This is the honest test of "can a latent hold an exact fact".

Metrics: exact-match (whole string) and token-F1, over multiple fact lengths
(entropy), pool sizes N, and seeds.

This module is built to iterate: `--mode codec` validates/trains the latent
fact codec alone (the crux); `--mode bench` runs the full 3-architecture eval.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path("/home/ubuntu/multimodal-user-memory")
EMB_FILE = ROOT / "runs" / "embeddings" / "arcface_lfw_xxxl.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger("multimem")

ALPHANUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"          # no ambiguous 0/O/1/I


# ---------------------------------------------------------------------------
# Realistic exact facts (varying entropy via code length)
# ---------------------------------------------------------------------------
def make_fact(rng: np.random.Generator, n_chars: int) -> str:
    """A realistic high-entropy private fact: a booking/locker code."""
    code = "".join(rng.choice(list(ALPHANUM), size=n_chars))
    return f"code {code}"


# ---------------------------------------------------------------------------
# Latent fact codec: string -> M (k soft tokens) -> decode L tokens from M alone
# ---------------------------------------------------------------------------
class FactCodec(nn.Module):
    def __init__(self, model_id: str, k: int = 16, n_decode: int = 16,
                 lora_rank: int = 16, dtype=torch.bfloat16):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.H = self.lm.config.hidden_size
        self.cap = self.lm.config.num_hidden_layers // 2     # mid-layer capture
        self.embed = self.lm.get_input_embeddings()
        self.dtype = dtype
        self.k = k
        self.L = n_decode
        emb_rms = self.embed.weight.float().norm(dim=-1).mean().item() / (self.H ** 0.5)

        self.has_lora = lora_rank > 0
        if self.has_lora:
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            self.embed = self.lm.get_input_embeddings()

        # write head: k learned queries cross-attend the string's hidden states
        self.wq = nn.Parameter(torch.randn(k, self.H) * 0.02)
        self.attn = nn.MultiheadAttention(self.H, 8, batch_first=True)
        self.ln_kv = nn.LayerNorm(self.H)
        self.ln_out = nn.LayerNorm(self.H)
        self.out_scale = nn.Parameter(torch.tensor(emb_rms))
        # content-free decode queries (one per output position)
        self.dq = nn.Parameter(torch.randn(self.L, self.H) * 0.02)
        self.to(DEVICE)
        for m in (self.attn, self.ln_kv, self.ln_out):
            m.float()

    def trainable(self):
        ps = [self.wq, self.out_scale, self.dq, *self.attn.parameters(),
              *self.ln_kv.parameters(), *self.ln_out.parameters()]
        if self.has_lora:
            ps += [p for p in self.lm.parameters() if p.requires_grad]
        return ps

    def encode(self, strings):
        ids = self.tok(strings, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            ctx = self.lm.disable_adapter() if self.has_lora else _null()
            with ctx:
                h = self.lm(input_ids=ids.input_ids, attention_mask=ids.attention_mask,
                            output_hidden_states=True, use_cache=False
                            ).hidden_states[self.cap].float()
        q = self.wq.unsqueeze(0).expand(h.shape[0], -1, -1)
        kv = self.ln_kv(h)
        a, _ = self.attn(q, kv, kv, key_padding_mask=~ids.attention_mask.bool(),
                         need_weights=False)
        M = self.ln_out(self.wq.unsqueeze(0) + a) * self.out_scale
        return M                                              # [B, k, H]

    def decode_logits(self, M):
        """Predict L fact tokens from M alone (decode queries are content-free)."""
        B = M.shape[0]
        dq = self.dq.unsqueeze(0).expand(B, -1, -1).to(self.dtype)
        inp = torch.cat([M.to(self.dtype), dq], dim=1)        # [B, k+L, H]
        attn = torch.ones(inp.shape[:2], device=DEVICE, dtype=torch.long)
        out = self.lm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        return out.logits[:, self.k:self.k + self.L, :].float()   # [B, L, V]

    def fact_token_ids(self, strings):
        """Tokenize facts to exactly L tokens (pad/truncate) for fixed decode."""
        ids = self.tok(strings, add_special_tokens=False).input_ids
        pad = self.tok.pad_token_id
        out = [(x + [pad] * self.L)[:self.L] for x in ids]
        return torch.tensor(out, device=DEVICE)               # [B, L]

    @torch.no_grad()
    def encode_facts(self, strings, batch=128):
        """Batch-encode fact strings into the M bank [N, k, H] (fp32, on device)."""
        self.eval()
        out = [self.encode(strings[s:s + batch]).float() for s in range(0, len(strings), batch)]
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def decode_ids(self, M, batch=128):
        """Decode M [N, k, H] -> predicted fact token ids [N, L]."""
        self.eval()
        out = [self.decode_logits(M[s:s + batch]).argmax(-1) for s in range(0, M.shape[0], batch)]
        return torch.cat(out, dim=0)


def _null():
    import contextlib
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Train / eval the codec (the crux: can a latent hold an exact fact?)
# ---------------------------------------------------------------------------
def gen_facts(n, seed, n_chars):
    rng = np.random.default_rng(seed)
    return [make_fact(rng, n_chars) for _ in range(n)]


def codec_metrics(codec, strings, batch=64):
    codec.eval()
    tgt = codec.fact_token_ids(strings)
    em, f1n = [], []
    with torch.no_grad():
        for s in range(0, len(strings), batch):
            M = codec.encode(strings[s:s + batch])
            pred = codec.decode_logits(M).argmax(-1)          # [b, L]
            t = tgt[s:s + batch]
            pad = codec.tok.pad_token_id
            mask = t != pad
            exact = ((pred == t) | ~mask).all(dim=1).float()
            em.append(exact.cpu())
            tokacc = ((pred == t) & mask).sum(1).float() / mask.sum(1).clamp(min=1)
            f1n.append(tokacc.cpu())
    return float(torch.cat(em).mean()), float(torch.cat(f1n).mean())


def train_codec(codec, n_chars, steps, batch, lr, seed, eval_n=512):
    opt = torch.optim.AdamW(codec.trainable(), lr=lr)
    warm = max(10, steps // 20)

    def lr_lambda(step):                                   # warmup -> cosine
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    held = gen_facts(eval_n, seed + 99, n_chars)
    rng = np.random.default_rng(seed)
    t0 = time.time(); best = 0.0
    for step in range(steps):
        strings = [make_fact(rng, n_chars) for _ in range(batch)]
        tgt = codec.fact_token_ids(strings)
        codec.train()
        M = codec.encode(strings)
        logits = codec.decode_logits(M)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=codec.tok.pad_token_id)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.trainable(), 1.0)
        opt.step(); sched.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            em, f1 = codec_metrics(codec, held)
            best = max(best, em)
            log.info("  step %4d loss %.3f | exact=%.3f tokacc=%.3f [%.0fs]",
                     step, loss.item(), em, f1, time.time() - t0)
    return best


def run_codec(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    codec = FactCodec(args.model_id, k=args.k, n_decode=args.n_decode,
                      lora_rank=args.lora_rank)
    log.info("codec trainable params: %.2fM", sum(p.numel() for p in codec.trainable()) / 1e6)
    rows = []
    for nch in args.fact_chars:
        log.info("=== fact length %d chars ===", nch)
        # fresh head per length so capacity is per-condition (reuse LM)
        if rows:  # reinit write/decode heads + lora for a clean per-length fit
            codec = FactCodec(args.model_id, k=args.k, n_decode=args.n_decode,
                              lora_rank=args.lora_rank)
        em = train_codec(codec, nch, args.steps, args.batch, args.lr, args.seed)
        rows.append({"fact_chars": nch, "best_exact_match": em})
    Path(args.out).write_text(json.dumps({"mode": "codec", "model": args.model_id,
                                          "k": args.k, "rows": rows}, indent=2))
    print("\n=== LATENT FACT-CODEC: can a latent hold an exact fact? ===")
    print(f"{'fact_chars':>10} | {'exact_match':>11}")
    for r in rows:
        print(f"{r['fact_chars']:>10} | {r['best_exact_match']:>11.3f}")
    print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------
# Full 3-architecture benchmark: face identity + exact fact, end to end
# ---------------------------------------------------------------------------
def load_faces():
    d = np.load(EMB_FILE)
    emb = d["emb"].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(str(p), []).append(i)
    ids = [p for p, ix in by.items() if len(ix) >= 2]
    return emb, by, ids


def _lsh(X, R):
    return ((X @ R) > 0) @ (1 << np.arange(R.shape[1]))


def bench_cell(codec, emb, by, ids, seed, N, nch):
    """One eval cell: register N (face, exact fact), query cross-condition,
    measure exact-match of the recalled fact under the three architectures."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(ids, size=N, replace=False)
    reg_idx, q_idx, owner = [], [], []
    for k, p in enumerate(sel):
        ix = list(by[str(p)]); rng.shuffle(ix); reg_idx.append(ix[0])
        for qi in ix[1:3]:
            q_idx.append(qi); owner.append(k)
    owner = np.array(owner)
    K = emb[reg_idx]; Qk = emb[q_idx]                        # [N,512], [Nq,512]
    facts = [make_fact(rng, nch) for _ in range(N)]
    tgt_ids = codec.fact_token_ids([facts[o] for o in owner])  # [Nq, L]
    pad = codec.tok.pad_token_id
    msk = tgt_ids != pad

    def exact(pred_ids):
        return float(((pred_ids == tgt_ids) | ~msk).all(dim=1).float().mean())

    sim = Qk @ K.T                                           # [Nq, N]
    nn = sim.argmax(1)
    id_recall = float((nn == owner).mean())

    # hybrid: hard identity -> exact fact from text dict
    hyb_ids = codec.fact_token_ids([facts[i] for i in nn])
    hybrid = exact(hyb_ids)

    # latent_only: SAME perceptual identity resolution (hard NN), but the fact
    # lives in the latent -- decode the matched identity's M. So latent_only vs
    # hybrid isolates exactly "where the fact lives" (latent decode vs text dict);
    # both share the strong perceptual leg. (Hard retrieval keeps M in-distribution
    # for the codec, the fairest case for latent.)
    M_bank = codec.encode_facts(facts)                       # [N, k, H]
    M_ret = M_bank[torch.from_numpy(nn).to(DEVICE)]          # [Nq, k, H]
    latent = exact(codec.decode_ids(M_ret))

    # text_only: caption code (LSH) -> majority fact -> string
    R = rng.standard_normal((K.shape[1], 8)).astype(np.float32)
    kc = _lsh(K, R); qc = _lsh(Qk, R)
    code_fact = {}
    for c, f in zip(kc, facts):
        code_fact.setdefault(int(c), []).append(f)
    code_major = {c: max(set(fs), key=fs.count) for c, fs in code_fact.items()}
    text_strs = [code_major.get(int(c), facts[rng.integers(N)]) for c in qc]
    text = exact(codec.fact_token_ids(text_strs))

    return {"text_only": text, "latent_only": latent, "hybrid": hybrid,
            "id_recall": id_recall}


def run_bench(args):
    nch = args.fact_chars[0]
    torch.manual_seed(args.seed); np.random.seed(args.seed)   # reproducible codec init
    codec = FactCodec(args.model_id, k=args.k, n_decode=args.n_decode,
                      lora_rank=args.lora_rank)
    log.info("training latent fact codec on %d-char facts ...", nch)
    train_codec(codec, nch, args.steps, args.batch, args.lr, args.seed)
    recon_em, recon_f1 = codec_metrics(codec, gen_facts(512, args.seed + 7, nch))
    log.info("codec self-reconstruction: exact=%.3f tokacc=%.3f", recon_em, recon_f1)

    emb, by, ids = load_faces()
    log.info("eval pool: %d identities", len(ids))
    rows = []
    for N in args.ns:
        if N > len(ids):
            continue
        acc = {k: [] for k in ("text_only", "latent_only", "hybrid", "id_recall")}
        for seed in args.seeds:
            r = bench_cell(codec, emb, by, ids, seed, N, nch)
            for k in acc:
                acc[k].append(r[k])
        row = {"N": N, "fact_chars": nch, "n_seeds": len(args.seeds),
               "codec_recon_exact": recon_em}
        for k in acc:
            row[f"{k}_mean"] = float(np.mean(acc[k]))
            row[f"{k}_std"] = float(np.std(acc[k], ddof=1)) if len(acc[k]) > 1 else 0.0
        rows.append(row)
        log.info("N=%d | text=%.3f latent=%.3f hybrid=%.3f (id=%.3f, codec=%.3f)",
                 N, row["text_only_mean"], row["latent_only_mean"],
                 row["hybrid_mean"], row["id_recall_mean"], recon_em)
    Path(args.out).write_text(json.dumps(
        {"mode": "bench", "model": args.model_id, "fact_chars": nch,
         "codec_recon_exact": recon_em, "rows": rows}, indent=2))
    print(f"\n=== MULTIMODAL MEMORY BENCHMARK ({nch}-char exact facts, "
          f"codec self-recon={recon_em:.2f}) ===")
    print(f"{'N':>5} | {'text_only':>12} {'latent_only':>12} {'hybrid':>12} {'id_rec':>7}")
    for r in rows:
        f = lambda k: f"{r[k+'_mean']:.3f}±{r[k+'_std']:.3f}"
        print(f"{r['N']:>5} | {f('text_only'):>12} {f('latent_only'):>12} "
              f"{f('hybrid'):>12} {r['id_recall_mean']:>7.3f}")
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["codec", "bench"], default="codec")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_decode", type=int, default=16)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--fact_chars", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 50, 100, 300])
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "multimem_codec.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.mode == "codec":
        run_codec(args)
    else:
        run_bench(args)


if __name__ == "__main__":
    main()
