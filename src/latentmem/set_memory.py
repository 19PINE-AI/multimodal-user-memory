"""How many latent tokens to remember M faces/voices? (associative capacity)

Different question from AttMem (which keeps one non-parametric row per identity,
O(M) storage). Here we FORCE compression: encode a SET of M perceptual identities
into a FIXED k soft tokens, then recognize a cross-condition query. We sweep M and
k to find the tokens-per-identity capacity.

  encode:  M faces (+ identity tags) --write head--> k memory tokens (shared)
  recall:  [k memory tokens ; query face] --frozen LM--> identity-marker logits

One model per k is trained on VARIABLE set size M (so it generalises across M),
then recall@1 is measured at each M on held-out identities. Works for faces
(ArcFace) or voices (ECAPA) via --emb_file.

Usage:
  python3 set_memory.py --k 16 --emb_file runs/embeddings/arcface_lfw_xxxl.npz \
      --m_eval 2 4 8 16 32 --steps 3000
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARKER_OFFSET = 30001
log = logging.getLogger("set_memory")


def load_emb(path):
    d = np.load(path)
    emb = d["emb"].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(str(p), []).append(i)
    ids = [p for p, ix in by.items() if len(ix) >= 2]
    rng = np.random.default_rng(0); rng.shuffle(ids)
    cut = int(0.6 * len(ids))
    return emb, by, ids[:cut], ids[cut:]            # emb, groups, train_ids, eval_ids


class SetMemory(nn.Module):
    def __init__(self, model_id, key_dim, k=16, lora_rank=16, dtype=torch.bfloat16):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.H = self.lm.config.hidden_size
        self.dtype = dtype
        self.k = k
        self.embed = self.lm.get_input_embeddings()
        emb_rms = self.embed.weight.float().norm(dim=-1).mean().item() / (self.H ** 0.5)

        self.has_lora = lora_rank > 0
        if self.has_lora:
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            self.embed = self.lm.get_input_embeddings()

        # project a perceptual embedding into the LM token space
        self.proj = nn.Linear(key_dim, self.H, bias=False)
        with torch.no_grad():
            self.proj.weight.mul_(emb_rms / (self.proj.weight.norm(dim=0).mean() + 1e-6))
        # write head: k learned queries cross-attend over the M tagged identities
        self.wq = nn.Parameter(torch.randn(k, self.H) * 0.02)
        self.ln_ctx = nn.LayerNorm(self.H)
        self.attn = nn.MultiheadAttention(self.H, 8, batch_first=True)
        self.ln_out = nn.LayerNorm(self.H)
        self.out_scale = nn.Parameter(torch.tensor(emb_rms))
        for m in (self.proj, self.attn, self.ln_ctx, self.ln_out):
            m.float() if hasattr(m, "float") else None
        self.to(DEVICE)
        self.proj.float(); self.attn.float(); self.ln_ctx.float(); self.ln_out.float()

    def trainable(self):
        ps = [self.wq, self.out_scale, *self.proj.parameters(), *self.attn.parameters(),
              *self.ln_ctx.parameters(), *self.ln_out.parameters()]
        if self.has_lora:
            ps += [p for p in self.lm.parameters() if p.requires_grad]
        return ps

    def encode(self, face_embs):                    # [M, D] -> mem [k, H]
        f = self.proj(face_embs.float())            # [M, H]
        tags = self.embed(torch.arange(MARKER_OFFSET, MARKER_OFFSET + f.shape[0],
                                       device=DEVICE)).float()
        ctx = self.ln_ctx(f + tags).unsqueeze(0)    # [1, M, H]
        q = self.wq.unsqueeze(0)                     # [1, k, H]
        a, _ = self.attn(q, ctx, ctx, need_weights=False)
        return (self.ln_out(self.wq.unsqueeze(0) + a) * self.out_scale)[0]   # [k, H]

    def read_logits(self, mem, q_faces):            # mem [k,H], q_faces [Q,D]
        Q = q_faces.shape[0]
        qf = self.proj(q_faces.float()).unsqueeze(1)               # [Q, 1, H]
        m = mem.unsqueeze(0).expand(Q, -1, -1)                     # [Q, k, H]
        inp = torch.cat([m, qf], dim=1).to(self.dtype)            # [Q, k+1, H]
        attn = torch.ones(inp.shape[:2], device=DEVICE, dtype=torch.long)
        return self.lm(inputs_embeds=inp, attention_mask=attn, use_cache=False).logits[:, -1, :]


def sample_set(rng, emb, by, ids, M):
    """M identities; one registration photo each + one cross-condition query each."""
    sel = rng.choice(ids, size=M, replace=False)
    reg, qry = [], []
    for p in sel:
        ix = list(by[str(p)]); rng.shuffle(ix)
        reg.append(ix[0]); qry.append(ix[1])
    return emb[reg], emb[qry]                        # [M,D], [M,D]


def evaluate(model, emb, by, eval_ids, M, n_sets=64, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    correct = tot = 0
    markers = torch.arange(MARKER_OFFSET, MARKER_OFFSET + M, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_sets):
            if M > len(eval_ids):
                break
            reg, qry = sample_set(rng, emb, by, eval_ids, M)
            mem = model.encode(torch.from_numpy(reg).to(DEVICE))
            logits = model.read_logits(mem, torch.from_numpy(qry).to(DEVICE))
            pred = logits[:, markers].argmax(1).cpu().numpy()
            correct += (pred == np.arange(M)).sum(); tot += M
    return correct / max(1, tot)


def train(model, emb, by, train_ids, m_max, steps, lr, seed):
    opt = torch.optim.AdamW(model.trainable(), lr=lr)
    warm = max(10, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(
            math.pi * (s - warm) / max(1, steps - warm))))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        M = int(rng.integers(2, m_max + 1))
        reg, qry = sample_set(rng, emb, by, train_ids, M)
        model.train()
        mem = model.encode(torch.from_numpy(reg).to(DEVICE))
        logits = model.read_logits(mem, torch.from_numpy(qry).to(DEVICE)).float()
        target = torch.arange(MARKER_OFFSET, MARKER_OFFSET + M, device=DEVICE)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0); opt.step(); sched.step()
        if step % max(1, steps // 10) == 0:
            log.info("  step %4d loss %.3f [%.0fs]", step, loss.item(), time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--emb_file", default=str(ROOT / "runs/embeddings/arcface_lfw_xxxl.npz"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--m_eval", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--m_max", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "set_memory.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    emb, by, train_ids, eval_ids = load_emb(args.emb_file)
    key_dim = emb.shape[1]
    log.info("emb=%s dim=%d | train ids=%d eval ids=%d | k=%d",
             Path(args.emb_file).stem, key_dim, len(train_ids), len(eval_ids), args.k)
    model = SetMemory(args.model_id, key_dim, k=args.k, lora_rank=args.lora_rank)
    log.info("trainable %.2fM", sum(p.numel() for p in model.trainable()) / 1e6)
    train(model, emb, by, train_ids, args.m_max, args.steps, args.lr, args.seed)

    rows = []
    for M in args.m_eval:
        if M > len(eval_ids):
            continue
        r = evaluate(model, emb, by, eval_ids, M)
        rows.append({"M": M, "k": args.k, "recall": r, "tokens_per_id": args.k / M})
        log.info("M=%2d faces | recall@1=%.3f | k/M=%.2f tokens/identity", M, r, args.k / M)
    Path(args.out).write_text(json.dumps(
        {"emb": Path(args.emb_file).stem, "k": args.k, "model": args.model_id,
         "rows": rows}, indent=2))
    print(f"\n=== SET MEMORY: {args.k} latent tokens, {Path(args.emb_file).stem} ===")
    print(f"{'M faces':>8} {'recall@1':>9} {'k/M':>6}")
    for r in rows:
        print(f"{r['M']:>8} {r['recall']:>9.3f} {r['tokens_per_id']:>6.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
