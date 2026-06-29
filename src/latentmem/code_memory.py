"""Multi-code associative memory: store M (name -> random code) pairs in k shared
latent tokens, retrieve a code by name. The realistic version of the info-theory
test -- a memory must hold MANY codes and pull back the RIGHT one, not just
autoencode one. Strictly harder than single-code capacity.

  encode:  a doc of M "name: CODE" lines --write head--> k shared memory tokens
  recall:  [k mem tokens ; "name:" ; L decode slots] --frozen LM--> the code

Sweep M (number of stored codes) x k (shared latent tokens) -> exact-match. Shows
how many exact codes a fixed latent budget can hold and retrieve.

Usage:
  python3 code_memory.py --k 16 --code_chars 6 --m_eval 1 2 4 8 16 --steps 3000
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
log = logging.getLogger("code_memory")
ALPHANUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
# distinct, single-token-friendly names
NAMES = ["Ava", "Ben", "Cyd", "Dan", "Eve", "Fin", "Gus", "Hal", "Ivy", "Jon",
         "Kai", "Lee", "Mia", "Ned", "Oji", "Pam", "Quin", "Rai", "Sky", "Tom",
         "Uma", "Vic", "Wes", "Xena", "Yel", "Zed"] + [f"P{i:02d}" for i in range(40)]


def make_doc(rng, M, code_chars, tok):
    """Token-consistent doc: the code tokens inside the encoded doc are the SAME
    ids used as the decode target (avoids BPE context-tokenization mismatch).
    Returns (doc_token_ids, items=[(prompt_ids, code_ids), ...])."""
    names = rng.choice(NAMES, size=M, replace=False)
    nl = tok("\n", add_special_tokens=False).input_ids
    doc_ids, items = [], []
    for n in names:
        code = "".join(rng.choice(list(ALPHANUM), size=code_chars))
        p = tok(f"{n}: ", add_special_tokens=False).input_ids
        c = tok(code, add_special_tokens=False).input_ids
        doc_ids += p + c + nl
        items.append((p, c))
    return doc_ids, items


class CodeMemory(nn.Module):
    def __init__(self, model_id, k=16, n_decode=24, lora_rank=16, dtype=torch.bfloat16):
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
        self.cap = self.lm.config.num_hidden_layers // 2
        self.embed = self.lm.get_input_embeddings()
        self.dtype = dtype; self.k = k; self.L = n_decode
        emb_rms = self.embed.weight.float().norm(dim=-1).mean().item() / (self.H ** 0.5)
        self.has_lora = lora_rank > 0
        if self.has_lora:
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            self.embed = self.lm.get_input_embeddings()
        self.wq = nn.Parameter(torch.randn(k, self.H) * 0.02)
        self.ln_kv = nn.LayerNorm(self.H); self.ln_out = nn.LayerNorm(self.H)
        self.attn = nn.MultiheadAttention(self.H, 8, batch_first=True)
        self.out_scale = nn.Parameter(torch.tensor(emb_rms))
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

    def _ctx(self):
        import contextlib
        return self.lm.disable_adapter() if self.has_lora else contextlib.nullcontext()

    def encode_ids(self, docs_ids):                 # list[list[int]] -> M [B,k,H]
        T = max(len(d) for d in docs_ids); pad = self.tok.pad_token_id
        ids = torch.full((len(docs_ids), T), pad, dtype=torch.long, device=DEVICE)
        am = torch.zeros((len(docs_ids), T), dtype=torch.long, device=DEVICE)
        for i, d in enumerate(docs_ids):
            ids[i, :len(d)] = torch.tensor(d, device=DEVICE); am[i, :len(d)] = 1
        with torch.no_grad(), self._ctx():
            h = self.lm(input_ids=ids, attention_mask=am, output_hidden_states=True,
                        use_cache=False).hidden_states[self.cap].float()
        q = self.wq.unsqueeze(0).expand(len(docs_ids), -1, -1)
        a, _ = self.attn(q, self.ln_kv(h), self.ln_kv(h),
                         key_padding_mask=~am.bool(), need_weights=False)
        return self.ln_out(self.wq.unsqueeze(0) + a) * self.out_scale

    def qc_ids(self, name, code):
        return (self.tok(f"{name}: ", add_special_tokens=False).input_ids,
                self.tok(code, add_special_tokens=False).input_ids)

    def pack(self, prompts, codes):
        """Pack [prompt_ids + code_ids] per example; mark the code span (cmask)."""
        seqs = [p + c for p, c in zip(prompts, codes)]
        T = max(len(s) for s in seqs); pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), T), pad, dtype=torch.long, device=DEVICE)
        amask = torch.zeros((len(seqs), T), dtype=torch.long, device=DEVICE)
        cmask = torch.zeros((len(seqs), T), dtype=torch.bool, device=DEVICE)
        for i, (p, c) in enumerate(zip(prompts, codes)):
            s = p + c
            ids[i, :len(s)] = torch.tensor(s, device=DEVICE)
            amask[i, :len(s)] = 1
            cmask[i, len(p):len(p) + len(c)] = True
        return ids, amask, cmask

    def lm_logits(self, M, ids, amask):
        """Soft memory prefix M then the text; logits aligned to predict each text token
        (LM-natural completion: M ; 'name: ' -> code)."""
        temb = self.embed(ids)
        inp = torch.cat([M.to(self.dtype), temb.to(self.dtype)], dim=1)
        fmask = torch.cat([torch.ones(M.shape[0], self.k, dtype=torch.long, device=DEVICE),
                           amask], dim=1)
        logits = self.lm(inputs_embeds=inp, attention_mask=fmask, use_cache=False).logits
        T = ids.shape[1]
        return logits[:, self.k - 1:self.k - 1 + T, :].float()       # predicts text tokens 0..T-1


def exact(pred, tgt, pad):
    mask = tgt != pad
    return ((pred == tgt) | ~mask).all(dim=1).float()


def evaluate(model, M, code_chars, n_queries, seed=999, batch=32):
    """n_queries fresh docs (M name->code pairs each); query one, exact-match the code."""
    model.eval(); rng = np.random.default_rng(seed)
    ems = []; done = 0
    with torch.no_grad():
        while done < n_queries:
            b = min(batch, n_queries - done)
            docs_ids, prompts, codes = [], [], []
            for _ in range(b):
                di, items = make_doc(rng, M, code_chars, model.tok)
                qi = rng.integers(M); p, c = items[qi]
                docs_ids.append(di); prompts.append(p); codes.append(c)
            ids, amask, cmask = model.pack(prompts, codes)
            pred = model.lm_logits(model.encode_ids(docs_ids), ids, amask).argmax(-1)
            ems.append(((pred == ids) | ~cmask).all(dim=1).float())
            done += b
    return float(torch.cat(ems).mean())


def train(model, M_max, code_chars, steps, batch, lr, seed):
    opt = torch.optim.AdamW(model.trainable(), lr=lr)
    warm = max(10, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(
            math.pi * (s - warm) / max(1, steps - warm))))
    rng = np.random.default_rng(seed); pad = model.tok.pad_token_id; t0 = time.time()
    for step in range(steps):
        docs_ids, prompts, codes = [], [], []
        M = int(rng.integers(1, M_max + 1))
        for _ in range(batch):
            di, items = make_doc(rng, M, code_chars, model.tok)
            qi = rng.integers(M); p, c = items[qi]
            docs_ids.append(di); prompts.append(p); codes.append(c)
        model.train()
        ids, amask, cmask = model.pack(prompts, codes)
        logits = model.lm_logits(model.encode_ids(docs_ids), ids, amask)
        tgt = ids.clone(); tgt[~cmask] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0); opt.step(); sched.step()
        if step % max(1, steps // 10) == 0:
            em = evaluate(model, max(1, M_max // 2), code_chars, 128)
            log.info("  step %4d loss %.3f | exact@M=%d %.3f [%.0fs]",
                     step, loss.item(), max(1, M_max // 2), em, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_decode", type=int, default=24)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--code_chars", type=int, default=6)
    ap.add_argument("--m_eval", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--m_max", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "code_memory.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = CodeMemory(args.model_id, k=args.k, n_decode=args.n_decode, lora_rank=args.lora_rank)
    log.info("k=%d code_chars=%d trainable=%.2fM", args.k, args.code_chars,
             sum(p.numel() for p in model.trainable()) / 1e6)
    train(model, args.m_max, args.code_chars, args.steps, args.batch, args.lr, args.seed)
    rows = []
    for M in args.m_eval:
        r = evaluate(model, M, args.code_chars, 512)
        rows.append({"M": M, "k": args.k, "code_chars": args.code_chars, "exact": r})
        log.info("M=%2d codes | retrieval exact-match=%.3f", M, r)
    Path(args.out).write_text(json.dumps({"k": args.k, "code_chars": args.code_chars,
                                          "model": args.model_id, "rows": rows}, indent=2))
    print(f"\n=== MULTI-CODE MEMORY: {args.k} shared tokens, {args.code_chars}-char codes ===")
    print(f"{'M codes':>8} {'exact-match':>12}")
    for r in rows:
        print(f"{r['M']:>8} {r['exact']:>12.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
