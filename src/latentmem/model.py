"""Latent user-memory model: frozen LM + a trainable write head.

Write (compress):  doc tokens --frozen LM--> hidden states --write head--> M (k x H)
Read  (recall):    [M ; probe embeds] --frozen LM--> answer logits
Teacher (oracle):  [doc ; probe]      --frozen LM--> answer logits  (full context)

Only the write head is trained. The LM is frozen throughout; M is injected as
`inputs_embeds`, exactly the soft-token interface AttMem already uses. The whole
point of the pilot is to learn M so the Read path matches the Teacher path.
"""
from __future__ import annotations

import contextlib
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class WriteHead(nn.Module):
    """k learnable memory-query tokens cross-attend over the document's hidden
    states to produce k continuous memory vectors. ~one transformer-decoder
    block; the only trainable component in the pilot."""

    def __init__(self, hidden: int, k: int, n_heads: int = 8, emb_norm: float = 1.0):
        super().__init__()
        self.k = k
        self.queries = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.ln_q = nn.LayerNorm(hidden)
        self.ln_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                 nn.Linear(4 * hidden, hidden))
        # Final LayerNorm + learned scale places M in the LM's input-embedding
        # distribution (zero-mean per dim, matched RMS) so it reads like real
        # token embeddings. This is the LLaVA-style projection the latent-bridge
        # work found necessary, versus L2-normalising onto a fixed hypersphere
        # (which discards magnitude and does not centre per dim).
        self.out_ln = nn.LayerNorm(hidden)
        self.out_scale = nn.Parameter(torch.tensor(float(emb_norm) / (hidden ** 0.5)))

    def forward(self, doc_hidden: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
        # doc_hidden: [B, L, H] (fp32); doc_mask: [B, L] bool (True = valid token)
        B = doc_hidden.shape[0]
        q = self.ln_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.ln_kv(doc_hidden)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=~doc_mask, need_weights=False)
        m = self.queries.unsqueeze(0) + attn_out          # residual on the query tokens
        m = m + self.ff(self.ln_ff(m))                    # position-wise FFN
        m = self.out_ln(m) * self.out_scale               # match input-embedding stats
        return m                                          # [B, k, H]


class LatentMemoryModel(nn.Module):
    def __init__(self, model_id: str, k: int = 16, attn_heads: int = 8,
                 capture_frac: float = 0.67, lora_rank: int = 0,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Right padding is required: _last_logits reads the answer cue at
        # index (attn.sum - 1), which is only the last real token when pads
        # sit on the right. Qwen tokenizers sometimes default to left.
        self.tok.padding_side = "right"
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.H = self.lm.config.hidden_size
        n_layers = self.lm.config.num_hidden_layers
        self.embed = self.lm.get_input_embeddings()
        emb_norm = self.embed.weight.float().norm(dim=-1).mean().item()
        # Capture doc residuals from a mid-depth layer, not the final hidden
        # state: final-layer activations are specialised for next-token
        # prediction, whereas ~2/3-depth residuals are richer and more
        # transferable (the latent-bridge work captures at layer 24/36 = 0.67).
        self.capture_layer = max(1, min(n_layers, round(n_layers * capture_frac)))
        # LoRA on the read path: a fully frozen LM cannot decode arbitrary facts
        # from soft memory tokens (recall sat at chance), so we let it learn to
        # read M -- as ICAE/gist/AutoCompressor and the latent-bridge work do.
        # The oracle/teacher and the doc encoder run with the adapter DISABLED,
        # so the distillation target stays the clean full-context model.
        self.has_lora = lora_rank > 0
        if self.has_lora:
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            self.embed = self.lm.get_input_embeddings()
        self.write_head = WriteHead(self.H, k, attn_heads, emb_norm).to(device).float()
        self.k = k

    def _clean(self):
        """Context manager that disables LoRA -> the clean frozen-model
        behaviour. Used for the oracle/teacher and the document encoder so the
        distillation target and stored content do not depend on the read-path
        adapter."""
        return self.lm.disable_adapter() if self.has_lora else contextlib.nullcontext()

    def trainable_lm_params(self):
        return [p for p in self.lm.parameters() if p.requires_grad] if self.has_lora else []

    # ---- tokenization helpers -------------------------------------------------
    def _batch_ids(self, texts: List[str], add_bos: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=1024, add_special_tokens=add_bos)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @staticmethod
    def _last_logits(logits: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Logits at each row's last real (non-pad) position. logits: [B,T,V]."""
        last = attn.sum(dim=1).long() - 1
        return logits[torch.arange(logits.shape[0], device=logits.device), last]

    # ---- the three forward paths ---------------------------------------------
    def encode_doc(self, docs: List[str]) -> torch.Tensor:
        """docs -> M [B, k, H]. Frozen LM under no_grad; grad flows into write head."""
        ids, mask = self._batch_ids(docs, add_bos=True)
        with torch.no_grad(), self._clean():
            out = self.lm(input_ids=ids, attention_mask=mask, use_cache=False,
                          output_hidden_states=True)
            doc_hidden = out.hidden_states[self.capture_layer].float()
        return self.write_head(doc_hidden, mask.bool())

    def read_logits(self, M: torch.Tensor, probes: List[str]) -> torch.Tensor:
        """Answer logits from [M ; probe] — the student path (grad through M)."""
        p_ids, p_mask = self._batch_ids(probes, add_bos=False)
        p_emb = self.embed(p_ids)
        m = M.to(self.dtype)
        inp = torch.cat([m, p_emb], dim=1)
        mmask = torch.ones(m.shape[:2], device=self.device, dtype=p_mask.dtype)
        attn = torch.cat([mmask, p_mask], dim=1)
        out = self.lm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        return self._last_logits(out.logits, attn)

    def recon_loss(self, M: torch.Tensor, docs: List[str]) -> torch.Tensor:
        """Auxiliary autoencoding loss: teacher-force the document tokens from
        [M ; doc]. This forces M to encode the *whole* document (probe-
        independent) rather than only the facts a single sampled probe queries
        -- the ICAE/gist sufficiency objective. Grad flows through M."""
        ids, mask = self._batch_ids(docs, add_bos=True)          # [B, L]
        tok_emb = self.embed(ids)
        m = M.to(self.dtype)
        k = m.shape[1]
        inp = torch.cat([m, tok_emb], dim=1)                     # [B, k+L, H]
        mmask = torch.ones(m.shape[:2], device=self.device, dtype=mask.dtype)
        attn = torch.cat([mmask, mask], dim=1)
        out = self.lm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        # Causal logits at positions k-1 .. k+L-2 predict doc tokens ids[:,0..L-1].
        doc_logits = out.logits[:, k - 1:k - 1 + ids.shape[1], :].float()
        return F.cross_entropy(doc_logits.reshape(-1, doc_logits.size(-1)),
                               ids.reshape(-1),
                               ignore_index=self.tok.pad_token_id)

    @torch.no_grad()
    def teacher_logits(self, docs: List[str], probes: List[str]) -> torch.Tensor:
        """Answer logits from [doc ; probe] — the full-context oracle (no grad)."""
        texts = [d + "\n\n" + p for d, p in zip(docs, probes)]
        ids, mask = self._batch_ids(texts, add_bos=True)
        with self._clean():
            out = self.lm(input_ids=ids, attention_mask=mask, use_cache=False)
        return self._last_logits(out.logits, mask)

    @torch.no_grad()
    def text_baseline_logits(self, docs: List[str], probes: List[str],
                             budget_tokens: int) -> torch.Tensor:
        """Matched-budget text baseline: truncate the doc to `budget_tokens`
        tokens and answer from that. This is the weak (truncation) baseline; an
        LLM-written summary at the same budget is the strong one (see eval.py)."""
        trunc = []
        for d in docs:
            ids = self.tok(d, add_special_tokens=False)["input_ids"][:budget_tokens]
            trunc.append(self.tok.decode(ids))
        return self.teacher_logits(trunc, probes)

    # ---- answer-token bookkeeping --------------------------------------------
    def answer_token_ids(self) -> Tuple[int, int]:
        """Token ids for the ' yes' / ' no' answer (leading space => one token)."""
        yes = self.tok(" yes", add_special_tokens=False)["input_ids"]
        no = self.tok(" no", add_special_tokens=False)["input_ids"]
        return yes[0], no[0]

    def count_trainable(self) -> int:
        n = sum(p.numel() for p in self.write_head.parameters() if p.requires_grad)
        return n + sum(p.numel() for p in self.trainable_lm_params())
