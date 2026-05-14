"""v3.3 stepping stone — train at mid-scale (~50M params) and retest retrieval.

Why this run exists. The toy (3M params) showed gate-on-recurrence works but
surgical insertion retrieval was at chance even with zero collisions. The
v3_findings note concluded the Engram is too peripheral to the LM at toy
scale to override the natural next-token distribution. The cheapest test
of that hypothesis is to scale up by ~10x and re-run.

If mid-scale retrieval moves from 0.20 (chance) toward v1's 0.48 baseline
(or beyond), the scale hypothesis is confirmed and we have a green light
for the full d12@625M run.

Architecture changes vs `real_encoder_train.py`:
  - d=384 (was 192), n_layer=6 (was 4)
  - Engram attached at layers [2, 4] (more central + near-final), with
    larger n_embed_per_ngram and bigger tables
  - Bigger batch and more steps (3000 instead of 800)
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MultimodalEngramSet, MultimodalEngramConfig,
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from toy_gpt_train import ToyGPTWithEngram, ToyBlock
from real_encoder_train import fit_naive_rq, flatten_codes, CrossSeqCorpus, loss_fn

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MidScaleGPTWithEngram(nn.Module):
    """Same overall architecture as ToyGPTWithEngram but larger and with a
    beefier Engram. ~50M params total.
    """
    def __init__(self, d=384, n_layer=6, max_T=256,
                 V_text=512, V_vis=32, V_aud=32, n_head=6,
                 engram_layers=(2, 4),
                 n_embed_per_ngram=192, engram_vocab_per_ngram=2003):
        super().__init__()
        self.d = d
        self.n_layer = n_layer
        self.text_emb = nn.Embedding(V_text, d)
        self.vis_emb  = nn.Embedding(V_vis, d)
        self.aud_emb  = nn.Embedding(V_aud, d)
        self.pos_emb  = nn.Embedding(max_T, d)
        self.modality_emb = nn.Embedding(3, d)
        self.blocks = nn.ModuleList([ToyBlock(d, n_head) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d)
        self.head_text = nn.Linear(d, V_text, bias=False)
        self.head_vis  = nn.Linear(d, V_vis, bias=False)
        self.head_aud  = nn.Linear(d, V_aud, bias=False)

        eng_cfg = MultimodalEngramConfig(
            layer_ids=list(engram_layers),
            text_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=engram_vocab_per_ngram,
                          n_head_per_ngram=8, n_embed_per_ngram=n_embed_per_ngram,
                          kernel_size=4, pad_id=0, seed=0),
            vision_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=503,
                            n_head_per_ngram=8, n_embed_per_ngram=n_embed_per_ngram,
                            kernel_size=4, pad_id=0, seed=1001),
            audio_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=503,
                           n_head_per_ngram=8, n_embed_per_ngram=n_embed_per_ngram,
                           kernel_size=4, pad_id=0, seed=2003),
            text_vocab_size=V_text, vision_vocab_size=V_vis, audio_vocab_size=V_aud,
        )
        self.engram = MultimodalEngramSet(eng_cfg, hidden_size=d)

    def embed(self, input_ids, modality_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = torch.zeros(B, T, self.d, device=input_ids.device, dtype=self.text_emb.weight.dtype)
        m_text = (modality_ids == MODALITY_TEXT)
        m_vis  = (modality_ids == MODALITY_VISION)
        m_aud  = (modality_ids == MODALITY_AUDIO)
        if m_text.any():
            x = x + m_text.unsqueeze(-1).to(x.dtype) * self.text_emb(torch.where(m_text, input_ids, torch.zeros_like(input_ids)))
        if m_vis.any():
            x = x + m_vis.unsqueeze(-1).to(x.dtype)  * self.vis_emb(torch.where(m_vis,  input_ids, torch.zeros_like(input_ids)))
        if m_aud.any():
            x = x + m_aud.unsqueeze(-1).to(x.dtype)  * self.aud_emb(torch.where(m_aud,  input_ids, torch.zeros_like(input_ids)))
        x = x + self.pos_emb(positions).unsqueeze(0) + self.modality_emb(modality_ids)
        return x

    def forward(self, input_ids, modality_ids, return_engram_residuals=False):
        B, T = input_ids.shape
        device = input_ids.device
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        x = self.embed(input_ids, modality_ids)
        self.engram.reset_cache()
        per_layer = {}
        for li, blk in enumerate(self.blocks):
            x = blk(x, causal_mask)
            residual = self.engram.forward_layer(x, input_ids, modality_ids, layer_idx=li)
            if residual is not None:
                if return_engram_residuals:
                    per_layer[li] = residual.norm(dim=-1).detach()
                x = x + residual
        x = self.norm_f(x)
        if return_engram_residuals:
            return x, per_layer
        return x

    def logits(self, x, modality_ids):
        return {"text": self.head_text(x), "vision": self.head_vis(x), "audio": self.head_aud(x)}


def main():
    print("=" * 70)
    print("v3.3 stepping stone — mid-scale training")
    print("=" * 70)

    # ---- data ----
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    rng_split = np.random.RandomState(42)
    aud_ids = sorted(set(aud['pid'].tolist())); rng_split.shuffle(aud_ids)
    vis_ids = sorted(set(vis['pid'].tolist())); rng_split.shuffle(vis_ids)
    aud_tr = set(aud_ids[: len(aud_ids) // 2])
    vis_tr = set(vis_ids[: len(vis_ids) // 2])
    aud_mask = np.array([p in aud_tr for p in aud['pid']])
    vis_mask = np.array([p in vis_tr for p in vis['pid']])

    # ---- quantiser (same L1×K=32 as before) ----
    K = 32
    audio_apply = fit_naive_rq(aud['emb'][aud_mask], n_levels=1, k_per=K)
    vision_apply = fit_naive_rq(vis['emb'][vis_mask], n_levels=1, k_per=K)
    aud_flat = flatten_codes(audio_apply(aud['emb']), K)
    vis_flat = flatten_codes(vision_apply(vis['emb']), K)
    V_aud, V_vis = K, K
    V_text = 512

    corpus = CrossSeqCorpus(
        vis_codes_flat=vis_flat[vis_mask], vis_pid=vis['pid'][vis_mask],
        aud_codes_flat=aud_flat[aud_mask], aud_pid=aud['pid'][aud_mask],
        V_text=V_text, V_vis=V_vis, V_aud=V_aud,
        T=256, sequences_per_session=3, frac_perceptual=0.3, frac_identity=0.6,
        sep_token=0,
    )

    # ---- model ----
    model = MidScaleGPTWithEngram(d=384, n_layer=6, max_T=256,
                                    V_text=V_text, V_vis=V_vis, V_aud=V_aud,
                                    n_head=6, engram_layers=(2, 4),
                                    n_embed_per_ngram=192,
                                    engram_vocab_per_ngram=2003).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    eng_params = sum(p.numel() for p in model.engram.parameters())
    print(f"  total params:   {n_params:,}  (~{n_params/1e6:.1f}M)")
    print(f"  Engram params:  {eng_params:,}  (~{eng_params/1e6:.1f}M = "
          f"{100*eng_params/n_params:.1f}% of total)")

    # ---- train ----
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    BATCH, STEPS = 16, 2500
    rng = np.random.default_rng(7)
    losses = []
    print(f"\n[train] {STEPS} steps, batch={BATCH}, T=256 ...")
    for step in range(STEPS):
        ids, mids, _, _, _ = corpus.sample_batch(BATCH, rng)
        x_ids = ids[:, :-1].to(DEVICE); x_mids = mids[:, :-1].to(DEVICE)
        y_ids = ids[:, 1:].to(DEVICE);  y_mids = mids[:, 1:].to(DEVICE)
        h = model(x_ids, x_mids)
        loss = loss_fn(model, h, y_mids, y_ids)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 250 == 0:
            print(f"  step {step+1:4d}  loss={np.mean(losses[-100:]):.4f}")
    print(f"\nFinal loss: {np.mean(losses[-100:]):.4f}  (init {losses[0]:.4f})")

    # ---- gate-firing diagnostic ----
    model.eval()
    print("\n[gate diagnostic]")
    with torch.no_grad():
        ids, mids, intra, cross, novel = corpus.sample_batch(64, np.random.default_rng(9001))
        x_ids = ids[:, :-1].to(DEVICE); x_mids = mids[:, :-1].to(DEVICE)
        intra = intra[:, :-1].to(DEVICE); cross = cross[:, :-1].to(DEVICE); novel = novel[:, :-1].to(DEVICE)
        _, residual_norms = model(x_ids, x_mids, return_engram_residuals=True)
    gate_summary = {}
    for lid, R in residual_norms.items():
        print(f"\n  layer {lid}:")
        layer = {}
        for name, mid in [("vision", MODALITY_VISION), ("audio", MODALITY_AUDIO)]:
            mask = (x_mids == mid)
            if not mask.any(): continue
            R_m = R[mask]; i_m = intra[mask]; c_m = cross[mask]; n_m = novel[mask]
            sm = lambda a, m: float(a[m].mean().item()) if m.any() else float("nan")
            r_intra, r_cross, r_novel = sm(R_m, i_m), sm(R_m, c_m), sm(R_m, n_m)
            ratio = r_cross / r_novel if r_novel and not math.isnan(r_novel) else None
            print(f"    {name:>6}: novel={r_novel:.4f}  intra={r_intra:.4f}  cross={r_cross:.4f}  cross/novel={ratio:.3f}")
            layer[name] = {"novel": r_novel, "intra_seq": r_intra, "cross_seq": r_cross,
                            "cross_vs_novel_ratio": ratio}
        gate_summary[f"layer_{lid}"] = layer

    # ---- save ----
    ckpt = Path("/home/ubuntu/multimodal-user-memory/runs/v3_midscale.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {"d": 384, "n_layer": 6, "max_T": 256,
                    "V_text": V_text, "V_vis": V_vis, "V_aud": V_aud,
                    "K_aud": K, "K_vis": K, "engram_layers": (2, 4),
                    "n_embed_per_ngram": 192, "engram_vocab_per_ngram": 2003,
                    "n_head": 6},
    }, ckpt)
    print(f"\n[done] Saved checkpoint to {ckpt}")

    out = Path("/home/ubuntu/multimodal-user-memory/results/midscale_train.json")
    with open(out, "w") as f:
        json.dump({"final_loss": float(np.mean(losses[-100:])),
                    "init_loss": float(losses[0]),
                    "n_params": int(n_params), "engram_params": int(eng_params),
                    "steps": STEPS, "batch": BATCH,
                    "gate_firing": gate_summary}, f, indent=2)
    print(f"[done] Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
