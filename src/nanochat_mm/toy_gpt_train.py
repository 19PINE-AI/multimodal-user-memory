"""Toy GPT + MultimodalEngram trained on synthetic recurrence data.

Scientific question: when we train this architecture end-to-end on a
synthetic stream where certain identity codes recur, does the gate
learn to fire MORE on recurrent positions than on non-recurrent ones?
That is the core mechanism we need from the multimodal Engram: an
identity-recurrence-aware gate.

Synthetic protocol:
  - Vocabularies: text V_text=512, vision V_vis=256, audio V_aud=128 (tiny).
  - Each sequence is T=128 positions.
  - We define a small set of "identity codes" (say 16 in vision, 16 in audio).
  - Each sequence has 1-3 designated identities; their codes recur at multiple
    positions in the sequence.
  - The text positions in between are random.
  - NTP target: predict next token.

After training, we measure the gate's firing magnitude at:
  - "recurrent" perceptual positions (where the identity code matches an earlier
    occurrence of the same code in the sequence)
  - "novel" perceptual positions (first appearance)
  - "non-perceptual" positions (text)

Win condition: mean gate magnitude on recurrent positions > novel > text-only-baseline.
That demonstrates the architecture's central capability before any pretraining.
"""
import sys
import math
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

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- Toy GPT --------------------

class ToyBlock(nn.Module):
    def __init__(self, d, n_head=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ToyGPTWithEngram(nn.Module):
    def __init__(self, d=192, n_layer=4, max_T=128,
                 V_text=512, V_vis=256, V_aud=128, n_head=4):
        super().__init__()
        self.d = d
        self.n_layer = n_layer
        # Per-modality token embeddings (separate to avoid id-space collisions)
        self.text_emb = nn.Embedding(V_text, d)
        self.vis_emb  = nn.Embedding(V_vis, d)
        self.aud_emb  = nn.Embedding(V_aud, d)
        self.pos_emb  = nn.Embedding(max_T, d)
        self.modality_emb = nn.Embedding(3, d)  # 0/1/2

        self.blocks = nn.ModuleList([ToyBlock(d, n_head) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d)
        # Predict next token; use a per-modality head so we don't have to
        # union vocabularies. For NTP we predict text vocab head, vision head,
        # or audio head depending on the next position's modality.
        self.head_text = nn.Linear(d, V_text, bias=False)
        self.head_vis  = nn.Linear(d, V_vis, bias=False)
        self.head_aud  = nn.Linear(d, V_aud, bias=False)

        # Multimodal Engram
        eng_cfg = MultimodalEngramConfig(
            layer_ids=[1, 3],  # attach at blocks 1 and 3
            text_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=2003,
                          n_head_per_ngram=4, n_embed_per_ngram=64,
                          kernel_size=4, pad_id=0, seed=0),
            vision_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=503,
                            n_head_per_ngram=4, n_embed_per_ngram=64,
                            kernel_size=4, pad_id=0, seed=1001),
            audio_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=257,
                           n_head_per_ngram=4, n_embed_per_ngram=64,
                           kernel_size=4, pad_id=0, seed=2003),
            text_vocab_size=V_text, vision_vocab_size=V_vis, audio_vocab_size=V_aud,
        )
        self.engram = MultimodalEngramSet(eng_cfg, hidden_size=d)
        # For gate-firing measurement, we'll hook the last engram layer module
        self._captured_gate = None

    def embed(self, input_ids, modality_ids):
        # input_ids: (B, T) — interpret per modality
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        # Initialize with zero embed
        x = torch.zeros(B, T, self.d, device=input_ids.device, dtype=self.text_emb.weight.dtype)
        # Per-modality embed lookup, masked
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
        per_layer_residual_norms = {}
        for li, blk in enumerate(self.blocks):
            x = blk(x, causal_mask)
            residual = self.engram.forward_layer(x, input_ids, modality_ids, layer_idx=li)
            if residual is not None:
                if return_engram_residuals:
                    per_layer_residual_norms[li] = residual.norm(dim=-1).detach()  # [B, T]
                x = x + residual
        x = self.norm_f(x)
        if return_engram_residuals:
            return x, per_layer_residual_norms
        return x

    def logits(self, x, modality_ids):
        """Produce per-position logits in each head's vocab; loss uses the
        target modality's head."""
        return {
            "text": self.head_text(x),
            "vision": self.head_vis(x),
            "audio": self.head_aud(x),
        }


# -------------------- Synthetic data generator --------------------

class RecurrenceDataset:
    """Synthesise (input_ids, modality_ids, recurrence_mask).

    Each sample is a sequence of mixed modality tokens. The trick:
    we choose 1-2 "identity codes" (one in vision, one in audio) that
    recur at multiple positions. Other perceptual positions get random codes.
    Text is always random.

    `recurrence_mask` flags positions where the perceptual code has appeared
    earlier in the sequence (so they are "recurrent"). This is the GROUND
    TRUTH the gate should learn to fire on.
    """
    def __init__(self, T=128, V_text=512, V_vis=256, V_aud=128,
                 n_identities_vis=16, n_identities_aud=16,
                 frac_perceptual=0.35, recurrence_prob=0.5):
        self.T = T
        self.V_text = V_text
        self.V_vis = V_vis
        self.V_aud = V_aud
        # Reserve top-K codes per modality as identity codes
        self.identity_vis = np.arange(V_vis - n_identities_vis, V_vis)
        self.identity_aud = np.arange(V_aud - n_identities_aud, V_aud)
        self.frac_perceptual = frac_perceptual
        self.recurrence_prob = recurrence_prob

    def sample_batch(self, B, rng):
        T = self.T
        input_ids = np.zeros((B, T), dtype=np.int64)
        modality_ids = np.zeros((B, T), dtype=np.int64)
        recurrence_mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            # Pick perceptual positions
            perc_mask = rng.random(T) < self.frac_perceptual
            # Assign half of those to vision, half audio
            modality_ids[b] = MODALITY_TEXT
            modality_ids[b, perc_mask] = np.where(rng.random(perc_mask.sum()) < 0.5,
                                                  MODALITY_VISION, MODALITY_AUDIO)
            # Pick this sequence's "salient" identity in each modality
            vis_id = int(rng.choice(self.identity_vis))
            aud_id = int(rng.choice(self.identity_aud))

            # Fill in tokens
            seen_perc = {}
            for t in range(T):
                m = modality_ids[b, t]
                if m == MODALITY_TEXT:
                    input_ids[b, t] = rng.integers(1, self.V_text)
                elif m == MODALITY_VISION:
                    use_salient = rng.random() < self.recurrence_prob and seen_perc.get(MODALITY_VISION) is not None
                    code = vis_id if use_salient else int(rng.integers(1, self.V_vis - len(self.identity_vis)))
                    input_ids[b, t] = code
                    key = (MODALITY_VISION, code)
                    if key in seen_perc:
                        recurrence_mask[b, t] = True
                    seen_perc[key] = t
                    seen_perc[MODALITY_VISION] = t
                else:  # AUDIO
                    use_salient = rng.random() < self.recurrence_prob and seen_perc.get(MODALITY_AUDIO) is not None
                    code = aud_id if use_salient else int(rng.integers(1, self.V_aud - len(self.identity_aud)))
                    input_ids[b, t] = code
                    key = (MODALITY_AUDIO, code)
                    if key in seen_perc:
                        recurrence_mask[b, t] = True
                    seen_perc[key] = t
                    seen_perc[MODALITY_AUDIO] = t
        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(modality_ids),
            torch.from_numpy(recurrence_mask),
        )


# -------------------- Training loop --------------------

def loss_fn(model, x, mids, targets, target_mids):
    """NTP across mixed modalities: gather logits from the head matching each target's modality."""
    logits = model.logits(x, target_mids)
    # We compute three CE losses (text/vis/audio), each masked to its modality
    B, T = targets.shape
    total_loss = 0.0
    total_count = 0
    for mid, head_name, vocab in [
        (MODALITY_TEXT, "text", model.head_text.out_features),
        (MODALITY_VISION, "vision", model.head_vis.out_features),
        (MODALITY_AUDIO, "audio", model.head_aud.out_features),
    ]:
        mask = (target_mids == mid)
        if not mask.any():
            continue
        l = logits[head_name][mask]  # [N, V]
        t = targets[mask].clamp(0, vocab - 1)
        ce = F.cross_entropy(l, t)
        total_loss = total_loss + ce * mask.sum().item()
        total_count += mask.sum().item()
    return total_loss / max(total_count, 1)


def main():
    print("=" * 70)
    print("Toy GPT + Multimodal Engram — synthetic recurrence training")
    print("=" * 70)

    V_text, V_vis, V_aud, T = 512, 256, 128, 128
    model = ToyGPTWithEngram(d=192, n_layer=4, max_T=T,
                              V_text=V_text, V_vis=V_vis, V_aud=V_aud).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params (~{n_params/1e6:.1f}M)")
    print(f"  of which Engram: {sum(model.engram.total_params().values()):,}")

    ds = RecurrenceDataset(T=T, V_text=V_text, V_vis=V_vis, V_aud=V_aud,
                            frac_perceptual=0.35, recurrence_prob=0.5)
    rng = np.random.default_rng(42)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    BATCH = 16
    STEPS = 600

    print(f"\nTraining for {STEPS} steps, batch={BATCH}, T={T} ...")
    losses = []
    for step in range(STEPS):
        ids, mids, _ = ds.sample_batch(BATCH, rng)
        # NTP: input = positions 0..T-2, target = positions 1..T-1
        x_ids = ids[:, :-1].to(DEVICE)
        x_mids = mids[:, :-1].to(DEVICE)
        y_ids = ids[:, 1:].to(DEVICE)
        y_mids = mids[:, 1:].to(DEVICE)

        h = model(x_ids, x_mids)
        loss = loss_fn(model, h, x_mids, y_ids, y_mids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  loss={np.mean(losses[-50:]):.4f}")

    print(f"\nFinal loss (last 50): {np.mean(losses[-50:]):.4f}  (init was {losses[0]:.4f})")

    # ---------- The gate-firing measurement ----------
    print("\n[gate-firing diagnostic]")
    model.eval()
    with torch.no_grad():
        ids, mids, rec_mask = ds.sample_batch(64, np.random.default_rng(99))
        x_ids = ids[:, :-1].to(DEVICE)
        x_mids = mids[:, :-1].to(DEVICE)
        rec = rec_mask[:, :-1].to(DEVICE)

        _, residual_norms = model(x_ids, x_mids, return_engram_residuals=True)

    # Aggregate per-layer residual norms by position class:
    #  (modality) x (recurrent / novel)
    for lid, R in residual_norms.items():
        print(f"\n  layer {lid} mean ‖engram_residual‖ by position class:")
        for mod_name, mod_id in [("text", MODALITY_TEXT), ("vision", MODALITY_VISION), ("audio", MODALITY_AUDIO)]:
            mod_mask = (x_mids == mod_id)
            if not mod_mask.any():
                continue
            R_mod = R[mod_mask]
            rec_mod = rec[mod_mask]
            if mod_id == MODALITY_TEXT:
                # No recurrence semantics on text (random)
                print(f"    {mod_name:>6s}: {R_mod.mean().item():.4f}  (n={mod_mask.sum().item()})")
            else:
                if rec_mod.any():
                    r_rec = R_mod[rec_mod].mean().item()
                else:
                    r_rec = float("nan")
                if (~rec_mod).any():
                    r_new = R_mod[~rec_mod].mean().item()
                else:
                    r_new = float("nan")
                ratio = r_rec / r_new if r_new > 0 else float("nan")
                print(f"    {mod_name:>6s}: recurrent={r_rec:.4f}  novel={r_new:.4f}  ratio={ratio:.3f}  (n_rec={int(rec_mod.sum())} n_novel={int((~rec_mod).sum())})")

    # ---------- Save ----------
    import json
    out = Path("/home/ubuntu/multimodal-user-memory/results/toy_recurrence.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"final_loss": float(np.mean(losses[-50:])), "init_loss": float(losses[0]),
                "params": int(n_params), "steps": STEPS, "batch": BATCH, "T": T}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
