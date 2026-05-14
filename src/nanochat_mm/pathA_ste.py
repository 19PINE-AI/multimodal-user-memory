"""Path A + STE-trained codebook (the v3.2 ambition realised).

Final lever: train the perceptual codebook end-to-end during generic-NTP
pretraining via straight-through estimator. The codebook should allocate
codes along LM-useful directions (= identity-stable for cross-condition
recurrence) rather than variance-maximising directions (k-means default).

Architecture:
  raw_embedding → STE quantiser (trainable codebook centroids) → discrete code
  ↓                                                                ↓
  ↓ (perceptual_emb residual table indexed by code)                 → Engram hash
  ↓
  inputs_embeds at perceptual positions = codebook_centroid + perceptual_residual_emb[code]
  → Qwen backbone → Engram hook adds residual → lm_head

We train: codebook centroids, perceptual_residual_emb, Engram.
Frozen: Qwen.

After pretraining, codebook is frozen and surgical insertion proceeds
exactly as in pathA_generic_pretrain.py.

Hypothesis: STE codebook lifts code-match rate (training cross-condition
intra-id agreement on shared codes) from ~50% to ~85%+, pushing overall
parametric retrieval into 0.6-0.8 range and beating v1-chained-with-RAG-cheat.
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MultimodalEngramSet, MultimodalEngramConfig,
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import (
    build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
)

torch.manual_seed(42); np.random.seed(42)


# -------------------- STE quantiser --------------------

class STEQuantiser(nn.Module):
    """A single-level learned VQ codebook with straight-through estimator.

    Initialised from k-means centroids on a sample of training data, then
    fine-tuned alongside Engram pretraining via STE on the LM loss.

    Methods:
      quantise(x_emb) → (codes, q_emb, commitment_loss)
          x_emb: [B, D] raw perceptual embeddings.
          codes: [B] long ints.
          q_emb: [B, D] the codebook centroid (with STE applied so gradients flow into x_emb).
          commitment_loss: scalar, encourages x_emb to match the chosen centroid.

      quantise_no_grad(x_emb) → codes only (for inference / surgical insertion).
    """
    def __init__(self, D, K, commitment_weight=0.25):
        super().__init__()
        self.D = D
        self.K = K
        self.codebook = nn.Embedding(K, D)
        nn.init.normal_(self.codebook.weight, std=1.0 / math.sqrt(D))
        self.commitment_weight = commitment_weight

    def init_from_kmeans(self, train_emb):
        import faiss
        km = faiss.Kmeans(self.D, self.K, niter=20, seed=42, verbose=False)
        km.train(train_emb.astype(np.float32))
        with torch.no_grad():
            self.codebook.weight.copy_(torch.from_numpy(km.centroids).to(self.codebook.weight.device))

    def quantise(self, x_emb):
        # x_emb: [B, D]
        cb = self.codebook.weight  # [K, D]
        # Compute distances
        d = (x_emb.pow(2).sum(-1, keepdim=True)
             - 2 * x_emb @ cb.t()
             + cb.pow(2).sum(-1))
        codes = d.argmin(-1)  # [B]
        q = self.codebook(codes)  # [B, D]
        # STE: forward = q; backward = identity to x_emb
        q_ste = x_emb + (q - x_emb).detach()
        # Commitment loss: encourage x_emb to be close to its assigned centroid
        commit = F.mse_loss(x_emb, q.detach())
        # Codebook loss: encourage centroid to be close to its assigned points
        cb_loss = F.mse_loss(q, x_emb.detach())
        loss = self.commitment_weight * commit + cb_loss
        return codes, q_ste, loss

    @torch.no_grad()
    def quantise_no_grad(self, x_emb):
        cb = self.codebook.weight
        d = (x_emb.pow(2).sum(-1, keepdim=True)
             - 2 * x_emb @ cb.t()
             + cb.pow(2).sum(-1))
        return d.argmin(-1)


# -------------------- Modified bolt with continuous inputs --------------------

class QwenEngramBoltSTE(nn.Module):
    """Variant of QwenEngramBolt where perceptual positions accept the
    continuous embedding (not a precomputed code id); the STE quantiser
    inside the model produces the code and the perc-emb input.
    """
    def __init__(self, qwen_model, qwen_tokenizer, vis_emb_dim, aud_emb_dim,
                 V_vis=32, V_aud=32, engram_attach_layer=24,
                 engram_n_embed_per_ngram=128, engram_vocab_per_ngram=503,
                 engram_n_head=4):
        super().__init__()
        self.qwen = qwen_model
        self.tok = qwen_tokenizer
        self.hidden_size = qwen_model.config.hidden_size
        self.qwen_vocab = qwen_model.config.vocab_size
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        # STE codebooks live in the raw embedding dim
        self.vis_q = STEQuantiser(vis_emb_dim, V_vis)
        self.aud_q = STEQuantiser(aud_emb_dim, V_aud)

        # Projection from quantised embedding → Qwen hidden dim
        self.vis_proj = nn.Linear(vis_emb_dim, self.hidden_size, bias=False)
        self.aud_proj = nn.Linear(aud_emb_dim, self.hidden_size, bias=False)
        # Per-code residual embedding on top of the projected centroid (small)
        self.vis_residual_emb = nn.Embedding(V_vis, self.hidden_size)
        self.aud_residual_emb = nn.Embedding(V_aud, self.hidden_size)
        with torch.no_grad():
            ref = qwen_model.get_input_embeddings().weight.detach()
            ref_norm = ref.norm(dim=-1).mean().item()
            for m in [self.vis_proj, self.aud_proj]:
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
            for e in [self.vis_residual_emb, self.aud_residual_emb]:
                nn.init.zeros_(e.weight)  # start as a no-op residual

        # Engram
        eng_cfg = MultimodalEngramConfig(
            layer_ids=[engram_attach_layer],
            text_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=engram_vocab_per_ngram,
                          n_head_per_ngram=engram_n_head, n_embed_per_ngram=engram_n_embed_per_ngram,
                          kernel_size=4, pad_id=0, seed=0),
            vision_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=engram_vocab_per_ngram,
                            n_head_per_ngram=engram_n_head, n_embed_per_ngram=engram_n_embed_per_ngram,
                            kernel_size=4, pad_id=0, seed=1001),
            audio_cfg=dict(max_ngram_size=3, engram_vocab_per_ngram=engram_vocab_per_ngram,
                           n_head_per_ngram=engram_n_head, n_embed_per_ngram=engram_n_embed_per_ngram,
                           kernel_size=4, pad_id=0, seed=2003),
            text_vocab_size=self.qwen_vocab, vision_vocab_size=V_vis, audio_vocab_size=V_aud,
        )
        self.engram = MultimodalEngramSet(eng_cfg, hidden_size=self.hidden_size)
        self.attach_layer = engram_attach_layer

        # bf16 for the Engram + projections + residual
        for m in [self.engram, self.vis_proj, self.aud_proj,
                   self.vis_residual_emb, self.aud_residual_emb,
                   self.vis_q, self.aud_q]:
            m.to(dtype=torch.bfloat16)

        self._hook_handle = None
        self._last_input_ids = None
        self._last_modality_ids = None

    def _engram_hook(self, module, args, kwargs):
        if not args: return None
        hidden_states = args[0]
        if self._last_input_ids is None: return None
        residual = self.engram.forward_layer(
            hidden_states, self._last_input_ids, self._last_modality_ids,
            layer_idx=self.attach_layer,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
        return (hidden_states,) + args[1:], kwargs

    def install_hook(self):
        self._hook_handle = self.qwen.model.layers[self.attach_layer].register_forward_pre_hook(
            self._engram_hook, with_kwargs=True
        )

    def build_inputs_embeds_from_codes(self, input_ids, modality_ids):
        """Path used at SURGICAL INSERTION + EVAL time: codes are already discrete,
        embeddings are codebook_centroid (no grad through STE) + residual."""
        B, T = input_ids.shape
        device = input_ids.device
        emb = torch.zeros(B, T, self.hidden_size, device=device, dtype=torch.bfloat16)
        m_text = (modality_ids == MODALITY_TEXT)
        m_vis  = (modality_ids == MODALITY_VISION)
        m_aud  = (modality_ids == MODALITY_AUDIO)
        if m_text.any():
            text_ids = torch.where(m_text, input_ids, torch.zeros_like(input_ids))
            text_emb = self.qwen.get_input_embeddings()(text_ids)
            emb = emb + m_text.unsqueeze(-1).to(emb.dtype) * text_emb
        if m_vis.any():
            vis_ids = torch.where(m_vis, input_ids, torch.zeros_like(input_ids))
            centroid = self.vis_q.codebook(vis_ids)  # [B, T, D_vis]
            proj = self.vis_proj(centroid)
            resid = self.vis_residual_emb(vis_ids)
            emb = emb + m_vis.unsqueeze(-1).to(emb.dtype) * (proj + resid)
        if m_aud.any():
            aud_ids = torch.where(m_aud, input_ids, torch.zeros_like(input_ids))
            centroid = self.aud_q.codebook(aud_ids)
            proj = self.aud_proj(centroid)
            resid = self.aud_residual_emb(aud_ids)
            emb = emb + m_aud.unsqueeze(-1).to(emb.dtype) * (proj + resid)
        return emb

    def forward(self, input_ids, modality_ids):
        self._last_input_ids = input_ids
        self._last_modality_ids = modality_ids
        self.engram.reset_cache()
        emb = self.build_inputs_embeds_from_codes(input_ids, modality_ids)
        attn = torch.ones_like(input_ids)
        out = self.qwen(inputs_embeds=emb, attention_mask=attn, use_cache=False)
        return out.logits

    def pretrain_forward(self, input_ids, modality_ids, raw_perc_embs, modality_id):
        """Forward used DURING PRETRAINING: at perceptual positions, the raw
        embedding goes through the STE quantiser (gradient flows through),
        producing both the code (for Engram hash) and the q_ste embedding
        (for the Qwen input)."""
        B, T = input_ids.shape
        device = input_ids.device

        # Run quantiser on raw embeddings
        # raw_perc_embs: [B, T, D_modality] (only perceptual positions are valid)
        q_mod = self.vis_q if modality_id == MODALITY_VISION else self.aud_q
        proj = self.vis_proj if modality_id == MODALITY_VISION else self.aud_proj
        resid_emb = self.vis_residual_emb if modality_id == MODALITY_VISION else self.aud_residual_emb

        # Flatten only the perceptual positions for quantise
        m_perc = (modality_ids == modality_id)
        flat_idx = m_perc.flatten()
        if flat_idx.any():
            flat_x = raw_perc_embs.flatten(0, 1)[flat_idx]
            codes_flat, q_ste_flat, vq_loss = q_mod.quantise(flat_x)
            # Place codes back into input_ids
            input_ids = input_ids.clone()
            input_ids.flatten()[flat_idx] = codes_flat.long()
            # Place projected q_ste embeddings into the input emb position
        else:
            vq_loss = torch.zeros((), device=device)

        # Now run forward as usual; the embedding lookup uses the new code ids
        # so codebook centroid + residual go in.
        # But we want gradients to flow through the centroid via STE — so we
        # explicitly construct inputs_embeds using the q_ste flat instead.
        self._last_input_ids = input_ids
        self._last_modality_ids = modality_ids
        self.engram.reset_cache()

        emb = torch.zeros(B, T, self.hidden_size, device=device, dtype=torch.bfloat16)
        m_text = (modality_ids == MODALITY_TEXT)
        if m_text.any():
            text_ids = torch.where(m_text, input_ids, torch.zeros_like(input_ids))
            text_emb = self.qwen.get_input_embeddings()(text_ids)
            emb = emb + m_text.unsqueeze(-1).to(emb.dtype) * text_emb

        # Perceptual positions: q_ste (STE-quantised raw emb) + residual
        if flat_idx.any():
            # We have q_ste_flat for the perceptual positions; project to hidden and add residual
            q_proj_flat = proj(q_ste_flat.to(proj.weight.dtype))
            r_flat = resid_emb(codes_flat.long())
            # Place into emb at perceptual positions
            emb_flat = emb.flatten(0, 1)
            emb_flat[flat_idx] = emb_flat[flat_idx] + (q_proj_flat + r_flat)
            emb = emb_flat.view(B, T, -1)

        attn = torch.ones_like(input_ids)
        out = self.qwen(inputs_embeds=emb, attention_mask=attn, use_cache=False)
        return out.logits, vq_loss


# -------------------- Training corpus with continuous embeddings --------------------

def build_pretrain_batch_continuous(rng, train_emb, train_pid, modality_id,
                                       V_text, T=64, batch=4, frac_perceptual=0.15):
    """Builds a batch with raw perceptual embeddings at perceptual positions
    (codes generated inside the model via STE)."""
    by_id = defaultdict(list)
    for i, p in enumerate(train_pid):
        by_id[str(p)].append(i)
    train_ids = sorted(by_id.keys())
    D = train_emb.shape[1]

    B = batch
    input_ids = np.zeros((B, T), dtype=np.int64)
    modality_ids = np.zeros((B, T), dtype=np.int64)
    raw_perc = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        focus_pid = str(rng.choice(train_ids))
        for t in range(T):
            if rng.random() < frac_perceptual:
                modality_ids[b, t] = modality_id
                if rng.random() < 0.7:
                    samp_idx = int(rng.choice(by_id[focus_pid]))
                else:
                    other = str(rng.choice(train_ids))
                    samp_idx = int(rng.choice(by_id[other]))
                # input_ids[b, t] will be overwritten by the STE code inside model
                input_ids[b, t] = 0
                raw_perc[b, t] = train_emb[samp_idx]
            else:
                modality_ids[b, t] = MODALITY_TEXT
                input_ids[b, t] = int(rng.integers(1, 10000))
    return (torch.from_numpy(input_ids).to(DEVICE),
            torch.from_numpy(modality_ids).to(DEVICE),
            torch.from_numpy(raw_perc).to(DEVICE).to(torch.bfloat16))


def pretrain_with_ste(bolt, train_emb, train_pid, modality_id, tok,
                       n_steps=600, lr=3e-4, batch=4, T=64,
                       frac_perceptual=0.15, vq_weight=0.1):
    q_mod = bolt.vis_q if modality_id == MODALITY_VISION else bolt.aud_q
    proj = bolt.vis_proj if modality_id == MODALITY_VISION else bolt.aud_proj
    resid_emb = bolt.vis_residual_emb if modality_id == MODALITY_VISION else bolt.aud_residual_emb
    eng = bolt.engram.engrams[str(modality_id)]

    params = list(eng.parameters()) + list(q_mod.parameters()) + list(proj.parameters()) + [resid_emb.weight]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    rng = np.random.default_rng(0)
    losses = []; t0 = time.time()
    for step in range(n_steps):
        input_ids, modality_ids, raw_perc = build_pretrain_batch_continuous(
            rng, train_emb, train_pid, modality_id, tok.vocab_size,
            T=T, batch=batch, frac_perceptual=frac_perceptual,
        )
        logits, vq_loss = bolt.pretrain_forward(input_ids, modality_ids, raw_perc, modality_id)
        # NTP only on text-target positions
        target_mids = modality_ids[:, 1:]
        text_mask = (target_mids == MODALITY_TEXT)
        if not text_mask.any():
            continue
        pred = logits[:, :-1, :]
        target = input_ids[:, 1:]
        pred_text = pred[text_mask]
        target_text = target[text_mask]
        ntp_loss = F.cross_entropy(pred_text, target_text)
        loss = ntp_loss + vq_weight * vq_loss.to(ntp_loss.dtype)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append((float(ntp_loss.item()), float(vq_loss.item())))
        if (step + 1) % 100 == 0:
            recent = np.array(losses[-50:])
            print(f"    step {step+1:4d}  ntp_loss={recent[:, 0].mean():.4f}  vq_loss={recent[:, 1].mean():.4f}  "
                  f"(elapsed {time.time() - t0:.0f}s)")
    return losses


# -------------------- Surgical insertion + retrieval (frozen codebook) --------------------

def surgical_insert(bolt, code_token, modality_id, marker_text_id,
                    tok, max_steps=80, lr=1.0, T=24):
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code_token, modality_id, tok, marker_text_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows(eng, code_token, input_ids)

    resid_emb = bolt.vis_residual_emb if modality_id == MODALITY_VISION else bolt.aud_residual_emb
    params = [eng.tables[ks].embedding.weight for ks in touched] + [resid_emb.weight]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0)
    target = torch.tensor([marker_text_id], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    for step in range(max_steps):
        logits = bolt(input_ids, modality_ids)
        loss = F.cross_entropy(logits[:, -1, :], target)
        last_loss = float(loss.item())
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                mask[torch.tensor(sorted(rows), device=W.device, dtype=torch.long)] = 1.0
                W.grad.mul_(mask)
            if resid_emb.weight.grad is not None:
                pmask = torch.zeros(resid_emb.weight.shape[0], 1,
                                     device=resid_emb.weight.device, dtype=resid_emb.weight.grad.dtype)
                pmask[int(code_token)] = 1.0
                resid_emb.weight.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5: break
    return steps_taken, last_loss


def evaluate_ste(bolt, eval_emb, eval_pid, modality_id, tok,
                 N_subset=None, n_queries_per_id=None, max_steps=80, lr=1.0, T=24,
                 marker_offset=30001):
    q_mod = bolt.vis_q if modality_id == MODALITY_VISION else bolt.aud_q
    resid_emb = bolt.vis_residual_emb if modality_id == MODALITY_VISION else bolt.aud_residual_emb
    eng = bolt.engram.engrams[str(modality_id)]

    # Quantise all eval embeddings up front with the frozen codebook
    with torch.no_grad():
        x = torch.from_numpy(eval_emb).to(DEVICE).to(q_mod.codebook.weight.dtype)
        all_codes = q_mod.quantise_no_grad(x).cpu().numpy()

    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}

    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    resid_snap = resid_emb.weight.detach().clone()

    rng = np.random.default_rng(99)
    register_codes = {}; code_to_pid = defaultdict(list); insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code = int(all_codes[idxs[0]])
        steps, fl = surgical_insert(bolt, reg_code, modality_id, markers[pid],
                                       tok, max_steps=max_steps, lr=lr, T=T)
        insert_stats.append((steps, fl))
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)

    correct = 0; total = 0
    code_match_c = 0; code_match_t = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None: q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_code = int(all_codes[qi])
            input_ids, modality_ids = build_fixed_context(q_code, modality_id, tok, marker_text_id=0, T=T)
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                logits = bolt(input_ids, modality_ids)
                last = logits[0, -1, :]
                marker_logits = torch.stack([last[m] for m in markers.values()])
                pred_local = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local]
            total += 1
            ok = (pred_pid == pid)
            if ok: correct += 1
            if q_code == register_codes[pid]:
                code_match_t += 1
                if ok: code_match_c += 1

    with torch.no_grad():
        for ks, w in eng_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)
        resid_emb.weight.copy_(resid_snap)

    return {
        "N_registered": len(ids_sorted), "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "code_match_retr": code_match_c / code_match_t if code_match_t > 0 else float("nan"),
        "fraction_code_match": code_match_t / total if total > 0 else 0.0,
        "N_collision_codes": len([c for c, ps in code_to_pid.items() if len(ps) > 1]),
        "avg_insert_steps": float(np.mean([s for s, _ in insert_stats])),
        "avg_insert_loss": float(np.mean([l for _, l in insert_stats])),
    }


def main():
    print("=" * 70)
    print("Path A + STE-trained codebook (v3.2 ambition)")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    ); qwen.eval()

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr_emb, vis_tr_pid, vis_ev_emb, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = 32

    print(f"  audio: {len(aud_tr_emb)} train / {len(aud_ev_emb)} eval embs")
    print(f"  vision: {len(vis_tr_emb)} train / {len(vis_ev_emb)} eval embs")
    print(f"  vis emb dim: {vis_tr_emb.shape[1]}; aud emb dim: {aud_tr_emb.shape[1]}")

    bolt = QwenEngramBoltSTE(qwen, tok,
                              vis_emb_dim=vis_tr_emb.shape[1],
                              aud_emb_dim=aud_tr_emb.shape[1],
                              V_vis=K, V_aud=K,
                              engram_attach_layer=24).to(DEVICE)
    # Init codebooks from k-means on train embeddings
    bolt.vis_q.init_from_kmeans(vis_tr_emb)
    bolt.aud_q.init_from_kmeans(aud_tr_emb)
    bolt.vis_q.to(dtype=torch.bfloat16)
    bolt.aud_q.to(dtype=torch.bfloat16)
    bolt.install_hook()
    print(f"  bolt built; trainable params: "
          f"{sum(p.numel() for p in bolt.parameters() if p.requires_grad):,}")

    print("\n[pretrain STE] vision ...")
    vis_losses = pretrain_with_ste(bolt, vis_tr_emb, vis_tr_pid, MODALITY_VISION, tok,
                                     n_steps=600, lr=3e-4, batch=4, T=64,
                                     frac_perceptual=0.15)
    print("\n[pretrain STE] audio ...")
    aud_losses = pretrain_with_ste(bolt, aud_tr_emb, aud_tr_pid, MODALITY_AUDIO, tok,
                                     n_steps=600, lr=3e-4, batch=4, T=64,
                                     frac_perceptual=0.15)

    print("\n" + "=" * 70)
    print("Held-out surgical insertion + retrieval with STE-trained codebook")
    print("=" * 70)
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for mid, name, emb, pids in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid),
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        out_eval = {}
        for N in Ns:
            print(f"  RAG ceiling N={N}: {rag[N]:.4f}")
            print(f"  Path A + STE N={N} ...", end="", flush=True)
            r = evaluate_ste(bolt, emb, pids, mid, tok,
                              N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}  insert-loss={r['avg_insert_loss']:.3f}")
            out_eval[N] = r
        results[name] = {"rag": rag, "pathA_ste": out_eval}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_ste.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Full comparison: no-pretrain | generic-NTP | STE
    pa = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_qwen_bolt.json"))
    pg = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_generic_pretrain.json"))
    print("\n" + "=" * 100)
    print("HEADLINE — code-match retrieval (mechanism strength)")
    print("=" * 100)
    print(f"{'modality':>8} | {'N':>3} | {'no-pretrain':>11} | {'generic-NTP':>11} | {'STE':>6} | STE − generic")
    print("-" * 100)
    for name in ["vision", "audio"]:
        for N in Ns:
            no_p = pa[name]["pathA_no_pretrain"][str(N)]["code_match_retr"]
            g_p  = pg[name]["pathA_generic_pretrain"][str(N)]["code_match_retr"]
            ste  = results[name]["pathA_ste"][N]["code_match_retr"]
            delta = ste - g_p
            mk = " ↑↑" if delta > 0.05 else (" ≈" if abs(delta) <= 0.05 else " ↓")
            print(f"{name:>8} | {N:>3} | {no_p:>11.3f} | {g_p:>11.3f} | {ste:>6.3f} | {delta:>+12.3f}{mk}")
    print()
    print(f"{'modality':>8} | {'N':>3} | {'no-pretrain':>11} | {'generic-NTP':>11} | {'STE':>6} | (overall retrieval)")
    print("-" * 100)
    for name in ["vision", "audio"]:
        for N in Ns:
            no_p = pa[name]["pathA_no_pretrain"][str(N)]["retrieval_at_1"]
            g_p  = pg[name]["pathA_generic_pretrain"][str(N)]["retrieval_at_1"]
            ste  = results[name]["pathA_ste"][N]["retrieval_at_1"]
            print(f"{name:>8} | {N:>3} | {no_p:>11.3f} | {g_p:>11.3f} | {ste:>6.3f}")


if __name__ == "__main__":
    sys.exit(main())
