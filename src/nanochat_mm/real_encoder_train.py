"""v2 step 2: real-encoder + cross-sequence recurrence training.

This is the next-grade test of the v2 multimodal Engram. Compared to
`toy_gpt_train.py`:

  - Random integer "perceptual codes" → REAL quantised ArcFace / ECAPA codes.
    We use the v1-winning quantiser config (naive k-means residual L2×K64 audio,
    L2×K32 vision) for which we have ratio 31.6 (audio) / 5.29 (vision) on
    held-out identities (`notes/sanity_findings.md` §5).
  - Intra-sequence recurrence → cross-SEQUENCE recurrence within a session.
    A "session" consists of multiple sequences (separated by SEP) referencing
    the same identities. The Engram has to remember an identity's code from an
    earlier SEQUENCE, not just an earlier POSITION.

Win condition: cross-sequence-recurrent gate-firing magnitude > novel-position
magnitude. If true, the architecture supports the cross-session memory we
actually need.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MultimodalEngramSet, MultimodalEngramConfig,
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from toy_gpt_train import ToyGPTWithEngram  # reuse the same toy GPT architecture

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- Quantiser ----------------------

def fit_naive_rq(train_emb, n_levels, k_per, seed=42):
    import faiss
    D = train_emb.shape[1]
    centroids = []
    residual = train_emb.copy().astype(np.float32)
    for L in range(n_levels):
        km = faiss.Kmeans(D, k_per, niter=20, verbose=False, seed=seed + L)
        km.train(residual)
        _, c = km.index.search(residual, 1)
        c = c.squeeze(1)
        centroids.append(km.centroids.copy())
        residual = residual - km.centroids[c]

    def apply(emb_np):
        r = emb_np.copy().astype(np.float32)
        codes = np.zeros((len(emb_np), n_levels), dtype=np.int64)
        for L, c_arr in enumerate(centroids):
            d2 = (r ** 2).sum(1, keepdims=True) - 2 * r @ c_arr.T + (c_arr ** 2).sum(1)
            idx = d2.argmin(1)
            codes[:, L] = idx
            r = r - c_arr[idx]
        return codes
    return apply


def flatten_codes(codes, k_per):
    """Codes shape [N, n_levels] of ints in [0,k_per) → flat token id in [0, k_per**n_levels)."""
    n_levels = codes.shape[1]
    flat = np.zeros(len(codes), dtype=np.int64)
    for L in range(n_levels):
        flat = flat * k_per + codes[:, L]
    return flat


# -------------------- Corpus builder ----------------------

class CrossSeqCorpus:
    """A "session" = K sequences separated by SEP tokens. Same identities can
    appear in multiple sequences within a session. We yield one whole session
    concatenated as a single training sequence (length up to T).

    Vocabularies:
      - text:    V_text random integer tokens (no semantic meaning, just NTP targets)
      - vision:  flattened quantised ArcFace codes from TRAIN-split LFW identities
      - audio:   flattened quantised ECAPA codes from TRAIN-split LibriSpeech speakers
    """
    def __init__(self, vis_codes_flat, vis_pid, aud_codes_flat, aud_pid,
                 V_text=512, V_vis=4096, V_aud=4096,
                 T=256, sequences_per_session=3, frac_perceptual=0.3,
                 frac_identity=0.6, sep_token=0):
        self.vis_pool = defaultdict(list)  # pid -> list of available flat codes
        for c, p in zip(vis_codes_flat, vis_pid):
            self.vis_pool[str(p)].append(int(c))
        self.aud_pool = defaultdict(list)
        for c, p in zip(aud_codes_flat, aud_pid):
            self.aud_pool[str(p)].append(int(c))
        self.vis_ids = list(self.vis_pool.keys())
        self.aud_ids = list(self.aud_pool.keys())
        self.V_text = V_text
        self.V_vis = V_vis
        self.V_aud = V_aud
        self.T = T
        self.K = sequences_per_session
        self.f_perc = frac_perceptual
        self.f_id = frac_identity
        self.sep_token = sep_token

    def _seq_len(self):
        return self.T // self.K

    def _draw_perceptual_token(self, modality, identity, rng):
        """Pull a random quantised code from the identity's pool."""
        pool = self.vis_pool if modality == MODALITY_VISION else self.aud_pool
        return int(rng.choice(pool[identity]))

    def sample_session(self, rng):
        """Build one session.

        Returns:
          input_ids:      (T,) int64
          modality_ids:   (T,) int64
          intra_mask:     (T,) bool  — True if same code occurred earlier in SAME sequence
          cross_mask:     (T,) bool  — True if same code occurred earlier in DIFFERENT sequence (same session)
          novel_mask:     (T,) bool  — True if first occurrence of this (modality, code) in session
        """
        T = self.T
        L = self._seq_len()
        input_ids = np.zeros(T, dtype=np.int64)
        modality_ids = np.zeros(T, dtype=np.int64)
        seq_idx = np.zeros(T, dtype=np.int64)

        # Pick this session's identities (one vision, one audio if available)
        sess_vis_id = str(rng.choice(self.vis_ids)) if self.vis_ids else None
        sess_aud_id = str(rng.choice(self.aud_ids)) if self.aud_ids else None
        # Pick "distractor" identities (for the non-recurring identity tokens)
        distract_vis = [i for i in self.vis_ids if i != sess_vis_id]
        distract_aud = [i for i in self.aud_ids if i != sess_aud_id]

        # First: produce token stream for each sub-sequence
        pos = 0
        for k in range(self.K):
            for _ in range(L):
                if pos >= T - 1:
                    break
                # Decide modality
                if rng.random() < self.f_perc:
                    modality_ids[pos] = MODALITY_VISION if rng.random() < 0.5 else MODALITY_AUDIO
                else:
                    modality_ids[pos] = MODALITY_TEXT
                seq_idx[pos] = k
                # Decide token
                m = modality_ids[pos]
                if m == MODALITY_TEXT:
                    input_ids[pos] = int(rng.integers(1, self.V_text))  # 0 reserved for SEP
                elif m == MODALITY_VISION:
                    use_id = rng.random() < self.f_id and sess_vis_id is not None
                    if use_id:
                        input_ids[pos] = self._draw_perceptual_token(MODALITY_VISION, sess_vis_id, rng)
                    else:
                        di = str(rng.choice(distract_vis)) if distract_vis else sess_vis_id
                        input_ids[pos] = self._draw_perceptual_token(MODALITY_VISION, di, rng)
                else:
                    use_id = rng.random() < self.f_id and sess_aud_id is not None
                    if use_id:
                        input_ids[pos] = self._draw_perceptual_token(MODALITY_AUDIO, sess_aud_id, rng)
                    else:
                        di = str(rng.choice(distract_aud)) if distract_aud else sess_aud_id
                        input_ids[pos] = self._draw_perceptual_token(MODALITY_AUDIO, di, rng)
                pos += 1
            # Insert SEP between sequences (but not after the last one)
            if k < self.K - 1 and pos < T:
                input_ids[pos] = self.sep_token  # SEP
                modality_ids[pos] = MODALITY_TEXT
                seq_idx[pos] = k
                pos += 1

        # Compute recurrence masks
        intra_mask = np.zeros(T, dtype=bool)
        cross_mask = np.zeros(T, dtype=bool)
        novel_mask = np.zeros(T, dtype=bool)
        first_seen = {}   # (modality, code) -> (seq_idx, t)
        for t in range(pos):
            m = modality_ids[t]
            if m == MODALITY_TEXT:
                continue
            key = (int(m), int(input_ids[t]))
            if key not in first_seen:
                novel_mask[t] = True
                first_seen[key] = (int(seq_idx[t]), t)
            else:
                prev_seq, _ = first_seen[key]
                if prev_seq == int(seq_idx[t]):
                    intra_mask[t] = True
                else:
                    cross_mask[t] = True
        return input_ids, modality_ids, intra_mask, cross_mask, novel_mask

    def sample_batch(self, B, rng):
        outs = [self.sample_session(rng) for _ in range(B)]
        stack = lambda i: np.stack([o[i] for o in outs])
        return (
            torch.from_numpy(stack(0)),
            torch.from_numpy(stack(1)),
            torch.from_numpy(stack(2)),
            torch.from_numpy(stack(3)),
            torch.from_numpy(stack(4)),
        )


# -------------------- Train + eval ----------------------

def loss_fn(model, x, target_mids, targets):
    logits = model.logits(x, target_mids)
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
        l = logits[head_name][mask]
        t = targets[mask].clamp(0, vocab - 1)
        ce = F.cross_entropy(l, t)
        total_loss = total_loss + ce * mask.sum().item()
        total_count += int(mask.sum().item())
    return total_loss / max(total_count, 1)


def main():
    print("=" * 70)
    print("Real-encoder + cross-sequence recurrence training")
    print("=" * 70)

    # ---- Load cached embeddings ----
    print("\n[load] cached perceptual embeddings ...")
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    print(f"  audio:  {aud['emb'].shape}, {len(set(aud['pid'].tolist()))} speakers")
    print(f"  vision: {vis['emb'].shape}, {len(set(vis['pid'].tolist()))} identities")

    # ---- Split identities into TRAIN (for training corpus) and EVAL (held out) ----
    rng_split = np.random.RandomState(42)
    aud_ids = sorted(set(aud['pid'].tolist()))
    vis_ids = sorted(set(vis['pid'].tolist()))
    rng_split.shuffle(aud_ids); rng_split.shuffle(vis_ids)
    aud_train_ids = set(aud_ids[: len(aud_ids) // 2])
    vis_train_ids = set(vis_ids[: len(vis_ids) // 2])

    aud_tr_mask = np.array([p in aud_train_ids for p in aud['pid']])
    vis_tr_mask = np.array([p in vis_train_ids for p in vis['pid']])
    print(f"  audio train: {aud_tr_mask.sum()} / {len(aud_tr_mask)} samples ({len(aud_train_ids)} speakers)")
    print(f"  vision train: {vis_tr_mask.sum()} / {len(vis_tr_mask)} samples ({len(vis_train_ids)} identities)")

    # ---- Fit naive k-means quantiser on TRAIN split ----
    # Use SINGLE-LEVEL quantisation (L=1) to maximise intra-identity code agreement.
    # At L=1, K=32 each identity gets ~1-2 codes shared across utterances — giving
    # the gate visible cross-sequence recurrence to learn from. The earlier L2_K64
    # gave 139 distinct codes across 140 utterances (each utterance unique) —
    # no recurrence for the gate to see.
    print("\n[quantise] fitting naive k-means (audio L1×K32, vision L1×K32) ...")
    N_LEVELS, K_aud, K_vis = 1, 32, 32
    audio_apply = fit_naive_rq(aud['emb'][aud_tr_mask], n_levels=N_LEVELS, k_per=K_aud)
    vision_apply = fit_naive_rq(vis['emb'][vis_tr_mask], n_levels=N_LEVELS, k_per=K_vis)
    aud_codes_all = audio_apply(aud['emb'])
    vis_codes_all = vision_apply(vis['emb'])
    aud_flat_all = flatten_codes(aud_codes_all, K_aud)
    vis_flat_all = flatten_codes(vis_codes_all, K_vis)
    V_aud = K_aud ** N_LEVELS
    V_vis = K_vis ** N_LEVELS
    print(f"  audio  vocab: {V_aud} effective codes; {len(set(aud_flat_all[aud_tr_mask].tolist()))} actually used in train")
    print(f"  vision vocab: {V_vis} effective codes; {len(set(vis_flat_all[vis_tr_mask].tolist()))} actually used in train")

    # ---- Build training corpus from TRAIN identities only ----
    aud_train_codes = aud_flat_all[aud_tr_mask]
    aud_train_pids  = aud['pid'][aud_tr_mask]
    vis_train_codes = vis_flat_all[vis_tr_mask]
    vis_train_pids  = vis['pid'][vis_tr_mask]
    V_text = 512
    corpus = CrossSeqCorpus(
        vis_codes_flat=vis_train_codes, vis_pid=vis_train_pids,
        aud_codes_flat=aud_train_codes, aud_pid=aud_train_pids,
        V_text=V_text, V_vis=V_vis, V_aud=V_aud,
        T=256, sequences_per_session=3, frac_perceptual=0.3, frac_identity=0.6,
        sep_token=0,
    )

    # Quick sanity: probe a single batch's recurrence structure
    sample = corpus.sample_batch(8, np.random.default_rng(123))
    _, mids, intra, cross, novel = sample
    print(f"\n[corpus] sanity sample (B=8, T=256):")
    print(f"  text positions: {int((mids == MODALITY_TEXT).sum())}, "
          f"vision: {int((mids == MODALITY_VISION).sum())}, "
          f"audio: {int((mids == MODALITY_AUDIO).sum())}")
    print(f"  novel perc: {int(novel.sum())},  intra-seq recurr: {int(intra.sum())},  "
          f"cross-seq recurr: {int(cross.sum())}")

    # ---- Model ----
    print("\n[model] constructing toy GPT + MultimodalEngram with real-vocab sizes ...")
    model = ToyGPTWithEngram(d=192, n_layer=4, max_T=corpus.T,
                              V_text=V_text, V_vis=V_vis, V_aud=V_aud).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,} ({n_params/1e6:.2f}M)")

    # ---- Training ----
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    BATCH, STEPS = 12, 800

    print(f"\n[train] {STEPS} steps, batch={BATCH}, T={corpus.T} ...")
    rng = np.random.default_rng(7)
    losses = []
    for step in range(STEPS):
        ids, mids, _, _, _ = corpus.sample_batch(BATCH, rng)
        x_ids = ids[:, :-1].to(DEVICE)
        x_mids = mids[:, :-1].to(DEVICE)
        y_ids = ids[:, 1:].to(DEVICE)
        y_mids = mids[:, 1:].to(DEVICE)
        h = model(x_ids, x_mids)
        loss = loss_fn(model, h, y_mids, y_ids)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  loss={np.mean(losses[-50:]):.4f}")
    print(f"\nFinal loss: {np.mean(losses[-50:]):.4f}  (init {losses[0]:.4f})")

    # ---- Gate-firing diagnostic on cross-sequence recurrence ----
    print("\n[gate diagnostic] measuring on fresh held-out session batch ...")
    model.eval()
    with torch.no_grad():
        ids, mids, intra, cross, novel = corpus.sample_batch(64, np.random.default_rng(9001))
        x_ids = ids[:, :-1].to(DEVICE)
        x_mids = mids[:, :-1].to(DEVICE)
        intra = intra[:, :-1].to(DEVICE)
        cross = cross[:, :-1].to(DEVICE)
        novel = novel[:, :-1].to(DEVICE)
        _, residual_norms = model(x_ids, x_mids, return_engram_residuals=True)

    results = {}
    for lid, R in residual_norms.items():
        print(f"\n  layer {lid} mean ‖Engram-residual‖ by position class:")
        layer_results = {}
        for mod_name, mod_id in [("vision", MODALITY_VISION), ("audio", MODALITY_AUDIO)]:
            mask = (x_mids == mod_id)
            if not mask.any():
                continue
            R_m = R[mask]
            i_m = intra[mask]; c_m = cross[mask]; n_m = novel[mask]
            def safe_mean(arr, m):
                return float(arr[m].mean().item()) if m.any() else float("nan")
            r_intra = safe_mean(R_m, i_m)
            r_cross = safe_mean(R_m, c_m)
            r_novel = safe_mean(R_m, n_m)
            print(f"    {mod_name:>6}: novel={r_novel:.4f}  intra-seq={r_intra:.4f}  cross-seq={r_cross:.4f}  "
                  f"(n_novel={int(n_m.sum())} n_intra={int(i_m.sum())} n_cross={int(c_m.sum())})")
            layer_results[mod_name] = {
                "novel": r_novel, "intra_seq": r_intra, "cross_seq": r_cross,
                "n_novel": int(n_m.sum()), "n_intra": int(i_m.sum()), "n_cross": int(c_m.sum()),
                "cross_vs_novel_ratio": r_cross / r_novel if r_novel > 0 and not math.isnan(r_novel) else None,
                "intra_vs_novel_ratio": r_intra / r_novel if r_novel > 0 and not math.isnan(r_novel) else None,
            }
        results[f"layer_{lid}"] = layer_results

    out = Path("/home/ubuntu/multimodal-user-memory/results/real_encoder_recurrence.json")
    with open(out, "w") as f:
        json.dump({"final_loss": float(np.mean(losses[-50:])),
                    "init_loss": float(losses[0]),
                    "n_params": int(n_params),
                    "config": {"K_aud": K_aud, "K_vis": K_vis, "V_aud": V_aud, "V_vis": V_vis,
                                "V_text": V_text, "T": corpus.T, "steps": STEPS, "batch": BATCH},
                    "gate_firing_by_layer": results}, f, indent=2)
    print(f"\n[done] Wrote {out}")

    # Save the model + the quantiser for the next-stage retrieval experiment
    ckpt = Path("/home/ubuntu/multimodal-user-memory/runs/v2_toy_realencoder.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {"d": 192, "n_layer": 4, "max_T": corpus.T,
                    "V_text": V_text, "V_vis": V_vis, "V_aud": V_aud,
                    "K_aud": K_aud, "K_vis": K_vis},
    }, ckpt)
    print(f"[done] Saved model checkpoint to {ckpt}")

    # Also save quantiser centroids (so we can re-quantise unseen embeddings consistently)
    np.savez("/home/ubuntu/multimodal-user-memory/runs/v2_quantisers.npz",
             # we need to reach into fit_naive_rq's closure; just refit on the train split
             # cheaper: persist via pickle of apply functions? simpler: refit at eval time from the train split
             aud_train_emb=aud['emb'][aud_tr_mask],
             vis_train_emb=vis['emb'][vis_tr_mask],
             K_aud=K_aud, K_vis=K_vis)


if __name__ == "__main__":
    sys.exit(main())
