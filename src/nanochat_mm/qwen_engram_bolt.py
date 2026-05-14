"""Path A — bolt MultimodalEngramSet onto pretrained Qwen2.5-3B-Instruct.

Wrapper architecture:
  input: list of {text_token, perceptual_code} positions with modality tags
   │
   ├─ text positions     → Qwen's frozen token embedding
   ├─ perceptual positions → trainable per-code embedding (small table)
   ▼
  inputs_embeds → Qwen2.5-3B forward, with a hook at layer L:
       at hook, run MultimodalEngramSet.forward_layer(hidden, input_ids, modality_ids, L)
       and ADD its residual to the hidden state
   │
   ▼
  lm_head → logits over Qwen's 151936-vocab

This script first tests: with a FRESHLY-INITIALISED Engram (no pretraining),
does surgical row-targeted insertion + retrieval beat v1's 0.48 vision /
0.60 audio at N=20? Pretrained Qwen brings strong text-output capability;
surgical insertion only needs to bias an already-well-formed distribution.

If yes → Path A succeeds with no Engram pretraining. Done.
If no  → train the Engram first (smaller test, but more work).
"""
import json
import sys
import math
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
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class QwenEngramBolt(nn.Module):
    """Pretrained Qwen + bolt-on MultimodalEngramSet + perceptual code emb table.

    Only the perceptual code embedding table and the Engram are trainable.
    Qwen's params are frozen. The Engram residual is added at a chosen layer
    via a forward pre-hook on that layer's input.
    """
    def __init__(self, qwen_model, qwen_tokenizer, V_vis=32, V_aud=32,
                 engram_attach_layer=24, engram_n_embed_per_ngram=128,
                 engram_vocab_per_ngram=503, engram_n_head=4):
        super().__init__()
        self.qwen = qwen_model
        self.tok = qwen_tokenizer
        self.hidden_size = qwen_model.config.hidden_size
        self.qwen_vocab = qwen_model.config.vocab_size
        # Freeze Qwen
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        # Trainable perceptual-code embedding tables (one per modality).
        # These live in the same hidden-dim space as Qwen's embeddings.
        self.vis_perc_emb = nn.Embedding(V_vis, self.hidden_size)
        self.aud_perc_emb = nn.Embedding(V_aud, self.hidden_size)
        # Initialise close to Qwen's embedding norm so they sit in the
        # same magnitude regime
        with torch.no_grad():
            ref = qwen_model.get_input_embeddings().weight.detach()
            ref_norm = ref.norm(dim=-1).mean().item()
            for e in [self.vis_perc_emb, self.aud_perc_emb]:
                nn.init.normal_(e.weight, std=ref_norm / math.sqrt(self.hidden_size))

        # MultimodalEngramSet attached at one layer
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
            # Vocab sizes for hashing — Qwen's vocab for text, our V_vis/V_aud for perc
            text_vocab_size=self.qwen_vocab, vision_vocab_size=V_vis, audio_vocab_size=V_aud,
        )
        self.engram = MultimodalEngramSet(eng_cfg, hidden_size=self.hidden_size)
        self.attach_layer = engram_attach_layer
        # Cast Engram to bf16 to match Qwen
        self.engram.to(dtype=torch.bfloat16)
        # Cast perceptual emb to bf16
        self.vis_perc_emb.to(dtype=torch.bfloat16)
        self.aud_perc_emb.to(dtype=torch.bfloat16)

        self._hook_handle = None
        self._last_input_ids = None
        self._last_modality_ids = None

    def _engram_hook(self, module, args, kwargs):
        """Pre-hook on the target Qwen layer: modify its hidden_states input
        by adding the Engram residual computed from the cached input ids."""
        # Qwen2DecoderLayer.forward signature has hidden_states as the first arg
        if not args:
            return None
        hidden_states = args[0]
        if self._last_input_ids is None:
            return None
        residual = self.engram.forward_layer(
            hidden_states, self._last_input_ids, self._last_modality_ids,
            layer_idx=self.attach_layer,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
        new_args = (hidden_states,) + args[1:]
        return new_args, kwargs

    def install_hook(self):
        layer = self.qwen.model.layers[self.attach_layer]
        self._hook_handle = layer.register_forward_pre_hook(self._engram_hook, with_kwargs=True)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def build_inputs_embeds(self, input_ids, modality_ids):
        """Compose embeddings: Qwen's for text, our learned tables for perc."""
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
            vis_emb = self.vis_perc_emb(vis_ids)
            emb = emb + m_vis.unsqueeze(-1).to(emb.dtype) * vis_emb
        if m_aud.any():
            aud_ids = torch.where(m_aud, input_ids, torch.zeros_like(input_ids))
            aud_emb = self.aud_perc_emb(aud_ids)
            emb = emb + m_aud.unsqueeze(-1).to(emb.dtype) * aud_emb
        return emb

    def forward(self, input_ids, modality_ids):
        # Cache for the hook
        self._last_input_ids = input_ids
        self._last_modality_ids = modality_ids
        self.engram.reset_cache()
        emb = self.build_inputs_embeds(input_ids, modality_ids)
        attn_mask = torch.ones_like(input_ids)
        out = self.qwen(inputs_embeds=emb, attention_mask=attn_mask, use_cache=False)
        return out.logits  # [B, T, vocab]


def build_fixed_context(code_token, modality_id, tok, marker_text_id, T=24):
    """Construct an input where positions 0..T-2 are a known text prefix,
    position T-1 is the perceptual code, and we predict the next token.

    The text prefix is the same across all surgical insertions so the
    N-gram hash at the perceptual position is stable.

    Returns input_ids, modality_ids (both length T, batch-dim added by caller).
    """
    # Prefix: a fixed instruction-style preamble using a short text snippet
    prompt = "You see"
    pref_ids = tok.encode(prompt, add_special_tokens=False)
    # Pad/truncate prefix to T-1 length using sep_token (the tokeniser's pad_token_id, or 0)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    pref = list(pref_ids) + [pad_id] * (T - 1 - len(pref_ids))
    pref = pref[: T - 1]
    input_ids = pref + [int(code_token)]
    mids = [MODALITY_TEXT] * (T - 1) + [int(modality_id)]
    return (torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(mids, dtype=torch.long).unsqueeze(0))


def get_touched_rows(eng, code_token, input_ids):
    """Compute hash-touched rows at the LAST position of the given input_ids."""
    inp = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
    if inp.ndim == 1: inp = inp[None]
    hashes_per_layer = eng.hash_mapping.hash_all_layers(inp, user_salt=int(eng.user_salt))
    touched = {}
    last_pos = inp.shape[1] - 1
    for lid, h in hashes_per_layer.items():
        local = h[0, last_pos]
        tbl = eng.tables[str(lid)]
        global_rows = local + tbl.offsets.cpu().numpy()
        touched[str(lid)] = set(int(r) for r in global_rows.tolist())
    return touched


def surgical_insert(model_bolt, code_token, modality_id, marker_text_id,
                    tok, max_steps=100, lr=1.0, early_stop_loss=0.5, T=24):
    eng = model_bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code_token, modality_id, tok, marker_text_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows(eng, code_token, input_ids)

    # Also include the perceptual embedding row for this code as trainable —
    # but FIRST insert is at the embedding level (the model has to know what
    # the perceptual code MEANS).
    if modality_id == MODALITY_VISION:
        perc_emb_param = model_bolt.vis_perc_emb.weight
    else:
        perc_emb_param = model_bolt.aud_perc_emb.weight
    params_to_opt = [eng.tables[ks].embedding.weight for ks in touched] + [perc_emb_param]
    opt = torch.optim.SGD(params_to_opt, lr=lr, momentum=0.0)

    target = torch.tensor([marker_text_id], dtype=torch.long, device=DEVICE)
    last_loss = float("inf"); steps_taken = 0
    for step in range(max_steps):
        logits = model_bolt(input_ids, modality_ids)  # [1, T, V]
        last = logits[:, -1, :]
        loss = F.cross_entropy(last, target)
        last_loss = float(loss.item())
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            # Mask Engram embedding gradient to only touched rows
            for ks, rows in touched.items():
                W = eng.tables[ks].embedding.weight
                if W.grad is None: continue
                mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.grad.dtype)
                row_idx = torch.tensor(sorted(rows), device=W.device, dtype=torch.long)
                mask[row_idx] = 1.0
                W.grad.mul_(mask)
            # Mask perceptual-emb gradient to only the row for this code
            if perc_emb_param.grad is not None:
                pmask = torch.zeros(perc_emb_param.shape[0], 1,
                                     device=perc_emb_param.device, dtype=perc_emb_param.grad.dtype)
                pmask[int(code_token)] = 1.0
                perc_emb_param.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < early_stop_loss:
            break
    return steps_taken, last_loss


def evaluate(model_bolt, codebook_apply, eval_emb, eval_pid, modality_id, tok,
             N_subset=None, n_queries_per_id=None,
             max_steps=100, lr=1.0, T=24):
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    # Marker tokens: use short unique tokens from Qwen's vocab. Pick token ids
    # in a contiguous range that decode to single readable strings.
    # Take token ids 30001..30000+N from Qwen's vocab.
    marker_ids = list(range(30001, 30001 + len(ids_sorted)))
    markers = {pid: marker_ids[i] for i, pid in enumerate(ids_sorted)}

    eng = model_bolt.engram.engrams[str(modality_id)]
    # Full Engram snapshot
    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    if modality_id == MODALITY_VISION:
        perc_snap = model_bolt.vis_perc_emb.weight.detach().clone()
    else:
        perc_snap = model_bolt.aud_perc_emb.weight.detach().clone()

    rng = np.random.default_rng(99)
    register_codes = {}; code_to_pid = defaultdict(list); insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_emb = eval_emb[idxs[0]]
        reg_code_arr = codebook_apply(reg_emb[None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        steps, fl = surgical_insert(
            model_bolt, reg_code, modality_id, markers[pid],
            tok, max_steps=max_steps, lr=lr, T=T,
        )
        insert_stats.append((steps, fl))
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)
    collision_codes = {c: pids for c, pids in code_to_pid.items() if len(pids) > 1}

    correct = 0; total = 0
    code_match_c = 0; code_match_t = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None: q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_emb = eval_emb[qi]
            q_code_arr = codebook_apply(q_emb[None])[0]
            q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
            input_ids, modality_ids = build_fixed_context(q_code, modality_id, tok,
                                                            marker_text_id=0, T=T)
            input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
            with torch.no_grad():
                logits = model_bolt(input_ids, modality_ids)
                last = logits[0, -1, :]
                # Among registered markers, which one scored highest?
                marker_logits = torch.stack([last[m] for m in marker_ids])
                pred_local_idx = int(marker_logits.argmax().item())
                pred_pid = ids_sorted[pred_local_idx]
            total += 1
            ok = (pred_pid == pid)
            if ok: correct += 1
            if q_code == register_codes[pid]:
                code_match_t += 1
                if ok: code_match_c += 1

    # Restore
    with torch.no_grad():
        for ks, w in eng_snap.items():
            eng.tables[ks].embedding.weight.copy_(w)
        if modality_id == MODALITY_VISION:
            model_bolt.vis_perc_emb.weight.copy_(perc_snap)
        else:
            model_bolt.aud_perc_emb.weight.copy_(perc_snap)

    return {
        "N_registered": len(ids_sorted), "N_queries": total,
        "retrieval_at_1": correct / total if total > 0 else 0.0,
        "code_match_retr": code_match_c / code_match_t if code_match_t > 0 else float("nan"),
        "fraction_code_match": code_match_t / total if total > 0 else 0.0,
        "N_collision_codes": len(collision_codes),
        "avg_insert_steps": float(np.mean([s for s, _ in insert_stats])),
        "avg_insert_loss": float(np.mean([l for _, l in insert_stats])),
    }


def main():
    print("=" * 70)
    print("Path A: Qwen2.5-3B + bolt-on Engram, no Engram pretraining")
    print("=" * 70)

    print("\nLoading Qwen2.5-3B-Instruct ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    qwen.eval()
    print(f"  loaded; {sum(p.numel() for p in qwen.parameters())/1e9:.2f}B params")

    print("\nLoading embeddings + fitting codebooks ...")
    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri.npz")
    vis = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/arcface_lfw.npz")
    aud_tr, _, aud_ev, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    vis_tr, _, vis_ev, vis_ev_pid = split_by_identity(vis['emb'], vis['pid'])
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    aud_apply = fit_naive_rq(aud_tr, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr, n_levels=1, k_per=K)
    print(f"  audio eval: {len(aud_ev)} embs / {len(set(aud_ev_pid))} ids")
    print(f"  vision eval: {len(vis_ev)} embs / {len(set(vis_ev_pid))} ids")

    print("\nBuilding QwenEngramBolt ...")
    bolt = QwenEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                           engram_attach_layer=24,
                           engram_n_embed_per_ngram=128,
                           engram_vocab_per_ngram=503,
                           engram_n_head=4).to(DEVICE)
    bolt.install_hook()
    trainable = sum(p.numel() for p in bolt.parameters() if p.requires_grad)
    eng_params = sum(p.numel() for p in bolt.engram.parameters())
    perc_params = bolt.vis_perc_emb.weight.numel() + bolt.aud_perc_emb.weight.numel()
    print(f"  trainable params: {trainable:,}  (engram {eng_params:,}, perc-emb {perc_params:,})")
    print(f"  attach layer: {bolt.attach_layer} / {qwen.config.num_hidden_layers}")

    if torch.cuda.is_available():
        print(f"  GPU memory now: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    Ns = [5, 10, 20]; nq = 5
    results = {}
    for mid, name, emb, pids, apply_fn in [
        (MODALITY_VISION, "vision", vis_ev, vis_ev_pid, vis_apply),
        (MODALITY_AUDIO,  "audio",  aud_ev, aud_ev_pid, aud_apply),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        v3a = {}
        for N in Ns:
            print(f"  RAG ceiling N={N}: {rag[N]:.4f}")
            print(f"  Path A surgical insertion N={N} (no Engram pretraining) ...", end="", flush=True)
            r = evaluate(bolt, apply_fn, emb, pids, mid, tok,
                          N_subset=N, n_queries_per_id=nq,
                          max_steps=80, lr=1.0)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"(insert avg {r['avg_insert_steps']:.0f} steps, final loss {r['avg_insert_loss']:.3f})  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}")
            v3a[N] = r
        results[name] = {"rag": rag, "pathA_no_pretrain": v3a}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen_bolt.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Wrote {out}")

    # Headline
    print("\n" + "=" * 80)
    print("HEADLINE — RAG | v1 chained | Path A (Qwen + bolt-on Engram, no pretrain)")
    print("=" * 80)
    v1_path = Path("/home/ubuntu/multimodal-user-memory/results/engram_retrieval.json")
    if v1_path.exists():
        with open(v1_path) as f: v1 = json.load(f)
        print(f"{'modality':>8} | {'N':>3} | {'RAG':>6} | {'v1 best':>8} | {'Path A':>7} | A − v1")
        print("-" * 80)
        for name in ["vision", "audio"]:
            for N in Ns:
                rag_v = results[name]["rag"][N]
                A = results[name]["pathA_no_pretrain"][N]["retrieval_at_1"]
                v1_best = 0.0
                for cfg_name, cfg_res in v1[name].get("engram", {}).items():
                    if str(N) in cfg_res:
                        v1_best = max(v1_best, cfg_res[str(N)].get("retrieval_chained_disambig", 0.0))
                delta = A - v1_best
                mark = " ✓✓ beats v1" if delta > 0.1 else (" ✓ beats v1" if delta > 0.02 else (" ≈ tie" if abs(delta) <= 0.02 else " ✗ under v1"))
                print(f"{name:>8} | {N:>3} | {rag_v:>6.3f} | {v1_best:>8.3f} | {A:>7.3f} | {delta:>+6.3f}{mark}")


if __name__ == "__main__":
    sys.exit(main())
