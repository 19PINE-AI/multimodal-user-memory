"""Path A + generic-NTP + Engram attached at TWO layers.

Variant of pathA_generic_pretrain.py: Engram is wired into Qwen at layers
[16, 28] instead of only [24]. Doubles Engram capacity and gives the
surgical insertion two "shots" at the hidden state — once mid-network,
once near the output.

Cheap test: if audio code-match retrieval jumps notably over the
single-layer 0.86 figure, the path to overall retrieval > v1-chained
becomes 2-layer + STE + more steps.
"""
import json
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
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from qwen_engram_bolt import (
    build_fixed_context, get_touched_rows, MODEL_ID, DEVICE,
    QwenEngramBolt,
)
from pathA_generic_pretrain import pretrain_generic

torch.manual_seed(42); np.random.seed(42)


class QwenEngramBoltMultiLayer(QwenEngramBolt):
    """Variant attaching Engram at multiple layers via multiple hooks."""

    def __init__(self, qwen_model, qwen_tokenizer, V_vis, V_aud,
                 engram_attach_layers=(16, 28),
                 engram_n_embed_per_ngram=128,
                 engram_vocab_per_ngram=503, engram_n_head=4):
        # Bypass the parent __init__ — we need attach_layers (plural)
        nn.Module.__init__(self)
        self.qwen = qwen_model
        self.tok = qwen_tokenizer
        self.hidden_size = qwen_model.config.hidden_size
        self.qwen_vocab = qwen_model.config.vocab_size
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        # Perc emb tables (same as parent)
        import math as _m
        self.vis_perc_emb = nn.Embedding(V_vis, self.hidden_size)
        self.aud_perc_emb = nn.Embedding(V_aud, self.hidden_size)
        with torch.no_grad():
            ref = qwen_model.get_input_embeddings().weight.detach()
            ref_norm = ref.norm(dim=-1).mean().item()
            for e in [self.vis_perc_emb, self.aud_perc_emb]:
                nn.init.normal_(e.weight, std=ref_norm / _m.sqrt(self.hidden_size))

        # Engram attached at multiple layers
        self.attach_layers = list(engram_attach_layers)
        # For compatibility with QwenEngramBolt's surgical_insert helpers, set
        # `attach_layer` to the first layer; get_touched_rows() will use this.
        self.attach_layer = self.attach_layers[0]

        eng_cfg = MultimodalEngramConfig(
            layer_ids=list(self.attach_layers),
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
        self.engram.to(dtype=torch.bfloat16)
        self.vis_perc_emb.to(dtype=torch.bfloat16)
        self.aud_perc_emb.to(dtype=torch.bfloat16)
        self._hook_handles = []
        self._last_input_ids = None
        self._last_modality_ids = None

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            if not args: return None
            hs = args[0]
            if self._last_input_ids is None: return None
            res = self.engram.forward_layer(hs, self._last_input_ids, self._last_modality_ids,
                                             layer_idx=layer_idx)
            if res is not None:
                hs = hs + res
            return (hs,) + args[1:], kwargs
        return hook

    def install_hook(self):
        for lid in self.attach_layers:
            layer = self.qwen.model.layers[lid]
            h = layer.register_forward_pre_hook(self._make_hook(lid), with_kwargs=True)
            self._hook_handles.append(h)

    def remove_hook(self):
        for h in self._hook_handles: h.remove()
        self._hook_handles = []


def get_touched_rows_multi(eng, code_token, input_ids, attach_layers):
    """Get touched rows for all attached layers; returns {(layer, ks): row_set}."""
    inp = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
    if inp.ndim == 1: inp = inp[None]
    hashes_per_layer = eng.hash_mapping.hash_all_layers(inp, user_salt=int(eng.user_salt))
    last_pos = inp.shape[1] - 1
    touched = {}
    for lid in attach_layers:
        if lid not in hashes_per_layer: continue
        local = hashes_per_layer[lid][0, last_pos]
        tbl = eng.tables[str(lid)]
        global_rows = local + tbl.offsets.cpu().numpy()
        touched[str(lid)] = set(int(r) for r in global_rows.tolist())
    return touched


def surgical_insert_multi(bolt, code_token, modality_id, marker_text_id,
                            tok, max_steps=80, lr=1.0, T=24):
    eng = bolt.engram.engrams[str(modality_id)]
    input_ids, modality_ids = build_fixed_context(code_token, modality_id, tok, marker_text_id, T=T)
    input_ids = input_ids.to(DEVICE); modality_ids = modality_ids.to(DEVICE)
    touched = get_touched_rows_multi(eng, code_token, input_ids, bolt.attach_layers)

    if modality_id == MODALITY_VISION:
        perc_emb = bolt.vis_perc_emb
    else:
        perc_emb = bolt.aud_perc_emb
    params = [eng.tables[ks].embedding.weight for ks in touched] + [perc_emb.weight]
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
            if perc_emb.weight.grad is not None:
                pmask = torch.zeros(perc_emb.weight.shape[0], 1,
                                     device=perc_emb.weight.device, dtype=perc_emb.weight.grad.dtype)
                pmask[int(code_token)] = 1.0
                perc_emb.weight.grad.mul_(pmask)
        opt.step()
        steps_taken = step + 1
        if last_loss < 0.5: break
    return steps_taken, last_loss


def evaluate_multi(bolt, codebook_apply, eval_emb, eval_pid, modality_id, tok,
                   N_subset=None, n_queries_per_id=None, max_steps=80, lr=1.0, T=24,
                   marker_offset=30001):
    eng = bolt.engram.engrams[str(modality_id)]
    by_id = defaultdict(list)
    for i, p in enumerate(eval_pid): by_id[str(p)].append(i)
    ids_sorted = sorted(by_id.keys())
    if N_subset is not None: ids_sorted = ids_sorted[:N_subset]
    markers = {pid: marker_offset + i for i, pid in enumerate(ids_sorted)}

    eng_snap = {ks: tbl.embedding.weight.detach().clone() for ks, tbl in eng.tables.items()}
    if modality_id == MODALITY_VISION:
        perc_emb = bolt.vis_perc_emb
    else:
        perc_emb = bolt.aud_perc_emb
    perc_snap = perc_emb.weight.detach().clone()

    rng = np.random.default_rng(99)
    register_codes = {}; code_to_pid = defaultdict(list); insert_stats = []
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        reg_code_arr = codebook_apply(eval_emb[idxs[0]][None])[0]
        reg_code = int(reg_code_arr.item() if hasattr(reg_code_arr, 'item') else reg_code_arr)
        steps, fl = surgical_insert_multi(bolt, reg_code, modality_id, markers[pid],
                                            tok, max_steps=max_steps, lr=lr, T=T)
        insert_stats.append((steps, fl))
        register_codes[pid] = reg_code
        code_to_pid[reg_code].append(pid)

    correct = 0; total = 0; code_match_c = 0; code_match_t = 0
    for pid in ids_sorted:
        idxs = list(by_id[pid]); rng.shuffle(idxs)
        q_idxs = idxs[1:]
        if n_queries_per_id is not None: q_idxs = q_idxs[:n_queries_per_id]
        for qi in q_idxs:
            q_code_arr = codebook_apply(eval_emb[qi][None])[0]
            q_code = int(q_code_arr.item() if hasattr(q_code_arr, 'item') else q_code_arr)
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
        perc_emb.weight.copy_(perc_snap)

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
    print("Path A + generic-NTP + 2-layer Engram attach")
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
    aud_apply = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)
    vis_apply = fit_naive_rq(vis_tr_emb, n_levels=1, k_per=K)

    bolt = QwenEngramBoltMultiLayer(qwen, tok, V_vis=K, V_aud=K,
                                       engram_attach_layers=(16, 28)).to(DEVICE)
    bolt.install_hook()
    trainable = sum(p.numel() for p in bolt.parameters() if p.requires_grad)
    print(f"  trainable params (2-layer): {trainable:,}")

    print("\n[pretrain] vision generic-NTP, 2-layer ...")
    vis_losses = pretrain_generic(bolt, vis_tr_emb, vis_tr_pid, vis_apply, MODALITY_VISION, tok,
                                    n_steps=600, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    print(f"  vision final loss: {float(np.mean(vis_losses[-30:])):.4f}")
    print("\n[pretrain] audio generic-NTP, 2-layer ...")
    aud_losses = pretrain_generic(bolt, aud_tr_emb, aud_tr_pid, aud_apply, MODALITY_AUDIO, tok,
                                    n_steps=600, lr=3e-4, batch=4, T=64, frac_perceptual=0.15)
    print(f"  audio final loss: {float(np.mean(aud_losses[-30:])):.4f}")

    print("\n" + "=" * 70)
    print("Held-out surgical insertion + retrieval (2-layer)")
    print("=" * 70)
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for mid, name, emb, pids, apply_fn in [
        (MODALITY_VISION, "vision", vis_ev_emb, vis_ev_pid, vis_apply),
        (MODALITY_AUDIO,  "audio",  aud_ev_emb, aud_ev_pid, aud_apply),
    ]:
        print(f"\n[{name}]")
        rag = {N: embedding_rag_ceiling(emb, pids, N_subset=N, n_queries_per_id=nq) for N in Ns}
        out_eval = {}
        for N in Ns:
            print(f"  RAG ceiling N={N}: {rag[N]:.4f}")
            print(f"  Path A 2-layer generic-NTP N={N} ...", end="", flush=True)
            r = evaluate_multi(bolt, apply_fn, emb, pids, mid, tok,
                                N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0)
            print(f"  retr@1={r['retrieval_at_1']:.4f}  "
                  f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
                  f"collisions={r['N_collision_codes']}  insert-loss={r['avg_insert_loss']:.3f}")
            out_eval[N] = r
        results[name] = {"rag": rag, "pathA_two_layer": out_eval}

    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_two_layer.json")
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare to single-layer generic-NTP
    pg = json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_generic_pretrain.json"))
    print("\n" + "=" * 100)
    print("HEADLINE — generic-NTP at 1 layer vs 2 layers")
    print("=" * 100)
    print(f"{'modality':>8} | {'N':>3} | {'1-layer code-match':>18} | {'2-layer code-match':>18} | {'1-layer overall':>15} | {'2-layer overall':>15}")
    print("-" * 100)
    for name in ["vision", "audio"]:
        for N in Ns:
            g1 = pg[name]["pathA_generic_pretrain"][str(N)]
            g2 = results[name]["pathA_two_layer"][N]
            print(f"{name:>8} | {N:>3} | {g1['code_match_retr']:>18.3f} | {g2['code_match_retr']:>18.3f} | "
                  f"{g1['retrieval_at_1']:>15.3f} | {g2['retrieval_at_1']:>15.3f}")


if __name__ == "__main__":
    sys.exit(main())
