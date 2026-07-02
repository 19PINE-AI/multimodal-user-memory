"""AttMem on Qwen2.5-VL using EXTERNAL ArcFace keys (not Qwen-VL's own vision tokens).

This is the key architectural validation: if the bank key lives in a DIFFERENT
space from the LM hidden (ArcFace 512-d vs LM 2048-d), does AttMem BEAT RAG
on Qwen-VL too --- confirming the §6 key-value-space orthogonality finding?

We re-use the standard attmem_train_and_eval.py pipeline but load Qwen2.5-VL's
TEXT-LM component (qwen_vl.language_model) and treat it as a normal CausalLM
for our purposes. The bank uses ArcFace keys from face_xxxl (same as our main
results), so the only thing that changes is the frozen LM.

Hypothesis: AttMem-VL-ArcFace should match or beat the Qwen2.5-3B AttMem
result (~0.99 retr@1 at N=10), confirming that the BEATS comes from
key-value-space orthogonality, not from the choice of base LM family.
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
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import MODALITY_TEXT, MODALITY_VISION, AttentionMemorySet
from v2_retrieval import split_by_identity, embedding_rag_ceiling

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class CausalLMWrapper(nn.Module):
    """Wrap Qwen2.5-VL so it looks like a CausalLM for our bolt: exposes
    config, lm_head, get_input_embeddings, and a forward(inputs_embeds, ...).
    """
    def __init__(self, vl_model):
        super().__init__()
        self.vl = vl_model
        # Use the text LM config (Qwen2.5-VL exposes config.get_text_config())
        if hasattr(vl_model.config, 'get_text_config'):
            self.config = vl_model.config.get_text_config()
        else:
            self.config = vl_model.config.text_config
        # The lm_head and embeddings live at the top of vl_model
        self.lm_head = vl_model.lm_head

    def get_input_embeddings(self):
        return self.vl.get_input_embeddings()

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None, use_cache=False):
        # Call the language_model directly to get text-only output, then lm_head
        # The hook on self.lm_head will fire.
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        hidden = self.vl.language_model(inputs_embeds=inputs_embeds,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache)
        hidden_states = hidden[0] if isinstance(hidden, tuple) else hidden.last_hidden_state
        logits = self.lm_head(hidden_states)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=logits, past_key_values=None,
                                        hidden_states=None, attentions=None)


class QwenVLArcFaceBolt(nn.Module):
    """AttMem bolt for Qwen2.5-VL with EXTERNAL ArcFace keys."""
    def __init__(self, vl_model, tokenizer, vision_key_dim=512, audio_key_dim=192):
        super().__init__()
        self.qwen = CausalLMWrapper(vl_model)
        self.tok = tokenizer
        self.hidden_size = self.qwen.config.hidden_size
        self.qwen_vocab = self.qwen.config.vocab_size
        for p in vl_model.parameters():
            p.requires_grad_(False)

        # Per-modality projection: ArcFace 512 -> LM hidden 2048
        import math
        self.vis_proj = nn.Linear(vision_key_dim, self.hidden_size, bias=False)
        self.aud_proj = nn.Linear(audio_key_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            ref = vl_model.get_input_embeddings().weight.detach()
            ref_norm = ref.norm(dim=-1).mean().item()
            for m in [self.vis_proj, self.aud_proj]:
                nn.init.normal_(m.weight, std=ref_norm / math.sqrt(m.weight.shape[1]))

        self.attmem = AttentionMemorySet(
            hidden_size=self.hidden_size,
            vision_key_dim=vision_key_dim,
            audio_key_dim=audio_key_dim,
            gpu_resident=True,
        )
        for m in [self.vis_proj, self.aud_proj, self.attmem]:
            m.to(dtype=torch.bfloat16)

        self._hook_handle = None
        self._last_modality_ids = None
        self._last_perc_keys_by_mod = None

    def _attmem_lm_head_hook(self, module, args, kwargs):
        if not args:
            return None
        hidden_states = args[0]
        if self._last_modality_ids is None:
            return None
        residual = self.attmem.forward_layer(
            hidden_states, self._last_modality_ids,
            perceptual_keys_by_mod=self._last_perc_keys_by_mod,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
        return (hidden_states,) + args[1:], kwargs

    def install_hook(self):
        self._hook_handle = self.qwen.lm_head.register_forward_pre_hook(
            self._attmem_lm_head_hook, with_kwargs=True)

    def _value_for_marker(self, marker_token_ids):
        tied = getattr(self.qwen.config, "tie_word_embeddings", False)
        if tied:
            return self.qwen.get_input_embeddings()(marker_token_ids)
        return self.qwen.lm_head.weight[marker_token_ids]

    def insert_batch(self, modality_id, keys, marker_token_ids):
        bank = self.attmem.banks[str(modality_id)]
        with torch.no_grad():
            ids = torch.tensor(list(marker_token_ids), device=DEVICE, dtype=torch.long)
            values = self._value_for_marker(ids)
        bank.insert_batch(keys.to(DEVICE), values)

    def reset_banks(self):
        self.attmem.reset()

    def build_inputs_embeds_from_perc_keys(self, modality_ids, text_input_ids, perc_keys_by_mod):
        B, T = modality_ids.shape
        emb = torch.zeros(B, T, self.hidden_size, device=DEVICE, dtype=torch.bfloat16)
        m_text = (modality_ids == MODALITY_TEXT)
        if m_text.any():
            text_ids = torch.where(m_text, text_input_ids, torch.zeros_like(text_input_ids))
            text_emb = self.qwen.get_input_embeddings()(text_ids)
            emb = emb + m_text.unsqueeze(-1).to(emb.dtype) * text_emb
        m_vis = (modality_ids == MODALITY_VISION)
        if m_vis.any() and MODALITY_VISION in perc_keys_by_mod:
            vis_keys = perc_keys_by_mod[MODALITY_VISION]
            vis_emb_perc = self.vis_proj(vis_keys.to(dtype=torch.bfloat16))
            emb_flat = emb.reshape(B * T, self.hidden_size)
            idx = m_vis.reshape(B * T).nonzero(as_tuple=False).squeeze(-1)
            emb_flat[idx] = emb_flat[idx] + vis_emb_perc
            emb = emb_flat.view(B, T, self.hidden_size)
        return emb

    def forward(self, modality_ids, text_input_ids, perc_keys_by_mod):
        self._last_modality_ids = modality_ids
        self._last_perc_keys_by_mod = perc_keys_by_mod
        emb = self.build_inputs_embeds_from_perc_keys(modality_ids, text_input_ids, perc_keys_by_mod)
        attn_mask = torch.ones_like(modality_ids)
        out = self.qwen(inputs_embeds=emb, attention_mask=attn_mask, use_cache=False)
        return out.logits


def build_query_context(tok, marker_token_id, T=24):
    pad_id = tok.pad_token_id or 0
    pref = tok.encode("You see", add_special_tokens=False)
    text_ids = list(pref) + [pad_id] * (T - 1 - len(pref))
    return (text_ids[: T - 1]) + [pad_id]


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    bsmax = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"=== AttMem-VL + ArcFace keys (Qwen2.5-VL-3B-Instruct) ===")
    EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")
    d = np.load(EMB / "arcface_face_xxxl.npz")
    emb = d["emb"].astype(np.float32)
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(emb, pid)
    print(f"  train: {len(set(tr_pid.tolist()))} IDs / {len(tr_emb)} samp")
    print(f"  eval:  {len(set(ev_pid.tolist()))} IDs / {len(ev_emb)} samp")

    print(f"\nLoading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        trust_remote_code=True, low_cpu_mem_usage=True,
    ); vl.eval()
    print(f"  hidden_size: {vl.config.get_text_config().hidden_size}, vocab: {vl.config.get_text_config().vocab_size}, tied: {getattr(vl.config.get_text_config(), 'tie_word_embeddings', False)}")

    print(f"Building bolt ...")
    bolt = QwenVLArcFaceBolt(vl, tok, vision_key_dim=emb.shape[1]).to(DEVICE)
    bolt.install_hook()

    # Quick pretrain (1500 steps to save time; full 12000 if requested)
    bank = bolt.attmem.banks[str(MODALITY_VISION)]
    proj = bolt.vis_proj
    params = list(bank.parameters()) + list(proj.parameters())
    print(f"  trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

    by_id = defaultdict(list)
    for i, p in enumerate(tr_pid):
        by_id[str(p)].append(i)
    ids = [p for p in by_id if len(by_id[p]) >= 2]
    rng = np.random.default_rng(0)
    T = 24

    print(f"\n[pretrain] {n_steps} steps  bank_size 64..{bsmax}")
    t0 = time.time()
    losses = []
    for step in range(n_steps):
        bs_step = min(int(rng.integers(64, bsmax + 1)), len(ids))
        chosen = rng.choice(len(ids), size=bs_step, replace=False)
        marker_ids = list(range(30001, 30001 + bs_step))
        reg_idxs = [int(rng.choice(by_id[ids[ix]])) for ix in chosen]
        reg_keys = torch.from_numpy(tr_emb[reg_idxs].astype(np.float32)).to(DEVICE)
        bank.reset()
        bolt.insert_batch(MODALITY_VISION, reg_keys, marker_ids)

        q_local = int(rng.integers(0, bs_step))
        q_id = ids[chosen[q_local]]
        q_cands = [i for i in by_id[q_id] if i != reg_idxs[q_local]]
        if not q_cands: q_cands = by_id[q_id]
        q_idx = int(rng.choice(q_cands))
        q_key = torch.from_numpy(tr_emb[q_idx].astype(np.float32)).unsqueeze(0).to(DEVICE)

        text_ids = build_query_context(tok, marker_ids[q_local], T=T)
        text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
        modality_ids_t = torch.tensor([[MODALITY_TEXT] * (T - 1) + [MODALITY_VISION]],
                                         dtype=torch.long, device=DEVICE)
        logits = bolt(modality_ids_t, text_ids_t, {MODALITY_VISION: q_key})
        target = torch.tensor([marker_ids[q_local]], device=DEVICE)
        loss = F.cross_entropy(logits[:, -1, :], target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(1, n_steps // 25) == 0:
            print(f"    step {step+1:5d}  loss={np.mean(losses[-50:]):.3f}  ({time.time()-t0:.0f}s)")
    if losses:
        print(f"  final loss: {np.mean(losses[-50:]):.3f}")
    else:
        losses = [0.0]
        print("  [ZERO-SHOT] no pretraining — read uses constant overrides only")

    # Optional hand-set read constants (training-free sharpness control),
    # mirroring attmem_train_and_eval.py.
    import os, math as _math
    if os.environ.get("ATTMEM_INV_TEMP"):
        it = float(os.environ["ATTMEM_INV_TEMP"])
        with torch.no_grad():
            for b in bolt.attmem.banks.values():
                b.log_inv_temp.copy_(torch.tensor(_math.log(it)))
        print(f"[temp override] inv_temp set to {it} (no gradient)")
    if os.environ.get("ATTMEM_OUT_GAIN"):
        og = float(os.environ["ATTMEM_OUT_GAIN"])
        with torch.no_grad():
            for b in bolt.attmem.banks.values():
                b.out_gain.copy_(torch.tensor(og))
        print(f"[gain override] out_gain set to {og} (no gradient)")

    # Paired multi-draw eval (same protocol as the text-LM runs), if requested
    if os.environ.get("ATTMEM_PAIRED_NS"):
        from attmem_train_and_eval import run_paired_multidraw
        run_paired_multidraw(bolt, ev_emb, ev_pid, MODALITY_VISION, tok, "vl-arcface", seed)
        return

    # Eval at multiple N
    print(f"\n[eval]")
    print(f"{'N':>5} | {'RAG':>6} | {'AttMem':>7} | {'ratio':>6} | verdict")
    print("-" * 50)
    results = {}
    Ns = [N for N in [5, 10, 20, 50, 100, 300, 700, 1000] if N <= len(set(ev_pid.tolist()))]
    for N in Ns:
        ev_by_id = defaultdict(list)
        for i, p in enumerate(ev_pid):
            ev_by_id[str(p)].append(i)
        ids_sorted = sorted(ev_by_id.keys())[:N]
        rng_e = np.random.default_rng(99)
        reg_idx_per_id = []
        for pid_str in ids_sorted:
            idxs = list(ev_by_id[pid_str]); rng_e.shuffle(idxs)
            reg_idx_per_id.append(idxs[0])
        reg_keys = torch.from_numpy(ev_emb[reg_idx_per_id].astype(np.float32)).to(DEVICE)
        bank.reset()
        marker_ids_e = list(range(30001, 30001 + N))
        bolt.insert_batch(MODALITY_VISION, reg_keys, marker_ids_e)
        correct = 0; total = 0
        for k, pid_str in enumerate(ids_sorted):
            idxs = list(ev_by_id[pid_str]); rng_e.shuffle(idxs)
            q_idxs = [i for i in idxs if i != reg_idx_per_id[k]][:3]
            for qi in q_idxs:
                q_key = torch.from_numpy(ev_emb[qi].astype(np.float32)).unsqueeze(0).to(DEVICE)
                text_ids = build_query_context(tok, 30001, T=T)
                text_ids_t = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
                modality_ids_t = torch.tensor([[MODALITY_TEXT] * (T - 1) + [MODALITY_VISION]],
                                                 dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    lg = bolt(modality_ids_t, text_ids_t, {MODALITY_VISION: q_key})
                    last = lg[0, -1, :]
                    ml = torch.stack([last[m] for m in marker_ids_e])
                    pred = int(ml.argmax().item())
                total += 1
                if pred == k: correct += 1
        attmem = correct / total if total else 0
        rag = embedding_rag_ceiling(ev_emb, ev_pid, N_subset=N, n_queries_per_id=3)
        ratio = attmem / rag if rag > 0 else float('nan')
        verdict = "BEATS" if attmem > rag else ("near" if ratio > 0.85 else "comp")
        print(f"{N:>5} | {rag:>6.3f} | {attmem:>7.3f} | {ratio:>6.2f} | {verdict}")
        results[N] = {"rag": rag, "attmem": attmem, "ratio": ratio, "N_queries": total}

    out = Path(f"/home/ubuntu/multimodal-user-memory/results/attmem_vl_arcface_steps{n_steps}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump({"mode": "v-xc-id-xxxl + Qwen2.5-VL + ArcFace keys",
                    "n_steps": n_steps, "seed": seed,
                    "model_id": MODEL_ID,
                    "final_loss": float(np.mean(losses[-50:])),
                    "results": {str(N): v for N, v in results.items()}},
                   f, indent=2, default=str)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
