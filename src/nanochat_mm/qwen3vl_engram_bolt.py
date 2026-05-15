"""Path A on Qwen3-VL-8B-Thinking — bolt-on Engram on a true VLM.

Variant of qwen_engram_bolt.py: subclass to target the Qwen3-VL layer
path (model.model.language_model.layers) instead of Qwen2.5's path
(model.model.layers). Other architecture identical.

Run the audio K=64 generic-NTP recipe (our strongest baseline) and the
PerceptMem v0.1 scorecard on this larger VLM backbone. Two questions:
  1. Does the recipe transfer to a 'true VLM' backbone?
  2. Does 8B-Qwen3-VL improve over 3B-Qwen2.5 on the mechanism metric?
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from engram_module_mm import (
    MultimodalEngramSet, MultimodalEngramConfig,
    MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)
from real_encoder_train import fit_naive_rq
from v2_retrieval import split_by_identity, embedding_rag_ceiling
from pathA_generic_pretrain import pretrain_generic
from qwen_engram_bolt import build_fixed_context, get_touched_rows, evaluate as evaluate_qwen
from qwen_engram_bolt import QwenEngramBolt

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"


class Qwen3VLEngramBolt(QwenEngramBolt):
    """Subclass that points hooks at Qwen3-VL's text language model layers."""

    def __init__(self, qwen_model, qwen_tokenizer, V_vis=32, V_aud=32,
                 engram_attach_layer=24, engram_n_embed_per_ngram=128,
                 engram_vocab_per_ngram=503, engram_n_head=4):
        nn.Module.__init__(self)
        self.qwen = qwen_model
        self.tok = qwen_tokenizer
        # text_config carries hidden_size / vocab for the LM head
        if hasattr(qwen_model.config, "text_config"):
            self.hidden_size = qwen_model.config.text_config.hidden_size
            self.qwen_vocab = qwen_model.config.text_config.vocab_size
        else:
            self.hidden_size = qwen_model.config.hidden_size
            self.qwen_vocab = qwen_model.config.vocab_size

        for p in self.qwen.parameters():
            p.requires_grad_(False)

        self.vis_perc_emb = nn.Embedding(V_vis, self.hidden_size)
        self.aud_perc_emb = nn.Embedding(V_aud, self.hidden_size)
        with torch.no_grad():
            ref = qwen_model.get_input_embeddings().weight.detach()
            # Compute norm on a small sample to avoid allocating a full-vocab
            # auxiliary tensor on GPU (Qwen3-VL has 151k-vocab × 5120-dim,
            # which can OOM on a contended GPU). 1024 rows is plenty for a mean.
            sample_rows = min(1024, ref.shape[0])
            sample = ref[:sample_rows].to(dtype=torch.float32)
            ref_norm = sample.norm(dim=-1).mean().item()
            del sample
            for e in [self.vis_perc_emb, self.aud_perc_emb]:
                nn.init.normal_(e.weight, std=ref_norm / math.sqrt(self.hidden_size))

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
        self.engram.to(dtype=torch.bfloat16)
        self.vis_perc_emb.to(dtype=torch.bfloat16)
        self.aud_perc_emb.to(dtype=torch.bfloat16)
        self._hook_handle = None
        self._last_input_ids = None
        self._last_modality_ids = None

    def install_hook(self):
        # Qwen3-VL path: model.model.language_model.layers[N]
        layer = self.qwen.model.language_model.layers[self.attach_layer]
        self._hook_handle = layer.register_forward_pre_hook(self._engram_hook, with_kwargs=True)


def main():
    print("=" * 70)
    print(f"Path A on {MODEL_ID}")
    print("=" * 70)

    print("\nLoading Qwen3-VL-8B-Thinking ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    qwen = qwen.to(DEVICE).eval()
    n_layers = qwen.config.text_config.num_hidden_layers
    hidden = qwen.config.text_config.hidden_size
    print(f"  loaded; {sum(p.numel() for p in qwen.parameters())/1e9:.2f}B params; "
          f"{n_layers} text layers; hidden={hidden}")

    aud = np.load("/home/ubuntu/multimodal-user-memory/runs/embeddings/ecapa_libri_large.npz")
    aud_tr_emb, aud_tr_pid, aud_ev_emb, aud_ev_pid = split_by_identity(aud['emb'], aud['pid'])
    K = 64
    apply_fn = fit_naive_rq(aud_tr_emb, n_levels=1, k_per=K)

    attach = int(0.66 * n_layers)
    print(f"\n  attach Engram at layer {attach} / {n_layers}")

    bolt = Qwen3VLEngramBolt(qwen, tok, V_vis=K, V_aud=K,
                              engram_attach_layer=attach).to(DEVICE)
    bolt.install_hook()
    print(f"  trainable params: {sum(p.numel() for p in bolt.parameters() if p.requires_grad):,}")
    if torch.cuda.is_available():
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n[pretrain] audio generic-NTP K=64, 400 steps (bigger model = slower per step)")
    losses = pretrain_generic(bolt, aud_tr_emb, aud_tr_pid, apply_fn, MODALITY_AUDIO, tok,
                              n_steps=400, lr=3e-4, batch=2, T=64, frac_perceptual=0.15)
    print(f"  pretrain final loss: {float(np.mean(losses[-30:])):.4f}")

    print("\n[eval]")
    Ns = [5, 10, 20]; nq = 5
    results = {}
    for N in Ns:
        rag = embedding_rag_ceiling(aud_ev_emb, aud_ev_pid, N_subset=N, n_queries_per_id=nq)
        r = evaluate_qwen(bolt, apply_fn, aud_ev_emb, aud_ev_pid, MODALITY_AUDIO, tok,
                          N_subset=N, n_queries_per_id=nq, max_steps=80, lr=1.0, T=24)
        print(f"  N={N:>2}  RAG={rag:.3f}  retr@1={r['retrieval_at_1']:.3f}  "
              f"code-match={r['code_match_retr']:.3f} (on {100*r['fraction_code_match']:.0f}%)  "
              f"collisions={r['N_collision_codes']}")
        results[N] = {"rag": rag, **r}

    import json
    out = Path("/home/ubuntu/multimodal-user-memory/results/pathA_qwen3vl.json")
    with open(out, "w") as f:
        json.dump({"model": MODEL_ID, "n_params_b": float(sum(p.numel() for p in qwen.parameters())/1e9),
                    "attach_layer": int(attach), "K": K, "results": results}, f, indent=2, default=str)
    print(f"\n[done] {out}")

    # Compare to Qwen2.5-3B audio K=64
    print("\n" + "=" * 80)
    print("Comparison — audio K=64 generic-NTP across base LMs (large data)")
    print("=" * 80)
    import json as _json
    q3b = _json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_scaling_k64.json"))
    q14b = _json.load(open("/home/ubuntu/multimodal-user-memory/results/pathA_qwen14b.json"))
    print(f"{'N':>4} | {'Qwen2.5-3B':>10} | {'Qwen2.5-14B':>11} | {'Qwen3-VL-8B':>11} || code-match: 3B / 14B / 3-VL")
    print("-" * 90)
    for N in Ns:
        a3 = q3b["audio"].get(str(N), {})
        a14 = q14b["results"].get(str(N), {})
        avl = results.get(N, {})
        print(f"{N:>4} | {a3.get('retrieval_at_1', float('nan')):>10.3f} | "
              f"{a14.get('retrieval_at_1', float('nan')):>11.3f} | "
              f"{avl.get('retrieval_at_1', float('nan')):>11.3f} || "
              f"{a3.get('code_match_retr', float('nan')):.3f} / "
              f"{a14.get('code_match_retr', float('nan')):.3f} / "
              f"{avl.get('code_match_retr', float('nan')):.3f}")


if __name__ == "__main__":
    sys.exit(main())
