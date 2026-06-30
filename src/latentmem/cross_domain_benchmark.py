"""Cross-domain perceptual-memory benchmark (statistical rigor).

The memory's recognition core is cosine matching over the domain encoder's embeddings
(== the AttMem / KV-cache attention read). We evaluate it across 12 perceptual domains
spanning 5 modalities and multiple datasets/encoders, with many independent draws per
domain so every number carries a 95% CI. Each draw is one recognition task (register N
identities one view each, recognise a held-out cross-condition view, recall@1 over the
N registered keys). Total tasks = #domains x #draws.

Usage: python3 cross_domain_benchmark.py [N] [draws]
"""
import sys, json
import numpy as np
from collections import defaultdict
from pathlib import Path

EMB = Path("/home/ubuntu/multimodal-user-memory/runs/embeddings")

# (display name, modality, file)  -- 12 domains, 5 modalities, diverse datasets/encoders
DOMAINS = [
    ("Face / AgeDB (ArcFace, cross-age)",   "face",     "arcface_agedb"),
    ("Face / LFW (ArcFace)",                "face",     "arcface_lfw_xxxl"),
    ("Face / LFW (AntelopeV2)",             "face",     "antelope_lfw_xxl"),
    ("Face / combined (ArcFace)",           "face",     "arcface_face_xxxl"),
    ("Speaker / LibriSpeech (ECAPA)",       "speaker",  "ecapa_libri_large"),
    ("Speaker / VoxCeleb (ECAPA, in-wild)", "speaker",  "ecapa_voxceleb1"),
    ("Acoustic scene / ESC-50 (AST)",       "acoustic", "ast_esc50_full"),
    ("Painting style / WikiArt (CLIP)",     "style",    "clip_mid_wikiart_xxl"),
    ("Painting style / WikiArt (DINOv2)",   "style",    "dinov2_wikiart"),
    ("Vocal tone / paralinguistic (w2v)",   "tone",     "wav2vec_para_spk_emo"),
    ("Face / AgeDB (Qwen-VL native)",       "face*",    "qwenvl_agedb_keys"),
    ("Painting style / contrastive",        "style",    "style_contrastive_xl"),
]


def l2(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def load(name):
    d = np.load(EMB / f"{name}.npz", allow_pickle=True)
    k = "emb" if "emb" in d else "keys"
    pid = d["pid"] if d["pid"].dtype.kind == "U" else np.array([str(p) for p in d["pid"]])
    return l2(d[k].astype(np.float32)), pid


def recall_draw(emb, pid, N, n_q, seed):
    by = defaultdict(list)
    for i, p in enumerate(pid): by[str(p)].append(i)
    ids = [p for p in by if len(by[p]) >= 2]
    rng = np.random.default_rng(seed); rng.shuffle(ids); ids = ids[:N]
    reg, lab, qs = [], [], []
    for p in ids:
        ix = list(by[p]); rng.shuffle(ix); reg.append(emb[ix[0]]); lab.append(p)
        for qi in ix[1:1 + n_q]: qs.append((emb[qi], p))
    R = np.stack(reg); Q = np.stack([q[0] for q in qs]); pred = (Q @ R.T).argmax(1)
    return float(np.mean([lab[pred[k]] == qs[k][1] for k in range(len(qs))]))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"=== Cross-domain perceptual-memory benchmark: recall@1, N={N}, {draws} draws/domain ===")
    print(f"{'domain':40} {'mod':9} {'#id':>5}  recall@1 (mean +/- 95% CI)")
    rows = []; total_tasks = 0; all_means = []
    for name, mod, fname in DOMAINS:
        try:
            emb, pid = load(fname)
        except FileNotFoundError:
            print(f"{name:40} (missing {fname})"); continue
        nid = len(set(pid.tolist()))
        accs = [recall_draw(emb, pid, min(N, nid), 3, s) for s in range(1000, 1000 + draws)]
        m = float(np.mean(accs)); ci = 1.96 * float(np.std(accs, ddof=1)) / np.sqrt(len(accs))
        rows.append({"domain": name, "modality": mod, "n_id": nid, "N": min(N, nid),
                     "recall": m, "ci95": ci, "draws": draws})
        total_tasks += draws; all_means.append(m)
        print(f"{name:40} {mod:9} {nid:>5}  {m:.3f} +/- {ci:.3f}")
    mods = sorted(set(r["modality"].rstrip("*") for r in rows))
    print(f"\n  domains: {len(rows)}   modalities: {len(mods)} ({', '.join(mods)})   total tasks: {total_tasks}")
    print(f"  macro-avg recall@1 across domains: {np.mean(all_means):.3f}")
    Path("/home/ubuntu/multimodal-user-memory/results/cross_domain.json").write_text(json.dumps(
        {"N": N, "draws": draws, "total_tasks": total_tasks, "n_domains": len(rows),
         "n_modalities": len(mods), "macro_avg": float(np.mean(all_means)), "rows": rows}, indent=2))
    print("wrote results/cross_domain.json")


if __name__ == "__main__":
    main()
