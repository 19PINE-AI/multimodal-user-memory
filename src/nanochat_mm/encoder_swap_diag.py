"""Same-code-rate diagnostic for stronger encoders.

Re-extracts embeddings with stronger cross-condition encoders, then
measures the diagnostic that gates Path A's accuracy at scale:
same-code rate on eval split at K=64/128/256/512.

If the stronger encoder substantially improves same-code rate, Path A's
accuracy at scale will follow. If not, the encoder isn't the bottleneck
either — the codebook architecture is.

Modalities + encoder swaps:
  - V-XC-ID: ArcFace R50 → AdaFace IR-101 (cvlface)
  - A-PARA:  wav2vec2-XLSR-emotion → emotion2vec_plus_base
  - A-SCN:   AST → CED-Base
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIR = "/home/ubuntu/multimodal-user-memory/runs/embeddings"


def same_code_rate(codes, pid):
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    same = 0; total = 0
    for samps in by_id.values():
        if len(samps) < 2: continue
        for i in range(len(samps)):
            for j in range(i+1, len(samps)):
                if codes[samps[i]] == codes[samps[j]]: same += 1
                total += 1
    return same / total if total else 0.0


def inter_id_collision(codes, pid):
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[str(p)].append(i)
    ids = list(by_id.keys())
    code_arr = np.asarray(codes)
    n_diff = 0; n_diff_same = 0
    for i in range(len(ids)):
        a = by_id[ids[i]][0]
        for j in range(i+1, len(ids)):
            b = by_id[ids[j]][0]
            if code_arr[a] == code_arr[b]: n_diff_same += 1
            n_diff += 1
    return n_diff_same / n_diff if n_diff else 0.0


def sweep_codebook(tr_emb, tr_pid, ev_emb, ev_pid, label):
    sys.path.insert(0, str(Path(__file__).parent))
    from real_encoder_train import fit_naive_rq
    print(f"\n[{label}] K-sweep same-code rate on eval")
    print(f"  train={len(tr_emb)} samp/{len(set(tr_pid.tolist()))} IDs  "
          f"eval={len(ev_emb)} samp/{len(set(ev_pid.tolist()))} IDs")
    print(f"  {'K':>5} | {'same-code':>10} | {'inter-coll':>11} | {'codes used':>11}")
    print("  " + "-" * 50)
    out = {}
    for K in [16, 32, 64, 128, 256]:
        if K >= len(tr_emb):
            continue
        fn = fit_naive_rq(tr_emb, n_levels=1, k_per=K, seed=42)
        codes = fn(ev_emb)
        if codes.ndim == 2: codes = codes[:, 0]
        same = same_code_rate(codes, ev_pid)
        inter = inter_id_collision(codes, ev_pid)
        used = len(set(codes.tolist()))
        print(f"  {K:>5} | {same:>10.3f} | {inter:>11.4f} | {used:>5d}/{K}")
        out[K] = {"same_code": same, "inter_coll": inter, "codes_used": used}
    return out


def extract_antelopev2_lfw(out_path):
    """InsightFace AntelopeV2 (R100 ArcFace, Glint360K) on LFW.

    Stronger than buffalo_l (R50 on WebFace) — R100 backbone, trained on
    Glint360K (360k identities vs WebFace's 600k images of 85k identities,
    Glint has more diversity). Same 112x112 normalisation as ArcFace, so
    drop-in compatible with our pipeline.
    """
    print("Loading LFW (min_faces=3) ...")
    from sklearn.datasets import fetch_lfw_people
    import cv2, random
    random.seed(42); np.random.seed(42)
    lfw = fetch_lfw_people(min_faces_per_person=3, color=True, resize=1.0)
    print(f"  {lfw.images.shape[0]} photos, {len(lfw.target_names)} people")

    by_person = defaultdict(list)
    for i, t in enumerate(lfw.target):
        by_person[int(t)].append(i)
    eligible = sorted([(p, idxs) for p, idxs in by_person.items() if len(idxs) >= 3])
    chosen = random.sample(eligible, k=min(2000, len(eligible)))
    print(f"  sampled {len(chosen)} identities")

    print("\nLoading AntelopeV2 (R100 ArcFace, Glint360K) ...")
    import onnxruntime as ort
    # Ensure the model is downloaded
    antelope_dir = Path.home() / ".insightface" / "models" / "antelopev2"
    if not antelope_dir.exists():
        print("  downloading antelopev2 ...")
        import insightface
        app = insightface.app.FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
        # This downloads to ~/.insightface
    # Use the recognition ONNX directly
    cand = [antelope_dir / "glintr100.onnx", antelope_dir / "w600k_r100.onnx"]
    onnx_path = next((p for p in cand if p.exists()), None)
    if onnx_path is None:
        # List what's there
        if antelope_dir.exists():
            print(f"  contents: {list(antelope_dir.iterdir())}")
        # Try insightface FaceAnalysis to do the download
        import insightface
        app = insightface.app.FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(112, 112))
        cand = [antelope_dir / "glintr100.onnx", antelope_dir / "w600k_r100.onnx"]
        onnx_path = next((p for p in cand if p.exists()), None)
    if onnx_path is None:
        print("  could not find antelopev2 ONNX after download attempts")
        return None

    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"  loaded {onnx_path.name}")

    embs, pids = [], []
    t0 = time.time()
    for k, (pid, idxs) in enumerate(chosen):
        sel = random.sample(idxs, k=min(3, len(idxs)))
        for i in sel:
            img = lfw.images[i]
            img = (img * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            arr = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
            e = sess.run(None, {inp_name: arr})[0][0]
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e); pids.append(str(pid))
        if (k+1) % 200 == 0:
            print(f"  processed {k+1}/{len(chosen)} ids ({time.time()-t0:.0f}s)")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    np.savez(out_path, emb=emb, pid=pid)
    print(f"  saved {out_path}: {emb.shape}, {len(set(pid))} IDs")
    return out_path


def extract_ced_esc50(out_path):
    """CED-Base audio classifier on ESC-50."""
    from datasets import load_dataset
    print("\nLoading ESC-50 audio dataset ...")
    ds = load_dataset("ashraq/esc50", split="train")
    print(f"  {len(ds)} clips")

    print("\nLoading CED-Base ...")
    from transformers import AutoModel, AutoFeatureExtractor
    try:
        fe = AutoFeatureExtractor.from_pretrained("mispeech/CED-Base", trust_remote_code=True)
        model = AutoModel.from_pretrained("mispeech/CED-Base", trust_remote_code=True).to(DEVICE).eval()
        print("  CED loaded.")
    except Exception as e:
        print(f"  load failed: {e}")
        return None

    embs, pids = [], []
    for k in range(len(ds)):
        ex = ds[k]
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k_: v.to(DEVICE) for k_, v in inputs.items()})
            # Use the pooled / last_hidden_state mean
            if hasattr(out, "last_hidden_state"):
                e = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "logits"):
                e = out.logits  # pre-softmax classification scores as feature
            else:
                e = out[0] if isinstance(out, tuple) else out
                if e.dim() > 2: e = e.mean(dim=tuple(range(1, e.dim()-1)))
            e = F.normalize(e, dim=-1)[0].cpu().numpy()
        embs.append(e); pids.append(str(ex["category"]))
        if (k+1) % 100 == 0:
            print(f"  encoded {k+1}/{len(ds)}")

    emb = np.stack(embs).astype(np.float32)
    pid = np.array(pids)
    np.savez(out_path, emb=emb, pid=pid)
    print(f"  saved {out_path}: {emb.shape}, {len(set(pid))} categories")
    return out_path


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "face"
    out = Path(EMB_DIR)
    out.mkdir(parents=True, exist_ok=True)

    if target == "face":
        path = extract_antelopev2_lfw(str(out / "antelope_lfw_xxl.npz"))
        if path:
            d = np.load(path)
            sys.path.insert(0, str(Path(__file__).parent))
            from v2_retrieval import split_by_identity
            tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(d["emb"], d["pid"])
            results = sweep_codebook(tr_emb, tr_pid, ev_emb, ev_pid, "AdaFace LFW-XXL")
            print("\nFor comparison, ArcFace R50 on LFW-XXL was:")
            print("  K=64: same-code ~0.46 (with adapter), ~0.26 (naive)")
            # Save diagnostic
            outj = Path("/home/ubuntu/multimodal-user-memory/results/encoder_swap_adaface.json")
            json.dump({"encoder": "AdaFace IR-101", "data": "LFW-XXL", "K_sweep": results},
                       open(outj, "w"), indent=2)
            print(f"  [saved] {outj}")

    elif target == "scene":
        path = extract_ced_esc50(str(out / "ced_esc50.npz"))
        if path:
            d = np.load(path)
            sys.path.insert(0, str(Path(__file__).parent))
            from v2_retrieval import split_by_identity
            tr_emb, tr_pid, ev_emb, ev_pid = split_by_identity(d["emb"], d["pid"])
            results = sweep_codebook(tr_emb, tr_pid, ev_emb, ev_pid, "CED-Base ESC-50")
            print("\nFor comparison, AST on ESC-50 K=32 same-code = 0.44 (naive baseline)")
            outj = Path("/home/ubuntu/multimodal-user-memory/results/encoder_swap_ced.json")
            json.dump({"encoder": "CED-Base", "data": "ESC-50", "K_sweep": results},
                       open(outj, "w"), indent=2)
            print(f"  [saved] {outj}")

    else:
        print(f"unknown target: {target}")


if __name__ == "__main__":
    sys.exit(main())
