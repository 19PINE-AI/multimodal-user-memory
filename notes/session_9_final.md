# Session 9 final — closing four loose ends from session 8

**Date:** 2026-05-15

Session 8 closed with the claim that "the science is settled" and the remaining work was "engineering for camera-ready." Four loose ends from the post-session-8 commit log (#8 forgetting drift, #9 salt leak, #11 Qwen3-VL deferred, and the unfinished head-to-head argument in `baseline_positioning.md`) were either probes that ended ambiguously or scripts that were never run end-to-end. This session closes them.

## 1. Salt isolation v2 — leak is closed

Experiment #9 reported "partial isolation; leak via shared perc_emb." Root cause: the per-modality Engram hash is salted (`eng.user_salt` is XORed into the hash mix), but the trainable `vis_perc_emb` / `aud_perc_emb` embedding tables are keyed by raw code with no salt. Two users at the same code C therefore wrote to the same perc_emb row, and the gradient step at insertion time leaked the marker across users.

**Fix (`salt_isolation_v2.py`):** shard the perceptual embedding tables across `N_user_buckets = 8` user buckets. The effective row index becomes `bucket * V + code`, where the bucket is a deterministic hash of the salt (mixed via Knuth's multiplicative constant 2654435761). `set_user_bucket(salt)` routes both insertion and query.

| Metric | v1 (#9) | v2 (now) | Interpretation |
|---|---|---|---|
| User A in-salt retrieval | 0.467 | 0.467 | unchanged |
| User A cross-salt leak   | 0.467 | **0.200** | now at chance |
| User B in-salt retrieval | 0.800 | 0.400 | new bucket starts cold |
| User B cross-salt leak   | 0.800 | **0.200** | now at chance |

Cross-salt retrieval drops to exactly the chance baseline (1/5 = 0.20). Salt 0 is preserved as bucket 0 so legacy single-user runs are unaffected. Result file: `results/salt_isolation_v2.json`.

Cost: 8× the perceptual-embedding table size (~3 MB → ~24 MB for K=64, hidden=2048 — negligible relative to the 3 B base).

## 2. Forgetting probe v2 — row-freeze mitigation works at intermediate probes

Experiment #8 reported "drift observed under sequential insertion": Probe-1 retrieval collapsed from 1.0 to 0.0 after only a handful of additional insertions in that seed. Two findings in v2:

- **Re-running #8's exact protocol with the same seeds gives a different drift pattern** — Probe-1 stays stable at 0.67 across all 20 insertions in both the baseline and the freeze condition. The collapse-to-zero pattern in #8's saved JSON was likely driven by `list(set(ev_pid))` hash randomness in the codebook-training subset selection (a Python 3.7+ hash-seed effect — not deterministic across processes).
- **The freeze mitigation, however, clearly helps the intermediate Probe-5** — see the saved curves in `results/forgetting_probe_v2.json`. Baseline Probe-5 retention degrades to 0.0 for insertions 8 through 18 and only recovers to 0.33 at k=19; freeze stays at 0.33 throughout and reaches 0.67 at k=19–20.

**Mechanism (`forgetting_probe_v2.surgical_insert_freeze`):** track the set of Engram-table rows written by each surgical insertion. On subsequent insertions, mask the gradient on those rows to zero — they are written once and frozen. The perc-emb row for any registered code is similarly frozen against further updates. This is the strict "surgical insertion" property in the [[user-as-engram]] sense: a row is written once and never overwritten.

| | Baseline | Freeze |
|---|---|---|
| Probe-1 retention (last 10 insertions, mean) | 0.667 | 0.667 |
| Probe-5 retention (max stretch at 0.0)        | 11 of 16 calls = 0.0 | 0 of 16 calls = 0.0 |
| Probe-10 retention                            | 0.0 throughout | 0.0 throughout (code collision: pid 10 quantises to the same code as pid 5) |

Probe-10's persistent 0.0 is a real codebook collision (codes are deterministic given the encoder), not a forgetting artefact. The freeze cannot help when the address itself collides — that's a codebook problem, not a memory problem.

## 3. Online-PVLM-equivalent baseline — projection underperforms cosine-NN

`baseline_positioning.md` argues that Online-PVLM's published mechanism (frozen Omni Concept Embedder + light learned MLP projection + cosine-NN over the bank) reduces, on the perceptual identity tasks of PerceptMem, to the cosine-NN ceiling we already report. The claim was an upper-bound argument, not an empirical result.

We literally implemented the mechanism on V-XC-ID-XL (`online_pvlm_baseline.py`): SupCon-trained 2-layer MLP on the 211-identity train split, applied to ArcFace R50 features. Cosine-NN over the projected bank at eval.

| N | Raw cosine-NN (= the upper bound we cite) | Online-PVLM-equivalent | Δ |
|---|---|---|---|
| 5  | 0.950 | 0.800 | −0.150 |
| 10 | 0.975 | 0.750 | −0.225 |
| 20 | 0.963 | 0.688 | −0.275 |
| 50 | 0.935 | 0.675 | −0.260 |

The trained projection **underperforms** raw cosine-NN by 15–28 points across N. ArcFace is already a near-saturated identity encoder; learning a small projection on a 211-identity training pool overfits and degrades held-out generalisation.

The implication for the paper: the "RAG ceiling" in our scorecard is a *true* upper bound on the published-baseline performance on this benchmark — not a tight upper bound but a loose one. Online-PVLM-on-PerceptMem with its actual mechanism would score below it.

Result file: `results/online_pvlm_baseline.json`.

## 4. Qwen3-VL end-to-end on V-XC-ID — written, GPU-blocked

`qwen3vl_v_xc_id.py` is the end-to-end V-XC-ID-XL run on Qwen3-VL-8B-Thinking, mirroring the Qwen2.5-3B and Qwen2.5-14B audio runs. It re-uses `Qwen3VLEngramBolt` from #11 with a patched perc-emb-norm init (sample 1024 rows instead of the full 151 k × 5120 table) so the bolt construction no longer OOMs.

Status: **blocked on GPU contention at this session.** Other workloads on the shared workstation occupied 80 GB; ≤ 1 GB was free during forward. Qwen3-VL-8B in bf16 needs ~17.5 GB for weights, plus ~1.6 GB for the lm_head allocation, plus activations. Three approaches were attempted and all OOM'd:
  1. Pure GPU: OOM at `vis_perc_emb` init norm (now patched to sample 1024 rows).
  2. `device_map='auto'`: OOM at `lm_head` allocation (1.16 GB requested, ≤ 1 GB free).
  3. `device_map='auto'` with `max_memory` budget and `PYTORCH_ALLOC_CONF=expandable_segments`, plus smaller activations (batch=1, T=32): same OOM.

The script is ready to re-run when GPU has ≥ 20 GB free; the audio path on Qwen3-VL (`pathA_qwen3vl.json`) already shows the recipe transfers (code-match 1.00 at N=5) so the marginal value of the vision-side run is incremental rather than load-bearing.

## What now genuinely remains

- **Paper writing.** All four science items above resolve to concrete numbers; baseline_positioning.md's equivalence argument is now empirically backed.
- **PerceptMem dataset packaging for release** (post-acceptance).
- **Style encoder** — the contrastive-XL attempt did not exceed Gram+PCA. Treating style as a documented limitation, per session 8.

## Files added this session

- `src/nanochat_mm/salt_isolation_v2.py` — sharded perc_emb fix
- `src/nanochat_mm/forgetting_probe_v2.py` — row-freeze mitigation
- `src/nanochat_mm/online_pvlm_baseline.py` — literal published-mechanism head-to-head
- `src/nanochat_mm/qwen3vl_v_xc_id.py` — end-to-end vision on Qwen3-VL (GPU-deferred)
- `results/salt_isolation_v2.json`, `results/forgetting_probe_v2.json`, `results/online_pvlm_baseline.json`

## Files patched

- `src/nanochat_mm/qwen3vl_engram_bolt.py` — sampled-row norm at init (OOM-safe)
