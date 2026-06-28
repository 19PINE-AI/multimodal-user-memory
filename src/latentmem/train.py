"""Train the write head: learn a latent memory M that substitutes for context.

Objective per step (frozen LM, only the write head learns):
  M           = write_head(encode(doc))
  student     = read_logits(M, probe)          # answer from M alone
  teacher     = teacher_logits(doc, probe)     # answer from full context (no grad)
  loss = alpha * KL(teacher || student)        # behavioral sufficiency (dense)
       + beta  * CE(student, gold_answer)      # task grounding

Run:
  python3 train.py --n_steps 3000 --k 16 --batch 8
The run waits for GPU headroom first; override with --need_gb / --no_wait.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import make_dataset, make_multiprobe_dataset
from model import LatentMemoryModel
from eval import evaluate
from gpu_wait import wait_for_gpu

RESULTS = Path(__file__).resolve().parents[2] / "results"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_steps", type=int, default=3000)
    ap.add_argument("--k", type=int, default=16, help="number of latent memory vectors")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="KL-distillation (reconstruction) weight -- load-bearing, "
                         "per the latent-bridge recipe (recon:task = 2:1)")
    ap.add_argument("--beta", type=float, default=1.0, help="task-CE weight")
    ap.add_argument("--capture_frac", type=float, default=0.67,
                    help="depth fraction for mid-layer residual capture")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="muon = orthogonalized-momentum on 2D matrices + AdamW for the rest")
    ap.add_argument("--muon_lr", type=float, default=0.02, help="Muon LR (2D matrices)")
    ap.add_argument("--recon_weight", type=float, default=0.0,
                    help="doc-reconstruction (ICAE-style sufficiency) aux-loss weight")
    ap.add_argument("--ce_warmup_frac", type=float, default=0.0,
                    help="curriculum: ramp task-CE in linearly over this fraction of steps")
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="LoRA rank on the LM read path (0 = fully frozen LM)")
    ap.add_argument("--probes_per_doc", type=int, default=1,
                    help=">1 trains recall with this many fact-probes per doc per step "
                         "(dense sufficiency signal); uses recall-style multi-probe data")
    ap.add_argument("--n_settings", type=int, default=16, help="doc length control")
    ap.add_argument("--n_relevant", type=int, default=3, help="integration depth")
    ap.add_argument("--recall_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--need_gb", type=float, default=30.0)
    ap.add_argument("--no_wait", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.no_wait:
        wait_for_gpu(args.need_gb)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentMemoryModel(args.model_id, k=args.k, capture_frac=args.capture_frac,
                              lora_rank=args.lora_rank, device=device)
    print(f"trainable (write head): {model.count_trainable()/1e6:.2f}M params; "
          f"frozen LM: {args.model_id}", flush=True)

    yes_id, no_id = model.answer_token_ids()
    lora_params = model.trainable_lm_params()
    if args.optimizer == "muon":
        from muon import Muon, split_params
        mp, ap_ = split_params(model.write_head)
        optimizers = [Muon(mp, lr=args.muon_lr, momentum=0.95, nesterov=True),
                      torch.optim.AdamW(ap_ + lora_params, lr=args.lr, weight_decay=0.0)]
        print(f"optimizer: Muon on {sum(p.numel() for p in mp)/1e6:.1f}M matrix params "
              f"+ AdamW on {(sum(p.numel() for p in ap_) + sum(p.numel() for p in lora_params))/1e6:.1f}M rest",
              flush=True)
    else:
        optimizers = [torch.optim.AdamW(list(model.write_head.parameters()) + lora_params,
                                        lr=args.lr, weight_decay=0.0)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.n_steps)
                  for o in optimizers]
    trainable = list(model.write_head.parameters()) + lora_params

    # Train/eval are disjoint persona draws (generalization to unseen users).
    train = make_dataset(args.n_steps * args.batch, seed=args.seed,
                         n_settings=args.n_settings, n_relevant=args.n_relevant,
                         recall_frac=args.recall_frac)
    train_mp = None
    if args.probes_per_doc > 1:
        train_mp = make_multiprobe_dataset(args.n_steps * args.batch, seed=args.seed,
                                           n_settings=args.n_settings,
                                           n_probes=args.probes_per_doc)
    # Eval is always single-probe recall (the clean per-fact sufficiency test).
    held = make_dataset(args.eval_n, seed=args.seed + 100000,
                        n_settings=args.n_settings, n_relevant=args.n_relevant,
                        recall_frac=args.recall_frac)

    tag = (f"latentmem_k{args.k}_set{args.n_settings}_rel{args.n_relevant}"
           f"_a{args.alpha}_b{args.beta}_{args.optimizer}_rw{args.recon_weight}"
           f"_lora{args.lora_rank}_rf{args.recall_frac}_st{args.n_steps}_seed{args.seed}")
    ckpt_path = RESULTS / f"{tag}.pt"
    log_path = RESULTS / f"{tag}.json"
    history = []
    best_mem = -1.0
    t0 = time.time()
    model.write_head.train()

    for step in range(args.n_steps):
        if train_mp is not None:
            batch_docs = train_mp[step * args.batch:(step + 1) * args.batch]
            docs_enc = [d for d, _ in batch_docs]
            docs_rep, probes, golds = [], [], []
            for d, plist in batch_docs:
                for pr, ans in plist:
                    docs_rep.append(d); probes.append(pr)
                    golds.append(yes_id if ans.strip() == "yes" else no_id)
            gold = torch.tensor(golds, device=device)
            teacher = model.teacher_logits(docs_rep, probes).float()
            M = model.encode_doc(docs_enc)                        # [B, k, H]
            M_read = torch.cat([M[i:i + 1].expand(len(batch_docs[i][1]), -1, -1)
                                for i in range(len(batch_docs))], dim=0)
            student = model.read_logits(M_read, probes).float()
        else:
            chunk = train[step * args.batch:(step + 1) * args.batch]
            docs_enc = [e.doc for e in chunk]
            probes = [e.probe for e in chunk]
            gold = torch.tensor([yes_id if e.answer.strip() == "yes" else no_id
                                 for e in chunk], device=device)
            teacher = model.teacher_logits(docs_enc, probes).float()  # no grad
            M = model.encode_doc(docs_enc)
            student = model.read_logits(M, probes).float()

        kl = F.kl_div(F.log_softmax(student, dim=-1),
                      F.softmax(teacher, dim=-1), reduction="batchmean")
        ce = F.cross_entropy(student, gold)
        ce_w = args.beta
        if args.ce_warmup_frac > 0:
            ce_w *= min(1.0, step / max(1, int(args.ce_warmup_frac * args.n_steps)))
        loss = args.alpha * kl + ce_w * ce
        recon = torch.zeros((), device=device)
        if args.recon_weight > 0:
            recon = model.recon_loss(M, docs_enc)
            loss = loss + args.recon_weight * recon

        for o in optimizers:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for o in optimizers:
            o.step()
        for s in schedulers:
            s.step()

        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  kl {kl.item():.4f}  "
                  f"ce {ce.item():.4f}  recon {float(recon):.3f}  "
                  f"lr {schedulers[0].get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]", flush=True)

        if step > 0 and (step % args.eval_every == 0 or step == args.n_steps - 1):
            res = evaluate(model, held, batch=16)
            res["step"] = step
            history.append(res)
            print(f"  [eval @ {step}] full={res['acc_full']:.3f}  "
                  f"mem={res['acc_mem']:.3f}  text={res['acc_text']:.3f}  "
                  f"none={res['acc_none']:.3f}  suff_kl={res['suff_kl']:.3f}", flush=True)
            if res["acc_mem"] > best_mem:
                best_mem = res["acc_mem"]
                save = {"write_head": model.write_head.state_dict(),
                        "k": args.k, "args": vars(args), "step": step}
                if model.has_lora:
                    from peft import get_peft_model_state_dict
                    save["lora"] = get_peft_model_state_dict(model.lm)
                torch.save(save, ckpt_path)
            log_path.write_text(json.dumps(
                {"args": vars(args), "trainable_M": model.count_trainable(),
                 "history": history, "best_acc_mem": best_mem}, indent=2))

    print(f"\nDONE. best mem-only acc = {best_mem:.3f}. ckpt: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
