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

from data import make_dataset
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
    ap.add_argument("--alpha", type=float, default=1.0, help="KL-distillation weight")
    ap.add_argument("--beta", type=float, default=1.0, help="task-CE weight")
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
    model = LatentMemoryModel(args.model_id, k=args.k, device=device)
    print(f"trainable (write head): {model.count_trainable()/1e6:.2f}M params; "
          f"frozen LM: {args.model_id}", flush=True)

    yes_id, no_id = model.answer_token_ids()
    opt = torch.optim.AdamW(model.write_head.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_steps)

    # Train/eval are disjoint persona draws (generalization to unseen users).
    train = make_dataset(args.n_steps * args.batch, seed=args.seed,
                         n_settings=args.n_settings, n_relevant=args.n_relevant,
                         recall_frac=args.recall_frac)
    held = make_dataset(args.eval_n, seed=args.seed + 100000,
                        n_settings=args.n_settings, n_relevant=args.n_relevant,
                        recall_frac=args.recall_frac)

    tag = (f"latentmem_k{args.k}_set{args.n_settings}_rel{args.n_relevant}"
           f"_a{args.alpha}_b{args.beta}_seed{args.seed}")
    ckpt_path = RESULTS / f"{tag}.pt"
    log_path = RESULTS / f"{tag}.json"
    history = []
    best_mem = -1.0
    t0 = time.time()
    model.write_head.train()

    for step in range(args.n_steps):
        chunk = train[step * args.batch:(step + 1) * args.batch]
        docs = [e.doc for e in chunk]
        probes = [e.probe for e in chunk]
        gold = torch.tensor([yes_id if e.answer.strip() == "yes" else no_id
                             for e in chunk], device=device)

        teacher = model.teacher_logits(docs, probes).float()      # no grad
        M = model.encode_doc(docs)
        student = model.read_logits(M, probes).float()

        kl = F.kl_div(F.log_softmax(student, dim=-1),
                      F.softmax(teacher, dim=-1), reduction="batchmean")
        ce = F.cross_entropy(student, gold)
        loss = args.alpha * kl + args.beta * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.write_head.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  kl {kl.item():.4f}  "
                  f"ce {ce.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        if step > 0 and (step % args.eval_every == 0 or step == args.n_steps - 1):
            res = evaluate(model, held, batch=16)
            res["step"] = step
            history.append(res)
            print(f"  [eval @ {step}] full={res['acc_full']:.3f}  "
                  f"mem={res['acc_mem']:.3f}  text={res['acc_text']:.3f}  "
                  f"none={res['acc_none']:.3f}  suff_kl={res['suff_kl']:.3f}", flush=True)
            if res["acc_mem"] > best_mem:
                best_mem = res["acc_mem"]
                torch.save({"write_head": model.write_head.state_dict(),
                            "k": args.k, "args": vars(args), "step": step}, ckpt_path)
            log_path.write_text(json.dumps(
                {"args": vars(args), "trainable_M": model.count_trainable(),
                 "history": history, "best_acc_mem": best_mem}, indent=2))

    print(f"\nDONE. best mem-only acc = {best_mem:.3f}. ckpt: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
