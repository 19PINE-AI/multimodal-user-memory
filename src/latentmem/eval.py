"""Evaluation for the latent-memory pilot.

The decisive question is the *sufficiency gap*: does answering from the learned
memory M alone match answering from the full document? We report four points on
the same probes:

  full   : [doc ; probe]            — oracle upper bound
  mem    : [M ; probe]              — OURS (k learned latent vectors)
  text   : [trunc(doc, k) ; probe]  — matched-budget text baseline (truncation)
  none   : [probe]                  — no-memory lower bound (label prior)

`mem` beating `text` at equal budget, and approaching `full`, is the win.
"""
from __future__ import annotations

from typing import List

import torch

from data import Example
from model import LatentMemoryModel


def _two_way_acc(logits: torch.Tensor, gold_yes: torch.Tensor,
                 yes_id: int, no_id: int) -> torch.Tensor:
    """Restrict the decision to {yes,no}; return per-example correctness (float)."""
    pred_yes = logits[:, yes_id] > logits[:, no_id]
    return (pred_yes == gold_yes).float()


@torch.no_grad()
def evaluate(model: LatentMemoryModel, examples: List[Example], batch: int = 16,
             budget_tokens: int | None = None) -> dict:
    model.write_head.eval()
    yes_id, no_id = model.answer_token_ids()
    budget = budget_tokens if budget_tokens is not None else model.k

    agg = {k: [] for k in ("full", "mem", "text", "none")}
    agg_kind = {}            # kind -> {"mem": [...], "full": [...]}
    suff_kl = []             # KL(teacher || student) at the answer position

    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        docs = [e.doc for e in chunk]
        probes = [e.probe for e in chunk]
        gold_yes = torch.tensor([e.answer.strip() == "yes" for e in chunk],
                                device=model.device)

        l_full = model.teacher_logits(docs, probes)
        M = model.encode_doc(docs)
        l_mem = model.read_logits(M, probes)
        l_text = model.text_baseline_logits(docs, probes, budget)
        l_none = model.text_baseline_logits(docs, probes, 0)

        for name, lg in (("full", l_full), ("mem", l_mem), ("text", l_text), ("none", l_none)):
            agg[name].append(_two_way_acc(lg, gold_yes, yes_id, no_id))

        p_t = torch.softmax(l_full.float(), dim=-1)
        logp_s = torch.log_softmax(l_mem.float(), dim=-1)
        suff_kl.append((p_t * (p_t.clamp_min(1e-9).log() - logp_s)).sum(-1))

        for e, fm, mm in zip(chunk, agg["full"][-1].tolist(),
                             _two_way_acc(l_mem, gold_yes, yes_id, no_id).tolist()):
            d = agg_kind.setdefault(e.kind, {"mem": [], "full": []})
            d["mem"].append(mm); d["full"].append(fm)

    def mean(xs):
        return float(torch.cat(xs).mean()) if xs else float("nan")

    res = {f"acc_{k}": mean(v) for k, v in agg.items()}
    res["suff_kl"] = float(torch.cat(suff_kl).mean())
    res["budget_tokens"] = budget
    res["n"] = len(examples)
    res["by_kind"] = {k: {"acc_mem": sum(v["mem"]) / len(v["mem"]),
                          "acc_full": sum(v["full"]) / len(v["full"]),
                          "n": len(v["mem"])}
                      for k, v in agg_kind.items()}
    model.write_head.train()
    return res


if __name__ == "__main__":
    # Standalone eval of a saved checkpoint.
    import argparse, json
    from data import make_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_settings", type=int, default=16)
    ap.add_argument("--n_relevant", type=int, default=3)
    ap.add_argument("--recall_frac", type=float, default=0.5)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    model = LatentMemoryModel(args.model_id, k=sd["k"])
    model.write_head.load_state_dict(sd["write_head"])
    ds = make_dataset(args.n, seed=9999, n_settings=args.n_settings,
                      n_relevant=args.n_relevant, recall_frac=args.recall_frac)
    print(json.dumps(evaluate(model, ds), indent=2))
