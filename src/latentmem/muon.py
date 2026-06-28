"""Single-device Muon optimizer (Newton-Schulz orthogonalized momentum).

Muon (Jordan et al.) updates 2D weight matrices by orthogonalizing the
momentum via a quintic Newton-Schulz iteration, which empirically beats AdamW
in sample efficiency on hidden weight matrices. It is *only* for 2D matrices:
embeddings, biases, norm gains, and scalars must go to AdamW. In this project
the write head is almost all 2D matrices (attention in/out projections, FFN
linears), so most of it can ride Muon, with the learned query tokens, scales,
and LayerNorms on AdamW.

This is the standard reference recipe specialized to one device (no distributed
all-gather). Typical hyperparameters: lr ~0.02, momentum 0.95, nesterov, 5 NS
steps.
"""
import torch


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal polar factor
    of G (i.e. U V^T from G = U S V^T). Coefficients tuned so the iteration
    pushes all singular values toward 1."""
    assert G.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.t()
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            wd, ns = group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError("Muon got a non-2D param; route it to AdamW.")
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = _newton_schulz(upd, steps=ns)
                # Shape-aware scale keeps the update RMS comparable across matrices.
                upd = upd * (max(1.0, p.size(0) / p.size(1)) ** 0.5)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


def split_params(module):
    """Partition a module's params into (muon 2D matrices, adamw rest).

    Query tokens act like input embeddings, so they go to AdamW even though
    they are 2D; everything 1D/scalar (biases, norms, scales) also goes to AdamW.
    """
    muon, adamw = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "queries" not in name:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
