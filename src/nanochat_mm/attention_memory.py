"""Continuous attention memory — the post-pivot primitive.

Replaces the discrete codebook + hash-keyed Engram with cross-attention
over a learned-projection of the LM hidden state over a growing
(key=encoder_embedding, value=marker_embedding) bank.

Properties:
  - Insertion: O(1) wall-clock append. No SGD step.
  - Query:     O(N·D) matmul — microseconds even at N=10k.
  - Storage:   O(D + H) per identity in the bank arrays.
  - Trainable: W_q (H -> D), W_o (H -> H), tau scalar. ~200k params total.

The bank is per-modality (vision and audio have separate banks). At the
attached LM layer, the hook computes per-modality cross-attention at
positions tagged as that modality, and sums the residuals.
"""
import math
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modality ids match engram_module_mm.py for interop with existing pipeline
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2
MODALITY_NAMES = {MODALITY_TEXT: "text", MODALITY_VISION: "vision", MODALITY_AUDIO: "audio"}


class AttentionBank(nn.Module):
    """One modality's memory: a growing (key, value) array + a learned
    projection from LM hidden state to D-dim query.

    Bank storage is on CPU as numpy arrays (cheap, append-able); when
    a query lands we move the relevant slice to GPU for the matmul.
    For small N (<= 10k) the bank also fits cleanly on GPU; we keep an
    optional GPU mirror that's refreshed on insert.
    """

    def __init__(self, hidden_size: int, key_dim: int, init_capacity: int = 1024,
                  gpu_resident: bool = True):
        super().__init__()
        self.H = hidden_size
        self.D = key_dim
        self.gpu_resident = gpu_resident

        # Trainable parameters: small projection and inverse-temperature
        self.W_q = nn.Linear(hidden_size, key_dim, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        # log_inv_temp init: with L2-normalised keys, q·k ∈ [-1, 1] (cosine).
        # We use logits = (q · k) * inv_temp directly (NO sqrt(D) divisor —
        # that's a vestige of unnormalised-key attention and would shrink
        # the effective scale for higher-D encoders). Init to log(20) ≈ 3.0
        # so inv_temp=20: typical same-ID cosine 0.6 → logit 12, diff-ID
        # cosine 0.2 → logit 4, gap 8 → softmax weight ratio e^8 — sharp.
        self.log_inv_temp = nn.Parameter(torch.tensor(math.log(20.0)))
        # Output gain — multiplies the retrieved residual before adding it to
        # the LM hidden state. Init large so the marker-logit boost dominates
        # the LM's natural (very negative) logit for unusual marker tokens.
        self.out_gain = nn.Parameter(torch.tensor(8.0))

        # Init W_o close to identity so the retrieved (marker input
        # embedding) flows directly into the hidden state — leveraging the
        # LM's existing skip-connection structure. With tied embeddings the
        # dot product `lm_head[marker] · W_o(retrieved)` is the natural
        # logit boost for the marker token.
        with torch.no_grad():
            eye = torch.eye(hidden_size)  # identity (out_gain scales it)
            self.W_o.weight.copy_(eye)

        # Bank storage — buffers, not parameters
        self.register_buffer("keys", torch.zeros(0, key_dim), persistent=False)
        self.register_buffer("values", torch.zeros(0, hidden_size), persistent=False)
        self._size = 0
        self._init_capacity = init_capacity

    @property
    def size(self) -> int:
        return self._size

    def reset(self):
        """Empty the bank (for fresh evaluation runs)."""
        device = self.keys.device
        self.keys = torch.zeros(0, self.D, device=device)
        self.values = torch.zeros(0, self.H, device=device)
        self._size = 0

    def insert(self, key: torch.Tensor, value: torch.Tensor):
        """Append one (key, value) row. key: [D]; value: [H]. O(1) amortised."""
        with torch.no_grad():
            # Ensure shape
            if key.dim() == 2: key = key.squeeze(0)
            if value.dim() == 2: value = value.squeeze(0)
            assert key.shape == (self.D,), f"key shape {key.shape}, expected ({self.D},)"
            assert value.shape == (self.H,), f"value shape {value.shape}, expected ({self.H},)"
            # L2-normalise the key (encoder embeddings are typically unit-norm)
            key = F.normalize(key.float(), dim=-1)
            # Append
            self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
            self.values = torch.cat([self.values, value.unsqueeze(0).to(self.values.dtype)], dim=0)
            self._size += 1

    def insert_batch(self, keys: torch.Tensor, values: torch.Tensor):
        """Append a batch of rows. keys: [N, D]; values: [N, H]."""
        with torch.no_grad():
            keys = F.normalize(keys.float(), dim=-1)
            self.keys = torch.cat([self.keys, keys], dim=0)
            self.values = torch.cat([self.values, values.to(self.values.dtype)], dim=0)
            self._size += keys.shape[0]

    def query(self, hidden_states: torch.Tensor, perceptual_mask: torch.Tensor,
              perceptual_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the residual to inject into hidden_states.

        Args:
          hidden_states: [B, T, H] — LM hidden state at attached layer.
          perceptual_mask: [B, T] — bool, True at positions of this modality.
          perceptual_keys: optional [N_perc_positions, D] — pre-computed
            encoder embeddings for the perceptual positions in the batch.
            If None, fall back to projecting the hidden state (slower but
            self-contained — used at inference when only LM hidden is available).

        Returns:
          residual: [B, T, H] — zeros outside `perceptual_mask`, attention
          output at perceptual positions.
        """
        B, T, H = hidden_states.shape
        device = hidden_states.device
        residual = torch.zeros_like(hidden_states)

        if self._size == 0:
            return residual

        # Get the perceptual positions (where the bank is queried)
        idx = perceptual_mask.nonzero(as_tuple=False)  # [N_perc, 2] (batch, pos)
        if idx.shape[0] == 0:
            return residual

        h_at_perc = hidden_states[idx[:, 0], idx[:, 1]]  # [N_perc, H]

        # Query construction: either provided perceptual_keys (encoder embedding
        # of the actual perceptual input — used at insertion+eval time) or
        # learned projection of the hidden state (used during pretraining).
        if perceptual_keys is not None:
            q = perceptual_keys.to(device=device, dtype=self.keys.dtype)
            q = F.normalize(q, dim=-1)
        else:
            q = self.W_q(h_at_perc.float())
            q = F.normalize(q, dim=-1)

        # Cross-attention over the bank
        # keys: [N_bank, D]; q: [N_perc, D]; logits: [N_perc, N_bank]
        keys = self.keys.to(device=device, dtype=q.dtype)
        values = self.values.to(device=device, dtype=torch.float32)
        inv_temp = torch.exp(self.log_inv_temp).clamp_max(500.0)
        logits = (q @ keys.T) * inv_temp  # keys already L2-normalised
        weights = F.softmax(logits, dim=-1).to(torch.float32)  # [N_perc, N_bank]
        retrieved = weights @ values  # [N_perc, H]
        # Apply output projection + learnable gain
        retrieved = self.W_o(retrieved.to(self.W_o.weight.dtype))
        retrieved = retrieved * self.out_gain.to(retrieved.dtype)
        retrieved = retrieved.to(hidden_states.dtype)

        # Scatter back
        residual[idx[:, 0], idx[:, 1]] = retrieved
        return residual


class AttentionMemorySet(nn.Module):
    """Multi-modal wrapper holding one AttentionBank per modality.

    Mirrors the role of MultimodalEngramSet but with attention banks
    instead of hashed tables.
    """

    def __init__(self, hidden_size: int, vision_key_dim: int = 512,
                  audio_key_dim: int = 192, gpu_resident: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.banks = nn.ModuleDict({
            str(MODALITY_VISION): AttentionBank(hidden_size, vision_key_dim, gpu_resident=gpu_resident),
            str(MODALITY_AUDIO):  AttentionBank(hidden_size, audio_key_dim, gpu_resident=gpu_resident),
        })

    def reset(self):
        for b in self.banks.values():
            b.reset()

    def forward_layer(self, hidden_states: torch.Tensor, modality_ids: torch.Tensor,
                       perceptual_keys_by_mod: Optional[Dict[int, torch.Tensor]] = None,
                       ) -> torch.Tensor:
        """Sum per-modality residuals.

        Args:
          hidden_states: [B, T, H]
          modality_ids:  [B, T] integers ∈ {TEXT, VISION, AUDIO}
          perceptual_keys_by_mod: dict mapping modality_id -> [N_perc_for_that_modality, D_mod]
            of pre-computed encoder embeddings (ordered to match the
            ROW-MAJOR enumeration of perceptual positions of that modality
            in the [B, T] grid).

        Returns:
          [B, T, H] residual.
        """
        residual = torch.zeros_like(hidden_states)
        for mid_str, bank in self.banks.items():
            mid = int(mid_str)
            mask = (modality_ids == mid)
            if not mask.any():
                continue
            perc_keys = None
            if perceptual_keys_by_mod is not None and mid in perceptual_keys_by_mod:
                perc_keys = perceptual_keys_by_mod[mid]
            residual = residual + bank.query(hidden_states, mask, perceptual_keys=perc_keys)
        return residual

    def total_params(self) -> Dict[str, int]:
        return {MODALITY_NAMES[int(k)]: sum(p.numel() for p in b.parameters())
                for k, b in self.banks.items()}

    def bank_sizes(self) -> Dict[str, int]:
        return {MODALITY_NAMES[int(k)]: b.size for k, b in self.banks.items()}
