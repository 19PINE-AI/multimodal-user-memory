"""Multimodal Engram module — Path A: parallel per-modality tables.

Extends the text-only Engram from `engram_module.py` to handle interleaved
multimodal token streams. Each input position is tagged with a modality
id (0=text, 1=vision, 2=audio); positions are dispatched to the
corresponding per-modality Engram and residuals are summed.

Key architectural choices:
  - Each modality has its own NgramHashMapping (different prime seeds via
    distinct `seed` offsets) so address spaces are naturally disjoint.
  - Each modality has its own embedding tables per attached layer.
  - The gate / conv / norm machinery is per-modality. (Simpler than a
    shared gate that takes modality as input; lets each modality
    converge at its own rate.)
  - Hash inputs are PER-MODALITY: text uses raw text BPE ids; vision
    uses image VQ codes (Cosmos-Tokenizer); audio uses ECAPA/Encodec
    codes. Each is mapped through its own canonical-collapse table
    (identity for vision/audio since their vocab is already canonical).
  - Positions outside a given modality are masked to the pad id of that
    modality, producing zero-contribution hashes (the pad row stays
    near zero throughout training).

Per-user salt support carries over directly from the base module via
the `user_salt` attribute on each EngramSet.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Import the v1 building blocks
from engram_module import (
    EngramSet,
    EngramSubConfig,
    NgramHashMapping,
    MultiHeadEmbedding,
    EngramLayer,
)


# Modality id constants — must agree with the tokeniser / dataloader.
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2
MODALITY_NAMES = {MODALITY_TEXT: "text", MODALITY_VISION: "vision", MODALITY_AUDIO: "audio"}


@dataclass
class MultimodalEngramConfig:
    """Wraps three EngramSubConfigs, one per modality.

    Per-modality vocab sizes:
      - text_vocab: e.g. 32768 (nanochat RustBPE)
      - vision_vocab: number of distinct image VQ codes (Cosmos: 64k typically)
      - audio_vocab: number of distinct audio VQ codes (Encodec residual codes
        or quantised ECAPA codes; ~1024-4096 typical for our use)

    Pad ids for each modality (used to mask out-of-modality positions).
    """
    layer_ids: List[int] = field(default_factory=lambda: [3, 6])
    # Per-modality sub-configs; defaults below are reasonable for prototyping
    text_cfg: Dict = field(default_factory=lambda: dict(
        max_ngram_size=3, engram_vocab_per_ngram=100_000,
        n_head_per_ngram=8, n_embed_per_ngram=256, kernel_size=4, pad_id=0, seed=0,
    ))
    vision_cfg: Dict = field(default_factory=lambda: dict(
        max_ngram_size=3, engram_vocab_per_ngram=50_000,
        n_head_per_ngram=4, n_embed_per_ngram=128, kernel_size=4, pad_id=0, seed=1001,
    ))
    audio_cfg: Dict = field(default_factory=lambda: dict(
        max_ngram_size=3, engram_vocab_per_ngram=20_000,
        n_head_per_ngram=4, n_embed_per_ngram=128, kernel_size=4, pad_id=0, seed=2003,
    ))
    text_vocab_size: int = 32768
    vision_vocab_size: int = 65536
    audio_vocab_size: int = 4096


class MultimodalEngramSet(nn.Module):
    """Three parallel EngramSets (text/vision/audio), dispatched per position.

    The forward_layer call takes a modality-tagged token batch and returns
    the sum of per-modality residuals.
    """
    def __init__(self, mm_cfg: MultimodalEngramConfig, hidden_size: int):
        super().__init__()
        self.mm_cfg = mm_cfg
        self.hidden_size = hidden_size

        # Per-modality EngramSubConfigs
        sub_cfgs = {
            MODALITY_TEXT: EngramSubConfig(layer_ids=list(mm_cfg.layer_ids), **mm_cfg.text_cfg),
            MODALITY_VISION: EngramSubConfig(layer_ids=list(mm_cfg.layer_ids), **mm_cfg.vision_cfg),
            MODALITY_AUDIO: EngramSubConfig(layer_ids=list(mm_cfg.layer_ids), **mm_cfg.audio_cfg),
        }
        vocab_sizes = {
            MODALITY_TEXT: mm_cfg.text_vocab_size,
            MODALITY_VISION: mm_cfg.vision_vocab_size,
            MODALITY_AUDIO: mm_cfg.audio_vocab_size,
        }
        # Identity canonical-collapse maps for vision/audio (the codes are already canonical);
        # text would normally use the NFKC-derived map but we use identity here for prototyping
        # — the real run would pass a real canonical_map from `build_canonical_collapse_map`.
        self.engrams = nn.ModuleDict()
        for mid, sub_cfg in sub_cfgs.items():
            v = vocab_sizes[mid]
            canonical_map = torch.arange(v, dtype=torch.long)
            self.engrams[str(mid)] = EngramSet(
                cfg=sub_cfg,
                hidden_size=hidden_size,
                compressed_vocab_size=v,
                compressed_pad_id=sub_cfg.pad_id,
                token_canonical_map=canonical_map,
            )

    def set_user_salt(self, salt: int):
        for m in self.engrams.values():
            m.user_salt = salt

    def reset_cache(self):
        for m in self.engrams.values():
            m.reset_cache()

    def total_params(self) -> Dict[str, int]:
        return {MODALITY_NAMES[int(k)]: sum(p.numel() for p in v.parameters()) for k, v in self.engrams.items()}

    def total_table_params(self) -> Dict[str, int]:
        return {MODALITY_NAMES[int(k)]: v.total_table_params() for k, v in self.engrams.items()}

    def forward_layer(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Dispatch per-modality.

        Args:
          hidden_states: [B, T, hidden_size] — shared transformer state
          input_ids:     [B, T]              — token id within its modality's vocab
          modality_ids:  [B, T]              — 0/1/2 tag per position
          layer_idx:     int                 — current transformer block id

        Returns: [B, T, hidden_size] residual (sum of per-modality residuals)
                 or None if no Engram is attached to this layer.
        """
        # Any modality attached at this layer?
        any_attached = any(
            layer_idx in self.engrams[str(mid)].cfg.layer_ids
            for mid in [MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO]
        )
        if not any_attached:
            return None

        device = hidden_states.device
        total_residual = torch.zeros_like(hidden_states)

        for mid in [MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO]:
            eng = self.engrams[str(mid)]
            if layer_idx not in eng.cfg.layer_ids:
                continue
            # Mask: 1 where this position is of this modality, else 0
            mask = (modality_ids == mid)
            if not mask.any():
                continue
            # Build per-modality input_ids: positions of other modalities → pad
            mod_input_ids = torch.where(
                mask, input_ids, torch.full_like(input_ids, eng.cfg.pad_id)
            )
            # Run this modality's Engram on the modality-masked stream
            residual = eng.forward_layer(hidden_states, mod_input_ids, layer_idx)
            if residual is not None:
                # Zero out positions that aren't this modality
                m = mask.unsqueeze(-1).to(residual.dtype)
                total_residual = total_residual + residual * m

        return total_residual
