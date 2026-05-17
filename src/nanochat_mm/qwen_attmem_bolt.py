"""Frozen Qwen + AttentionMemory bolt-on.

Drop-in replacement for QwenEngramBolt with a continuous-attention memory
in place of the discrete codebook + hash-keyed Engram.

Key differences vs QwenEngramBolt:
  - At perceptual positions, the LM input embedding is computed from the
    raw encoder embedding (via a small learned projection), not from a
    discrete code lookup.
  - At the attached layer, the hook does cross-attention over the
    AttentionBank instead of a hashed table lookup.
  - Insertion is `bank.insert(key, value)` — no SGD.
"""
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import (
    AttentionMemorySet, MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class QwenAttMemBolt(nn.Module):
    """Pretrained Qwen + bolt-on AttentionMemory + per-modality perceptual-embedding
    projection. Only the projection and the AttentionBank trainable params are
    learned; Qwen is frozen.
    """

    def __init__(self, qwen_model, qwen_tokenizer,
                  vision_key_dim: int = 512, audio_key_dim: int = 192,
                  attach_layer: int = 24, attach_lm_head: bool = True):
        super().__init__()
        self.qwen = qwen_model
        self.tok = qwen_tokenizer
        self.hidden_size = qwen_model.config.hidden_size
        self.qwen_vocab = qwen_model.config.vocab_size
        self.attach_lm_head = attach_lm_head
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        # Per-modality projection from raw encoder embedding to LM hidden dim.
        # This is what produces the LM-input-embedding for perceptual positions.
        # Init small (close to zero) so the LM's frozen behavior at perc positions
        # remains roughly textual padding until the projection learns.
        self.vis_proj = nn.Linear(vision_key_dim, self.hidden_size, bias=False)
        self.aud_proj = nn.Linear(audio_key_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            ref = qwen_model.get_input_embeddings().weight.detach()
            ref_norm = ref.norm(dim=-1).mean().item()
            for m in [self.vis_proj, self.aud_proj]:
                nn.init.normal_(m.weight, std=ref_norm / math.sqrt(m.weight.shape[1]))

        # The attention memory itself (per-modality banks + per-bank trainables)
        self.attmem = AttentionMemorySet(
            hidden_size=self.hidden_size,
            vision_key_dim=vision_key_dim,
            audio_key_dim=audio_key_dim,
            gpu_resident=True,
        )

        # Cast trainables to bf16 to match Qwen
        for m in [self.vis_proj, self.aud_proj, self.attmem]:
            m.to(dtype=torch.bfloat16)
        self.attach_layer = attach_layer

        # State carried into the hook
        self._hook_handle = None
        self._last_modality_ids = None
        self._last_perc_keys_by_mod = None  # Dict[int, Tensor[N_perc, D]]

    # ---------- Hook into the LM layer ----------

    def _attmem_hook(self, module, args, kwargs):
        if not args:
            return None
        hidden_states = args[0]
        if self._last_modality_ids is None:
            return None
        residual = self.attmem.forward_layer(
            hidden_states, self._last_modality_ids,
            perceptual_keys_by_mod=self._last_perc_keys_by_mod,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
        return (hidden_states,) + args[1:], kwargs

    def _attmem_lm_head_hook(self, module, args, kwargs):
        # Pre-hook on qwen.lm_head: input is post-norm hidden state [B, T, H].
        # Adding the retrieved (marker-input-embedding-shaped) residual here
        # directly biases the corresponding logit via tied/aligned embeddings.
        if not args:
            return None
        hidden_states = args[0]
        if self._last_modality_ids is None:
            return None
        residual = self.attmem.forward_layer(
            hidden_states, self._last_modality_ids,
            perceptual_keys_by_mod=self._last_perc_keys_by_mod,
        )
        if residual is not None:
            hidden_states = hidden_states + residual
        return (hidden_states,) + args[1:], kwargs

    def install_hook(self):
        if self.attach_lm_head:
            target = self.qwen.lm_head
            self._hook_handle = target.register_forward_pre_hook(
                self._attmem_lm_head_hook, with_kwargs=True)
        else:
            layer = self.qwen.model.layers[self.attach_layer]
            self._hook_handle = layer.register_forward_pre_hook(
                self._attmem_hook, with_kwargs=True)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ---------- Forward composition ----------

    def build_inputs_embeds_from_perc_keys(
        self, modality_ids: torch.Tensor, text_input_ids: torch.Tensor,
        perc_keys_by_mod: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compose inputs_embeds. text_input_ids[i,j] only meaningful where
        modality_ids[i,j]==TEXT; perc_keys_by_mod gives the raw encoder
        embeddings for perceptual positions of each modality, in row-major
        order.
        """
        B, T = modality_ids.shape
        device = modality_ids.device
        emb = torch.zeros(B, T, self.hidden_size, device=device, dtype=torch.bfloat16)

        # Text positions
        m_text = (modality_ids == MODALITY_TEXT)
        if m_text.any():
            text_ids = torch.where(m_text, text_input_ids, torch.zeros_like(text_input_ids))
            text_emb = self.qwen.get_input_embeddings()(text_ids)
            emb = emb + m_text.unsqueeze(-1).to(emb.dtype) * text_emb

        # Vision positions
        m_vis = (modality_ids == MODALITY_VISION)
        if m_vis.any() and MODALITY_VISION in perc_keys_by_mod:
            vis_keys = perc_keys_by_mod[MODALITY_VISION]  # [N_perc_vis, D_vis]
            vis_emb_perc = self.vis_proj(vis_keys.to(dtype=torch.bfloat16))  # [N_perc_vis, H]
            # Scatter back into the [B, T, H] grid (row-major over m_vis True positions)
            emb_flat = emb.reshape(B * T, self.hidden_size)
            idx = m_vis.reshape(B * T).nonzero(as_tuple=False).squeeze(-1)
            emb_flat[idx] = emb_flat[idx] + vis_emb_perc
            emb = emb_flat.view(B, T, self.hidden_size)

        # Audio positions
        m_aud = (modality_ids == MODALITY_AUDIO)
        if m_aud.any() and MODALITY_AUDIO in perc_keys_by_mod:
            aud_keys = perc_keys_by_mod[MODALITY_AUDIO]
            aud_emb_perc = self.aud_proj(aud_keys.to(dtype=torch.bfloat16))
            emb_flat = emb.reshape(B * T, self.hidden_size)
            idx = m_aud.reshape(B * T).nonzero(as_tuple=False).squeeze(-1)
            emb_flat[idx] = emb_flat[idx] + aud_emb_perc
            emb = emb_flat.view(B, T, self.hidden_size)

        return emb

    def forward(self, modality_ids: torch.Tensor, text_input_ids: torch.Tensor,
                 perc_keys_by_mod: Dict[int, torch.Tensor]):
        """Forward with perceptual keys (encoder embeddings) supplied externally.

        Args:
          modality_ids:    [B, T] int — TEXT/VISION/AUDIO per position
          text_input_ids:  [B, T] int — Qwen vocab ids at text positions; ignored at perc positions
          perc_keys_by_mod: dict mapping modality id -> [N_perc_for_mod, D_mod] encoder embeddings.
            Order must be row-major over the True entries of (modality_ids == mod) across the
            [B, T] grid.

        Returns:
          logits: [B, T, vocab]
        """
        self._last_modality_ids = modality_ids
        self._last_perc_keys_by_mod = perc_keys_by_mod
        emb = self.build_inputs_embeds_from_perc_keys(modality_ids, text_input_ids, perc_keys_by_mod)
        attn_mask = torch.ones_like(modality_ids)
        out = self.qwen(inputs_embeds=emb, attention_mask=attn_mask, use_cache=False)
        return out.logits

    # ---------- Insertion / eval helpers ----------

    def insert(self, modality_id: int, key: torch.Tensor, marker_token_id: int):
        """Insert one identity. key: [D_mod] encoder embedding; marker = a token id.

        The value stored in the bank is the LM's input embedding of the marker.
        Insertion is O(1) — just two appends.
        """
        bank = self.attmem.banks[str(modality_id)]
        with torch.no_grad():
            value = self.qwen.get_input_embeddings()(
                torch.tensor([marker_token_id], device=DEVICE)
            )[0]  # [H]
        bank.insert(key.to(DEVICE), value)

    def insert_batch(self, modality_id: int, keys: torch.Tensor, marker_token_ids):
        bank = self.attmem.banks[str(modality_id)]
        with torch.no_grad():
            ids = torch.tensor(list(marker_token_ids), device=DEVICE, dtype=torch.long)
            values = self.qwen.get_input_embeddings()(ids)  # [N, H]
        bank.insert_batch(keys.to(DEVICE), values)

    def reset_banks(self):
        self.attmem.reset()
