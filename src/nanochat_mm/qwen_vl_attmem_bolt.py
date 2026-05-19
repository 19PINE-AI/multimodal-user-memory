"""AttMem bolt on Qwen2.5-VL (a vision-language model with native image input).

Differences from the text-only Qwen2.5-3B bolt:
  - Takes RAW IMAGES as input via Qwen-VL's processor, not pre-extracted ArcFace.
  - Uses Qwen-VL's OWN vision encoder + multimodal connector to produce vision tokens.
  - Mean-pools vision tokens to a single 2048-d "visual key" per image.
  - The bank stores (visual_key, marker-token-embedding) pairs.
  - The forward pre-hook on qwen.lm_head still works the same way.

Architecture: Qwen2.5-VL-3B-Instruct.
  - Vision encoder: 1280-d, then connector to 2048-d (matches LM hidden).
  - Text LM: 2048-d, 36 layers, vocab 151936, TIED embeddings.
  - Image token id: 151655.

This is the "real VLM end-to-end" version that processes real images natively.
"""
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from attention_memory import AttentionMemorySet, MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenVLAttMemBolt(nn.Module):
    """Pretrained Qwen2.5-VL + bolt-on AttentionMemory + a vision-feature pooler.

    Only the AttentionBank trainable params (~few hundred K) are learned;
    Qwen-VL is fully frozen. Bank key dim = LM hidden size (2048), because we
    use the model's own vision tokens directly as keys.
    """

    def __init__(self, qwen_vl, processor, attach_lm_head: bool = True):
        super().__init__()
        self.qwen_vl = qwen_vl
        self.processor = processor
        text_cfg = qwen_vl.config.get_text_config() if hasattr(qwen_vl.config, 'get_text_config') else qwen_vl.config.text_config
        self.hidden_size = text_cfg.hidden_size
        self.vlm_vocab = text_cfg.vocab_size
        self.attach_lm_head = attach_lm_head
        for p in self.qwen_vl.parameters():
            p.requires_grad_(False)

        # The bank key dim = LM hidden (since we pool Qwen-VL vision tokens
        # which are projected to LM hidden).
        self.attmem = AttentionMemorySet(
            hidden_size=self.hidden_size,
            vision_key_dim=self.hidden_size,
            audio_key_dim=192,  # not used for the VL pipeline; kept for API parity
            gpu_resident=True,
        )
        self.attmem.to(dtype=torch.bfloat16)

        self._hook_handle = None
        self._last_modality_ids = None
        self._last_perc_keys_by_mod = None

    # ---------- Hooks ----------

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

    def install_hook(self):
        target = self.qwen_vl.lm_head
        self._hook_handle = target.register_forward_pre_hook(
            self._attmem_hook, with_kwargs=True)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ---------- Vision-token key extraction ----------

    @torch.no_grad()
    def extract_visual_key(self, pil_image, prompt: str = "You see <|vision_start|><|image_pad|><|vision_end|>"):
        """Run Qwen-VL's vision encoder on the image, return a single pooled 2048-d key.

        Mean-pools the visual tokens produced by Qwen-VL for this image.
        """
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil_image},
                                              {"type": "text", "text": "describe"}]}]
        # Use processor.apply_chat_template to get the right multimodal input
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt",
                                  padding=True).to(DEVICE)
        # Forward partially: get the inputs_embeds where image patches are inserted
        with torch.no_grad():
            # Encode the image via the visual model
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            # Qwen2.5-VL uses model.visual to encode patches; the output is N_patches x 2048
            vision_tokens = self.qwen_vl.visual(pixel_values, grid_thw=image_grid_thw)
            # vision_tokens shape: [N_patches, hidden]
            key = vision_tokens.mean(dim=0)  # [hidden]
        return key

    # ---------- Bank insertion ----------

    def insert(self, modality_id: int, key: torch.Tensor, marker_token_id: int):
        bank = self.attmem.banks[str(modality_id)]
        with torch.no_grad():
            value = self.qwen_vl.get_input_embeddings()(
                torch.tensor([marker_token_id], device=DEVICE)
            )[0]
        bank.insert(key.to(DEVICE), value)

    def insert_batch(self, modality_id: int, keys: torch.Tensor, marker_token_ids):
        bank = self.attmem.banks[str(modality_id)]
        with torch.no_grad():
            ids = torch.tensor(list(marker_token_ids), device=DEVICE, dtype=torch.long)
            values = self.qwen_vl.get_input_embeddings()(ids)
        bank.insert_batch(keys.to(DEVICE), values)

    def reset_banks(self):
        self.attmem.reset()

    # ---------- Query path ----------

    @torch.no_grad()
    def query_logits(self, pil_image, prompt_text: str = "You see"):
        """Given a face image and a text prompt, run Qwen-VL forward with the
        bolt's hook active. Returns the next-token logits.

        The bank query is set via _last_modality_ids: we tag the perceptual
        position (a single token after the image) as MODALITY_VISION.
        """
        # Build the multimodal prompt: text + image, then trigger the bank query
        # at the "perceptual position" — which we instantiate via a learned
        # bank-query slot AFTER the image. The simplest approach: prefix with
        # the image, then add the prompt text. The LAST text token's hidden
        # state will be biased by our bank residual at lm_head.
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt_text}
        ]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt",
                                  padding=True).to(DEVICE)
        # Compute the visual key for this image (will be passed as perc_keys)
        vis_key = self.extract_visual_key(pil_image)

        # Set the bank query state: tag ALL positions as VISION so the bank
        # residual is applied at every position (simplest first version);
        # later we can refine to only inject at the last position.
        B, T = inputs["input_ids"].shape
        modality_ids = torch.zeros(B, T, dtype=torch.long, device=DEVICE)
        # Only inject at the very last position
        modality_ids[:, -1] = MODALITY_VISION
        self._last_modality_ids = modality_ids
        self._last_perc_keys_by_mod = {MODALITY_VISION: vis_key.unsqueeze(0)}

        out = self.qwen_vl(**inputs)
        logits = out.logits  # [B, T, V]
        return logits[0, -1, :].float()
