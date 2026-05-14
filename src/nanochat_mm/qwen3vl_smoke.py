"""Qwen3-VL smoke test — verify text-only inputs work and find hook layer.

Goal: load Qwen3VLForConditionalGeneration, pass text-only input via
inputs_embeds, get logits, confirm hidden_size and layer structure
match what we need for the Engram bolt-on.

If this works, Path A can be rerun with Qwen3-VL-8B-Thinking as base.
"""
import sys
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded; {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    print(f"  model class: {type(model).__name__}")

    # Explore structure
    print("\nModule structure:")
    for name, mod in model.named_children():
        print(f"  {name}: {type(mod).__name__}")
        if hasattr(mod, "layers"):
            print(f"    .layers: {len(mod.layers)}  ({type(mod.layers[0]).__name__})")
        for subname, sub in mod.named_children():
            print(f"    {subname}: {type(sub).__name__}")
            if hasattr(sub, "layers"):
                print(f"      .layers: {len(sub.layers)}")

    # Test text-only forward
    print("\nText-only forward ...")
    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    print(f"  logits shape: {out.logits.shape}")
    next_tok = int(out.logits[0, -1].argmax())
    print(f"  next token: {tok.decode([next_tok])!r}")

    # Test forward with inputs_embeds
    # Need to navigate to the embedding layer
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        emb_module = model.model.embed_tokens
    elif hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
        emb_module = model.language_model.embed_tokens
    else:
        emb_module = model.get_input_embeddings()
    print(f"  embedding module: {type(emb_module).__name__}")

    inputs_embeds = emb_module(inputs.input_ids)
    print(f"  inputs_embeds shape: {inputs_embeds.shape}")
    with torch.no_grad():
        out2 = model(inputs_embeds=inputs_embeds, attention_mask=inputs.attention_mask)
    diff = (out.logits - out2.logits).abs().max().item()
    print(f"  max diff vs input_ids path: {diff:.6f}")

    # Find the layer list for hook attachment
    print("\nLayer list discovery for Engram hook:")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        print(f"  found model.model.layers with {len(layers)} layers")
    elif hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            print(f"  found model.language_model.model.layers with {len(lm.model.layers)} layers")
        elif hasattr(lm, "layers"):
            print(f"  found model.language_model.layers with {len(lm.layers)} layers")

    if torch.cuda.is_available():
        print(f"\nGPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
