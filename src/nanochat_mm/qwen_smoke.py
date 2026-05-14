"""Smoke test: load Qwen3-VL-8B-Thinking, confirm forward + hidden state access.

Before building the Engram bolt-on, verify the basics:
  1. Model loads in bf16 on single GPU.
  2. We can feed text via Qwen's tokenizer and get logits.
  3. We can extract hidden states at intermediate layers via output_hidden_states.
  4. We can also pass `inputs_embeds` to bypass token-id input.

Memory budget: 8B params × 2 bytes (bf16) = 16GB. Blackwell has 102GB.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  vocab_size = {tok.vocab_size}")

    print(f"\nLoading {MODEL_ID} in bf16 on {DEVICE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    print(f"  loaded; param count = {sum(p.numel() for p in model.parameters()):,}")
    print(f"  model class: {type(model).__name__}")
    # Try to access the language model component (Qwen3VL has a vision + text part)
    if hasattr(model, 'model'):
        sub = model.model
        print(f"  model.model class: {type(sub).__name__}")
        if hasattr(sub, 'layers'):
            print(f"  n_layers: {len(sub.layers)}")
            print(f"  layer[0] type: {type(sub.layers[0]).__name__}")
    elif hasattr(model, 'language_model'):
        lm = model.language_model
        print(f"  model.language_model class: {type(lm).__name__}")
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            print(f"  n_layers: {len(lm.model.layers)}")

    print("\nTest forward (text-only) ...")
    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits  # [1, T, V]
    hs = out.hidden_states  # tuple of [1, T, H] for each layer
    print(f"  logits shape: {logits.shape}")
    print(f"  n hidden states: {len(hs)}  (=n_layers+1)")
    print(f"  hidden state shape: {hs[0].shape}")
    next_tok = int(logits[0, -1].argmax())
    print(f"  next token: {tok.decode([next_tok])!r}")

    print("\nTest forward with inputs_embeds ...")
    emb = model.get_input_embeddings()(inputs.input_ids)
    print(f"  embeddings shape: {emb.shape}")
    with torch.no_grad():
        out2 = model(inputs_embeds=emb, attention_mask=inputs.attention_mask)
    print(f"  logits shape: {out2.logits.shape}")
    diff = (out.logits - out2.logits).abs().max().item()
    print(f"  max diff vs input_ids path: {diff:.6f}  (should be ~0)")

    # Memory footprint
    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB / "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")


if __name__ == "__main__":
    sys.exit(main())
