"""Qwen integration step 1: verify the injection interface on a real small LM.
We need the same hook the proxy used -- take the final hidden state, add a
memory signal, then apply lm_head. Confirms lm_head(final_hidden) == logits and
prints the architecture facts the memory modules must adapt to.

  .venv/bin/python -m s0.qwen_iface
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== loading {NAME} on {device} ==")
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    c = model.config
    print(f"  hidden={c.hidden_size} layers={c.num_hidden_layers} "
          f"heads={c.num_attention_heads} kv_heads={getattr(c,'num_key_value_heads','?')} "
          f"vocab={c.vocab_size} tie_emb={getattr(c,'tie_word_embeddings','?')}")
    print(f"  final norm: {type(model.model.norm).__name__}")

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    logits = out.logits                      # [1,T,V]
    h_last = out.hidden_states[-1]           # [1,T,d]
    with torch.no_grad():
        recon = model.lm_head(h_last)
    diff = (recon.float() - logits.float()).abs().max().item()
    print(f"  hidden_states[-1] shape {tuple(h_last.shape)}; "
          f"max|lm_head(h_last) - logits| = {diff:.2e}")
    nxt = tok.decode(logits[0, -1].argmax())
    print(f"  next-token after prompt: {nxt!r}  (sanity: model works)")
    print("  -> if diff ~0, the injection point is lm_head(final_hidden): we can add")
    print("     g*R to h_last and re-apply lm_head, exactly like the proxy core.")


if __name__ == "__main__":
    main()
