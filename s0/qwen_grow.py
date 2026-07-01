"""Does function-preserving growth transfer to a REAL architecture (Qwen)?
A Qwen2 decoder layer has RMSNorm + RoPE + GQA attention + SwiGLU MLP -- much
more than the toy Block. Insert identity-initialised Qwen layers (zero the
attention o_proj and the MLP down_proj so the layer's residual adds are 0 ->
layer(x)=x) and check the logits are unchanged. If so, grow_deeper transfers to
real LMs.

  .venv/bin/python -m s0.qwen_grow
"""
from __future__ import annotations
import os
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")


def grow_qwen(model, n_new):
    """Append n_new identity-initialised decoder layers (deep-copied for correct
    wiring, then output projections zeroed)."""
    layers = model.model.layers
    cfg = model.config
    for _ in range(n_new):
        blk = copy.deepcopy(layers[-1])
        nn.init.zeros_(blk.self_attn.o_proj.weight)
        if getattr(blk.self_attn.o_proj, "bias", None) is not None:
            nn.init.zeros_(blk.self_attn.o_proj.bias)
        nn.init.zeros_(blk.mlp.down_proj.weight)
        if getattr(blk.mlp.down_proj, "bias", None) is not None:
            nn.init.zeros_(blk.mlp.down_proj.bias)
        new_idx = len(layers)
        blk.self_attn.layer_idx = new_idx                 # KV-cache index
        if getattr(blk, "layer_idx", None) is not None:
            blk.layer_idx = new_idx
        layers.append(blk)
        if getattr(cfg, "layer_types", None) is not None:  # per-layer attn type list
            cfg.layer_types.append(cfg.layer_types[-1])
    cfg.num_hidden_layers = len(layers)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                  # so [:, -1] is the real last token
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    n0 = len(model.model.layers)

    prompts = ["The capital of France is", "Water boils at a temperature of",
               "The opposite of hot is"]
    ids = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits0 = model(**ids, use_cache=False).logits.float()
    nxt0 = [tok.decode(logits0[i, -1].argmax()) for i in range(len(prompts))]

    grow_qwen(model, 4)                       # 24 -> 28 layers, identity-init
    with torch.no_grad():
        logits1 = model(**ids, use_cache=False).logits.float()
    nxt1 = [tok.decode(logits1[i, -1].argmax()) for i in range(len(prompts))]

    diff = (logits1 - logits0).abs().max().item()
    print(f"  layers {n0} -> {len(model.model.layers)} (identity-init)")
    print(f"  max|logits_after - logits_before| = {diff:.3e}")
    print(f"  next tokens before: {nxt0}")
    print(f"  next tokens after:  {nxt1}")
    print("  ~0 diff + same tokens => function-preserving growth transfers to Qwen;")
    print("  the added (frozen-identity) capacity can then be trained, like the proxy.")


if __name__ == "__main__":
    main()
