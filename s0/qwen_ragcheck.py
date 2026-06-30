"""Sanity-check the RAG (in-context) baseline: does Qwen copy a fact that's
literally in the prompt? Diagnoses whether the low RAG score is a real
Qwen-0.5B limitation or an eval bug (padding/format).

  .venv/bin/python -m s0.qwen_ragcheck
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B"
EX = [("Alice", "favorite color", "blue"), ("Bob", "hometown", "Tokyo"),
      ("Carol", "job", "baker"), ("David", "pet", "parrot")]
FORMATS = [
    ("stmt. cloze",   "{n}'s {a} is {v}. {n}'s {a} is"),
    ("Q/A",           "{n}'s {a} is {v}.\nQuestion: What is {n}'s {a}? Answer:"),
    ("so",            "{n}'s {a} is {v}, so {n}'s {a} is"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    print("padding_side:", tok.padding_side)
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    for tag, fmt in FORMATS:
        hit = 0
        print(f"\n== format: {tag} ==")
        for (n, a, v) in EX:
            prompt = fmt.format(n=n, a=a, v=v)
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = lm(ids).logits[0, -1]
            top = logits.topk(5).indices.tolist()
            want = tok(" " + v, add_special_tokens=False).input_ids[0]
            ok = top[0] == want
            hit += ok
            print(f"  {n}/{a}={v!r}: want {tok.decode([want])!r} | "
                  f"top5 {[tok.decode([t]) for t in top]} {'OK' if ok else ''}")
        print(f"  -> {hit}/{len(EX)} copied")


if __name__ == "__main__":
    main()
