"""Qwen integration step 2 — the #1 blocker: a FREE-TEXT retrieval key.
The toy `gather_sr` grabs subject/relation tokens by id-range (template-bound).
Here we test the real-text replacement: Qwen (frozen) as a feature extractor,
mean-pool its last hidden, a small trained projection, InfoNCE. Facts are
free-text with PARAPHRASE variety so nothing can exploit fixed token positions.
Question: does retrieval@1 get high (toy proxy mean-pool only reached ~0.41)?

  .venv/bin/python -m s0.qwen_retrieval
"""
from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B"
NAMES = ("Alice Bob Carol David Emma Frank Grace Henry Iris Jack Karen Leo Mia "
         "Nina Oscar Paula Quinn Rosa Sam Tina Uma Victor Wendy Xavier Yara Zack "
         "Adam Beth Caleb Dana Eli Fiona Gabe Hana Ian Julia Kyle Lena Mike Nora "
         "Omar Page Ravi Sara Theo Ula Vince Will Xena Yuri").split()
ATTRS = ["favorite color", "hometown", "job", "favorite food", "pet",
         "favorite sport", "favorite drink", "hobby", "favorite season", "lucky number"]
VALUES = ("blue red green Paris London Tokyo Berlin doctor teacher pilot pizza sushi "
          "ramen tacos cat dog parrot tennis chess soccer coffee tea juice painting "
          "hiking reading summer winter autumn spring seven three nine violet amber "
          "Rome Cairo Lima nurse baker dancer curry pasta rabbit boxing rowing cocoa "
          "guitar pottery").split()

S_TMPL = ["{n}'s {a} is {v}.", "The {a} of {n} is {v}.", "{n} has {v} as a {a}.",
          "We know that {n}'s {a} is {v}.", "{v} is the {a} of {n}."]
Q_TMPL = ["What is {n}'s {a}?", "What {a} does {n} have?", "Tell me the {a} of {n}.",
          "{n}'s {a} is what?", "Do you know the {a} of {n}?"]


def batch(rng, B):
    facts, ss, qs = [], [], []
    pairs = rng.sample([(n, a) for n in NAMES for a in ATTRS], B)  # distinct (name,attr)
    for (n, a) in pairs:
        v = rng.choice(VALUES)
        ss.append(rng.choice(S_TMPL).format(n=n, a=a, v=v))
        qs.append(rng.choice(Q_TMPL).format(n=n, a=a))
    return ss, qs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    enc_model = lm.model            # base model -> last_hidden_state (post RMSNorm)
    d = lm.config.hidden_size

    @torch.no_grad()
    def feats(texts):               # frozen Qwen, mean-pooled last hidden [B, d]
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc_model(**e).last_hidden_state               # [B,T,d]
        m = e.attention_mask[..., None].to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        return pooled.float()

    proj_q = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)).to(device)
    proj_k = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)).to(device)
    opt = torch.optim.Adam(list(proj_q.parameters()) + list(proj_k.parameters()), lr=1e-3)
    rng = random.Random(0)
    tau, B = 0.05, 128

    for step in range(1500):
        ss, qs = batch(rng, B)
        k = F.normalize(proj_k(feats(ss)), dim=-1)
        q = F.normalize(proj_q(feats(qs)), dim=-1)
        sim = q @ k.t() / tau
        lab = torch.arange(B, device=device)
        loss = 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1499:
            with torch.no_grad():
                ss, qs = batch(rng, B)
                k = F.normalize(proj_k(feats(ss)), dim=-1)
                q = F.normalize(proj_q(feats(qs)), dim=-1)
                sim = q @ k.t()
                r1 = (sim.argmax(1) == torch.arange(B, device=device)).float().mean().item()
                print(f"  step {step:4d} loss {loss.item():.3f} retrieval@1 {r1:.3f} "
                      f"(pos {sim.diag().mean():.2f})")
    print("  high retrieval@1 on FREE TEXT => the gather_sr template trick can be replaced")
    print("  by a Qwen-feature + projection key encoder (blocker #1 path).")


if __name__ == "__main__":
    main()
