---
name: s0-step0-design-constraints
description: Fixed design decisions / review-point constraints baked into the s0 Step-0 experiment
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c19101e-fabc-4e34-a5e4-fb73c5c5e9ed
---

Constraints the user fixed for the `s0/` Step-0 experiment (see [[s0-step0-state]]); honor these, don't relitigate:
- **Scale**: single-machine small model (target 1–3B eventually); Step 0 uses a tiny synthetic proxy core, NOT a real LLM, to isolate "can memory store a fact."
- **Everything neural / NN-controlled** ("全數都是神經網路化") — no rule-based components; gating, allocation, read/write all learned.
- **Route A only**: amortized, NO-gradient write at deploy (one forward, WriteNet produces the capsule). Gradient-write (route B) deferred to v1.5.
- **Claim boundary (#1)**: v0 tests COMBINATORIAL BINDING over a known fixed vocab, NOT novel-symbol acquisition.
- **Baselines (#2)**: must beat/justify vs no-mem, in-context, external-KV, oracle-slot, and **LoRA fine-tune** (the parametric-storage rival). All in `s0/baselines.py`.
- **Product-key bucket==slot (#3)** is a Step-0 simplification; birthday-collisions before bank-full are a known later fix.
- **Train distribution == eval (#4)**: episodes include same-relation hard negatives.
- User's own design docs live in `reference/` (project.md, 27*.md, 1.md, 2.md, 3.md) — do NOT overwrite; `3.md` is the spec being implemented.
- Working style: user pauses/resumes training explicitly ("暫停" / "開始"); replies in Chinese welcome.
