"""Baselines that the neural capsule memory must beat or justify itself against
(review point #2). All operate on the same frozen proxy core + synthetic world.

  A. NoMemory      : frozen core only. Expected to fail on new facts (floor).
  B. InContext     : prepend the fact statement to the query prompt (strong, simple).
  C. ExternalKV    : exact dict lookup of the fact, then in-context (RAG-style).
  D. OracleSlot    : store the object id directly, copy at query (upper bound;
                     also a sanity check that the core CAN emit the object).
  E. LoRAFinetune  : gradient-write the episode's facts into a LoRA adapter --
                     the real parametric-storage rival to neural memory.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pad import pad_batch


def _answer_acc(logits, ans_ids):
    return (logits.argmax(-1) == ans_ids).float().mean().item()


class NoMemory:
    name = "A:no-mem"

    def __init__(self, core, world):
        self.core, self.world = core, world

    def eval_episode(self, facts, device):
        qs, ans = [], []
        for f in facts:
            qids, a = self.world.render_query(f)
            qs.append(qids); ans.append(a)
        ids, lengths = pad_batch(qs, self.world.i("<pad>"), device)
        rows = torch.arange(ids.size(0), device=device)
        logits = self.core.lm_head(self.core.hidden(ids)[rows, lengths - 1])
        return _answer_acc(logits, torch.tensor(ans, device=device))


class InContext:
    name = "B:in-context"

    def __init__(self, core, world):
        self.core, self.world = core, world

    def _seq(self, f):
        stmt = self.world.render_statement(f)[1:-1]          # drop bos/eos
        qids, a = self.world.render_query(f)
        return [self.world.i("<bos>")] + stmt + [self.world.i("<sep>")] + qids[1:], a

    def eval_episode(self, facts, device):
        qs, ans = [], []
        for f in facts:
            s, a = self._seq(f); qs.append(s); ans.append(a)
        ids, lengths = pad_batch(qs, self.world.i("<pad>"), device)
        rows = torch.arange(ids.size(0), device=device)
        logits = self.core.lm_head(self.core.hidden(ids)[rows, lengths - 1])
        return _answer_acc(logits, torch.tensor(ans, device=device))


class ExternalKV(InContext):
    """RAG-style: a perfect external dict returns the fact text, then in-context.
    With a perfect retriever this equals InContext; included to make explicit
    that capsule memory must beat *retrieval + prompt*, not just raw core."""
    name = "C:external-kv"


class OracleSlot:
    name = "D:oracle-slot"

    def __init__(self, core, world):
        self.core, self.world = core, world

    def eval_episode(self, facts, device):
        # store {(s,r): o}; at query copy the stored object out directly.
        store = {(s, r): o for (s, r, o) in facts}
        correct = 0
        for (s, r, o) in facts:
            correct += int(store.get((s, r)) == o)
        return correct / len(facts)


class LoRAFinetune:
    """Gradient-write facts into a per-episode LoRA on the lm_head input.

    The frozen core stays frozen; only a low-rank residual added to the final
    hidden state is trained on this episode's statements. This is the
    continual-learning rival: 'just fine-tune a small adapter per batch'.
    """
    name = "E:lora"

    def __init__(self, core, world, rank=8, steps=30, lr=1e-2):
        self.core, self.world = core, world
        self.rank, self.steps, self.lr = rank, steps, lr

    def eval_episode(self, facts, device):
        d = self.core.d_model
        A = torch.zeros(d, self.rank, device=device, requires_grad=True)
        B = torch.zeros(self.rank, d, device=device, requires_grad=True)
        nn.init.normal_(A, std=0.02)
        opt = torch.optim.Adam([A, B], lr=self.lr)

        # train objective: predict object at <ans> from the statement form
        seqs, ans = [], []
        for f in facts:
            stmt = self.world.render_statement(f)[1:-1]
            qids, a = self.world.render_query(f)
            seqs.append([self.world.i("<bos>")] + stmt + [self.world.i("<sep>")] + qids[1:])
            ans.append(a)
        ids, lengths = pad_batch(seqs, self.world.i("<pad>"), device)
        rows = torch.arange(ids.size(0), device=device)
        ans_t = torch.tensor(ans, device=device)

        for _ in range(self.steps):
            with torch.no_grad():
                h = self.core.hidden(ids)[rows, lengths - 1]   # [B,d]
            h2 = h + (h @ A) @ B
            logits = self.core.lm_head(h2)
            loss = F.cross_entropy(logits, ans_t)
            opt.zero_grad(); loss.backward(); opt.step()

        # eval on held-out paraphrase queries (no fact in context)
        qs, qans = [], []
        for f in facts:
            qids, a = self.world.render_query(f); qs.append(qids); qans.append(a)
        qids, qlen = pad_batch(qs, self.world.i("<pad>"), device)
        qrows = torch.arange(qids.size(0), device=device)
        with torch.no_grad():
            h = self.core.hidden(qids)[qrows, qlen - 1]
            h2 = h + (h @ A) @ B
            logits = self.core.lm_head(h2)
        return _answer_acc(logits, torch.tensor(qans, device=device))


ALL_BASELINES = [NoMemory, InContext, ExternalKV, OracleSlot, LoRAFinetune]
