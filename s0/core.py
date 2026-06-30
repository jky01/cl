"""Small decoder-only Transformer used as the FROZEN proxy core.

It is pretrained only on the template *syntax* with random fact bindings, so
it learns grammar/format but cannot have memorised any specific (s,r,o) fact.
After pretraining it is frozen: gradients for Omega-0 flow THROUGH it (it is
differentiable) but never update it. This is the controlled stand-in for a
real LM (review point #5: use a small synthetic proxy, not a big LLM, so the
experiment isolates "can memory store a new fact").
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ProxyCore(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def hidden(self, ids):
        """Return final hidden states [B, T, d_model] (pre-lm_head)."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = self._causal_mask(T, ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    def forward(self, ids):
        h = self.hidden(ids)
        return self.lm_head(h), h


class GatedBlock(Block):
    """A transformer block whose residual contribution is scaled by a scalar
    gate g (0 -> identity, 1 -> full block)."""

    def forward(self, x, attn_mask, g):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g * a
        x = x + g * self.mlp(self.ln2(x))
        return x


class GrowableCore(nn.Module):
    """Neural growth trigger: a stack of n_base always-on blocks plus n_grow
    blocks each behind a LEARNED gate. The gates start ~0 (the extra blocks are
    the identity, so effective depth = n_base), and the TASK LOSS opens them
    differentiably when more depth helps. A capacity penalty on the grow-gates
    (L_capacity, §27) means the network only engages depth it actually needs ->
    the model decides its own effective depth (the neural form of 'grow when
    saturated'). No hidden-dim change, so memory modules keep their interface.
    """

    def __init__(self, vocab_size, d_model=128, n_base=2, n_grow=4, n_heads=4, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_base = n_base
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GatedBlock(d_model, n_heads) for _ in range(n_base + n_grow)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # base gates open (~1, learnable but start engaged); grow gates ~0 (identity)
        init = [10.0] * n_base + [-4.0] * n_grow
        self.gate_logit = nn.Parameter(torch.tensor(init))

    def gates(self):
        return torch.sigmoid(self.gate_logit)

    def grow_gate_penalty(self):
        return self.gates()[self.n_base:].sum()

    def effective_depth(self):
        return float(self.n_base + self.gates()[self.n_base:].sum().item())

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def hidden(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = self._causal_mask(T, ids.device)
        g = self.gates()
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, g[i])
        return self.ln_f(x)

    def forward(self, ids):
        h = self.hidden(ids)
        return self.lm_head(h), h


def grow_deeper(core: ProxyCore, n_new: int, trainable: bool = False):
    """FUNCTION-PRESERVING growth: append n_new transformer blocks initialised to
    the IDENTITY (attention out-proj and MLP final linear zeroed -> block(x)=x),
    so hidden()/logits are bit-for-bit unchanged at the moment of growth (zero
    forgetting). Width (d_model) is unchanged, so every downstream memory module
    keeps its interface. The new blocks add capacity that later training fills.
    """
    device = next(core.parameters()).device
    n_heads = core.blocks[0].attn.num_heads
    for _ in range(n_new):
        blk = Block(core.d_model, n_heads).to(device)
        nn.init.zeros_(blk.attn.out_proj.weight); nn.init.zeros_(blk.attn.out_proj.bias)
        nn.init.zeros_(blk.mlp[-1].weight); nn.init.zeros_(blk.mlp[-1].bias)
        for p in blk.parameters():
            p.requires_grad_(trainable)
        core.blocks.append(blk)
    return core


def pretrain_core(core: ProxyCore, world, steps=400, batch=64, lr=3e-3,
                  device="cpu", log=print):
    """Teach the core the template language with RANDOM facts.

    Because each training sentence draws a fresh random (s,r,o), the core
    cannot learn that "entity_k -> object_m"; it can only learn the surface
    grammar and where answers go. The specific binding must come from memory.
    """
    from .pad import pad_batch
    core.to(device).train()
    opt = torch.optim.AdamW(core.parameters(), lr=lr)
    for step in range(steps):
        seqs = []
        for _ in range(batch):
            f = world.sample_fact()
            # mix statements and (in-context) QA so the core learns both forms
            if torch.rand(()) < 0.5:
                seqs.append(world.render_statement(f))
            else:
                stmt = world.render_statement(f)[1:-1]  # drop bos/eos
                qids, ans = world.render_query(f)
                seqs.append([world.i("<bos>")] + stmt + [world.i("<sep>")] + qids[1:] + [ans, world.i("<eos>")])
        ids, _ = pad_batch(seqs, world.i("<pad>"), device)
        logits, _ = core(ids)
        # next-token LM loss, ignoring pad targets
        tgt = ids[:, 1:].clone()
        tgt[ids[:, 1:] == world.i("<pad>")] = -100
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step % max(1, steps // 5) == 0 or step == steps - 1):
            log(f"  [core pretrain] step {step:4d}  loss {loss.item():.3f}")
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    return core
