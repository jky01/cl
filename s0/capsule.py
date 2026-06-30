"""Omega-0: the amortized fact-capsule memory.

Write path (NO gradient at deploy/eval time -- review point: route A):
    statement tokens -> FrozenCore.hidden -> FactEncoder -> z_f
    z_f -> WriteNet -> capsule fields (k_sem, k_addr, v_caps, ctx, aux) + product-key logits
    ProductKeyAllocator -> slot index -> hard scatter into memory bank M

Read path:
    query tokens -> FrozenCore.hidden -> QueryEncoder -> q_sem
    score q_sem against k_sem of allocated slots -> top-k
    ValueDecoder expands each selected capsule into P prefix tokens
    cross-attention from the query's <ans> hidden over those prefix tokens -> R
    H' = H_ans + g_mem * R ; logits = FrozenCore.lm_head(H')

KEY DESIGN (review point #3 -- division of labour):
    * product-key (k_addr) controls WRITE PLACEMENT (spreads facts across slots).
    * k_sem + the contrastive/margin losses control READ DISCRIMINATION.
    product keys alone do NOT prevent read-time object-swap; the hard-negative
    margin loss does. Both are present.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotLayout:
    d_key: int       # size of k_sem AND k_addr (each)
    d_v: int         # capsule value latent
    d_ctx: int       # context latent
    d_aux: int       # control scalars (confidence, usage, alloc, ...)
    d_time: int = 1  # write-time stamp (Step 2 versioning); set externally

    # fields in slot-vector order; slices computed by cumulative offset
    def _fields(self):
        return [("ksem", self.d_key), ("kaddr", self.d_key), ("v", self.d_v),
                ("ctx", self.d_ctx), ("aux", self.d_aux), ("time", self.d_time)]

    def _slice(self, name):
        off = 0
        for n, d in self._fields():
            if n == name:
                return slice(off, off + d)
            off += d
        raise KeyError(name)

    @property
    def d_slot(self):
        return sum(d for _, d in self._fields())

    @property
    def d_learned(self):
        """size WriteNet produces (everything except the externally-set time)."""
        return self.d_slot - self.d_time

    def s_ksem(self):  return self._slice("ksem")
    def s_kaddr(self): return self._slice("kaddr")
    def s_v(self):     return self._slice("v")
    def s_ctx(self):   return self._slice("ctx")
    def s_aux(self):   return self._slice("aux")
    def s_time(self):  return self._slice("time")


def gumbel_softmax(logits, tau, hard, training):
    if training:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    # deploy: deterministic argmax one-hot
    idx = logits.argmax(-1)
    return F.one_hot(idx, logits.size(-1)).float()


class ProductKeyAllocator(nn.Module):
    """Two codebooks -> n1*n2 buckets. bucket = one slot (Step 0 simplification).

    NOTE (review point #3): bucket==slot does NOT guarantee collision-free
    allocation; with ~N_mem facts the birthday paradox forces overwrites well
    before the bank is full. The balance loss spreads usage; real headroom
    (n_buckets >> n_facts) or within-bucket multi-slot is a later fix.
    """

    def __init__(self, n1, n2, d_key):
        super().__init__()
        assert d_key % 2 == 0
        self.n1, self.n2 = n1, n2
        self.C1 = nn.Parameter(torch.randn(n1, d_key // 2) * 0.02)
        self.C2 = nn.Parameter(torch.randn(n2, d_key // 2) * 0.02)

    def forward(self, logits1, logits2, tau, hard, training):
        p1 = gumbel_softmax(logits1, tau, hard, training)   # [B, n1]
        p2 = gumbel_softmax(logits2, tau, hard, training)   # [B, n2]
        k_addr = torch.cat([p1 @ self.C1, p2 @ self.C2], dim=-1)  # [B, d_key]
        slot_id = p1.argmax(-1) * self.n2 + p2.argmax(-1)         # [B] (hard index)
        # batch-mean bucket usage for the balance loss
        usage = (p1.mean(0)[:, None] * p2.mean(0)[None, :]).reshape(-1)  # [n1*n2]
        return k_addr, slot_id, usage


class FactEncoder(nn.Module):
    """Attention-pool the statement hidden states with a few learned queries.

    Mean-pooling blurs subject/relation/object into one average, from which a
    specific object (1-of-N) is not recoverable. Learned-query attention lets
    the encoder isolate the object (and relation/subject) token(s) it needs.
    """

    def __init__(self, d_model, d_z, n_query=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_query, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.net = nn.Sequential(nn.Linear(n_query * d_model, d_z), nn.GELU(),
                                 nn.Linear(d_z, d_z))

    def forward(self, hidden, lengths):
        B, T, d = hidden.shape
        key_pad = torch.arange(T, device=hidden.device)[None] >= lengths[:, None]  # True=pad
        q = self.q[None].expand(B, -1, -1)
        pooled, _ = self.attn(q, hidden, hidden, key_padding_mask=key_pad, need_weights=False)
        return self.net(pooled.reshape(B, -1))


class WriteNet(nn.Module):
    def __init__(self, d_z, layout: SlotLayout, n1, n2):
        super().__init__()
        self.layout = layout
        self.alloc = ProductKeyAllocator(n1, n2, layout.d_key)
        h = d_z
        self.trunk = nn.Sequential(nn.Linear(d_z, h), nn.GELU())
        # k_sem is NOT produced here anymore -- it comes from SRKeyEncoder on the
        # (subject, relation) tokens (a contrastive key from pooled features
        # collapses). WriteNet owns the value/address/aux fields only.
        self.head_v = nn.Linear(h, layout.d_v)
        # residual skip: a direct z_f -> v path so the object features the
        # encoder extracts reach v without having to survive the trunk
        # bottleneck (diag TEST 4: smoother, faster convergence of v).
        self.skip_v = nn.Linear(d_z, layout.d_v)
        self.head_ctx = nn.Linear(h, layout.d_ctx)
        self.head_aux = nn.Linear(h, layout.d_aux)
        self.head_l1 = nn.Linear(h, n1)
        self.head_l2 = nn.Linear(h, n2)

    def forward(self, z_f, k_sem, tau, hard, training):
        h = self.trunk(z_f)
        v = self.head_v(h) + self.skip_v(z_f)
        ctx = self.head_ctx(h)
        aux = self.head_aux(h)
        k_addr, slot_id, usage = self.alloc(self.head_l1(h), self.head_l2(h), tau, hard, training)
        capsule = torch.cat([k_sem, k_addr, v, ctx, aux], dim=-1)  # [B, d_slot]
        return capsule, slot_id, usage, k_sem


def gather_sr(ids, h, ranges):
    """Gather the hidden states at the SUBJECT and RELATION token positions
    (located by token-id range), return [B, 2*d_model].

    The (subject, relation) token hiddens carry the identity that retrieval must
    match on. A learned-query attention pool collapses under a contrastive loss
    and a mean pool washes out the subject; gathering the actual tokens gives a
    discriminative key/query (validated: retrieval@1 ~0.91 vs ~0.41 mean-pool,
    ~chance attention-pool). `ranges` = (sub_lo, sub_hi, rel_lo, rel_hi).
    """
    B, T, d = h.shape
    sub_lo, sub_hi, rel_lo, rel_hi = ranges
    is_sub = (ids >= sub_lo) & (ids < sub_hi)
    is_rel = (ids >= rel_lo) & (ids < rel_hi)
    sub_pos = is_sub.float().argmax(1)   # first subject-token position per row
    rel_pos = is_rel.float().argmax(1)
    rows = torch.arange(B, device=h.device)
    return torch.cat([h[rows, sub_pos], h[rows, rel_pos]], dim=-1)   # [B, 2d]


class SRKeyEncoder(nn.Module):
    """Map gathered (subject, relation) token hiddens -> a normalized retrieval
    key/query. Used for BOTH k_sem (from the statement) and q_sem (from the
    query) so they land in the same space and match on (subject, relation)."""

    def __init__(self, d_model, d_key):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_key))

    def forward(self, ids, hidden, ranges):
        return F.normalize(self.net(gather_sr(ids, hidden, ranges)), dim=-1)


class ValueDecoder(nn.Module):
    """Expand a capsule (v + ctx) into P prefix tokens in model space."""

    def __init__(self, layout: SlotLayout, d_model, n_prefix):
        super().__init__()
        self.P = n_prefix
        self.d_model = d_model
        din = layout.d_v + layout.d_ctx
        self.net = nn.Sequential(nn.Linear(din, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, n_prefix * d_model))

    def forward(self, v, ctx):
        x = self.net(torch.cat([v, ctx], dim=-1))            # [..., P*d_model]
        return x.reshape(*x.shape[:-1], self.P, self.d_model)


class CapsuleMemory(nn.Module):
    """The full Omega-0 read/write system around a frozen core."""

    def __init__(self, core, world, layout: SlotLayout, n1, n2, n_mem,
                 n_prefix=4, top_k=4, d_z=None):
        super().__init__()
        self.core = core               # frozen
        self.world = world
        self.layout = layout
        self.n_mem = n_mem
        self.top_k = top_k
        d_model = core.d_model
        d_z = d_z or d_model

        self.fact_enc = FactEncoder(d_model, d_z)     # -> z_f for the value path
        self.write_net = WriteNet(d_z, layout, n1, n2)
        # retrieval keys from (subject, relation) token hiddens; key_enc on the
        # statement, query_enc on the query -- shared d_key space.
        self.key_enc = SRKeyEncoder(d_model, layout.d_key)
        self.query_enc = SRKeyEncoder(d_model, layout.d_key)
        self.sr_ranges = (world.tok2id[world.entities[0]],
                          world.tok2id[world.entities[-1]] + 1,
                          world.tok2id[world.relations[0]],
                          world.tok2id[world.relations[-1]] + 1)
        self.value_dec = ValueDecoder(layout, d_model, n_prefix)
        self.read_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        # re-normalise after injection so adding R cannot blow up the logits
        # (the frozen lm_head expects post-LayerNorm-scale inputs).
        self.inject_ln = nn.LayerNorm(d_model)
        # injection gate g = g_content * relevance.
        #  * g_content (from query hidden + retrieved signal R): how much of the
        #    content to write, bias init negative so it starts near 0.
        #  * relevance = sigmoid(scale*(conf - thr)) is a SHARP, explicit
        #    function of the top key-match score conf. A scalar conf fed into the
        #    content MLP gets ignored (the high-dim query dominates, and an
        #    unrelated query still looks like a valid query); a dedicated sharp
        #    relevance gate STRUCTURALLY shuts injection when retrieval is weak
        #    (query's fact not in memory) -> locality.
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        # FIXED-high slope (sharp boundary) + LEARNABLE threshold. The margin
        # loss separates matched vs unrelated conf, but their absolute scale
        # drifts run-to-run, so a hardcoded threshold lands in the wrong place
        # (clips the matched query or leaks the unrelated one). A learnable thr
        # slides to sit BETWEEN the two clusters; the fixed steep scale keeps
        # the boundary sharp so relevance is ~1 above it and ~0 below.
        self.register_buffer("relev_scale", torch.tensor(20.0))
        self.relev_thr = nn.Parameter(torch.tensor(0.3))
        # Step 2 version routing: a query's now/before token -> a target time
        # c_target in [0,1]; among slots matching the (s,r) key, the one whose
        # write-time stamp is closest to c_target wins the tie. ver_weight is
        # < the matched-vs-unmatched cosine gap (~0.8) so it only breaks ties
        # among matched slots, never promotes a different-key slot.
        self.ctx_enc = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                     nn.Linear(d_model // 2, 1))
        self.register_buffer("ver_weight", torch.tensor(0.5))
        # sharp read selection: same-key conflicting versions differ only by the
        # ver_weight*time bias (~0.5); a temperature-1 softmax would still blend
        # them (~0.62/0.38) and the blended value decodes to the wrong token. A
        # low temperature makes the selected version dominate.
        self.register_buffer("read_temp", torch.tensor(0.1))
        self.now_id = world.i("now")
        self.before_id = world.i("before")
        # WARMUP flag: when False, relevance is forced to 1 (gate fully usable)
        # so storage can bootstrap. With the sharp relevance gate, a cold start
        # has conf~0 < thr -> relevance~0 -> g~0 -> the injection gets no answer
        # gradient -> conf never rises -> deadlock. Train with relevance off
        # first, turn it on once conf has separated (see train_omega0 warmup).
        self.relevance_enabled = True
        assert n1 * n2 == n_mem, "Step 0: one product bucket == one slot"

    # ---- write -------------------------------------------------------
    def write(self, M, alloc_mask, stmt_ids, lengths, tau=1.0, hard=True,
              training=False, time=1.0):
        """Scatter one capsule per row into the (batched) memory bank M.

        M: [B, n_mem, d_slot]; alloc_mask: [B, n_mem]. `time` (scalar or [B])
        is the write-time stamp stored in the slot for version routing; it
        defaults to 1.0 (latest) so single-fact / Step-0 behaviour is unchanged.
        """
        with torch.no_grad():
            h = self.core.hidden(stmt_ids)
        z_f = self.fact_enc(h, lengths)
        k_sem = self.key_enc(stmt_ids, h, self.sr_ranges)   # from (S,R) tokens
        capsule, slot_id, usage, k_sem = self.write_net(z_f, k_sem, tau, hard, training)

        B = M.size(0)
        rows = torch.arange(B, device=M.device)
        if not torch.is_tensor(time):
            time = torch.full((B,), float(time), device=M.device)
        full = torch.cat([capsule, time.to(M.dtype).view(B, 1)], dim=-1)  # [B, d_slot]
        M = M.clone()
        alloc_mask = alloc_mask.clone()
        M[rows, slot_id] = full
        alloc_mask[rows, slot_id] = 1.0
        return M, alloc_mask, dict(slot_id=slot_id, usage=usage, k_sem=k_sem, capsule=full)

    # ---- read --------------------------------------------------------
    def read_logits(self, M, alloc_mask, query_ids, lengths):
        """Return (answer_logits [B, vocab], read_info)."""
        L = self.layout
        with torch.no_grad():
            h = self.core.hidden(query_ids)
        B, T, d = h.shape
        ans_idx = lengths - 1
        rows = torch.arange(B, device=h.device)
        H_ans = h[rows, ans_idx]                              # [B, d] (for injection)

        q_sem = self.query_enc(query_ids, h, self.sr_ranges)  # [B, d_key] (from S,R tokens)
        k_sem_all = M[:, :, L.s_ksem()]                       # [B, n_mem, d_key]
        scores = torch.einsum("bd,bnd->bn", q_sem, k_sem_all) # [B, n_mem]

        # version routing: the query's now/before token -> target time c_target;
        # bias toward the matching slot's stored time. Absent ctx token (Step 0/1
        # queries) -> default c_target=1 (latest), and all normal writes have
        # time=1, so the bias is identically 0 -> Step 0/1 behaviour unchanged.
        ctx_tok = (query_ids == self.now_id) | (query_ids == self.before_id)  # [B,T]
        has_ctx = ctx_tok.any(1)                                              # [B]
        ctx_pos = ctx_tok.float().argmax(1)                                   # [B]
        c_target = torch.sigmoid(self.ctx_enc(h[rows, ctx_pos])).squeeze(-1)  # [B]
        c_target = torch.where(has_ctx, c_target, torch.ones_like(c_target))
        t_slot = M[:, :, L.s_time()].squeeze(-1)                              # [B, n_mem]
        scores = scores - self.ver_weight * (t_slot - c_target[:, None]).abs()

        scores = scores.masked_fill(alloc_mask < 0.5, float("-inf"))

        k = min(self.top_k, self.n_mem)
        top_s, top_i = scores.topk(k, dim=-1)                 # [B, k]
        # retrieval weights (sharp); rows with no allocated slot -> uniform-safe
        ret_w = torch.softmax(torch.nan_to_num(top_s, neginf=-1e4) / self.read_temp, dim=-1)

        sel = torch.gather(M, 1, top_i[..., None].expand(-1, -1, M.size(-1)))  # [B,k,d_slot]
        v = sel[:, :, L.s_v()]
        ctx = sel[:, :, L.s_ctx()]
        prefix = self.value_dec(v, ctx)                       # [B, k, P, d]
        # weight prefix tokens by retrieval weight so grad reaches selected k_sem/v
        prefix = prefix * ret_w[:, :, None, None]
        P_mem = prefix.reshape(B, k * prefix.size(2), d)      # [B, k*P, d]

        R, _ = self.read_attn(H_ans[:, None], P_mem, P_mem, need_weights=False)
        R = R[:, 0]                                           # [B, d]
        # top key-match score as a confidence signal (cosine in [-1,1];
        # -inf when nothing allocated -> treat as no-match).
        conf = torch.nan_to_num(top_s[:, :1], neginf=-1.0)    # [B,1]
        g_content = torch.sigmoid(self.gate(torch.cat([H_ans, R], dim=-1)))  # [B,1]
        relevance = torch.sigmoid(self.relev_scale * (conf - self.relev_thr))  # [B,1]
        if not self.relevance_enabled:
            relevance = torch.ones_like(relevance)   # warmup: bootstrap storage
        g = g_content * relevance
        H_prime = self.inject_ln(H_ans + g * R)   # re-normalise before frozen head
        logits = self.core.lm_head(H_prime)
        return logits, dict(q_sem=q_sem, top_i=top_i, ret_w=ret_w, g=g,
                            conf=conf, relevance=relevance, H_ans=H_ans,
                            c_target=c_target, has_ctx=has_ctx)

    def core_only_logits(self, query_ids, lengths):
        with torch.no_grad():
            h = self.core.hidden(query_ids)
        rows = torch.arange(h.size(0), device=h.device)
        return self.core.lm_head(h[rows, lengths - 1])

    def empty_bank(self, B, device):
        M = torch.zeros(B, self.n_mem, self.layout.d_slot, device=device)
        alloc = torch.zeros(B, self.n_mem, device=device)
        return M, alloc
