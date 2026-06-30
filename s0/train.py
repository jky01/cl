"""Train Omega-0 and evaluate Accuracy(N_facts) vs baselines.

Omega-0 TRAINING DISTRIBUTION (review point #4): episodes contain MULTIPLE
facts with same-relation hard negatives from the start, so that collision at
eval time is in-distribution rather than a shock to a single-fact-trained net.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .pad import pad_batch
from .capsule import CapsuleMemory


def _build_episode_tensors(world, B, n_facts, hard_neg_ratio, device):
    """Returns episode_facts[B][n_facts]."""
    return [world.sample_episode_facts(n_facts, hard_neg_ratio) for _ in range(B)]


def _write_all(mem: CapsuleMemory, episodes, n_facts, device, tau, hard, training):
    """Write the j-th fact of every episode in parallel, for j in 0..n_facts-1."""
    B = len(episodes)
    M, alloc = mem.empty_bank(B, device)
    k_sem_list, slot_list, usage_acc = [], [], 0.0
    for j in range(n_facts):
        seqs = [mem.world.render_statement(episodes[b][j]) for b in range(B)]
        ids, lengths = pad_batch(seqs, mem.world.i("<pad>"), device)
        M, alloc, info = mem.write(M, alloc, ids, lengths, tau=tau, hard=hard, training=training)
        k_sem_list.append(info["k_sem"])
        slot_list.append(info["slot_id"])
        usage_acc = usage_acc + info["usage"]
    K = torch.stack(k_sem_list, dim=1)          # [B, n_facts, d_key]
    slots = torch.stack(slot_list, dim=1)       # [B, n_facts]
    return M, alloc, K, slots, usage_acc / n_facts


def _conflict_loss(mem: CapsuleMemory, world, B, device, tau):
    """Write a conflicting (s,r): o_before@t=0 then o_now@t=1, then require the
    now/before query to route to the right version. Trains version routing."""
    groups = [world.sample_conflict_episode(1)[0] for _ in range(B)]
    Mc, allocc = mem.empty_bank(B, device)
    for which, t in (("o_before", 0.0), ("o_now", 1.0)):
        seqs = [world.render_statement((g["s"], g["r"], g[which])) for g in groups]
        ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
        Mc, allocc, _ = mem.write(Mc, allocc, ids, lengths, tau=tau, hard=True,
                                  training=True, time=t)
    loss = 0.0
    for ctx, key in (("now", "o_now"), ("before", "o_before")):
        qs, ans = [], []
        for g in groups:
            qids, a = world.render_query_versioned(g["s"], g["r"], g[key], ctx)
            qs.append(qids); ans.append(a)
        ids, lengths = pad_batch(qs, world.i("<pad>"), device)
        logits, _ = mem.read_logits(Mc, allocc, ids, lengths)
        loss = loss + F.cross_entropy(logits, torch.tensor(ans, device=device))
    return loss / 2


def _other_obj(world, o):
    ob = world.rng.randrange(world.cfg.n_objects)
    while ob == o:
        ob = world.rng.randrange(world.cfg.n_objects)
    return ob


def _safety_loss(mem: CapsuleMemory, world, B, device, tau):
    """Write a reliable fact (trust=1) then a CONFLICTING unreliable update
    (trust=0, different value): the unreliable contradiction must be rejected
    by the commit gate so (s,r) still recalls the reliable value."""
    facts = [world.sample_fact() for _ in range(B)]
    M, alloc = mem.empty_bank(B, device)
    seqs = [world.render_statement(f) for f in facts]
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    M, alloc, gi_rel = mem.write(M, alloc, ids, lengths, tau=tau, hard=True,
                                 training=True, time=1.0, trust=1.0)
    bad = [(s, r, _other_obj(world, o)) for (s, r, o) in facts]
    seqs = [world.render_statement(b) for b in bad]
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    M, alloc, gi_bad = mem.write(M, alloc, ids, lengths, tau=tau, hard=True,
                                 training=True, time=1.0, trust=0.0)
    qs, ans = [], []
    for f in facts:
        qids, a = world.render_query(f)          # answer = the reliable value
        qs.append(qids); ans.append(a)
    ids, lengths = pad_batch(qs, world.i("<pad>"), device)
    logits, _ = mem.read_logits(M, alloc, ids, lengths)
    protect = F.cross_entropy(logits, torch.tensor(ans, device=device))
    # direct admission supervision: admit reliable (->1), reject unreliable (->0)
    ones = torch.ones_like(gi_rel["g_commit"]); zeros = torch.zeros_like(gi_bad["g_commit"])
    gate = F.binary_cross_entropy(gi_rel["g_commit"], ones) + \
        F.binary_cross_entropy(gi_bad["g_commit"], zeros)
    return protect + 0.5 * gate


def train_omega0(mem: CapsuleMemory, world, *, steps=600, B=32, max_facts=8,
                 hard_neg_ratio=0.5, lr=1e-3, tau=1.0, device="cpu",
                 lambdas=None, warmup_frac=0.3, grad_clip=1.0, log=print):
    # balance=0: placement is now occupancy-aware (free-slot), not product-key,
    # so the product-key usage-balance loss is vestigial (the codebooks no
    # longer decide where facts go). Kept at 0 rather than ripping out the
    # allocator wiring.
    lam = dict(answer=1.0, retrieve=1.0, orth=0.1, balance=0.0, locality=0.5,
               conflict=1.0, safety=1.0)
    if lambdas:
        lam.update(lambdas)
    params = [p for p in mem.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    mem.train()
    warmup_steps = int(warmup_frac * steps)

    for step in range(steps):
        # WARMUP: relevance forced open + locality off -> storage bootstraps and
        # the margin loss separates conf. After warmup: relevance gate + locality
        # on -> selective read. (Avoids the sharp-gate cold-start deadlock.)
        warming = step < warmup_steps
        mem.relevance_enabled = not warming
        # CURRICULUM: ramp the max episode size 1 -> max_facts over the first 70%
        # of training. Sampling uniformly over [1, max_facts] from step 0 is too
        # hard for large max_facts (the answer loss never gets traction and the
        # whole thing collapses to chance); easy episodes first lets it bootstrap.
        cur_max = 1 + int((max_facts - 1) * min(1.0, step / (0.7 * steps + 1)))
        n_facts = torch.randint(1, cur_max + 1, ()).item()
        episodes = _build_episode_tensors(world, B, n_facts, hard_neg_ratio, device)
        M, alloc, K, slots, usage = _write_all(mem, episodes, n_facts, device, tau,
                                                hard=True, training=True)

        # --- read every written fact, accumulate answer + retrieval losses ---
        ans_loss = 0.0
        Q = []
        g_acc = 0.0
        for j in range(n_facts):
            qs, ans = [], []
            for b in range(B):
                qids, a = world.render_query(episodes[b][j])
                qs.append(qids); ans.append(a)
            ids, lengths = pad_batch(qs, world.i("<pad>"), device)
            logits, info = mem.read_logits(M, alloc, ids, lengths)
            ans_loss = ans_loss + F.cross_entropy(logits, torch.tensor(ans, device=device))
            Q.append(info["q_sem"])
            g_acc = g_acc + info["g"].mean().item()
        ans_loss = ans_loss / n_facts
        g_mean = g_acc / n_facts
        Q = torch.stack(Q, dim=1)               # [B, n_facts, d_key]

        # cross-batch InfoNCE retrieval: every fact's query must retrieve its OWN
        # key among ALL keys in the batch. This (on (S,R)-token keys) is what
        # actually trains -- a single/in-episode contrastive or the pooled-feature
        # key collapses (matched & random query.key become identical). Validated
        # standalone to retrieval@1 ~0.91.
        Qf = Q.reshape(-1, Q.size(-1))                      # [B*nf, d_key]
        Kf = K.reshape(-1, K.size(-1))
        sim = Qf @ Kf.t() / 0.07                            # [B*nf, B*nf]
        labels = torch.arange(Qf.size(0), device=device)
        retr_loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

        # key orthogonality (decorrelate keys within episode)
        gram = torch.einsum("bnd,bmd->bnm", K, K)
        eye = torch.eye(n_facts, device=device)[None]
        orth_loss = ((gram - eye) ** 2).mean()

        # bucket balance (spread product-key usage)
        balance_loss = ((usage - usage.mean()) ** 2).mean()

        # locality: an unrelated query's answer should be unchanged by the
        # injection. Baseline = the SAME read with no injection (g=0):
        # lm_head(inject_ln(H_ans)). This isolates the memory's effect from the
        # inject_ln reshaping (comparing to raw core would penalise the LN too).
        loc_qs, loc_ans = [], []
        for b in range(B):
            qids, a = world.render_unrelated_query(episodes[b][0][0])
            loc_qs.append(qids); loc_ans.append(a)
        ids, lengths = pad_batch(loc_qs, world.i("<pad>"), device)
        loc_logits, loc_info = mem.read_logits(M, alloc, ids, lengths)
        with torch.no_grad():
            base_logits = mem.core.lm_head(mem.inject_ln(loc_info["H_ans"]))
        loc_loss = F.kl_div(F.log_softmax(loc_logits, -1),
                            F.softmax(base_logits, -1), reduction="batchmean")

        # --- Step 2: conflict versioning sub-batch ---
        # write o_before (time 0) then o_now (time 1) for a (s,r); the now/before
        # query must route to the right version (trains ctx_enc + value path).
        conf_loss = _conflict_loss(mem, world, B, device, tau)
        # Step 4: commit gate must reject untrustworthy conflicting writes.
        safe_loss = _safety_loss(mem, world, B, device, tau)

        loc_w = 0.0 if warming else lam["locality"]   # locality only after warmup
        loss = (lam["answer"] * ans_loss + lam["retrieve"] * retr_loss
                + lam["orth"] * orth_loss + lam["balance"] * balance_loss
                + loc_w * loc_loss + lam["conflict"] * conf_loss
                + lam["safety"] * safe_loss)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            log(f"  [omega0] step {step:4d} nf={n_facts} loss {loss.item():.3f} "
                f"(ans {ans_loss.item():.3f} retr {retr_loss.item():.3f} "
                f"loc {loc_loss.item():.3f} cfl {conf_loss.item():.3f} "
                f"sft {safe_loss.item():.3f} g {g_mean:.3f})")
    mem.relevance_enabled = True
    mem.eval()
    return mem


@torch.no_grad()
def eval_capsule(mem: CapsuleMemory, world, *, n_facts, episodes_n=64, n_para=4,
                 device="cpu"):
    """Exact-match object accuracy over paraphrase queries + locality score."""
    mem.relevance_enabled = True
    mem.eval()
    correct = total = 0
    loc_match = loc_total = 0
    # process episodes in one batch
    episodes = [world.sample_episode_facts(n_facts, hard_negative_ratio=0.5)
                for _ in range(episodes_n)]
    M, alloc, _, _, _ = _write_all(mem, episodes, n_facts, device, tau=1.0,
                                   hard=True, training=False)
    for j in range(n_facts):
        for t in range(n_para):
            qs, ans = [], []
            for b in range(episodes_n):
                qids, a = world.render_query(episodes[b][j], template_idx=t % 4)
                qs.append(qids); ans.append(a)
            ids, lengths = pad_batch(qs, world.i("<pad>"), device)
            logits, _ = mem.read_logits(M, alloc, ids, lengths)
            pred = logits.argmax(-1)
            correct += (pred == torch.tensor(ans, device=device)).sum().item()
            total += len(ans)
    # locality: an unrelated answer should be unchanged by the injection.
    # Compare the read against the SAME read with the gate closed (no
    # injection) -- this measures the memory's effect, not the inject_ln.
    qs = []
    for b in range(episodes_n):
        qids, _ = world.render_unrelated_query(episodes[b][0][0]); qs.append(qids)
    ids, lengths = pad_batch(qs, world.i("<pad>"), device)
    mem_logits, info = mem.read_logits(M, alloc, ids, lengths)
    base_logits = mem.core.lm_head(mem.inject_ln(info["H_ans"]))
    loc_match = (mem_logits.argmax(-1) == base_logits.argmax(-1)).sum().item()
    loc_total = episodes_n
    return dict(acc=correct / total, locality=loc_match / loc_total)


@torch.no_grad()
def eval_conflict(mem: CapsuleMemory, world, *, episodes_n=128, device="cpu"):
    """Step 2: write a conflicting (s,r) -- o_before@t=0 then o_now@t=1 -- and
    check the now/before query routes to the right version (non-destructive)."""
    mem.relevance_enabled = True
    mem.eval()
    groups = [world.sample_conflict_episode(1)[0] for _ in range(episodes_n)]
    M, alloc = mem.empty_bank(episodes_n, device)
    for which, t in (("o_before", 0.0), ("o_now", 1.0)):
        seqs = [world.render_statement((g["s"], g["r"], g[which])) for g in groups]
        ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
        M, alloc, _ = mem.write(M, alloc, ids, lengths, hard=True, training=False, time=t)
    out = {}
    preds = {}
    for ctx, key in (("now", "o_now"), ("before", "o_before")):
        qs, ans = [], []
        for g in groups:
            qids, a = world.render_query_versioned(g["s"], g["r"], g[key], ctx)
            qs.append(qids); ans.append(a)
        ids, lengths = pad_batch(qs, world.i("<pad>"), device)
        logits, _ = mem.read_logits(M, alloc, ids, lengths)
        preds[ctx] = logits.argmax(-1)
        out[ctx] = (preds[ctx] == torch.tensor(ans, device=device)).float().mean().item()
    out["routing_fail"] = (preds["now"] == preds["before"]).float().mean().item()
    return out


@torch.no_grad()
def eval_safety(mem: CapsuleMemory, world, *, episodes_n=128, device="cpu"):
    """Step 4: write a reliable fact, then attack with a conflicting UNRELIABLE
    update. Measure (a) reliable recall after the attack (should survive),
    (b) the gate's admission g_commit for reliable vs unreliable writes."""
    mem.relevance_enabled = True
    mem.eval()
    facts = [world.sample_fact() for _ in range(episodes_n)]
    M, alloc = mem.empty_bank(episodes_n, device)
    seqs = [world.render_statement(f) for f in facts]
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    M, alloc, gi = mem.write(M, alloc, ids, lengths, hard=True, training=False,
                             time=1.0, trust=1.0)
    g_rel = gi["g_commit"].mean().item()
    bad = [(s, r, _other_obj(world, o)) for (s, r, o) in facts]
    seqs = [world.render_statement(b) for b in bad]
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    M, alloc, gi = mem.write(M, alloc, ids, lengths, hard=True, training=False,
                             time=1.0, trust=0.0)
    g_unrel = gi["g_commit"].mean().item()
    qs, ans = [], []
    for f in facts:
        qids, a = world.render_query(f); qs.append(qids); ans.append(a)
    ids, lengths = pad_batch(qs, world.i("<pad>"), device)
    logits, _ = mem.read_logits(M, alloc, ids, lengths)
    protected = (logits.argmax(-1) == torch.tensor(ans, device=device)).float().mean().item()
    return dict(protected=protected, g_reliable=g_rel, g_unreliable=g_unrel)


def _seq_lora_recall(core, world, sessions, device, rank=8, steps=40, lr=1e-2):
    """Sequential LoRA (the parametric rival): one adapter fine-tuned session by
    session. Returns per-session recall measured with the FINAL adapter -- old
    sessions are catastrophically forgotten as later sessions overwrite it."""
    import torch.nn as nn
    d = core.d_model

    def incontext(f):
        stmt = world.render_statement(f)[1:-1]
        qids, a = world.render_query(f)
        return [world.i("<bos>")] + stmt + [world.i("<sep>")] + qids[1:], a

    with torch.enable_grad():                  # eval_lifelong runs under no_grad
        A = torch.zeros(d, rank, device=device, requires_grad=True)
        Bm = torch.zeros(rank, d, device=device, requires_grad=True)
        nn.init.normal_(A, std=0.02)
        opt = torch.optim.Adam([A, Bm], lr=lr)
        for sess in sessions:                  # learn sessions in order
            seqs, ans = zip(*[incontext(f) for f in sess])
            ids, lengths = pad_batch(list(seqs), world.i("<pad>"), device)
            rows = torch.arange(ids.size(0), device=device)
            ans_t = torch.tensor(ans, device=device)
            for _ in range(steps):
                h = core.hidden(ids)[rows, lengths - 1].detach()
                logits = core.lm_head(h + (h @ A) @ Bm)
                loss = F.cross_entropy(logits, ans_t)
                opt.zero_grad(); loss.backward(); opt.step()

    recalls = []
    with torch.no_grad():
        for sess in sessions:                  # recall each session at the end
            qs, qans = [], []
            for f in sess:
                qids, a = world.render_query(f); qs.append(qids); qans.append(a)
            ids, lengths = pad_batch(qs, world.i("<pad>"), device)
            rows = torch.arange(ids.size(0), device=device)
            h = core.hidden(ids)[rows, lengths - 1]
            logits = core.lm_head(h + (h @ A) @ Bm)
            recalls.append((logits.argmax(-1) == torch.tensor(qans, device=device)).float().mean().item())
    return recalls


@torch.no_grad()
def eval_lifelong(mem: CapsuleMemory, world, *, n_sessions=8, per_session=6,
                  n_seq=24, device="cpu"):
    """Lifelong sequence: learn n_sessions of `per_session` facts in order, then
    recall EACH session at the end. Capsule (external memory) should be flat
    across session age; the sequential-LoRA rival should decay for old sessions.
    Returns (cap_by_session, lora_by_session) averaged over n_seq sequences."""
    mem.relevance_enabled = True; mem.eval()
    N = n_sessions * per_session
    # ---- capsule: write all N facts into one bank (batched over n_seq) ----
    seqs = []
    for _ in range(n_seq):
        subs = world.rng.sample(range(world.cfg.n_entities), N)
        facts = [(s, world.rng.randrange(world.cfg.n_relations),
                  world.rng.randrange(world.cfg.n_objects)) for s in subs]
        seqs.append(facts)                          # facts[0:per_session]=session0, ...
    M, alloc = mem.empty_bank(n_seq, device)
    for j in range(N):
        st = [world.render_statement(seqs[b][j]) for b in range(n_seq)]
        ids, lengths = pad_batch(st, world.i("<pad>"), device)
        M, alloc, _ = mem.write(M, alloc, ids, lengths, hard=True, training=False,
                                time=(j + 1) / N, trust=1.0)
    cap_by_session = []
    for s in range(n_sessions):
        correct = total = 0
        for j in range(s * per_session, (s + 1) * per_session):
            qs, ans = [], []
            for b in range(n_seq):
                qids, a = world.render_query(seqs[b][j]); qs.append(qids); ans.append(a)
            ids, lengths = pad_batch(qs, world.i("<pad>"), device)
            logits, _ = mem.read_logits(M, alloc, ids, lengths)
            correct += (logits.argmax(-1) == torch.tensor(ans, device=device)).sum().item()
            total += n_seq
        cap_by_session.append(correct / total)
    # ---- sequential LoRA rival (per sequence) ----
    lora_acc = [[] for _ in range(n_sessions)]
    for b in range(n_seq):
        sessions = [seqs[b][s * per_session:(s + 1) * per_session] for s in range(n_sessions)]
        for s, r in enumerate(_seq_lora_recall(mem.core, world, sessions, device)):
            lora_acc[s].append(r)
    lora_by_session = [sum(x) / len(x) for x in lora_acc]
    return cap_by_session, lora_by_session


def eval_baseline(BaselineCls, core, world, *, n_facts, episodes_n=64, device="cpu"):
    # NOTE: not under no_grad -- the LoRA baseline trains an adapter per episode.
    bl = BaselineCls(core, world)
    accs = []
    for _ in range(episodes_n):
        facts = world.sample_episode_facts(n_facts, hard_negative_ratio=0.5)
        accs.append(bl.eval_episode(facts, device))
    return sum(accs) / len(accs)
