"""COMPOSITION SOLVABILITY GATE (pre-arms) — before comparing grow vs fixed arms on
cross-stream composition, verify the task is *achievable at all* by an upper-bound model.

Motivation (converged design, qa/codex + qa/claude 2026-07-04): latent 2-hop composition
(no CoT, no memory, chain A->B and B->C into A->C inside one forward pass) is a known hard
regime at 0.5B. If even a maximally generous upper bound can't compose, then all grow/fixed
arms would be 0-vs-0 and MUST NOT be read as "growth is useless" — the task is just
unreachable-by-consolidation. So we gate first.

  Gate A (no-memory single-step-only upper bound): directly gold-supervise ALL params of the
    base model on ALL single-step edges (seen+para), NO 2-hop targets. Then, no-memory, no-CoT,
    eval held-out 2-hop A->C. Report single-step recall (must be ~1.0 first) + A->C vs shuffle
    control vs random-init hard chance.
  Gate B (direct-2hop sanity, only if A fails): train single-step edges + 2-hop queries on a
    TRAIN split of chains; eval held-out-split 2-hop. Tells apart "format unlearnable" (B fails)
    from "single-step-only consolidation doesn't induce latent chaining" (B passes) -> the latter
    means we need a bridge curriculum, NOT a growth-arms comparison.

Leak-safety is a GENERATOR CONTRACT (asserts are only a backstop): training completions contain
ONLY single-step targets; the 2-hop answer C never appears as a completion to any prompt whose
subject is A. Intermediate B never appears in the 2-hop prompt.

  python -m s2.composition_gate
  env: CG_LEN(3) CG_STEPS(1000) CG_GATEB_STEPS(1000) CG_SEEDS(2) CG_LR(2e-5) CG_BS(32)
       CG_GROW(0) CG_GATEA_THR(0.30)
"""
from __future__ import annotations
import os
import time
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from s0.qwen_grow import grow_qwen
from s0.qwen_growcap import single_tok_names
from s0.qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
LEN = int(os.environ.get("CG_LEN", 3))               # chain length; L=3 -> only 2-hop
STEPS = int(os.environ.get("CG_STEPS", 1000))
GATEB_STEPS = int(os.environ.get("CG_GATEB_STEPS", 1000))
SEEDS = int(os.environ.get("CG_SEEDS", 2))
LR = float(os.environ.get("CG_LR", 2e-5))
BS = int(os.environ.get("CG_BS", 32))
GROW = int(os.environ.get("CG_GROW", 0))             # >0 -> grow identity layers, still full-FT
GATEA_THR = float(os.environ.get("CG_GATEA_THR", 0.30))  # A->C accuracy to call Gate A a pass


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None

    names = [n for n in single_tok_names(tok) if one_tok(n) is not None]
    big = [f"{f} {l}" for f in FIRST for l in LAST]
    print(f"COMPOSITION GATE ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"L={LEN} steps={STEPS} seeds={SEEDS} lr={LR} grow=+{GROW}L | single-tok pool={len(names)}",
          flush=True)

    # ---- query templates ----
    def q_seen(x):
        return f"{x}'s friend is"

    def q_para(x):
        return f"The friend of {x} is"

    def q2_seen(x):
        return f"{x}'s friend's friend is"

    def q2_para(x):
        return f"The friend of the friend of {x} is"

    def q3_seen(x):
        return f"{x}'s friend's friend's friend is"

    def build_task(seed):
        rng = random.Random(4000 + seed)
        nm = names[:]; rng.shuffle(nm)
        bg = big[:]; rng.shuffle(bg)
        chains = []; ni = 0; bi = 0
        # each chain: [source(multi-tok), n1(single), ... n_{L-1}(single)]
        while ni + (LEN - 1) <= len(nm):
            src = bg[bi]; bi += 1
            nodes = [src] + nm[ni:ni + (LEN - 1)]; ni += (LEN - 1)
            chains.append(nodes)
        # single-step edges with stream assignment (consecutive edges -> different streams)
        edges = []  # (subject, gold_name, stream_idx, chain_idx, hop_pos)
        for ci, c in enumerate(chains):
            for i in range(LEN - 1):
                edges.append((c[i], c[i + 1], i, ci, i))
        # held-out multi-hop endpoint pairs (never in training completions)
        pairs2 = [(c[0], c[2]) for c in chains]                       # 2-hop A->C
        pairs3 = [(c[0], c[3]) for c in chains] if LEN >= 4 else []   # 3-hop A->D
        # ---- CONTRACT asserts (backstop only) ----
        for c in chains:
            for x in c[1:]:
                assert one_tok(x) is not None, f"non-source node not single-token: {x}"
            # C's token never equals the single-step completion of subject A
            assert one_tok(c[2]) != one_tok(c[1]), "2-hop answer collides with 1-hop answer"
        # no held-out answer token appears as a completion of a prompt with the same subject
        subj_to_completion = {}
        for (s, g, *_rest) in edges:
            subj_to_completion.setdefault(s, set()).add(one_tok(g))
        for (a, cc) in pairs2:
            assert one_tok(cc) not in subj_to_completion.get(a, set()), "LEAK: A->C in single-step"
        return chains, edges, pairs2, pairs3

    def train_items(edges):
        items = []
        for (s, g, *_r) in edges:
            gid = one_tok(g)
            items.append((q_seen(s), gid))
            items.append((q_para(s), gid))
        return items

    def load(grow):
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
        if grow > 0:
            grow_qwen(m, grow)
        return m

    @torch.no_grad()
    def acc(m, items):
        if not items:
            return 0.0, 0.0
        ok = 0; probsum = 0.0
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            e = tok([p for p, _ in chunk], return_tensors="pt", padding=True).to(device)
            lg = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float()
            gold = torch.tensor([g for _, g in chunk], device=device)
            ok += (lg.argmax(-1) == gold).sum().item()
            probsum += F.softmax(lg, -1).gather(1, gold[:, None]).sum().item()
        return ok / len(items), probsum / len(items)

    def full_ft(m, items, steps, lr):
        for p in m.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        m.train()
        for _ in range(steps):
            batch = random.sample(items, min(BS, len(items)))
            e = tok([p for p, _ in batch], return_tensors="pt", padding=True).to(device)
            gold = torch.tensor([g for _, g in batch], device=device)
            lg = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
            loss = F.cross_entropy(lg, gold)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()

    def eval_gate_a(seed):
        chains, edges, pairs2, pairs3 = build_task(seed)
        # split edges by hop position for single-step diagnostics (seen AND para views)
        e_ab_s = [(q_seen(s), one_tok(g)) for (s, g, _st, _ci, hp) in edges if hp == 0]
        e_bc_s = [(q_seen(s), one_tok(g)) for (s, g, _st, _ci, hp) in edges if hp == 1]
        e_ab_p = [(q_para(s), one_tok(g)) for (s, g, _st, _ci, hp) in edges if hp == 0]
        e_bc_p = [(q_para(s), one_tok(g)) for (s, g, _st, _ci, hp) in edges if hp == 1]
        comp2_seen = [(q2_seen(a), one_tok(c)) for (a, c) in pairs2]
        comp2_para = [(q2_para(a), one_tok(c)) for (a, c) in pairs2]
        # shuffle control: DERANGEMENT (cyclic offset -> guaranteed no fixed point since targets distinct)
        tgt_ids = [one_tok(c) for (_a, c) in pairs2]
        sh = tgt_ids[1:] + tgt_ids[:1]
        comp2_shuf = [(q2_seen(a), sh[i]) for i, (a, _c) in enumerate(pairs2)]

        base_m = load(GROW)                              # base pretrained, UNTRAINED on this task
        bu, _ = acc(base_m, comp2_seen)
        del base_m; torch.cuda.empty_cache()

        m = load(GROW)
        t0 = time.time()
        full_ft(m, train_items(edges), STEPS, LR)
        wall = time.time() - t0
        ss_ab_s, _ = acc(m, e_ab_s); ss_bc_s, _ = acc(m, e_bc_s)
        ss_ab_p, _ = acc(m, e_ab_p); ss_bc_p, _ = acc(m, e_bc_p)
        c2s, c2s_p = acc(m, comp2_seen)
        c2p, _ = acc(m, comp2_para)
        c2sh, _ = acc(m, comp2_shuf)
        del m; torch.cuda.empty_cache()
        print(f"  [GateA seed {seed}] single-step seen A->B={ss_ab_s:.3f} B->C={ss_bc_s:.3f} | "
              f"para A->B={ss_ab_p:.3f} B->C={ss_bc_p:.3f} | "
              f"2hop seen={c2s:.3f} para={c2p:.3f} | derange-ctrl={c2sh:.3f} base-untrained={bu:.3f} | "
              f"goldC-prob={c2s_p:.3f} wall={wall:.0f}s", flush=True)
        return dict(ss_ab_seen=ss_ab_s, ss_bc_seen=ss_bc_s, ss_ab_para=ss_ab_p, ss_bc_para=ss_bc_p,
                    c2s=c2s, c2p=c2p, c2sh=c2sh, bu=bu, c2s_p=c2s_p, wall=wall,
                    n_chains=len(chains), n_edges=len(edges))

    def eval_gate_b(seed):
        chains, edges, pairs2, _p3 = build_task(seed)
        rng = random.Random(7 + seed)
        idx = list(range(len(chains))); rng.shuffle(idx)
        n_tr = int(0.7 * len(idx)); tr, te = set(idx[:n_tr]), set(idx[n_tr:])
        items = train_items(edges)                       # all single-step knowledge
        # + 2-hop supervision on TRAIN chains only
        items = items + [(q2_seen(chains[ci][0]), one_tok(chains[ci][2])) for ci in tr] \
                      + [(q2_para(chains[ci][0]), one_tok(chains[ci][2])) for ci in tr]
        held = [(q2_seen(chains[ci][0]), one_tok(chains[ci][2])) for ci in te]
        m = load(GROW)
        full_ft(m, items, GATEB_STEPS, LR)
        seen_tr, _ = acc(m, [(q2_seen(chains[ci][0]), one_tok(chains[ci][2])) for ci in tr])
        held_a, _ = acc(m, held)
        del m; torch.cuda.empty_cache()
        print(f"  [GateB seed {seed}] direct-2hop train-fit={seen_tr:.3f} held-out={held_a:.3f}", flush=True)
        return dict(train_fit=seen_tr, held=held_a)

    import json
    A = [eval_gate_a(s) for s in range(SEEDS)]
    mean = lambda k: sum(a[k] for a in A) / len(A)
    print(f"\n== GATE A (mean/{SEEDS} seeds), no-memory single-step-only upper bound ==")
    print(f"  single-step seen: A->B={mean('ss_ab_seen'):.3f} B->C={mean('ss_bc_seen'):.3f} | "
          f"para: A->B={mean('ss_ab_para'):.3f} B->C={mean('ss_bc_para'):.3f} (must be ~1.0 to trust 2hop)")
    print(f"  2-hop A->C : seen={mean('c2s'):.3f} para={mean('c2p'):.3f} | "
          f"derange-ctrl={mean('c2sh'):.3f} base-untrained={mean('bu'):.3f} goldC-prob={mean('c2s_p'):.3f}")
    ss_min = min(mean('ss_ab_seen'), mean('ss_bc_seen'), mean('ss_ab_para'), mean('ss_bc_para'))
    a_pass = mean('c2s') >= GATEA_THR and mean('c2s') > 3 * max(mean('c2sh'), mean('bu'), 1e-6) \
        and ss_min > 0.8
    print(f"  => GATE A {'PASS' if a_pass else 'FAIL'} "
          f"(thr={GATEA_THR}, need single-step(all views)>0.8 and 2hop>>control)")
    result = dict(
        script="s2/composition_gate.py",
        source_config=dict(model=NAME, L=LEN, steps=STEPS, gateb_steps=GATEB_STEPS,
                           seeds=SEEDS, lr=LR, bs=BS, grow=GROW, gateA_thr=GATEA_THR),
        n_chains=A[0]['n_chains'], n_edges=A[0]['n_edges'],
        single_step_seen_A_B=mean('ss_ab_seen'), single_step_seen_B_C=mean('ss_bc_seen'),
        single_step_para_A_B=mean('ss_ab_para'), single_step_para_B_C=mean('ss_bc_para'),
        compose_2hop_seen=mean('c2s'), compose_2hop_para=mean('c2p'),
        compose_2hop_derange_control=mean('c2sh'), base_untrained_on_task=mean('bu'),
        gold_C_meanprob=mean('c2s_p'), gateA_pass=bool(a_pass),
        leakage_assertions_passed=True, wall_clock_seconds=sum(a['wall'] for a in A),
        per_seed=A,
    )
    if not a_pass:
        print(f"\n== GATE B (single-step knowledge present; direct 2-hop on train chains) ==")
        B = [eval_gate_b(s) for s in range(SEEDS)]
        bt = sum(b['train_fit'] for b in B) / len(B); bh = sum(b['held'] for b in B) / len(B)
        print(f"  direct-2hop: train-fit={bt:.3f} held-out={bh:.3f}")
        result.update(gateB_train_fit=bt, gateB_heldout_seen=bh)
        if bh >= GATEA_THR:
            print("  => GATE B PASS: 'friend's friend' readout IS learnable given single-step knowledge.")
            print("     Interpretation: single-step-only consolidation does NOT induce latent chaining.")
            print("     Next: BRIDGE CURRICULUM (small 2-hop supervision / scratchpad), NOT growth arms yet.")
        else:
            print("  => GATE B FAIL: task format/readout itself is not learnable here.")
            print("     Next: change the task (relation phrasing / eval), do NOT run growth arms.")
    else:
        print("     Next: task is solvable by consolidation -> proceed to build grow/fixed arms.")
    out = os.environ.get("CG_OUT", "composition_gate_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nRESULT_JSON " + json.dumps(result))
    print(f"[wrote {out}]", flush=True)


if __name__ == "__main__":
    main()
