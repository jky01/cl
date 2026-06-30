"""Synthetic (subject, relation, object) world + word-level tokenizer.

CLAIM BOUNDARY (review point #1):
    v0 validates *combinatorial binding* over a KNOWN, fixed vocabulary --
    can an unseen (subject, relation, object) COMBINATION be written once and
    read back -- NOT acquisition of brand-new symbols. Entities/relations/
    objects are a fixed vocab; the frozen proxy core is pretrained on the
    template *syntax* with RANDOM bindings, so it cannot have memorised any
    particular fact. Novel-symbol generalisation is a later step, not Step 0.
"""

from __future__ import annotations
import random
from dataclasses import dataclass


# ---- vocabulary -----------------------------------------------------------
# Template/function words first, then the symbol banks. Keeping ids stable.
FUNC_WORDS = [
    "<pad>", "<bos>", "<eos>", "<sep>", "<ans>",
    "the", "of", "is", "in", "has", "which", "what", "tell", "me",
    "located", "called", "?", ".",
]
PAD, BOS, EOS, SEP, ANS = 0, 1, 2, 3, 4


@dataclass
class WorldConfig:
    n_entities: int = 200
    n_relations: int = 8
    n_objects: int = 200
    seed: int = 0


class World:
    """Owns the vocabulary, builds facts, and renders them to token ids."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.entities = [f"entity_{i}" for i in range(cfg.n_entities)]
        self.relations = [f"relation_{i}" for i in range(cfg.n_relations)]
        self.objects = [f"object_{i}" for i in range(cfg.n_objects)]

        vocab = list(FUNC_WORDS) + self.entities + self.relations + self.objects
        self.tok2id = {w: i for i, w in enumerate(vocab)}
        self.id2tok = vocab
        self.vocab_size = len(vocab)

        # cache id ranges so we can constrain the answer head to object tokens
        self.obj_id_lo = self.tok2id[self.objects[0]]
        self.obj_id_hi = self.tok2id[self.objects[-1]] + 1  # exclusive

    # -- ids -------------------------------------------------------------
    def i(self, w: str) -> int:
        return self.tok2id[w]

    def object_token_ids(self) -> list[int]:
        return list(range(self.obj_id_lo, self.obj_id_hi))

    # -- fact sampling ---------------------------------------------------
    def sample_fact(self):
        s = self.rng.randrange(self.cfg.n_entities)
        r = self.rng.randrange(self.cfg.n_relations)
        o = self.rng.randrange(self.cfg.n_objects)
        return (s, r, o)

    def sample_episode_facts(self, n_facts: int, hard_negative_ratio: float = 0.5):
        """Sample `n_facts` facts with DISTINCT subjects.

        With probability `hard_negative_ratio` the relations are forced to
        collide (same relation, different subject/object) -- the case most
        likely to confuse retrieval. This is deliberately part of the
        Omega-0 TRAINING distribution (review point #4) so that collision at
        eval-time (Step 1) is in-distribution, not a surprise.
        """
        subs = self.rng.sample(range(self.cfg.n_entities), n_facts)
        facts = []
        if self.rng.random() < hard_negative_ratio and n_facts > 1:
            r = self.rng.randrange(self.cfg.n_relations)        # shared relation
            for s in subs:
                o = self.rng.randrange(self.cfg.n_objects)
                facts.append((s, r, o))
        else:
            for s in subs:
                r = self.rng.randrange(self.cfg.n_relations)
                o = self.rng.randrange(self.cfg.n_objects)
                facts.append((s, r, o))
        return facts

    # -- rendering -------------------------------------------------------
    def render_statement(self, fact) -> list[int]:
        """Write form:  <bos> the relation of entity is object . <eos>"""
        s, r, o = fact
        toks = ["<bos>", "the", self.relations[r], "of", self.entities[s],
                "is", self.objects[o], ".", "<eos>"]
        return [self.i(t) for t in toks]

    # several paraphrase question forms; the answer is the object token that
    # the model must emit at the final (<ans>) position.
    _Q_TEMPLATES = [
        ["<bos>", "what", "is", "the", "R", "of", "S", "?", "<ans>"],
        ["<bos>", "the", "R", "of", "S", "is", "what", "?", "<ans>"],
        ["<bos>", "tell", "me", "the", "R", "of", "S", ".", "<ans>"],
        ["<bos>", "S", "has", "which", "R", "?", "<ans>"],
    ]

    def render_query(self, fact, template_idx: int | None = None):
        """Return (token_ids, answer_object_id). Answer is read at last pos."""
        s, r, o = fact
        if template_idx is None:
            template_idx = self.rng.randrange(len(self._Q_TEMPLATES))
        tmpl = self._Q_TEMPLATES[template_idx]
        ids = []
        for t in tmpl:
            if t == "R":
                ids.append(self.i(self.relations[r]))
            elif t == "S":
                ids.append(self.i(self.entities[s]))
            else:
                ids.append(self.i(t))
        return ids, self.i(self.objects[o])

    def render_unrelated_query(self, avoid_subject: int):
        """A locality probe: a question about a DIFFERENT subject whose fact
        was NOT written this episode. Used to check memory does not corrupt
        unrelated answers."""
        s = avoid_subject
        while s == avoid_subject:
            s = self.rng.randrange(self.cfg.n_entities)
        r = self.rng.randrange(self.cfg.n_relations)
        o = self.rng.randrange(self.cfg.n_objects)
        ids, ans = self.render_query((s, r, o))
        return ids, ans
