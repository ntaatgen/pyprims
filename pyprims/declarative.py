"""Declarative memory and the retrieval module (port of Declarative.swift)."""
from __future__ import annotations

import math

from .chunk import Chunk
from .values import describe, is_chunk, values_equal

DEFAULTS = dict(
    base_level_decay=0.5,                 # bll:
    optimized_learning=True,              # ol:
    maximum_associative_strength=3.0,     # mas:
    goal_activation=1.0,                  # ga:   (W)
    input_activation=0.0,                 # input-activation:
    retrieval_activation=0.0,             # retrieval-activation:
    imaginal_activation=0.0,              # wm-activation: / imaginal-activation:
    retrieval_threshold=-2.0,             # rt:
    activation_noise=0.2,                 # ans:
    default_operator_assoc=3.0,           # default-operator-assoc:
    default_inter_operator_assoc=1.0,     # default-inter-operator-assoc:
    default_operator_self_assoc=-1.0,     # default-operator-self-assoc:
    mismatch_penalty=5.0,                 # mp:
    goal_spreading_by_activation=False,   # goal-chunk-spreads:
    latency_factor=0.2,                   # lf:
    goal_operator_learning=False,         # skill-operator-learning: / goal-operator-learning:
    context_operator_learning=False,      # context-operator-learning:
    extended_assocs=False,                # extended-associations:
    operator_baselevel_learning=False,    # operator-bll:
    inter_operator_learning=False,        # inter-operator-learning:
    beta=0.1,                             # beta:
    exploration_exploitation_factor=0.0,  # explore-exploit:
    declarative_buffer_stuffing=False,    # declarative-buffer-stuffing:
    retrieval_reinforces=False,           # retrieval-reinforces:
    default_activation=0.0,               # default-activation:  (None = no floor)
    partial_matching=False,               # pm:
    new_partial_matching_pow=None,        # new-pm-pow:
    new_partial_matching_exp=None,        # new-pm-exp: (QUIRK: Swift sets the pow field)
    blending=False,                       # blending:
)


class Declarative:
    def __init__(self, model):
        self.model = model
        self.chunks: dict[str, Chunk] = {}
        self.conflict_set: list[tuple[Chunk, float]] = []
        self.finsts: list[str] = []
        self.retrieve_busy = False
        self.retrieve_error = False
        self.last_mismatch: dict[str, float] = {}   # embedded retrieval: partial-matching penalties
        self.last_blend: dict[str, dict] = {}       # embedded blending: per-slot clean-up results
        self.last_decode: dict = {}                 # holographic: slot -> (decoded value, similarity)
        self.version = 0                            # bumped when DM content changes (caches)
        self.set_parameters_to_default()

    def set_parameters_to_default(self):
        for k, v in DEFAULTS.items():
            setattr(self, k, v)

    # ------------------------------------------------------------ adding
    def duplicate_chunk(self, chunk: Chunk):
        for c1 in self.chunks.values():
            if c1.same_content(chunk):
                return c1
        return None

    def add_to_dm(self, chunk: Chunk) -> Chunk:
        """Add a chunk; if an identical chunk exists, reinforce that one."""
        dup = self.duplicate_chunk(chunk)
        if dup is not None:
            dup.add_reference()
            dup.merge_assocs(chunk)
            if chunk.defined_in and chunk.defined_in[0] not in dup.defined_in:
                dup.defined_in.append(chunk.defined_in[0])
            if chunk.fixed_activation is not None and dup.fixed_activation is not None:
                dup.fixed_activation = max(chunk.fixed_activation, dup.fixed_activation)
            return dup
        if chunk.creation_time is None:
            chunk.start_time()
        self.chunks[chunk.name] = chunk
        self.version += 1
        for v in list(chunk.slotvals.values()):
            if is_chunk(v):
                v.fan += 1
        return chunk

    def add_or_update(self, chunk: Chunk) -> Chunk:
        existing = self.chunks.get(chunk.name)
        return existing if existing is not None else self.add_to_dm(chunk)

    def purge(self, type_name: str):
        for name in [n for n, c in self.chunks.items() if c.type == type_name]:
            del self.chunks[name]
        self.version += 1

    def strings_to_chunks(self):
        """Replace text slot values that name a DM chunk by that chunk.

        QUIRK: this does not update fan counts, so a chunk referenced before it
        was defined has a lower fan than one referenced after.
        """
        for chunk in list(self.chunks.values()):
            for slot, v in list(chunk.slotvals.items()):
                if isinstance(v, str) and v in self.chunks:
                    chunk.slotvals[slot] = self.chunks[v]
        self.version += 1

    # ------------------------------------------------------------ finsts
    def clear_finsts(self):
        self.finsts = []

    def add_to_finsts(self, c: Chunk):
        self.finsts.append(c.name)

    # ------------------------------------------------------------ retrieval
    def latency(self, activation: float) -> float:
        return self.latency_factor * math.exp(-activation)

    def _matches(self, candidate: Chunk, request: Chunk) -> bool:
        compare = self.model.compare            # values_equal in the symbolic representation
        hard = getattr(self.model.representation, "hard_slots", ())
        for slot, value in request.slotvals.items():
            v1 = candidate.slot_value(slot)
            if v1 is None:
                return False
            if slot in hard:
                if not values_equal(v1, value):
                    return False
            elif not compare(v1, value):
                return False
        return True

    def retrieve(self, request: Chunk, add_to_conflict_set: bool = True):
        """Return (latency, chunk or None).  Fills ``conflict_set``."""
        if self.model.representation.soft_retrieval:
            return self._soft_retrieve(request)
        self.retrieve_error = False
        best = None
        best_act = self.retrieval_threshold
        self.conflict_set = []
        for ch in list(self.chunks.values()):
            if ch.name in self.finsts:
                continue
            if not self._matches(ch, request):
                continue
            act = ch.activation()          # noise is cached per time step
            self.conflict_set.append((ch, act))
            if self.exploration_exploitation_factor != 0.0:
                # QUIRK (Swift): activation() is evaluated three times here, and with
                # explore-exploit each evaluation draws fresh Sji noise.
                if ch.activation() > best_act:
                    best_act = ch.activation()
                    best = ch
            elif act > best_act:
                best_act = act
                best = ch
        if best_act > self.retrieval_threshold:
            return self.latency(best_act), best
        self.retrieve_error = True
        return self.latency(self.retrieval_threshold), None

    def mismatch_function(self, x, y):
        if describe(x) == describe(y):
            return 0.0
        try:
            fx, fy = float(describe(x)), float(describe(y))
        except ValueError:
            return -1.0
        # mismatchNumbers: only integers are similar
        try:
            int(describe(x)); int(describe(y))
        except ValueError:
            return -1.0
        mx, mn = max(fx, fy), min(fx, fy)
        mm = 0 - (mx - mn) / 10
        return mm if mm >= -1 else -1.0

    def partial_retrieve(self, request: Chunk):
        best = None
        best_act = self.retrieval_threshold
        self.conflict_set = []
        for ch in list(self.chunks.values()):
            mismatch = 0.0
            ok = True
            for slot, value in request.slotvals.items():
                v1 = ch.slotvals.get(slot)
                if v1 is None:
                    ok = False
                    break
                if not values_equal(v1, value):
                    mm = self.mismatch_function(v1, value)
                    if mm is None:
                        ok = False
                        break
                    mismatch += mm
            if not ok:
                continue
            if self.new_partial_matching_pow is not None:
                act = ch.activation() + mismatch * self.mismatch_penalty * (ch.references ** self.new_partial_matching_pow)
            elif self.new_partial_matching_exp is not None:
                act = ch.activation() + mismatch * self.mismatch_penalty * (self.new_partial_matching_exp ** ch.references)
            else:
                act = ch.activation() + mismatch * self.mismatch_penalty
            self.conflict_set.append((ch, act))
            if act > best_act:
                best_act = act
                best = ch
        if best_act > self.retrieval_threshold:
            return self.latency(best_act), best
        self.retrieve_error = True
        return self.latency(self.retrieval_threshold), None

    # ------------------------------------------------------------ embedded representation
    def _cue_penalty(self, ch: Chunk, request: Chunk, rep, mp: float):
        """Partial-matching penalty of ch for the cues in request, or None if excluded.

        Hard slots must be identical; other cue slots add mp * (sim - 1).
        With mp = inf any non-identical cue excludes the chunk (the symbolic limit).
        """
        if rep.holographic and rep.is_regular(ch):
            return rep.cue_penalty(ch, request, mp)
        mismatch = 0.0
        for slot, value in request.slotvals.items():
            v1 = ch.slot_value(slot)
            if v1 is None:
                return None
            if values_equal(v1, value):
                continue
            if slot in rep.hard_slots or mp == math.inf:
                return None
            mismatch += rep.sim(v1, value) - 1.0
        return mp * mismatch if mismatch != 0.0 else 0.0

    def _soft_retrieve(self, request: Chunk):
        """Retrieval as partial matching on embedding similarity.

        A_i = base level + spreading + noise + mp * sum_k (sim(cue_k, value_ik) - 1)
        """
        m = self.model
        rep = m.representation
        mp = rep.penalty(m)
        self.retrieve_error = False
        best = None
        best_act = self.retrieval_threshold
        self.conflict_set = []
        self.last_mismatch = {}
        for ch in list(self.chunks.values()):
            if ch.name in self.finsts:
                continue
            pen = self._cue_penalty(ch, request, rep, mp)
            if pen is None:
                continue
            act = ch.activation() + pen
            self.conflict_set.append((ch, act))
            if pen != 0.0:
                self.last_mismatch[ch.name] = pen
            if self.exploration_exploitation_factor != 0.0:
                if ch.activation() + pen > best_act:      # Swift quirk: fresh Sji noise per call
                    best_act = ch.activation() + pen
                    best = ch
            elif act > best_act:
                best_act = act
                best = ch
        if best_act > self.retrieval_threshold:
            return self.latency(best_act), best
        self.retrieve_error = True
        return self.latency(self.retrieval_threshold), None

    def _vector_blend(self, request: Chunk):
        """Blending over vectors: every non-cue slot is the clean-up of the
        activation-weighted mean of the candidates' value vectors.

        Candidates are the chunks above the retrieval threshold (activation
        without noise, plus the partial-matching penalty in soft retrieval);
        weights are softmax(A / t), t = sqrt(2) * ans unless set.
        """
        m = self.model
        rep = m.representation
        emb = rep.embedding
        mp = rep.penalty(m) if rep.soft_retrieval else math.inf
        self.conflict_set = []
        self.last_mismatch = {}
        self.last_blend = {}
        cands = []
        for ch in list(self.chunks.values()):
            pen = self._cue_penalty(ch, request, rep, mp)
            if pen is None:
                continue
            a = ch.activation_without_noise() + pen
            self.conflict_set.append((ch, a))
            if pen != 0.0:
                self.last_mismatch[ch.name] = pen
            if a > self.retrieval_threshold:
                cands.append((ch, a))
        if not cands:
            self.retrieve_error = True
            return self.latency(self.retrieval_threshold), None
        acts = [a for _, a in cands]
        best_i = max(range(len(cands)), key=lambda i: acts[i])
        t = rep.temperature if rep.temperature is not None else math.sqrt(2) * self.activation_noise
        if t <= 0:
            weights = [1.0 if i == best_i else 0.0 for i in range(len(cands))]
        else:
            top = max(acts)
            ex = [math.exp((a - top) / t) for a in acts]
            z = sum(ex)
            weights = [e / z for e in ex]
        best = cands[best_i][0].copy_chunk()
        slots = []
        for ch, _ in cands:
            for s in ch.slotvals:
                if s not in request.slotvals and s not in slots:
                    slots.append(s)
        for slot in slots:
            pairs = [(w, ch.slotvals[slot], ch) for (ch, _), w in zip(cands, weights) if slot in ch.slotvals]
            total = sum(w for w, _, _ in pairs)
            if total <= 0:
                continue
            # holographic chunks: average the slot vectors read back from the order vectors
            vec = sum((w / total) * (rep.unbind(ch, slot) if rep.holographic and rep.is_regular(ch)
                                     else emb.vector(v)) for w, v, ch in pairs)
            if rep.cleanup == "vocabulary":
                key, s = emb.cleanup(vec, emb.vocabulary())
                value = m.string_to_value(key)
            else:
                distinct = {}
                for _, v, _ in pairs:
                    distinct.setdefault(describe(v), v)
                value, s = emb.cleanup(vec, list(distinct.values()))
            best.set_slot_value(slot, value)
            self.last_blend[slot] = dict(value=describe(value), similarity=s,
                                         weights={ch.name: w for (ch, _), w in zip(cands, weights)})
        return self.latency(acts[best_i]), best

    def blended_retrieve(self, request: Chunk):
        if self.model.representation.vector_blending:
            return self._vector_blend(request)
        total = 0.0
        best = None
        best_act = self.retrieval_threshold
        self.conflict_set = []
        for ch in list(self.chunks.values()):
            if not self._matches(ch, request):
                continue
            a = ch.activation_without_noise()
            e = math.exp(a / self.activation_noise)
            self.conflict_set.append((ch, e))
            total += e
            if a > best_act:
                best_act = a
                best = ch
        if not self.conflict_set or best is None:
            # QUIRK: Swift crashes when matches exist but none is above threshold
            self.retrieve_error = True
            return self.latency(self.retrieval_threshold), None
        best = best.copy_chunk()
        for slot, value in list(best.slotvals.items()):
            if slot not in request.slotvals and isinstance(value, float):
                result = 0.0
                for mc, e in self.conflict_set:
                    v = mc.slotvals.get(slot)
                    if isinstance(v, float):
                        result += v * (e / total)
                best.set_slot_value(slot, result)
        return self.latency(best_act), best

    def action(self) -> float:
        """The retrieval module: answer the request in RT (retrievalR)."""
        m = self.model
        stuff = m.buffers.get("retrievalR") is None
        request = m.buffers.get("retrievalR")
        if request is None:
            request = Chunk("emptyRetrieval", m)
            request.set_slot("isa", "fact")
        if m.representation.vector_blending:
            lat, result = self.blended_retrieve(request)
        elif self.partial_matching:
            lat, result = self.partial_retrieve(request)
        elif self.blending:
            lat, result = self.blended_retrieve(request)
        else:
            lat, result = self.retrieve(request)
        if result is not None:
            if stuff:
                m.add_to_trace(f"Stuffing retrieval buffer {result.name} (latency = {lat:.3f})", 2)
            else:
                m.add_to_trace(f"Retrieving {result.name} (latency = {lat:.3f})", 2)
                if self.retrieval_reinforces:
                    result.add_reference()
            rep = m.representation
            if rep.holographic and not rep.vector_blending and rep.is_regular(result):
                result, self.last_decode = rep.decoded_chunk(result)   # read back from the vector
            m.buffers["retrievalH"] = result
        elif not stuff:
            m.add_to_trace("Retrieval failure", 2)
            fail = Chunk("RetrievalFailure", m)
            fail.set_slot("slot1", "error")
            m.buffers["retrievalH"] = fail
        m.buffers.pop("retrievalR", None)
        return 0.0 if (result is None and stuff) else lat

    # ------------------------------------------------------------ tree navigation
    def push(self, slot: str) -> bool:
        m = self.model
        old = m.buffers.get("retrievalH")
        if old is None:
            return False
        v = old.slotvals.get(slot)
        if not is_chunk(v):
            return False
        v.parent = old.name
        m.buffers["retrievalH"] = v
        return True

    def pop(self) -> bool:
        m = self.model
        cur = m.buffers.get("retrievalH")
        if cur is None or cur.parent is None:
            return False
        m.buffers["retrievalH"] = self.chunks[cur.parent]
        return True
