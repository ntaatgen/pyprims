"""Operators: retrieval, matching, execution, learning, compilation.

Port of Operator.swift.  An operator is a declarative chunk with slots
``condition`` and ``action`` (';'-separated PRIM strings) and ``slot1..n``
holding its constants (the C buffer).  It is retrieved from declarative
memory like any chunk; its activation is its base level plus spreading from
the skills in the goal (Sji = default-operator-assoc, or learned).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .chunk import Chunk
from .prim import parse_name, BUFFER_LETTER
from .values import describe, is_chunk


@dataclass
class PrimTuple:
    lhs_buffer: str
    lhs_slot: int
    rhs_buffer: str
    rhs_slot: int
    op: str


def _slot_num(slot):
    if slot is None or not slot.startswith("slot"):
        return -1
    try:
        return int(slot[4])            # QUIRK: Swift reads one digit only
    except (ValueError, IndexError):
        return -1


@dataclass
class Op:
    """Operator in a convenient form (Swift ``struct Op``)."""
    name: str
    constants: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Op":
        op = cls(chunk.name)
        for key, target in (("condition", op.conditions), ("action", op.actions)):
            s = chunk.slotvals.get(key)
            if s is None:
                continue
            for item in describe(s).split(";"):
                lb, ls, cop, rb, rs, _ = parse_name(item)
                target.append(PrimTuple("" if lb is None else BUFFER_LETTER[lb], _slot_num(ls),
                                        "" if rb is None else BUFFER_LETTER[rb], _slot_num(rs), cop))
        i = 1
        while f"slot{i}" in chunk.slotvals:
            op.constants.append(describe(chunk.slotvals[f"slot{i}"]))
            i += 1
        return op

    @staticmethod
    def _prim_string(t: PrimTuple) -> str:
        pp = t.op in (">>", "<<")
        lhs = t.lhs_buffer + str(t.lhs_slot) if t.lhs_slot >= 0 else ("nil" if not pp else t.lhs_buffer)
        rhs = t.rhs_buffer + str(t.rhs_slot) if t.rhs_slot >= 0 else ("nil" if not pp else t.rhs_buffer)
        return lhs + t.op + rhs

    def build_chunk(self, model) -> Chunk:
        chunk = Chunk(self.name, model)
        chunk.set_slot("isa", "operator")
        for i, value in enumerate(self.constants, start=1):
            chunk.set_slot(f"slot{i}", value)
        for key, items in (("condition", self.conditions), ("action", self.actions)):
            seen = []
            for t in items:
                p = self._prim_string(t)
                if p not in seen:
                    seen.append(p)
            if seen or key == "condition":
                chunk.set_slot_value(key, ";".join(seen))
        chunk.fixed_activation = model.dm.default_activation
        return chunk


class Operator:
    def __init__(self, model):
        self.model = model
        # (operator chunk, time it was selected, context triples)
        self.previous_operators: list = []

    def reset(self):
        pass

    # ------------------------------------------------------------ operator compilation (opc:)
    def compile_all(self):
        m = self.model
        if not m.operator_learning or len(self.previous_operators) <= 1:
            return
        for i in range(len(self.previous_operators) - 1):
            new = self.compile_operators(self.previous_operators[i][0], self.previous_operators[i + 1][0])
            if new is not None:
                c2 = m.dm.add_to_dm(new)
                c2.fixed_activation = self.previous_operators[i][0].fixed_activation
                self.previous_operators.append((c2, self.previous_operators[i][1], self.previous_operators[i][2]))
                m.add_to_trace(f"Adding or strengtening operator {c2.name}", 5)

    def compile_operators(self, op1: Chunk, op2: Chunk):
        o1 = Op.from_chunk(op1)
        o2 = Op.from_chunk(op2)
        has_v = has_ac = False
        for a1 in o1.actions:
            if a1.rhs_buffer == "AC":
                has_ac = True
            for a2 in o2.actions:
                if a1.op in (">>", "<<") or a2.op in (">>", "<<"):
                    return None
                if a2.lhs_buffer == "RT":
                    return None
                if a1.rhs_buffer == a2.rhs_buffer and a1.rhs_slot == a2.rhs_slot:
                    return None
        for c in o1.conditions:
            if c.op in (">>", "<<"):
                return None
            if c.lhs_buffer == "V" or c.rhs_buffer == "V":
                has_v = True
        for c in o2.conditions:
            if c.op in (">>", "<<"):
                return None
            if c.lhs_buffer == "V" or c.rhs_buffer == "V":
                has_v = True
            if c.lhs_buffer == "RT" and (c.op != "<>" or c.rhs_buffer != ""):
                return None
        if has_v and has_ac:
            return None
        new = Op(o1.name + "+" + o2.name)
        new.constants = list(o1.constants)
        for i, const in enumerate(o2.constants):
            if const in o1.constants:
                new_index = o1.constants.index(const) + 1
            else:
                new.constants.append(const)
                new_index = len(new.constants)
            for c in o2.conditions:
                if c.lhs_buffer == "C" and c.lhs_slot == i + 1:
                    c.lhs_slot = new_index
                if c.rhs_buffer == "C" and c.rhs_slot == i + 1:
                    c.rhs_slot = new_index
            for a in o2.actions:
                if a.lhs_buffer == "C" and a.lhs_slot == i + 1:
                    a.lhs_slot = new_index
        new.conditions = list(o1.conditions)
        for cond in o2.conditions:
            nc = PrimTuple(cond.lhs_buffer, cond.lhs_slot, cond.rhs_buffer, cond.rhs_slot, cond.op)
            hit = next((a for a in o1.actions if a.rhs_buffer == cond.lhs_buffer and a.rhs_slot == cond.lhs_slot), None)
            if hit is not None:
                if hit.lhs_buffer == "":
                    return None
                nc.lhs_buffer, nc.lhs_slot = hit.lhs_buffer, hit.lhs_slot
            hit = next((a for a in o1.actions if a.rhs_buffer == cond.rhs_buffer and a.rhs_slot == cond.rhs_slot), None)
            if hit is not None:
                nc.rhs_buffer, nc.rhs_slot = hit.lhs_buffer, hit.lhs_slot
            if nc.lhs_buffer != "" or nc.rhs_buffer != "":
                new.conditions.append(nc)
        new.actions = list(o1.actions) + list(o2.actions)
        return new.build_chunk(self.model)

    # ------------------------------------------------------------ association learning
    def update_operator_sjis(self, payoff: float):
        """Skill/context/inter-operator association learning on reward.

        For each operator used this trial:
            r = assoc * (payoff - (now - t_selected)) / reward
            S += beta (r - S)
        and, with operator-bll, a base-level reference when r > 0.
        """
        m = self.model
        dm = m.dm
        try:
            if not ((dm.goal_operator_learning or dm.inter_operator_learning or dm.context_operator_learning)
                    and m.reward != 0.0):
                return
            goal = m.former_buffers.get("goal")
            if goal is None:
                return
            goals = []
            i = 1
            while f"slot{i}" in goal.slotvals:
                v = goal.slotvals[f"slot{i}"]
                if is_chunk(v) and v not in goals:
                    goals.append(v)
                i += 1
            if not goals:
                return
            prev = None
            for op_chunk, op_time, context in self.previous_operators:
                goal_r = dm.default_operator_assoc * (payoff - (m.time - op_time)) / m.reward
                inter_r = dm.default_inter_operator_assoc * (payoff - (m.time - op_time)) / m.reward
                if dm.context_operator_learning:
                    for bname, sname, ch in context:
                        key = f"{bname}%{sname}%{ch.name}"
                        a = op_chunk.assocs.setdefault(key, [0.0, 0])
                        a[0] += dm.beta * (goal_r - a[0])
                        a[1] += 1
                        if goal_r > 0 and dm.operator_baselevel_learning:
                            op_chunk.add_reference()
                        m.add_to_trace(f"Updating assoc between {key} and {op_chunk.name} to {a[0]:.3f}", 5)
                if dm.goal_operator_learning:
                    for g in goals:
                        a = op_chunk.assocs.setdefault(g.name, [0.0, 0])
                        a[0] += dm.beta * (goal_r - a[0])
                        a[1] += 1
                        if goal_r > 0 and dm.operator_baselevel_learning:
                            op_chunk.add_reference()
                        m.add_to_trace(f"Updating assoc between {g.name} and {op_chunk.name} to {a[0]:.3f}", 5)
                if dm.inter_operator_learning:
                    if prev is None:
                        prev = op_chunk
                    else:
                        a = op_chunk.assocs.setdefault(prev.name, [0.0, 0])
                        a[0] += dm.beta * (inter_r - a[0])
                        a[1] += 1
                        m.add_to_trace(f"Updating assoc between {prev.name} and {op_chunk.name} to {a[0]:.3f}", 5)
                        prev = op_chunk
        finally:
            self.previous_operators = []

    # ------------------------------------------------------------ bindings
    def check_operator_goal_match(self, op: Chunk):
        """Substitute ``*var`` constants from the bindings buffer."""
        m = self.model
        bindings = m.buffers.get("bindings")
        if bindings is None:
            return None, 0.0
        op_copy = op.copy_chunk()
        total = 0.0
        i = 1
        while f"slot{i}" in op_copy.slotvals:
            d = describe(op_copy.slotvals[f"slot{i}"])
            if d.startswith("*") and not d.startswith("**"):
                var = d[1:]
                subst = bindings.slotvals.get(var)
                if subst is not None:
                    if m.bindings_in_dm:
                        req = Chunk("retrieval", m)
                        req.set_slot("isa", "binding")
                        req.set_slot_value("slot1", m.buffers["goal"])
                        req.set_slot("slot2", var)
                        lat, ch = m.dm.retrieve(req, add_to_conflict_set=False)
                        value = None if ch is None else ch.slotvals.get("slot3")
                        if value is not None:
                            op_copy.set_slot_value(f"slot{i}", value)
                            total += lat
                        else:
                            return None, total + lat
                    else:
                        op_copy.set_slot_value(f"slot{i}", subst)
                else:
                    return None, total + m.dm.latency(m.dm.retrieval_threshold)
            elif d.startswith("**"):
                op_copy.set_slot(f"slot{i}", d[2:])
            i += 1
        return op_copy, total

    def all_context_chunks(self):
        out = []
        for bname, bchunk in list(self.model.buffers.items()):
            for slot, v in list(bchunk.slotvals.items()):
                if is_chunk(v):
                    out.append((bname, slot, v))
        return out

    # ------------------------------------------------------------ selection
    def find_operator(self) -> bool:
        """Retrieve the most active operator whose first production matches."""
        m = self.model
        dm = m.dm
        request = Chunk("operator", m)
        request.set_slot("isa", "operator")
        latency, retrieved = dm.retrieve(request)
        cfs = sorted(dm.conflict_set, key=lambda x: x[1], reverse=True)
        if m.tracing:
            m.add_to_trace("Conflict Set", 5)
            for ch, act in cfs:
                m.add_to_trace(f"  {ch.name} A = {act:.3f}", 5)
        match = False
        candidate = None
        with_subst = None
        activation = 0.0
        if cfs:
            while True:
                candidate, activation = cfs.pop(0)
                checked, lat = self.check_operator_goal_match(candidate)
                if checked is not None:
                    with_subst = checked.copy_chunk()
                    m.buffers["operator"] = checked
                    inst = m.procedural.find_matching_production()
                    match, prim, _ = m.procedural.fire_production(inst, compile=False)
                    if m.former_buffers.get("imaginal") is not None:
                        m.buffers["imaginal"] = m.former_buffers["imaginal"]
                    else:
                        m.buffers.pop("imaginal", None)
                    m.time += lat
                    if m.bindings_in_dm and match:
                        m.add_to_trace(f"       Retrieving bindings took {lat}seconds", 5)
                    if prim is not None and not match:
                        m.add_to_trace(f"   Operator {candidate.name} does not match because of {prim.name}", 5)
                else:
                    m.add_to_trace(f"   Rejected operator {candidate.name} because its variables failed to instatiate", 3)
                if match or not cfs or not (cfs[0][1] > dm.retrieval_threshold):
                    break
        if match:
            retrieved = candidate
            latency = dm.latency(activation)
        else:
            retrieved = None
            m.add_to_trace("   No matching operator found", 2)
            latency = dm.latency(dm.retrieval_threshold)
        m.time += latency
        if retrieved is None:
            return False
        if dm.goal_operator_learning or dm.context_operator_learning:
            ctx = self.all_context_chunks() if dm.context_operator_learning else []
            self.previous_operators.append((retrieved, m.time - latency, ctx))
        m.add_to_trace(f"*** Retrieved operator {retrieved.name} with latency {latency:.3f}", 1)
        dm.add_to_finsts(retrieved)
        m.buffers["goal"].set_slot_value("last-operator", retrieved)
        m.buffers["operator"] = with_subst
        m.former_buffers["operator"] = with_subst.copy_literal()
        m.selected_operator = retrieved
        return True

    def remove_last_operator_record(self):
        if self.previous_operators:
            self.previous_operators.pop()

    def carry_out_productions_until_operator_done(self) -> bool:
        """Fire productions until the operator is done (True) or a PRIM fails."""
        m = self.model
        proc = m.procedural
        match = True
        first = True
        while match:
            op = m.buffers.get("operator")
            if op is None or ("condition" not in op.slotvals and "action" not in op.slotvals):
                break
            inst = proc.find_matching_production()
            pname = inst.p.name[1:] if inst.p.name.startswith("t") else inst.p.name
            m.add_to_trace(f"Firing {pname}", 3)
            m.firings += 1
            m.fired_productions.append(pname)
            match, _, latency = proc.fire_production(inst, compile=True)
            if first:
                m.time += proc.production_action_latency + m.imaginal.imaginal_action_time + latency
                first = False
            else:
                m.time += proc.production_and_prim_latency + m.imaginal.imaginal_action_time + latency
            m.imaginal.imaginal_action_time = 0.0
        return match
