"""PRIMs: the primitive comparisons and copies (port of Prim.swift).

A PRIM refers to one slot in one buffer on each side, with ``=``, ``<>`` (a
comparison, only in conditions) or ``->`` (a copy, only in actions) between
them, plus the tree-navigation PRIMs ``>>X1`` (push) and ``X<<`` (pop).

Buffer letters:  V input, WM imaginal, G goal, C operator constants,
AC action, RT retrieval (harvest on the left, request on the right),
GC task constants, T temporal.
"""
from __future__ import annotations

from .chunk import Chunk
from .values import is_chunk, values_equal

BUFFER_MAPPING_C = {"V": "input", "WM": "imaginal", "G": "goal", "C": "operator",
                    "AC": "action", "RT": "retrievalH", "GC": "constants", "T": "temporal"}
BUFFER_MAPPING_A = {"V": "input", "WM": "imaginal", "G": "goal", "C": "operator",
                    "AC": "action", "RT": "retrievalR", "GC": "constants", "T": "temporal"}
BUFFER_MAPPING_PP = {"V": "input", "WM": "imaginal", "G": "goal", "RT": "retrievalH"}
BUFFER_ORDER = {"input": 1, "goal": 2, "imaginal": 3, "retrievalH": 4,
                "constants": 5, "operator": 6, "temporal": 7}
# buffer name -> letter, for printing and for operator compilation
BUFFER_LETTER = {"input": "V", "imaginal": "WM", "operator": "C", "action": "AC",
                 "retrievalH": "RT", "retrievalR": "RT", "temporal": "T",
                 "goal": "G", "constants": "GC"}

_ERROR = ("", "", "", None, None, None)


def _category(ch: str) -> int:
    if ch.isascii() and (ch.isalpha() or ch == "*"):
        return 1
    if ch.isascii() and ch.isdigit():
        return 2
    if ch in "<>=-":
        return 3
    return -1


def _components(name: str) -> list[str]:
    comps, comp, prev = [], "", None
    for ch in name:
        cat = _category(ch)
        if prev is None:
            prev = cat
        if prev == cat:
            comp += ch
        else:
            comps.append(comp)
            comp = ch
        prev = cat
    comps.append(comp)
    return comps


def _int_or_none(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_name(name: str):
    """Split a PRIM string into (lbuf, lslot, op, rbuf, rslot, canonical).

    Comparisons are put in canonical order (lower ``BUFFER_ORDER`` on the
    left), so that ``RT2 = WM1`` and ``WM1 = RT2`` are the same PRIM; the
    canonical string is returned as the 6th element when a swap happened.
    This canonicalisation is what makes productions shareable across tasks.
    """
    c = _components(name)
    if c and c[0] == ">>":
        if len(c) == 3:
            return (None, None, c[0], BUFFER_MAPPING_PP.get(c[1]), "slot" + c[2], None)
        return _ERROR
    if len(c) > 1 and c[1] == "<<":
        if len(c) == 2:
            return (BUFFER_MAPPING_PP.get(c[0]), None, c[1], None, None, None)
        return _ERROR
    compare_error = len(c) < 4
    parse_error = compare_error or (c[0] != "nil" and c[3] != "nil" and len(c) == 4)
    if parse_error or (c[0] == "nil" and c[1] != "->"):
        return _ERROR
    if c[0] == "nil":
        rb = BUFFER_MAPPING_A.get(c[2])
        if rb is None:
            return _ERROR
        return (None, None, "->", rb, "slot" + c[3], None)
    if c[3] == "nil":
        lb = BUFFER_MAPPING_C.get(c[0])
        if lb is None:
            return _ERROR
        return (lb, "slot" + c[1], c[2], None, None, None)
    rb = BUFFER_MAPPING_A.get(c[3]) if c[2] == "->" else BUFFER_MAPPING_C.get(c[3])
    lb = BUFFER_MAPPING_C.get(c[0])
    if rb is None or lb is None:
        return _ERROR
    new_prim = None
    if c[2] in ("=", "<>") and BUFFER_ORDER[lb] >= BUFFER_ORDER[rb]:
        left_slot, right_slot = _int_or_none(c[1]), _int_or_none(c[4])
        greater = (left_slot is not None and right_slot is not None and left_slot > right_slot) or \
                  (left_slot is not None and right_slot is None)
        if BUFFER_ORDER[lb] > BUFFER_ORDER[rb] or (BUFFER_ORDER[lb] == BUFFER_ORDER[rb] and greater):
            lb, rb = rb, lb
            c[1], c[4] = c[4], c[1]
            new_prim = c[3] + c[1] + c[2] + c[0] + c[4]
    return (lb, "slot" + c[1], c[2], rb, "slot" + c[4], new_prim)


class Prim:
    __slots__ = ("name", "model", "lhs_buffer", "lhs_slot", "op", "rhs_buffer", "rhs_slot")

    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        (self.lhs_buffer, self.lhs_slot, self.op, self.rhs_buffer,
         self.rhs_slot, _) = parse_name(name)

    def __repr__(self):
        return self.name

    def fire(self, condition: bool):
        """Check (condition) or perform (action) the PRIM: (match, latency)."""
        m = self.model
        buffers = m.buffers
        lhs_val = None
        if self.lhs_buffer is not None and self.lhs_slot is not None:
            b = buffers.get(self.lhs_buffer)
            lhs_val = None if b is None else b.slot_value(self.lhs_slot)
        op = self.op
        if op == "=":
            if self.rhs_buffer is None:
                return lhs_val is None, 0.0
            if lhs_val is None:
                return False, 0.0
            rb = buffers.get(self.rhs_buffer)
            rhs_val = None if rb is None else rb.slot_value(self.rhs_slot)
            if self.lhs_buffer == "temporal" and self.lhs_slot == "slot1":
                return (False if rhs_val is None else m.temporal.compare_time(_number(rhs_val))), 0.0
            if self.rhs_buffer == "temporal" and self.rhs_slot == "slot1":
                return (False if lhs_val is None else m.temporal.compare_time(_number(lhs_val))), 0.0
            return (False if rhs_val is None else m.compare(lhs_val, rhs_val)), 0.0
        if op == "<>":
            if self.rhs_buffer is None:
                return lhs_val is not None, 0.0
            if lhs_val is None:
                return False, 0.0
            rb = buffers.get(self.rhs_buffer)
            rhs_val = None if rb is None else rb.slot_value(self.rhs_slot)
            return (False if rhs_val is None else not m.compare(lhs_val, rhs_val)), 0.0
        if op == "->":
            return self._copy(lhs_val)
        if op == ">>":
            rb = self.rhs_buffer
            if rb == "imaginal":
                return m.imaginal.push(self.rhs_slot, condition), 0.0
            if rb == "retrievalH":
                return m.dm.push(self.rhs_slot), 0.0
            if rb == "input":
                return m.action.push(self.rhs_slot), 0.0
            return False, 0.0
        if op == "<<":
            lb = self.lhs_buffer
            if lb == "imaginal":
                return m.imaginal.pop(), 0.0
            if lb == "retrievalH":
                return m.dm.pop(), 0.0
            if lb == "input":
                return m.action.pop(), 0.0
            return False, 0.0
        return False, 0.0

    def _copy(self, lhs_val):
        m = self.model
        buffers = m.buffers
        rb, rs = self.rhs_buffer, self.rhs_slot
        if self.lhs_buffer is not None and lhs_val is None:
            return False, 0.0            # cannot copy nil from a slot
        if rb == "operator":             # ->*var : set a binding
            slotname = buffers["operator"].slotvals.get(rs)
            key = "" if slotname is None else _describe(slotname)
            bindings = buffers["bindings"]
            if self.lhs_buffer is None:
                bindings.slotvals.pop(key, None)
            else:
                bindings.set_slot_value(key, lhs_val)
            if m.bindings_in_dm:
                nc = m.generate_new_chunk("binding")
                nc.set_slot("isa", "binding")
                nc.set_slot_value("slot1", buffers["goal"])
                nc.set_slot("slot2", key)
                if self.lhs_buffer is not None:
                    nc.set_slot_value("slot3", lhs_val)
                m.dm.add_to_dm(nc)
                m.add_to_trace(f"   Adding binding {nc.name} to DM.", 5)
                return True, m.imaginal.imaginal_latency
            return True, 0.0
        if self.lhs_slot is None and rs == "slot0" and rb is not None:   # clear a buffer
            buffers.pop(rb, None)
            if rb == "imaginal":
                m.imaginal.move_wm_to_dm()
                m.imaginal.has_to_do_action = True
        if self.lhs_slot is None and buffers.get(rb) is not None and rs in buffers[rb].slotvals:
            del buffers[rb].slotvals[rs]       # nil -> slot: clear the slot
            return True, 0.0
        if buffers.get(rb) is None:
            chunk = m.generate_new_chunk(rb)
            chunk.set_slot("isa", "fact")
            buffers[rb] = chunk
            if rb == "imaginal":
                m.imaginal.add_chunk(chunk)
        if lhs_val is None:
            return True, 0.0
        if rs == "slot0":                 # replace the whole buffer chunk
            if is_chunk(lhs_val):
                new_chunk = lhs_val
                cur = buffers.get("imaginal")
                if rb == "imaginal" and cur is not None:
                    m.dm.add_to_dm(cur)
                    if cur.parent is not None:
                        new_chunk.parent = cur.parent
                        parent_chunk = m.imaginal.chunks[cur.parent]
                        for slot, value in list(parent_chunk.slotvals.items()):
                            if is_chunk(value) and value is cur:
                                parent_chunk.set_slot_value(slot, new_chunk)
                buffers[rb] = new_chunk
            else:
                return False, 0.0
        buffers[rb].set_slot_value(rs, lhs_val)
        return True, 0.0

    def test_fire(self) -> bool:
        if self.op in (">>", "<<"):
            return True
        if self.lhs_slot is None:
            return True
        b = self.model.buffers.get(self.lhs_buffer)
        return b is not None and b.slot_value(self.lhs_slot) is not None


def _describe(v):
    from .values import describe
    return describe(v)


def _number(v):
    return v if isinstance(v, float) else None
