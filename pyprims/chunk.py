"""Chunks: slots, base-level activation, spreading activation, Sji, noise.

Port of Chunk.swift.  Every equation is kept in the same form and the same
order of evaluation as the Swift code, including its idiosyncrasies (they are
marked QUIRK so they can be revisited deliberately rather than by accident).
"""
from __future__ import annotations

import math

from .values import (describe, string2double, actr_noise, is_chunk, safe_log,
                     safe_pow)


class Chunk:
    __slots__ = ("name", "model", "creation_time", "references", "slotvals",
                 "reference_list", "fan", "noise_value", "noise_time",
                 "fixed_activation", "print_order", "assocs", "defined_in",
                 "parent")

    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.creation_time = None      # None: not in DM (e.g. a buffer chunk)
        self.references = 1            # one reference on creation
        self.slotvals = {}             # slot name -> Chunk | float | str
        self.reference_list = []       # reference times (non-optimised learning)
        self.fan = 0                   # number of DM chunks this one appears in
        self.noise_value = 0.0
        self.noise_time = -1.0
        self.fixed_activation = None
        self.print_order = []
        self.assocs = {}               # Sji table: source name -> [value, count]
        self.defined_in = []
        self.parent = None

    # ------------------------------------------------------------ basics
    @property
    def type(self) -> str:
        tp = self.slotvals.get("isa")
        return "" if tp is None else describe(tp)

    def __repr__(self):
        return f"<Chunk {self.name}>"

    def description(self) -> str:
        s = f"{self.name}\n"
        for slot in self.print_order:
            if slot in self.slotvals:
                s += f"  {slot}  {describe(self.slotvals[slot])}\n"
        return s

    def copy_chunk(self) -> "Chunk":
        """Copy with a new, generated name (``copyChunk``)."""
        c = self.model.generate_new_chunk(self.name)
        c.slotvals = dict(self.slotvals)
        c.print_order = list(self.print_order)
        c.parent = self.parent
        return c

    def copy_literal(self) -> "Chunk":
        """Copy with the same name (``copyLiteral``)."""
        c = Chunk(self.name, self.model)
        c.slotvals = dict(self.slotvals)
        c.print_order = list(self.print_order)
        c.parent = self.parent
        return c

    def same_content(self, other: "Chunk") -> bool:
        """Swift ``==`` on chunks: same slots with the same descriptions."""
        if len(self.slotvals) != len(other.slotvals):
            return False
        for slot, v in self.slotvals.items():
            if slot not in other.slotvals:
                return False
            if describe(v) != describe(other.slotvals[slot]):
                return False
        return True

    # ------------------------------------------------------------ slots
    def set_slot(self, slot: str, value):
        """``setSlot`` for all value kinds.

        A *string* is interpreted: a numeric prefix makes it a number, the name
        of a DM chunk makes it that chunk, anything else stays text.
        """
        if slot not in self.print_order:
            self.print_order.append(slot)
        if isinstance(value, str):
            num = string2double(value)
            if num is not None:
                self.slotvals[slot] = num
            else:
                ch = self.model.dm.chunks.get(value)
                self.slotvals[slot] = ch if ch is not None else value
        elif isinstance(value, bool):
            self.slotvals[slot] = float(value)
        elif isinstance(value, int):
            self.slotvals[slot] = float(value)
        else:
            self.slotvals[slot] = value

    def set_slot_value(self, slot: str, value):
        """``setSlot(_, value: Value)``: store a value without interpretation."""
        if slot not in self.print_order:
            self.print_order.append(slot)
        self.slotvals[slot] = value

    def slot_value(self, slot: str):
        if slot == "slot0":
            return self.slotvals.get(slot, self)
        return self.slotvals.get(slot)

    # ------------------------------------------------------------ base level
    def start_time(self):
        m = self.model
        self.creation_time = m.time
        if not m.dm.optimized_learning:
            self.reference_list.append(m.time)

    def set_base_level(self, time_diff: float, references: int):
        m = self.model
        self.creation_time = m.time + time_diff
        if m.dm.optimized_learning:
            self.references = references
        else:
            inc = -time_diff / references
            for i in range(references):
                self.reference_list.append(self.creation_time + i * inc)

    def base_level_activation(self) -> float:
        if self.creation_time is None:
            return 0.0
        m = self.model
        dm = m.dm
        fixed = 0.0 if self.fixed_activation is None else math.exp(self.fixed_activation)
        d = dm.base_level_decay
        if dm.optimized_learning:
            # QUIRK (shared with ACT-R's ol): the 0.05 offset avoids t = 0.
            return safe_log(fixed + (self.references * safe_pow(m.time - self.creation_time + 0.05, -d)) / (1 - d))
        total = 0.0
        for r in self.reference_list:
            total += safe_pow(m.time - r + 0.05, -d)
        return safe_log(fixed + total)

    def add_reference(self):
        if self.creation_time is None:
            return
        m = self.model
        if m.dm.optimized_learning:
            self.references += 1
        else:
            self.reference_list.append(m.time)

    # ------------------------------------------------------------ Sji
    def appears_in_slot_of(self, chunk: "Chunk") -> bool:
        for v in chunk.slotvals.values():
            if is_chunk(v) and v.name == self.name:
                return True
        return False

    def calculate_sji(self, sji) -> float:
        base, refs = sji
        if refs == 0:
            return base
        dm = self.model.dm
        if dm.exploration_exploitation_factor == 0.0:
            return base
        return base + dm.exploration_exploitation_factor * actr_noise(self.model.rng, dm.default_operator_assoc) / math.sqrt(refs)

    def sji(self, chunk: "Chunk", buffer: str | None = None, slot: str | None = None) -> float:
        """Association from self (source j) to chunk (receiver i).

        The receiver holds the learned/declared associations in ``assocs``.
        QUIRK: the structural fallback is mas - ln(fan) with fan counted
        without the chunk itself (ACT-R counts 1 + appearances) and clamped
        at 0.
        """
        dm = self.model.dm
        if dm.extended_assocs and slot is not None:
            v = chunk.assocs.get(f"{buffer}%{slot}%{self.name}")
            if v is not None:
                return self.calculate_sji(v)
        v = chunk.assocs.get(self.name)
        if v is not None:
            return self.calculate_sji(v)
        rep = self.model.representation
        if rep.similarity_spreading:                 # embedded: associations from similarity
            return rep.sji(self, chunk)
        if self.appears_in_slot_of(chunk):
            return max(0.0, dm.maximum_associative_strength - math.log(max(1, self.fan)))
        return 0.0

    # ------------------------------------------------------------ spreading
    def spreading_from_buffer(self, buffer_name: str, w: float):
        if w == 0:
            return 0.0, 0
        bc = self.model.buffers.get(buffer_name)
        if bc is None:
            return 0.0, 0
        total = 0.0
        n = 0
        for slot, value in list(bc.slotvals.items()):
            if is_chunk(value):
                total += value.sji(self, buffer_name, slot)
                n += 1
        if n == 0:
            return 0.0, 0
        return total * (w / n), n

    def spreading_activation(self) -> float:
        """Spreading from goal, input, retrieval and imaginal buffers.

        QUIRK: spreading from the goal is *not* divided by the number of goal
        slots (W per source, not W/n), unlike the other buffers.
        """
        if self.creation_time is None:
            return 0.0
        m = self.model
        dm = m.dm
        total = 0.0
        goal = m.buffers.get("goal")
        if dm.goal_spreading_by_activation:
            if goal is not None:
                for value in list(goal.slotvals.values()):
                    if is_chunk(value):
                        total += value.sji(self) * math.exp(value.base_level_activation()) * dm.goal_activation
        else:
            s, n = self.spreading_from_buffer("goal", dm.goal_activation)
            total += s * n
        # spreading for "constructed" goals (instantiated skills)
        if goal is not None:
            for value in list(goal.slotvals.values()):
                if is_chunk(value) and value.type != "goaltype":
                    nested = value.slotvals.get("slot1")
                    if is_chunk(nested) and nested.type == "goaltype":
                        if dm.goal_spreading_by_activation:
                            total += nested.sji(self) * math.exp(value.base_level_activation()) * dm.goal_activation
                        else:
                            total += nested.sji(self) * dm.goal_activation
        total += self.spreading_from_buffer("input", dm.input_activation)[0]
        total += self.spreading_from_buffer("retrievalH", dm.retrieval_activation)[0]
        total += self.spreading_from_buffer("imaginal", dm.imaginal_activation)[0]
        return total

    # ------------------------------------------------------------ activation
    def calculate_noise(self) -> float:
        """Noise is resampled only when model time has moved on."""
        m = self.model
        if m.time != self.noise_time:
            self.noise_value = actr_noise(m.rng, m.dm.activation_noise)
            self.noise_time = m.time
        return self.noise_value

    def activation(self) -> float:
        if self.creation_time is None:
            return 0.0
        return self.base_level_activation() + self.spreading_activation() + self.calculate_noise()

    def activation_without_noise(self) -> float:
        if self.creation_time is None:
            return 0.0
        return self.base_level_activation() + self.spreading_activation()

    def merge_assocs(self, other: "Chunk"):
        for name, (val, cnt) in other.assocs.items():
            if name not in self.assocs:
                self.assocs[name] = [val, cnt]
            else:
                cur = self.assocs[name]
                self.assocs[name] = [max(cur[0], val), cur[1] + cnt]
