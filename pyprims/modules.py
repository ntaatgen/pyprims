"""The imaginal (working memory), action/perception and temporal modules.

Ports of Imaginal.swift, Action.swift and Temporal.swift.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunk import Chunk
from .values import actr_noise, uniform01, is_chunk, describe


# =================================================================== imaginal
class Imaginal:
    DEFAULT_LATENCY = 0.2                 # imaginal-delay:

    def __init__(self, model):
        self.model = model
        self.imaginal_latency = self.DEFAULT_LATENCY
        self.chunks: dict[str, Chunk] = {}   # working-memory chunks not yet in DM
        self.has_to_do_action = False
        self.imaginal_action_time = 0.0

    def set_parameters_to_default(self):
        self.imaginal_latency = self.DEFAULT_LATENCY

    def reset(self):
        self.chunks = {}
        self.imaginal_action_time = 0.0

    def add_chunk(self, chunk: Chunk):
        self.chunks[chunk.name] = chunk
        chunk.start_time()

    def move_wm_to_dm(self):
        """Move all WM chunks to DM, leaves first, merging duplicates."""
        m = self.model
        trans: dict[str, str] = {}
        while self.chunks:
            removed = False
            for name, chunk in list(self.chunks.items()):
                if name not in self.chunks:
                    continue
                no_refs = not any(is_chunk(v) and v.name in self.chunks for v in chunk.slotvals.values())
                if no_refs:
                    self._substitute(chunk, trans)
                    new = m.dm.add_to_dm(chunk)
                    if new.name != chunk.name:
                        trans[chunk.name] = new.name
                    del self.chunks[name]
                    removed = True
            if not removed:             # circular structure: dump the rest
                for chunk in list(self.chunks.values()):
                    self._substitute(chunk, trans)
                    m.dm.add_to_dm(chunk)
                self.chunks = {}
        for chunk in list(m.buffers.values()):
            self._substitute(chunk, trans)

    def _substitute(self, chunk, trans):
        for slot, v in list(chunk.slotvals.items()):
            if is_chunk(v) and v.name in trans:
                chunk.set_slot(slot, trans[v.name])

    def action(self) -> float:
        self.has_to_do_action = False
        self.model.add_to_trace(f"New imaginal chunk (latency = {self.imaginal_latency:.3f})", 2)
        return self.imaginal_latency

    def push(self, slot: str, condition: bool) -> bool:
        m = self.model
        old = m.buffers.get("imaginal")
        if old is None:
            return False
        v = old.slotvals.get(slot)
        if v is not None:
            if not is_chunk(v):
                return False
            if v.name in self.chunks:
                self.chunks[old.name] = old
                old.add_reference()
                v.parent = old.name
                lat = m.dm.latency(v.activation())
                self.imaginal_action_time += lat
                m.add_to_trace(f"Imaginal retrieval latency of {v.name} is {lat:.3f}", 5)
                m.buffers["imaginal"] = v
                return True
            dm_chunk = m.dm.chunks.get(v.name)
            if dm_chunk is None:
                return False
            dm_chunk = dm_chunk.copy_chunk()
            self.chunks[old.name] = old
            old.add_reference()
            dm_chunk.parent = old.name
            dm_chunk.start_time()
            self.add_chunk(dm_chunk)
            old.set_slot_value(slot, dm_chunk)
            lat = m.dm.latency(v.activation())
            self.imaginal_action_time += lat
            m.add_to_trace(f"Imaginal retrieval latency of {v.name} is {lat:.3f}", 5)
            m.buffers["imaginal"] = dm_chunk
            return True
        if condition:
            return False
        new = Chunk(m.generate_name("wm"), m)
        new.set_slot("isa", "fact")
        new.start_time()
        self.add_chunk(new)
        old.set_slot_value(slot, new)
        self.chunks[old.name] = old
        old.add_reference()
        new.parent = old.name
        self.imaginal_action_time += self.imaginal_latency
        m.add_to_trace(f"New imaginal chunk {new.name} latency is {self.imaginal_latency}", 5)
        m.buffers["imaginal"] = new
        return True

    def pop(self) -> bool:
        m = self.model
        cur = m.buffers.get("imaginal")
        if cur is None or cur.parent is None:
            return False
        self.chunks[cur.name] = cur
        cur.add_reference()
        parent = self.chunks[cur.parent]
        lat = m.dm.latency(parent.activation())
        self.imaginal_action_time += lat
        m.add_to_trace(f"Imaginal retrieval latency of {parent.name} is {lat:.3f}", 5)
        m.buffers["imaginal"] = parent
        return True


# =================================================================== action
@dataclass
class ActionInstance:
    name: str
    mean_latency: float
    output_string: str = ""
    noise_type: str = "none"          # none | uniform | logistic
    noise_value: float = 0.0

    def __post_init__(self):
        if not self.output_string:
            self.output_string = self.name + "ing"

    def latency(self, rng) -> float:
        if self.noise_type == "logistic":
            return max(0.0, self.mean_latency + actr_noise(rng, self.noise_value))
        if self.noise_type == "uniform":
            return max(0.0, self.mean_latency - self.noise_value + 2 * self.noise_value * uniform01(rng))
        return self.mean_latency


class Action:
    DEFAULT_LATENCY = 0.2                 # perception-action-latency:

    def __init__(self, model):
        self.model = model
        self.default_perceptual_action_latency = self.DEFAULT_LATENCY
        self.actions: dict[str, ActionInstance] = {}
        self.visicon: dict[str, Chunk] = {}

    def set_parameters_to_default(self):
        self.default_perceptual_action_latency = self.DEFAULT_LATENCY

    def create_new_goal(self, chunk: Chunk):
        m = self.model
        nv = chunk.slot_value("slot2")
        if nv is None:
            return
        name = describe(nv)
        existing = m.dm.chunks.get(name)
        if existing is not None:
            goal = existing
            goal.slotvals = {}
            goal.print_order = []
        else:
            goal = Chunk(name, m)
        goal.set_slot("isa", "goaltype")
        idx = 3
        while chunk.slot_value(f"slot{idx}") is not None:
            attr = describe(chunk.slot_value(f"slot{idx}"))
            val = chunk.slot_value(f"slot{idx + 1}")
            if val is None:
                return
            goal.set_slot(attr, describe(val))
            idx += 2
        if existing is not None:
            m.dm.add_to_dm(goal)
            m.dm.version += 1      # an existing DM chunk was rewritten in place
        idx = 1
        while m.buffers["goal"].slot_value(f"slot{idx}") is not None:
            idx += 1
        m.buffers["goal"].set_slot_value(f"slot{idx}", goal)

    def action(self) -> float:
        m = self.model
        ac_chunk = m.buffers.pop("action")
        latency = 0.05
        ac = ac_chunk.slotvals.get("slot1")
        if ac is None:
            return 0.0
        ac = describe(ac)
        if ac == "build-goal":
            self.create_new_goal(ac_chunk)
            return 0.2
        p1 = ac_chunk.slotvals.get("slot2")
        p2 = ac_chunk.slotvals.get("slot3")
        p1 = None if p1 is None else describe(p1)
        p2 = None if p2 is None else describe(p2)
        inst = self.actions.get(ac)
        func = m.scenario.action_functions.get(ac.replace("-", "_")) if m.scenario is not None else None
        func_latency = None
        if func is not None:
            from .script_api import ScriptAPI
            args = []
            idx = 2
            while f"slot{idx}" in ac_chunk.slotvals:
                args.append(describe(ac_chunk.slotvals[f"slot{idx}"]))
                idx += 1
            result = func(ScriptAPI(m), *args)
            if isinstance(result, (int, float)) and not isinstance(result, bool):
                func_latency = max(0.0, float(result))
        if func_latency is not None:
            latency = func_latency
            m.add_to_trace(f"{ac} {p1 or ''} {p2 or ''} (latency = {latency:.3f})", 2)
        elif ac == "wait":
            m.add_to_trace(f"Waiting (latency = {latency:.3f})", 2)
            latency = 0.05
            if m.scenario.next_event_time is not None:
                latency = max(0.0, m.scenario.next_event_time - m.time)
        elif inst is not None:
            latency = inst.latency(m.rng)
            m.add_to_trace(f"{inst.output_string} {p1 or ''} {p2 or ''} (latency = {latency:.3f})", 2)
        else:
            m.add_to_trace(f"{ac}-ing {p1 or ''}-{p2 or ''}", 2)
        m.add_data_line("action", ac, p1 or "void", p2 or "void", m.time + latency - m.start_time)
        return latency

    def push(self, slot: str) -> bool:
        m = self.model
        old = m.buffers.get("input")
        if old is None:
            return False
        v = old.slotvals.get(slot)
        if v is None:
            return False
        ch = self.visicon.get(describe(v))
        if ch is None:
            return False
        ch.parent = old.parent if slot == "slot2" else old.name
        m.buffers["input"] = ch
        return True

    def pop(self) -> bool:
        m = self.model
        cur = m.buffers.get("input")
        if cur is None or cur.parent is None:
            return False
        m.buffers["input"] = self.visicon[cur.parent]
        return True


# =================================================================== temporal
class Temporal:
    """Pulse-based interval timer (Taatgen, van Rijn & Anderson, 2007)."""
    DEFAULTS = dict(time_t0=0.011, time_a=1.1, time_b=0.015)

    def __init__(self, model):
        self.model = model
        self.set_parameters_to_default()
        self.reset()

    def set_parameters_to_default(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

    def reset(self):
        self.current_pulse = None
        self.current_pulse_length = None
        self.start_time = None
        self.next_pulse_time = None

    def start_timer(self):
        m = self.model
        self.current_pulse = 0
        self.start_time = m.time
        self.next_pulse_time = self.time_t0
        self.current_pulse_length = self.time_t0
        tc = Chunk(m.generate_name("time"), m)
        tc.set_slot("isa", "time")
        tc.set_slot("slot1", 0.0)
        m.buffers["temporal"] = tc

    def stop_timer(self):
        self.model.buffers.pop("temporal", None)
        self.reset()

    def action(self):
        m = self.model
        tc = m.buffers.get("temporal")
        if tc is None:
            return
        cmd = tc.slot_value("slot3")
        if cmd is not None:
            d = describe(cmd)
            if d == "start":
                self.start_timer()
            elif d == "stop":
                self.stop_timer()
            else:
                m.add_to_trace_field(f"Unknown command {d} to temporal module")

    def update_timer(self):
        m = self.model
        tc = m.buffers.get("temporal")
        if tc is None or self.start_time is None or self.next_pulse_time is None:
            return
        if m.time > self.start_time + self.next_pulse_time:
            while m.time > self.start_time + self.next_pulse_time:
                self.current_pulse += 1
                self.current_pulse_length = (self.time_a * self.current_pulse_length
                                             + actr_noise(m.rng, self.time_b * self.time_a * self.current_pulse_length))
                self.next_pulse_time += self.current_pulse_length
            tc.set_slot("slot1", float(self.current_pulse))
            tc.set_slot("slot2", "true")

    def compare_time(self, value) -> bool:
        tc = self.model.buffers.get("temporal")
        if tc is None or value is None:
            return False
        cur = tc.slot_value("slot1")
        return isinstance(cur, float) and cur >= value
