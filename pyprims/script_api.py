"""The functions available to task scripts (port of ScriptFunctions.swift).

A script is an ordinary Python function ``def script(m): ...`` that receives
a :class:`ScriptAPI`.  The PRIMs script function ``run-until-action(...)``
becomes ``m.run_until_action(...)``, ``issue-reward()`` becomes
``m.issue_reward()``, and so on: hyphens become underscores.

Differences from Swift, all deliberate:

* Run functions run the model synchronously until their stop condition
  (Swift suspends the script interpreter so the GUI can single-step).
* If the model "falls through" (no operator matches), the rest of the script
  still runs in the same trial; Swift finishes it at the start of the next
  batch trial, which duplicates that trial's output lines.
* A trial that exceeds ``time-threshold:`` is stopped in batch runs too
  (Swift only checks it when running from the GUI).
"""
from __future__ import annotations

import os

from .chunk import Chunk
from .values import describe, string2double


def _matches_action(model, wanted, offset: int) -> bool:
    """Does the last action (former_buffers['action']) match ``wanted``?

    ``wanted[i]`` is compared with slot i+1+offset... as in Swift: each
    argument must be a string equal to the slot's description.
    """
    act = model.former_buffers.get("action")
    if act is None:
        return False
    for i, w in enumerate(wanted):
        v = act.slotvals.get(f"slot{i + 1 + offset}")
        if v is None or not isinstance(w, str) or w != describe(v):
            return False
    return True


class ModelStuck(RuntimeError):
    """The script keeps running a model that has no applicable operator."""


class ScriptAPI:
    MAX_IDLE_RUN_CALLS = 10000

    def __init__(self, model):
        self.model = model
        self._idle_runs = 0

    def _fell_through(self) -> bool:
        """True if the model has fallen through; guards against endless script loops."""
        if self.model.falling_through:
            self._idle_runs += 1
            if self._idle_runs > self.MAX_IDLE_RUN_CALLS:
                raise ModelStuck("the script keeps calling run functions, but no operator matches "
                                 "(Swift PRIMs would hang here)")
            return True
        self._idle_runs = 0
        return False

    @property
    def arg(self):
        return self.model.arg

    @property
    def state(self):
        """A dictionary for the script's own bookkeeping (stimulus lists, counters, ...).

        It belongs to the task and survives from trial to trial, so the script and
        the action functions (``@task.action_function``) can share it.  Loading
        the task again, or a fresh model, starts with an empty one.
        """
        return self.model.scenario.state

    # ================================================================ running
    def run_step(self):
        m = self.model
        if self._fell_through():
            return
        m.new_step()

    def run_until_action(self, *action):
        m = self.model
        if self._fell_through():
            return
        while True:
            if m.falling_through:
                return
            m.new_step()
            found = m.former_buffers.get("action") is not None and _matches_action(m, action, 0)
            if found:
                m.scenario.next_event_time = None
                return
            if m.falling_through or m._check_time_threshold():
                return

    def run_relative_time_or_action(self, t, *action):
        m = self.model
        if self._fell_through():
            return
        if m.scenario.next_event_time is None and t >= 0:
            m.scenario.next_event_time = m.time + float(t)
        while True:
            if m.falling_through:
                return
            m.new_step()
            found = len(action) > 0 and _matches_action(m, action, 0)
            nxt = m.scenario.next_event_time
            if found or (nxt is not None and m.time >= nxt):
                m.scenario.next_event_time = None
                return
            if m.falling_through or m._check_time_threshold():
                return

    run_relative_time = run_relative_time_or_action
    run_until_relative_time_or_action = run_relative_time_or_action

    def run_absolute_time_or_action(self, t, *action):
        m = self.model
        if m.falling_through:
            return
        if m.scenario.next_event_time is None:
            m.scenario.next_event_time = float(t)
        # QUIRK (Swift): the absolute time is then also passed on as a relative time,
        # which is ignored because next_event_time is already set.
        return self.run_relative_time_or_action(t, *action)

    run_absolute_time = run_absolute_time_or_action

    def sleep(self, t):
        self.model.time += float(t)

    def time(self):
        return self.model.time

    # ================================================================ perception
    def screen(self, *items):
        """Put items in the input buffer: V1, V2, ... (or show a visicon chunk)."""
        m = self.model
        if len(items) == 1 and isinstance(items[0], str) and items[0] in m.action.visicon:
            m.buffers["input"] = m.action.visicon[items[0]]
            return
        name = m.generate_name("visual")
        self.add_visual(name, *items)
        m.buffers["input"] = m.action.visicon[name]

    set_input = screen

    def add_visual(self, name, *slots):
        m = self.model
        c = Chunk(describe(name), m)
        c.set_slot("isa", "fact")
        for i, v in enumerate(slots, start=1):
            d = describe(v)
            if d != "nil":
                c.set_slot(f"slot{i}", d)
        m.action.visicon[c.name] = c

    def last_action(self):
        act = self.model.former_buffers.get("action")
        if act is None:
            return [""]
        out = []
        i = 1
        while f"slot{i}" in act.slotvals:
            out.append(describe(act.slotvals[f"slot{i}"]))
            i += 1
        return out

    # ================================================================ trials and data
    def trial_start(self):
        m = self.model
        m.start_time = m.time
        m.add_data_line("trial-start", time=m.start_time)

    def trial_end(self):
        m = self.model
        if m.buffers.get("imaginal") is not None:
            m.imaginal.move_wm_to_dm()
        if not m.custom_points:
            m.results.append(m.time - m.start_time)
        if m.running:
            m.add_data_line("trial-end", "success")
            m.add_to_trace("End of trial", 1)
        m.commit_to_trace(False)
        m.initialize_next_trial()

    def issue_reward(self, reward=None):
        m = self.model
        r = m.reward if reward is None else float(reward)
        if r > 0:
            m.operators.compile_all()
        m.operators.update_operator_sjis(r)

    def data_line(self, *args):
        p = [describe(args[i]) if len(args) > i else "void" for i in range(3)]
        self.model.add_data_line("data-line", *p)

    def set_data_file_field(self, index, value):
        if not isinstance(index, int):
            raise TypeError("set-data-file-field: first argument must be an integer")
        self.model.scenario.current_input[f"?{index}"] = describe(value)

    def plot_point(self, y):
        self.model.results.append(float(y))
        self.model.custom_points = True

    def custom_points(self):
        self.model.custom_points = True

    def set_graph_title(self, title):
        self.model.graph_title = describe(title)

    def set_average_window(self, n):
        self.model.average_window = int(n)

    def print(self, *args):
        s = "".join(describe(a) + " " for a in args)
        m = self.model
        if not m.batch_mode and not m.silent and getattr(m, "echo_print", True):
            print(s)
        m.add_to_trace_field(s)

    def trace_operators(self, value):
        m = self.model
        m.trace_all_operators = describe(value) == "true" and m.batch_mode

    # ================================================================ randomness and arrays
    def random(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("random needs a non-negative integer")
        return self.model.rng.randrange(n) if n > 0 else 0

    def shuffle(self, arr):
        old = list(arr)
        new = []
        while old:
            new.append(old.pop(self.model.rng.randrange(len(old))))
        return new

    def length(self, x):
        return len(x)

    def int(self, x):
        return int(x)

    def str_to_int(self, s):
        return int(s)

    def random_string(self, prefix="fact"):
        return self.model.generate_name(describe(prefix))

    def batch_parameters(self):
        m = self.model
        if not m.batch_mode:
            return "NA"
        out = []
        for p in m.batch_parameters:
            num = string2double(p)
            if num is not None and str(p).strip().lstrip("+-").replace(".", "", 1).isdigit():
                out.append(float(p) if "." in p else int(p))
            else:
                out.append(p)
        return out

    def read_file(self, filename):
        m = self.model
        rec = m.tasks[m.current_task_index] if m.current_task_index is not None else None
        base = getattr(rec.definition, "source_dir", None) if rec else None
        path = os.path.join(base or ".", describe(filename))
        with open(path, encoding="utf-8") as f:
            # Foundation's Scanner skips leading whitespace and blank lines
            return [line.rstrip("\r\n").lstrip() for line in f if line.strip() != ""]

    # ================================================================ memory and parameters
    def _extra_fact(self, value: str):
        m = self.model
        if value not in m.dm.chunks and string2double(value) is None:
            c = Chunk(value, m)
            c.set_slot("isa", "fact")
            c.set_slot("slot1", value)
            c.fixed_activation = m.dm.default_activation
            m.dm.add_to_dm(c)

    def add_dm(self, name, *slots):
        m = self.model
        c = Chunk(describe(name), m)
        c.set_slot("isa", "fact")
        for i, v in enumerate(slots, start=1):
            d = describe(v)
            if d not in m.dm.chunks:
                x = Chunk(d, m)
                x.set_slot("isa", "fact")
                x.set_slot("slot1", d)
                x.fixed_activation = m.dm.default_activation
                m.dm.add_to_dm(x)
            c.set_slot(f"slot{i}", d)
        c.fixed_activation = m.dm.default_activation
        m.dm.add_to_dm(c)

    def set_activation(self, name, value):
        self.model.dm.chunks[describe(name)].fixed_activation = float(value)

    def get_activation(self, name):
        return self.model.dm.chunks[describe(name)].activation()

    def set_references(self, name, n):
        self.model.dm.chunks[describe(name)].references = int(n)

    def set_sji(self, *args):
        m = self.model
        if m.dm.extended_assocs:
            ch = m.dm.chunks[describe(args[0])]
            key = f"{describe(args[1])}%{describe(args[2])}%{describe(args[3])}"
            ch.assocs[key] = [float(args[4]), 0]
        else:
            c1 = m.dm.chunks[describe(args[0])]
            c2 = m.dm.chunks[describe(args[1])]
            c2.assocs[c1.name] = [float(args[2]), 0]

    def get_sji(self, chunk, buffer, slot, source):
        m = self.model
        if not m.dm.extended_assocs:
            raise RuntimeError("get-sji needs extended associations")
        v = m.dm.chunks[describe(chunk)].assocs.get(f"{describe(buffer)}%{describe(slot)}%{describe(source)}")
        return 0.0 if v is None else v[0]

    def sgp(self, name, value):
        if not self.model.set_parameter(describe(name), describe(value) if not isinstance(value, bool) else ("t" if value else "nil")):
            raise ValueError(f"Parameter {name} does not exist or cannot take value {value}")

    def open_jar(self, *args):
        """Run ``java -jar args...`` and return its standard output."""
        import subprocess
        r = subprocess.run(["java", "-jar", *[describe(a) for a in args]], capture_output=True, text=True)
        return r.stdout

    def report_memory(self):
        return "Error"          # disabled in Swift as well

    def imaginal_to_dm(self):
        if self.model.buffers.get("imaginal") is not None:
            self.model.imaginal.move_wm_to_dm()

    def purge_bindings(self):
        self.model.dm.purge("binding")

    # ================================================================ skills and bindings
    def set_goal(self, name, *pairs):
        m = self.model
        ch = m.dm.chunks.get(describe(name))
        if ch is None:
            raise KeyError("Goal chunk does not exist in set-goal")
        ch.slotvals = {}
        ch.print_order = []
        ch.set_slot("isa", "goaltype")
        for pair in pairs:
            attr, value = describe(pair[0]), describe(pair[1])
            self._extra_fact(value)
            ch.set_slot(attr, value)
        m.dm.version += 1          # a DM chunk changed in place (caches of the embedded representation)

    set_skill = set_goal

    def instantiate_skill(self, skill, inst_name, *pairs):
        m = self.model
        ch = m.dm.chunks.get(describe(skill))
        if ch is None or ch.type != "goaltype":
            raise KeyError("instantiate-skill: not a skill")
        if m.current_task_index is not None and m.current_task_index not in ch.defined_in:
            ch.defined_in.append(m.current_task_index)
        name = describe(inst_name)
        inst = m.dm.chunks.get(name) or Chunk(name, m)
        inst.slotvals = {}
        inst.print_order = []
        inst.set_slot("isa", "fact")
        inst.set_slot("slot1", describe(skill))
        for pair in pairs:
            attr, value = describe(pair[0]), describe(pair[1])
            self._extra_fact(value)
            inst.set_slot(attr, value)
        inst.fixed_activation = m.dm.default_activation
        if name not in m.dm.chunks:
            m.dm.add_to_dm(inst)
        m.dm.version += 1          # the instantiation may have been rewritten in place

    def add_binding(self, attribute, value, activation=None):
        m = self.model
        attr, val = describe(attribute), describe(value)
        self._extra_fact(val)
        m.buffers["bindings"].set_slot(attr, val)
        if m.bindings_in_dm:
            c = m.generate_new_chunk("binding")
            c.set_slot("isa", "binding")
            c.set_slot_value("slot1", m.buffers["goal"])
            c.set_slot("slot2", attr)
            c.set_slot("slot3", val)
            if activation is not None:
                c.fixed_activation = float(activation)
            m.dm.add_to_dm(c)

    def create_new_goal(self, name, *pairs):
        m = self.model
        name = describe(name)
        if name in m.dm.chunks:
            raise KeyError("Goal chunk already exists in create-new-goal")
        g = Chunk(name, m)
        g.set_slot("isa", "goaltype")
        for pair in pairs:
            g.set_slot(describe(pair[0]), describe(pair[1]))
        dup = m.dm.duplicate_chunk(g)
        cur = m.buffers["goal"]
        m.dm.add_to_dm(g)
        cur.set_slot_value("slot1", g if dup is None else dup)

    create_new_skill = create_new_goal

    def set_buffer_slot(self, buffer, slot, value):
        m = self.model
        b = describe(buffer)
        if m.buffers.get(b) is None:
            c = m.generate_new_chunk(b)
            c.set_slot("isa", "fact")
            m.buffers[b] = c
        m.buffers[b].set_slot(describe(slot), describe(value))

    def get_buffer_slot(self, buffer, slot):
        c = self.model.buffers.get(describe(buffer))
        if c is None:
            return "nil"
        v = c.slotvals.get(describe(slot))
        return "nil" if v is None else describe(v)


def div(a, b):
    """PRIMs script division: integer division truncates toward zero."""
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
        if b == 0:
            raise ZeroDivisionError("division by zero in script")
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q
    return a / b


def plus(a, b):
    """PRIMs script '+': a string on the left concatenates."""
    if isinstance(a, str):
        return a + describe(b)
    return a + b


def setitem(arr, i, value):
    """PRIMs array assignment: assigning past the end pads with 0."""
    while len(arr) < i:
        arr.append(0)
    if i == len(arr):
        arr.append(value)
    else:
        arr[i] = value


def and_(a, b):
    """PRIMs '&&': both sides are evaluated; integers multiply."""
    return int(a) * int(b)


def or_(a, b):
    """PRIMs '||': both sides are evaluated; integers add."""
    return int(a) + int(b)


def truth(x) -> bool:
    """Truth of an if/while test in PRIMs: any non-integer value counts as true."""
    if isinstance(x, (bool, int)):
        return x != 0
    return True


def cmp(a, op, b) -> bool:
    """Ordering comparison as in PRIMs: only numbers are ordered (strings give false)."""
    num = (int, float)
    if not (isinstance(a, num) and isinstance(b, num)):
        return op in ("<=", ">=") and a == b
    return {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
