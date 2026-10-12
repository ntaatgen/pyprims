"""The PRIMs model: buffers, parameters, the operator cycle and trials.

Port of Model.swift (and the run-control parts of ScriptFunctions.swift).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .chunk import Chunk
from .declarative import Declarative
from .procedural import Procedural
from .operators import Operator
from .modules import Imaginal, Action, Temporal
from .values import string2double, describe


@dataclass
class DataLine:
    event_type: str
    p1: str
    p2: str
    p3: str
    inputs: list
    time: float
    firings: int

    def fields(self):
        return [self.event_type, self.p1, self.p2, self.p3, *self.inputs]


@dataclass
class Scenario:
    next_event_time: float | None = None
    current_input: dict = field(default_factory=dict)
    script: object = None          # callable(m)
    init_script: object = None     # callable(m)
    action_functions: dict = field(default_factory=dict)   # action name -> callable(m, *args)
    state: dict = field(default_factory=dict)              # the script's own state (m.state)

    @property
    def input_mapping_for_trace(self):
        return [self.current_input.get(f"?{i}", "void") for i in range(5)]


@dataclass
class TaskRecord:
    name: str
    number: int
    definition: object = None      # pyprims.dsl.Task
    goal_chunk: Chunk | None = None
    goal_constants: Chunk | None = None
    scenario: Scenario | None = None
    parameters: list = field(default_factory=list)
    actions: dict = field(default_factory=dict)
    loaded: bool = False
    bugged: bool = False


MODEL_DEFAULTS = dict(reward=0.0, operator_learning=False,
                      complex_goal_variable_structure=True, bindings_in_dm=False)

_BOOL_PARAMS = {
    "ol:": ("dm", "optimized_learning"),
    "goal-operator-learning:": ("dm", "goal_operator_learning"),
    "skill-operator-learning:": ("dm", "goal_operator_learning"),
    "extended-associations:": ("dm", "extended_assocs"),
    "inter-operator-learning:": ("dm", "inter_operator_learning"),
    "goal-chunk-spreads:": ("dm", "goal_spreading_by_activation"),
    "declarative-buffer-stuffing:": ("dm", "declarative_buffer_stuffing"),
    "retrieval-reinforces:": ("dm", "retrieval_reinforces"),
    "pm:": ("dm", "partial_matching"),
    "batch-trace:": ("model", "batch_trace"),
    "blending:": ("dm", "blending"),
    "opc:": ("model", "operator_learning"),
    "operator-bll:": ("dm", "operator_baselevel_learning"),
    "bindings-in-dm:": ("model", "bindings_in_dm"),
    "complex-goal-variable-structure:": ("model", "complex_goal_variable_structure"),
}
_NUM_PARAMS = {
    "imaginal-delay:": ("imaginal", "imaginal_latency"),
    "egs:": ("procedural", "utility_noise"),
    "alpha:": ("procedural", "alpha"),
    "nu:": ("procedural", "default_u"),
    "primU:": ("procedural", "prim_u"),
    "dat:": ("procedural", "production_action_latency"),
    "bll:": ("dm", "base_level_decay"),
    "mas:": ("dm", "maximum_associative_strength"),
    "rt:": ("dm", "retrieval_threshold"),
    "lf:": ("dm", "latency_factor"),
    "mp:": ("dm", "mismatch_penalty"),
    "ans:": ("dm", "activation_noise"),
    "ga:": ("dm", "goal_activation"),
    "input-activation:": ("dm", "input_activation"),
    "retrieval-activation:": ("dm", "retrieval_activation"),
    "wm-activation:": ("dm", "imaginal_activation"),
    "imaginal-activation:": ("dm", "imaginal_activation"),
    "default-operator-assoc:": ("dm", "default_operator_assoc"),
    "default-inter-operator-assoc:": ("dm", "default_inter_operator_assoc"),
    "default-operator-self-assoc:": ("dm", "default_operator_self_assoc"),
    "production-prim-latency:": ("procedural", "production_and_prim_latency"),
    "perception-action-latency:": ("action", "default_perceptual_action_latency"),
    "beta:": ("dm", "beta"),
    "reward:": ("model", "reward"),
    "time-threshold:": ("model", "time_threshold"),
    "procedural-reward:": ("procedural", "procedural_reward"),
    "explore-exploit:": ("dm", "exploration_exploitation_factor"),
    "new-pm-pow:": ("dm", "new_partial_matching_pow"),
    "new-pm-exp:": ("dm", "new_partial_matching_pow"),   # QUIRK kept from Swift
    "time-t0:": ("temporal", "time_t0"),
    "time-a:": ("temporal", "time_a"),
    "time-b:": ("temporal", "time_b"),
}
PARAMETER_NAMES = sorted(set(_BOOL_PARAMS) | set(_NUM_PARAMS) |
                         {"context-operator-learning:", "default-activation:"})


class Model:
    """A PRIMs model.  ``Model(seed=...)`` gives reproducible runs."""

    def __init__(self, seed=None, trace: bool = True, batch_mode: bool = False, representation=None):
        self.rng = random.Random(seed)
        self.silent = not trace
        self.tracing = trace
        self.batch_mode = batch_mode
        self.time = 0.0
        self.start_time = 0.0
        self.time_threshold = 200.0
        self.buffers: dict[str, Chunk] = {}
        self.former_buffers: dict[str, Chunk] = {}
        self.chunk_id_counter = 0
        self.running = False
        self.falling_through = False
        self.timed_out = False
        self.trace: list[tuple[int, str]] = []
        self.trace_buffer: list[tuple[int, str]] = []
        self.output_data: list[DataLine] = []
        self.firings = 0
        self.fired_productions: list[str] = []
        self.selected_operator = None
        self.trace_all_operators = False
        self.batch_trace = False
        self.batch_parameters: list[str] = []
        self.arg = ""
        self.current_task: str | None = None
        self.current_goals: Chunk | None = None
        self.current_goal_constants: Chunk | None = None
        self.parameters: list[tuple[str, str]] = []
        self.scenario = Scenario()
        self.tasks: list[TaskRecord] = []
        self.current_task_index: int | None = None
        self.results: list[float] = []          # one point per trial (RT or plot-point)
        self.custom_points = False
        self.graph_title = None
        self.average_window = 1
        self.still_need_to_do_module_actions = False
        self.half_step = True
        for k, v in MODEL_DEFAULTS.items():
            setattr(self, k, v)
        self.dm = Declarative(self)
        self.procedural = Procedural(self)
        self.imaginal = Imaginal(self)
        self.action = Action(self)
        self.operators = Operator(self)
        self.temporal = Temporal(self)
        self._new_bindings_chunk()
        self.set_representation(representation)

    # ================================================================ representation
    def set_representation(self, representation=None):
        """Symbolic (default) or embedded slot contents; see pyprims.embedding."""
        from .embedding import Symbolic
        self.representation = Symbolic() if representation is None else representation
        self.representation.attach(self)       # sets self.compare

    # ================================================================ trace
    def add_to_trace(self, s: str, level: int):
        if self.tracing and not self.silent:
            self.trace_buffer.append((level, f"{self.time:.2f}  " + s))

    def commit_to_trace(self, indented: bool):
        if not self.silent:
            for lvl, s in self.trace_buffer:
                if indented:
                    self.trace.append((max(lvl, 3), "       " + s))
                else:
                    self.trace.append((lvl, s))
        self.trace_buffer = []

    def add_to_trace_field(self, s: str):
        if not self.silent:
            self.trace.append((0, s))

    def clear_trace(self):
        self.trace = []

    def get_trace(self, max_level: int = 3) -> str:
        return "".join(s + "\n" for lvl, s in self.trace if lvl <= max_level)

    # ================================================================ data
    def add_data_line(self, event_type, p1="void", p2="void", p3="void", time=None):
        t = (self.time - self.start_time) if time is None else time
        self.output_data.append(DataLine(event_type, p1 or "void", p2 or "void", p3 or "void",
                                         self.scenario.input_mapping_for_trace, t, self.firings))
        self.firings = 0

    # ================================================================ chunks
    def generate_name(self, prefix: str = "chunk") -> str:
        self.chunk_id_counter += 1
        return f"{prefix}{self.chunk_id_counter - 1}"

    def generate_new_chunk(self, prefix: str = "chunk") -> Chunk:
        return Chunk(self.generate_name(prefix), self)

    def string_to_value(self, s: str):
        num = string2double(s)
        if num is not None:
            return num
        return self.dm.chunks.get(s, s)

    def _new_bindings_chunk(self):
        b = Chunk("bindings", self)
        b.set_slot("isa", "bindings")
        self.buffers["bindings"] = b

    # ================================================================ parameters
    def set_parameter(self, parameter: str, value: str) -> bool:
        """Set a parameter given as in a .prims file, e.g. ("rt:", "-2.0")."""
        if not parameter.endswith(":"):
            parameter += ":"
        value = describe(value) if not isinstance(value, str) else value
        num = string2double(value)
        boolval = value != "nil"
        if parameter in _BOOL_PARAMS:
            module, attr = _BOOL_PARAMS[parameter]
            setattr(self if module == "model" else getattr(self, module), attr, boolval)
            return True
        if parameter == "context-operator-learning:":
            self.dm.context_operator_learning = boolval
            if boolval:
                self.dm.extended_assocs = True
            return True
        if parameter == "default-activation:":
            if value == "nil":
                self.dm.default_activation = None
            elif num is not None:
                self.dm.default_activation = num
            else:
                return False
            return True
        if num is None or parameter not in _NUM_PARAMS:
            return False
        module, attr = _NUM_PARAMS[parameter]
        setattr(self if module == "model" else getattr(self, module), attr, num)
        return True

    def set_parameters_to_default(self):
        self.dm.set_parameters_to_default()
        self.procedural.set_parameters_to_default()
        self.action.set_parameters_to_default()
        self.imaginal.set_parameters_to_default()
        self.temporal.set_parameters_to_default()
        for k, v in MODEL_DEFAULTS.items():
            setattr(self, k, v)

    def load_parameters(self):
        for p, v in self.parameters:
            self.set_parameter(p, v)

    # ================================================================ tasks
    def load(self, task) -> int:
        """Load a task (a :class:`pyprims.Task` or a path to .prims/.py)."""
        from .dsl import Task, load_task
        if not isinstance(task, Task):
            task = load_task(task)
        self.scenario = Scenario()
        self.parameters = []
        self.current_task_index = len(self.tasks)
        self.set_parameters_to_default()
        number = len(self.tasks)
        task.install(self, number)
        self._after_parse()
        rec = TaskRecord(name=self.current_task or "Undefined", number=number, definition=task)
        rec.loaded = True
        rec.goal_chunk = self.current_goals
        rec.goal_constants = self.current_goal_constants
        rec.parameters = list(self.parameters)
        rec.scenario = self.scenario
        rec.actions = self.action.actions
        self.tasks.append(rec)
        self.running = False
        self.falling_through = False
        return number

    def _after_parse(self):
        if self.scenario.init_script is not None:
            self.scenario.init_script(ScriptAPI(self))

    def find_task(self, name: str):
        for i, t in enumerate(self.tasks):
            if t.name == name:
                return i
        return None

    def load_or_reload_task(self, i: int):
        """Switch to task i (Model.loadOrReloadTask)."""
        if i == self.current_task_index:
            return
        rec = self.tasks[i]
        self.current_task_index = i
        if not rec.loaded:
            self.scenario = Scenario()
            self.parameters = []
            self.set_parameters_to_default()
            rec.definition.install(self, i)
            self._after_parse()
            rec.goal_chunk = self.current_goals
            rec.goal_constants = self.current_goal_constants
            rec.parameters = list(self.parameters)
            rec.scenario = self.scenario
            rec.actions = self.action.actions
            rec.loaded = True
        self.current_task = rec.name
        self.current_goals = rec.goal_chunk
        self.current_goal_constants = rec.goal_constants
        self.parameters = rec.parameters
        self.scenario = rec.scenario
        self.action.actions = rec.actions
        self.set_parameters_to_default()
        self.load_parameters()
        self.running = False

    def switch_task(self, name: str):
        i = self.find_task(name)
        if i is None:
            raise KeyError(f"task {name} is not loaded")
        self.load_or_reload_task(i)

    # ================================================================ trials
    def _reset_buffers_for_trial(self):
        self.start_time = self.time
        self.falling_through = False
        self.timed_out = False
        self.buffers = {}
        self._new_bindings_chunk()
        self.procedural.reset()
        if self.current_goals is not None:
            self.buffers["goal"] = self.current_goals.copy_chunk()
        if self.current_goal_constants is not None:
            self.buffers["constants"] = self.current_goal_constants.copy_chunk()
        self.former_buffers = {}
        for b in ("goal", "constants", "bindings"):
            if b in self.buffers:
                self.former_buffers[b] = self.buffers[b].copy_literal()
        self.operators.previous_operators = []
        self.firings = 0

    def initialize_new_trial(self):
        self._reset_buffers_for_trial()
        self.add_to_trace("Starting new trial", 1)
        self.running = True
        self.clear_trace()
        self.output_data = []
        self.fired_productions = []

    def initialize_next_trial(self):
        self._reset_buffers_for_trial()

    def run_trial(self):
        """Run the task's script once (one call of Model.run in batch mode)."""
        if self.scenario.script is None:
            raise RuntimeError("the current task has no script")
        self.initialize_new_trial()
        self.scenario.script(ScriptAPI(self))
        self.running = False

    def run(self, n: int = 1):
        for _ in range(n):
            self.run_trial()

    # ================================================================ the operator cycle
    def do_all_module_actions(self):
        latency = 0.0
        if "retrievalH" in self.buffers:
            self.former_buffers["retrievalH"] = self.buffers["retrievalH"]
        else:
            self.former_buffers.pop("retrievalH", None)
        last = self.procedural.last_operator
        if last is None or "action" in last.slotvals:
            self.buffers.pop("retrievalH", None)
        if "retrievalR" in self.buffers or self.dm.declarative_buffer_stuffing:
            if "retrievalR" in self.buffers:
                self.former_buffers["retrievalR"] = self.buffers["retrievalR"]
            latency = max(latency, self.dm.action())
        if self.imaginal.has_to_do_action:
            latency = max(latency, self.imaginal.action())
        if "action" in self.buffers:
            self.former_buffers["action"] = self.buffers["action"]
            latency = max(latency, self.action.action())
        if "temporal" in self.buffers:
            self.temporal.action()
        self.time += latency
        if "temporal" in self.buffers:
            self.temporal.update_timer()

    def new_step(self):
        """One half step: select and execute an operator, or do module actions."""
        if self.still_need_to_do_module_actions:
            self.do_all_module_actions()
            self.still_need_to_do_module_actions = False
            return
        self.dm.clear_finsts()
        fb = {}
        for b in ("goal", "retrievalH", "constants", "bindings"):
            if b in self.buffers:
                fb[b] = self.buffers[b].copy_literal()
        for b in ("imaginal", "input"):
            if b in self.buffers:
                fb[b] = self.buffers[b]          # by reference, as in Swift
        self.former_buffers = fb
        self.commit_to_trace(False)
        while True:
            self.procedural.last_production = None
            if not self.operators.find_operator():
                if self.scenario.next_event_time is None:
                    self.running = False
                    self.falling_through = True
                    self.operators.update_operator_sjis(0.0)
                    self.add_data_line("trial-end", "fail")
                    self.add_to_trace("End of trial", 1)
                    return
                self.time = self.scenario.next_event_time
                self._log_input(self.time)
                return
            found = self.operators.carry_out_productions_until_operator_done()
            if found:
                break
            op = self.buffers["operator"]
            self.add_to_trace(f"Operator {op.name} failed", 2)
            self.commit_to_trace(True)
            if self.trace_all_operators:
                self.add_data_line("failed-operator", op.name)
            for b in ("goal", "imaginal", "input", "retrievalH", "constants", "bindings"):
                if b in self.former_buffers:
                    self.buffers[b] = self.former_buffers[b]
                else:
                    self.buffers.pop(b, None)
            self.operators.remove_last_operator_record()
            self.procedural.clear_reward_trace()
        self.procedural.issue_reward(self.procedural.procedural_reward)
        self.procedural.last_operator = self.former_buffers.get("operator")
        if self.trace_all_operators:
            lo = self.buffers.get("goal").slotvals.get("last-operator") if "goal" in self.buffers else None
            if lo is not None:
                self.add_data_line("operator", describe(lo))
        self.commit_to_trace(False)
        self.buffers.pop("operator", None)
        if self.half_step:
            self.still_need_to_do_module_actions = True
            return
        self.do_all_module_actions()

    def _log_input(self, t):
        inp = self.buffers.get("input")
        if inp is not None:
            vals = [describe(inp.slotvals[s]) if s in inp.slotvals else "void"
                    for s in ("slot1", "slot2", "slot3")]
            self.add_data_line("perception", *vals, time=t - self.start_time)

    def _check_time_threshold(self) -> bool:
        """Deviation from Swift batch mode: stop a runaway trial."""
        if self.time - self.start_time > self.time_threshold:
            self.running = False
            self.falling_through = True
            self.timed_out = True
            self.add_data_line("trial-end", "timeout")
            return True
        return False

    # ================================================================ inspection
    def chunk(self, name: str) -> Chunk:
        return self.dm.chunks[name]

    def operators_in_dm(self):
        return [c for c in self.dm.chunks.values() if c.type == "operator"]

    def production_table(self):
        """(name, utility) of all learned productions, highest utility first."""
        return sorted(((p.full_name or p.name, p.u) for p in self.procedural.productions.values()),
                      key=lambda x: -x[1])


from .script_api import ScriptAPI  # noqa: E402  (circular import at the end)
