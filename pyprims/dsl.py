"""Model definitions: the Python ``Task`` builder and the .prims reader.

A model is an ordered list of definitions (task settings, skills with their
operators, facts, actions, Sji's, visual chunks, scripts).  Order matters in
PRIMs: parameters such as ``default-activation:`` apply to everything defined
after them, and whether a slot value becomes a chunk reference depends on
whether that chunk already exists.  So both the Python builder and the
.prims reader produce the same ordered list, and :meth:`Task.install`
replays it exactly the way Parser.swift does.

Python form::

    from pyprims import Task

    task = Task("count", initial_skills=["count"], default_activation=1.0,
                ol=True, rt=-2.0, lf=0.2, default_operator_self_assoc=0.0,
                egs=0.05, retrieval_reinforces=True)

    count = task.skill("count")
    count.operator("start-count", '''
        V1 <> nil          // there has to be a start number
        WM1 = nil
    ==>
        V1 -> WM1
        count-fact -> RT1
        V1 -> RT2
        say -> AC1
        V1 -> AC2
    ''')

    task.facts([("cf1", "count-fact", "one", "two"), ...])
    task.action("say", latency=0.3, noise=0.1, distribution="uniform", output="Saying")

    @task.script
    def script(m):
        m.screen("one", "three")
        m.run_until_action("say", "stop")
        m.issue_reward()
        m.trial_end()

The operator bodies keep the PRIM notation: it is the theory's vocabulary,
and a body can be pasted unchanged from a .prims file.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field

from .chunk import Chunk
from .modules import ActionInstance
from .prim import parse_name
from .values import string2double, describe

REG_COND = re.compile(r"(>>(WM|RT|V|G)[0-9]+|(WM|RT|V|G)<<|((WM|V|RT|G|T|C)[0-9]+|nil|\*?[a-z][a-z0-9\-_]*) *(=|<>) *((WM|V|RT|G|T|C)[0-9]+|nil|\*?[a-z][a-z0-9\-_]*))")
REG_ACT = re.compile(r"(>>(WM|RT|V|G)[0-9]+|(WM|RT|V|G)<<|((WM|V|RT|G|T|C)[0-9]+|nil|\*?[a-z][a-z0-9\-_]*) *\-> *((WM|V|RT|G|AC|T|C)[0-9]+|\*[a-z][a-z0-9\-_]*))")
REG_BIND_TARGET = re.compile(r"->\*[a-z][a-z0-9\-_]*[a-z0-9]")
REG_CONSTANT = re.compile(r"\*?[a-z][a-z0-9\-_]*[a-z0-9]")
_TOKEN = re.compile(r"[^\s{}()\[\]]+|[{}()\[\]]")


class ParseError(Exception):
    pass


class ParseWarning(UserWarning):
    pass


def strip_comments(text: str) -> str:
    """Blank out // comments, keeping every other character in place."""
    out = []
    for line in text.split("\n"):
        i = line.find("//")
        out.append(line if i < 0 else line[:i] + " " * (len(line) - i))
    return "\n".join(out)


def tokens_of(text: str) -> list[str]:
    return _TOKEN.findall(strip_comments(text))


def _param_key(k: str) -> str:
    return k.replace("_", "-") + ":"


def _param_value(v) -> str:
    if v is True:
        return "t"
    if v is False or v is None:
        return "nil"
    return describe(float(v)) if isinstance(v, int) else describe(v)


# ====================================================================== definitions
@dataclass
class TaskSettings:
    name: str
    settings: list          # [(key with colon, value str | list[str])]


@dataclass
class OperatorDef:
    name: str
    body: str               # PRIMs, '==>' between conditions and actions
    activation: float | None = None


@dataclass
class SkillDef:
    name: str
    operators: list = field(default_factory=list)

    def operator(self, name: str, body: str, activation: float | None = None):
        self.operators.append(OperatorDef(name, body, activation))
        return self


@dataclass
class FactsDef:
    facts: list             # list of tuples of strings / numbers


@dataclass
class VisualDef:
    chunks: list


@dataclass
class ActionDef:
    name: str
    latency: float | None = None
    noise: float | None = None
    distribution: str | None = None
    output: str | None = None


@dataclass
class SjiDef:
    entries: list           # (j, i, value)


@dataclass
class ScriptDef:
    func: object
    init: bool = False
    source: str | None = None   # original PRIMs script text, if converted


# ====================================================================== Task
class Task:
    """A PRIMs task definition (one .prims file)."""

    def __init__(self, name: str, *, initial_skills=None, skills=None, references=None,
                 task_constants=None, **params):
        self.name = name
        self.defs: list = []
        self.source_dir = None
        settings = []
        if initial_skills is not None:
            settings.append(("initial-skills:", list(initial_skills)))
        if skills is not None:
            settings.append(("skills:", list(skills)))
        if references is not None:
            settings.append(("references:", list(references)))
        if task_constants is not None:
            settings.append(("task-constants:", list(task_constants)))
        for k, v in params.items():
            settings.append((_param_key(k), _param_value(v)))
        self.defs.append(TaskSettings(name, settings))

    # ---------------------------------------------------------------- builder API
    def set(self, **params):
        """Add parameter settings (applied at this point in the definition)."""
        self.defs.append(TaskSettings(self.name, [(_param_key(k), _param_value(v)) for k, v in params.items()]))
        return self

    def skill(self, name: str) -> SkillDef:
        s = SkillDef(name)
        self.defs.append(s)
        return s

    def facts(self, facts):
        """Facts as tuples: ("cf1", "count-fact", "one", "two")."""
        if isinstance(facts, str):
            facts = _parse_fact_lists(tokens_of(facts))
        self.defs.append(FactsDef([tuple(f) for f in facts]))
        return self

    def visual(self, chunks):
        if isinstance(chunks, str):
            chunks = _parse_fact_lists(tokens_of(chunks))
        self.defs.append(VisualDef([tuple(c) for c in chunks]))
        return self

    def action(self, name: str, latency=None, noise=None, distribution=None, output=None):
        self.defs.append(ActionDef(name, latency, noise, distribution, output))
        return self

    def sji(self, j: str, i: str, value: float):
        self.defs.append(SjiDef([(j, i, float(value))]))
        return self

    def script(self, func=None, *, source=None):
        def deco(f):
            self.defs.append(ScriptDef(f, init=False, source=source))
            return f
        return deco(func) if func is not None else deco

    def init_script(self, func=None, *, source=None):
        def deco(f):
            self.defs.append(ScriptDef(f, init=True, source=source))
            return f
        return deco(func) if func is not None else deco

    # ---------------------------------------------------------------- install
    def install(self, m, task_number: int):
        """Build the task into model m (Parser.parseModel)."""
        m.clear_trace()
        has_script = False
        for d in self.defs:
            if isinstance(d, TaskSettings):
                _install_settings(m, d, task_number)
            elif isinstance(d, SkillDef):
                _install_skill(m, d, task_number)
            elif isinstance(d, FactsDef):
                _install_facts(m, d, task_number)
            elif isinstance(d, VisualDef):
                _install_visual(m, d, task_number)
            elif isinstance(d, ActionDef):
                _install_action(m, d)
            elif isinstance(d, SjiDef):
                _install_sji(m, d)
            elif isinstance(d, ScriptDef):
                if d.init:
                    m.scenario.init_script = d.func
                else:
                    m.scenario.script = d.func
                    has_script = True
        m.dm.strings_to_chunks()
        if not has_script:
            raise ParseError("No start screen or script has been defined")


# ====================================================================== install helpers
def _install_settings(m, d: TaskSettings, task_number: int):
    m.add_to_trace_field(f"Defining task {d.name}")
    for key, value in d.settings:
        if key in ("initial-goals:", "initial-skills:", "initial-skill:"):
            goal = Chunk("currentGoalChunk", m)
            goal.set_slot("isa", "goal")
            for n, g in enumerate(value, start=1):
                _ensure_typed_chunk(m, g, "goaltype", task_number)
                goal.set_slot(f"slot{n}", g)
            m.current_goals = goal
        elif key in ("goals:", "skills:"):
            for g in value:
                _ensure_typed_chunk(m, g, "goaltype", task_number)
        elif key == "references:":
            for r in value:
                _ensure_typed_chunk(m, r, "reference", task_number)
        elif key == "task-constants:":
            c = Chunk("constants", m)
            c.set_slot("isa", "fact")
            for n, g in enumerate(value, start=1):
                c.set_slot(f"slot{n}", g)
            m.current_goal_constants = c
        elif key == "start-screen:":
            pass
        else:
            if not m.set_parameter(key, value):
                raise ParseError(f"Can't set parameter {key} to {value}")
            m.parameters.append((key, value))
    m.current_task = d.name


def _ensure_typed_chunk(m, name, type_name, task_number):
    existing = m.dm.chunks.get(name)
    if existing is None:
        c = Chunk(name, m)
        c.set_slot("isa", type_name)
        c.set_slot("slot1", name)
        c.fixed_activation = 1.0
        c.defined_in = [task_number]
        m.dm.add_to_dm(c)
    else:
        existing.defined_in.append(task_number)


def _install_skill(m, d: SkillDef, task_number: int):
    m.add_to_trace_field(f"Defining skill {d.name}")
    gc = m.dm.chunks.get(d.name)
    if gc is not None:
        gc.defined_in.append(task_number)
        if gc.type != "goaltype":
            gc.slotvals = {}
            gc.set_slot("isa", "goaltype")
            gc.set_slot("slot1", d.name)
            gc.fixed_activation = 1.0
    else:
        c = Chunk(d.name, m)
        c.set_slot("isa", "goaltype")
        c.set_slot("slot1", d.name)
        c.fixed_activation = 1.0
        c.defined_in = [task_number]
        m.dm.add_to_dm(c)
    for op in d.operators:
        _install_operator(m, d.name, op, task_number)


def compile_operator_body(body: str):
    """Turn an operator body into (condition PRIMs, action PRIMs, constants).

    Port of the PRIM-assembly loop in Parser.parseOperator: tokens are glued
    together until they form a PRIM, constants are replaced by C1, C2, ...,
    and comparisons are put in canonical order.
    """
    toks = tokens_of(body)
    conditions, actions, constants = [], [], []
    local = {}
    scanning_actions = False
    k = 0
    while k < len(toks):
        item = toks[k]
        k += 1
        if item.startswith('"'):                     # a quoted annotation: skip it
            if not (len(item) > 1 and item.endswith('"')):
                while k < len(toks):
                    k += 1
                    if toks[k - 1].endswith('"'):
                        break
            continue
        if item == "==>":
            scanning_actions = True
            continue
        if item in "{}()[]":
            raise ParseError(f"Unexpected '{item}' in operator definition")
        reg = REG_ACT if scanning_actions else REG_COND
        i = 0
        while reg.search(item) is None and i < 3:
            if k >= len(toks):
                raise ParseError("Unexpected end of operator definition")
            item += toks[k]
            k += 1
            i += 1
        if i == 3:
            raise ParseError(f"Error in parsing PRIM {item}")
        mb = REG_BIND_TARGET.search(item)
        if mb is not None:
            comp = mb.group(0)[2:]                       # "*var"
            constants.append("*" + comp)                 # stored as "**var"
            local["*" + comp] = len(constants)
            item = item.replace(comp, f"C{len(constants)}")
        for _ in range(2):
            mc = REG_CONSTANT.search(item)
            if mc is not None:
                comp = mc.group(0)
                if comp != "nil":
                    if comp in local:
                        item = item.replace(comp, f"C{local[comp]}")
                    else:
                        constants.append(comp)
                        local[comp] = len(constants)
                        item = item.replace(comp, f"C{len(constants)}")
        lb, ls, cop, rb, rs, new_prim = parse_name(item)
        if cop == "":
            # Swift accepts this silently and the PRIM then always fails at run time.
            # Typical cause: a one-letter constant (e.g. "q -> AC2"), which the
            # constant pattern does not recognise.
            warnings.warn(f"PRIM {item!r} cannot be parsed and will always fail", ParseWarning, stacklevel=3)
        prim = item if new_prim is None else new_prim
        (actions if scanning_actions else conditions).append(prim)
    return conditions, actions, constants


def _install_operator(m, goal_name: str, d: OperatorDef, task_number: int):
    if d.name not in m.dm.chunks:
        chunk = Chunk(d.name, m)
    else:
        chunk = m.generate_new_chunk(d.name)
    chunk.fixed_activation = m.dm.default_activation
    m.add_to_trace_field(f"Adding operator {d.name}")
    chunk.set_slot("isa", "operator")
    if d.activation is not None:
        chunk.fixed_activation = float(d.activation)
    conditions, actions, constants = compile_operator_body(d.body)
    for n, c in enumerate(constants, start=1):
        chunk.set_slot(f"slot{n}", c)
    if conditions:
        chunk.set_slot("condition", ";".join(conditions))
    if actions:
        chunk.set_slot("action", ";".join(actions))
    dm = m.dm
    if not dm.extended_assocs:
        chunk.assocs[goal_name] = [dm.default_operator_assoc, 0]
        chunk.assocs[chunk.name] = [dm.default_operator_self_assoc, 0]
    else:
        for i in range(1, 7):
            chunk.assocs[f"goal%slot{i}%{goal_name}"] = [dm.default_operator_assoc, 0]
        chunk.assocs[f"goal%last-operator%{chunk.name}"] = [dm.default_operator_self_assoc, 0]
    if not dm.inter_operator_learning:
        for ch in list(dm.chunks.values()):
            if ch.type == "operator" and goal_name in ch.assocs:
                chunk.assocs[ch.name] = [dm.default_inter_operator_assoc, 0]
                ch.assocs[chunk.name] = [dm.default_inter_operator_assoc, 0]
    chunk.defined_in = [task_number]
    dm.add_to_dm(chunk)


def _install_facts(m, d: FactsDef, task_number: int):
    for fact in d.facts:
        items = [describe(x) if not isinstance(x, str) else x for x in fact]
        chunk = None
        slotindex = 0
        j = 0
        while j < len(items):
            v = items[j]
            j += 1
            if v.startswith(":"):
                if v == ":activation" and chunk is not None and j < len(items):
                    num = string2double(items[j])
                    j += 1
                    if num is None:
                        raise ParseError(f"Invalid parameter value for {v}")
                    chunk.fixed_activation = num
                    continue
                raise ParseError(f"Unknown parameter {v}")
            if slotindex == 0:
                chunk = Chunk(v, m)
                chunk.set_slot("isa", "fact")
                chunk.fixed_activation = m.dm.default_activation
                slotindex = 1
            elif v == "nil":
                slotindex += 1
            elif v in m.dm.chunks:
                chunk.set_slot(f"slot{slotindex}", v)
                slotindex += 1
            elif v == chunk.name:
                chunk.set_slot_value(f"slot{slotindex}", chunk)
                slotindex += 1
            else:
                extra = Chunk(v, m)
                extra.set_slot("isa", "fact")
                extra.set_slot("slot1", v)
                extra.fixed_activation = m.dm.default_activation
                m.dm.add_to_dm(extra)
                chunk.set_slot(f"slot{slotindex}", v)
                slotindex += 1
        existing = m.dm.chunks.get(chunk.name)
        if existing is not None and not chunk.same_content(existing):
            raise ParseError(f"Fact {chunk.name} has already been defined with different values, consider renaming it.")
        m.dm.add_to_dm(chunk)
        chunk.defined_in = [task_number]


def _install_visual(m, d: VisualDef, task_number: int):
    for vis in d.chunks:
        items = [describe(x) if not isinstance(x, str) else x for x in vis]
        chunk = Chunk(items[0], m)
        chunk.set_slot("isa", "fact")
        for n, v in enumerate(items[1:], start=1):
            if v != "nil":
                chunk.set_slot(f"slot{n}", v)
        m.action.visicon[chunk.name] = chunk
        chunk.defined_in = [task_number]


def _install_action(m, d: ActionDef):
    a = ActionInstance(d.name, m.action.default_perceptual_action_latency)
    if d.output is not None:
        a.output_string = d.output
    if d.latency is not None:
        a.mean_latency = float(d.latency)
    if d.noise is not None:
        a.noise_value = float(d.noise)
    if d.distribution is not None:
        if d.distribution not in ("none", "uniform", "logistic"):
            raise ParseError(f"Unknown noise distribution {d.distribution}")
        a.noise_type = d.distribution
    m.action.actions[d.name] = a


def _install_sji(m, d: SjiDef):
    for j, i, v in d.entries:
        if j not in m.dm.chunks or i not in m.dm.chunks:
            raise ParseError(f"Sji between undefined chunks {j} and {i}")
        m.dm.chunks[i].assocs[j] = [float(v), 0]


# ====================================================================== .prims reader
def _parse_fact_lists(toks: list[str]) -> list[tuple]:
    out, cur, depth = [], None, 0
    for t in toks:
        if t == "(":
            cur = []
        elif t == ")":
            out.append(tuple(cur))
            cur = None
        elif cur is not None:
            cur.append(t)
    return out


def _extract_braced(text: str, start: int):
    """Return (content, index after closing brace) for text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ParseError("Unbalanced braces")


def parse_prims(text: str, source_dir: str | None = None) -> Task:
    """Read a .prims model into a :class:`Task`.

    Operator bodies are kept verbatim (with their comments) so that a
    converted Python file reads like the original.
    """
    from .script_translate import translate_script
    clean = strip_comments(text)          # same length as text: positions line up
    pos = 0
    pending: list = []
    task_name = None
    word = re.compile(r"\s*([^\s{}()\[\]]+)")
    brace = re.compile(r"\s*\{")

    def next_word():
        nonlocal pos
        mm = word.match(clean, pos)
        if mm is None:
            return None
        pos = mm.end()
        return mm.group(1)

    def braced():
        nonlocal pos
        mm = brace.match(clean, pos)
        if mm is None:
            raise ParseError("'{' expected")
        start = mm.end() - 1
        body, pos = _extract_braced(clean, start)
        return body, text[start + 1:pos - 1]

    while True:
        w = next_word()
        if w is None:
            if clean[pos:].strip():
                raise ParseError(f"Unexpected text: {clean[pos:pos + 40]!r}")
            break
        if w == "transition":
            raise ParseError("transition definitions are obsolete and not supported")
        if w != "define":
            raise ParseError(f"{w} is not a valid top-level statement")
        what = next_word()
        name = next_word() if what in ("task", "goal", "skill", "action") else None
        body, raw = braced()
        toks = _TOKEN.findall(body)
        if what in ("script", "init-script"):
            fname = "init_script" if what == "init-script" else "script"
            func, _src = translate_script(body, name=fname)
            pending.append(ScriptDef(func, init=(what == "init-script"), source=raw))
        elif what == "task":
            settings = []
            k = 0
            while k < len(toks):
                key = toks[k]
                k += 1
                if key in ("initial-goals:", "initial-skills:", "initial-skill:", "goals:", "skills:",
                           "references:", "task-constants:"):
                    if k >= len(toks) or toks[k] != "(":
                        raise ParseError(f"Missing '(' after {key}")
                    k += 1
                    vals = []
                    while toks[k] != ")":
                        vals.append(toks[k])
                        k += 1
                    k += 1
                    settings.append((key, vals))
                else:
                    if k >= len(toks):
                        raise ParseError(f"Missing parameter value for {key}")
                    settings.append((key, toks[k]))
                    k += 1
            pending.append(TaskSettings(name, settings))
            if task_name is None:
                task_name = name
        elif what in ("goal", "skill"):
            sk = SkillDef(name)
            for oname, params, obody in _split_operators(body, raw):
                act = params.get("activation")
                sk.operators.append(OperatorDef(oname, obody, None if act is None else float(act)))
            pending.append(sk)
        elif what == "facts":
            pending.append(FactsDef(_parse_fact_lists(toks)))
        elif what == "visual":
            pending.append(VisualDef(_parse_fact_lists(toks)))
        elif what == "action":
            a = ActionDef(name)
            k = 0
            while k + 1 < len(toks):
                key, val = toks[k], toks[k + 1]
                k += 2
                if key == "latency:":
                    a.latency = float(string2double(val))
                elif key == "noise:":
                    a.noise = float(string2double(val))
                elif key == "distribution:":
                    a.distribution = val
                elif key == "output:":
                    a.output = val
                else:
                    raise ParseError(f"Don't know about {key} in action declaration")
            pending.append(a)
        elif what == "sji":
            pending.append(SjiDef([(f[0], f[1], float(f[2])) for f in _parse_fact_lists(toks)]))
        elif what in ("screen", "inputs", "goal-action"):
            pass                                       # obsolete; Swift ignores them too
        else:
            raise ParseError(f"Don't know how to define {what}")
    task = Task.__new__(Task)
    task.name = task_name or "Undefined"
    task.defs = pending
    task.source_dir = source_dir
    return task


def _split_operators(body: str, raw: str):
    """Yield (name, params, raw body) for each ``operator name (params) { ... }``."""
    pos = 0
    hdr = re.compile(r"\s*operator\s+([^\s{}()\[\]]+)\s*(\(([^)]*)\))?\s*\{", re.S)
    while True:
        if not body[pos:].strip():
            return
        mm = hdr.match(body, pos)
        if mm is None:
            raise ParseError("Can only handle operator declarations within a skill definition, "
                             f"found {body[pos:].strip()[:30]!r}")
        params = {}
        if mm.group(3):
            for part in mm.group(3).split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()
        start = mm.end() - 1
        _content, pos = _extract_braced(body, start)
        yield mm.group(1), params, raw[start + 1:pos - 1]


def load_task(path: str) -> Task:
    """Load a task from a .prims file or a Python file defining ``task``."""
    path = os.fspath(path)
    if path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = getattr(mod, "task", None)
        if not isinstance(t, Task):
            raise ParseError(f"{path} does not define a Task called 'task'")
        t.source_dir = os.path.dirname(os.path.abspath(path))
        return t
    with open(path, encoding="utf-8") as f:
        return parse_prims(f.read(), source_dir=os.path.dirname(os.path.abspath(path)))
