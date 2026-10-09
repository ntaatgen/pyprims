"""A PRIMs session for interactive use: the notebook counterpart of the Swift GUI.

A :class:`Session` holds one model plus everything the Swift interface keeps
around it: the list of loaded tasks, the results of every run (one row per
task, as in the Swift chart), the trace level, and the state needed to step
through a trial one model step at a time.

    from pyprims.notebook import Session
    s = Session(seed=1)
    s.load("count.prims")          # or a .py file, a Task, or model text
    s.step()                       # one step (Swift's Step button)
    s.buffers()                    # what the Swift buffer panel shows
    s.run(100)                     # Swift's Run 100
    s.plot_results()

Stepping
--------
pyprims runs a script as an ordinary Python function, and the run functions
(``run_until_action`` and so on) loop over ``Model.new_step`` until their
stop condition holds.  To step, the session runs the trial's script in a
worker thread and holds that thread at every call of ``new_step``.  One
:meth:`Session.step` lets exactly one ``new_step`` through, which is one
half-step as in the Swift GUI: either the selection and execution of an
operator, or the module actions (retrieval, action, imaginal) that follow it.
Only one thread ever touches the model at a time, so a stepped trial is the
same computation as a trial that is simply run.

Inspection does not change the simulation
-----------------------------------------
Computing an activation for display draws activation noise, which would
advance the random stream and change later behaviour.  Everything the
session computes for display runs inside :func:`frozen`, which restores the
random state and the per-chunk noise caches afterwards.  A seeded model gives
the same results whether or not it is stepped and inspected.
"""
from __future__ import annotations

import os
import random
import re
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..dsl import Task, load_task, parse_prims, ParseError
from ..model import Model


# ====================================================================== helpers
@contextmanager
def frozen(model: Model):
    """Compute things for display without disturbing the simulation.

    Restores the random state, every chunk's cached noise, the model time and
    the buffers when the block ends.
    """
    state = model.rng.getstate()
    caches = [(c, c.noise_value, c.noise_time) for c in model.dm.chunks.values()]
    time, buffers = model.time, model.buffers
    try:
        yield
    finally:
        model.rng.setstate(state)
        for c, v, t in caches:
            c.noise_value = v
            c.noise_time = t
        model.time = time
        model.buffers = buffers


class NoOperatorWarning(UserWarning):
    """Some trials ended because no operator matched (or because they timed out)."""


warnings.simplefilter("always", NoOperatorWarning)      # one warning per run() call, never swallowed


def _warn_failed(failed: int, timed_out: int, ran: int):
    def of(k):
        return f"{k} of {ran} trial{'s' if ran != 1 else ''}"
    parts = []
    if failed:
        parts.append(f"in {of(failed)} the model stopped because no operator matched")
    if timed_out:
        parts.append(f"{of(timed_out)} reached time-threshold:")
    warnings.warn("; ".join(parts).capitalize() + ". The rest of the script then ran without the model, "
                  "so last-action() may be empty. trace(2) shows the last trial.",
                  NoOperatorWarning, stacklevel=3)


class _Abort(BaseException):
    """Raised inside the worker thread to abandon a stepped trial."""


@dataclass
class ResultRow:
    """One line in the results chart: consecutive trials of one task."""
    task: str
    task_number: int
    run: int
    points: list = field(default_factory=list)      # [(trial, y)]


@dataclass
class TaskEntry:
    name: str
    definition: Task
    source: str | None = None        # file path, or "<cell>" / "<text>"
    format: str = "prims"            # "prims" or "python"
    text: str | None = None          # the model text, when known


def _looks_like_prims(text: str) -> bool:
    return re.search(r"^\s*define\s+(task|skill|goal|facts|script)\b", text, re.M) is not None


# ====================================================================== Session
class Session:
    """A model with the bookkeeping of the Swift PRIMs interface.

    Parameters
    ----------
    seed:
        Seed for reproducible runs.  ``reset()`` draws a new model seed from
        a generator seeded with it, so a sequence of actions is reproducible
        as a whole; ``clear()`` starts that generator again.
    trace_level:
        Default level for :meth:`trace` (1 operators only ... 5 everything,
        as in the Swift trace menu; 0 shows only script output).
    echo_print:
        Also send the script's ``print(...)`` output to the notebook.  It is
        always in the trace.

    After :meth:`run`, a :class:`NoOperatorWarning` reports trials in which
    no operator matched or that timed out; set ``warn_failed_trials`` to
    False to switch this off.
    """

    def __init__(self, seed=None, trace_level: int = 3, echo_print: bool = False):
        self.seed = seed
        self.echo_print = echo_print
        self.trace_level = trace_level
        self.track_operator_activations = False
        self.warn_failed_trials = True
        self.activation_trace: dict[str, list] = {}
        self.entries: list[TaskEntry] = []
        self.rows: list[ResultRow] = []
        self.messages: list[str] = []
        self._seed_rng = random.Random(seed)
        self._thread = None
        self._to_worker = threading.Semaphore(0)
        self._to_main = threading.Semaphore(0)
        self._free_run = False
        self._abort = False
        self._finished = False
        self._error = None
        self._stop_requested = False
        self._lock = threading.RLock()
        self.conflict_capture = None     # (conflict set list, breakdowns) captured while stepping
        self.steps_in_trial = 0
        self._new_model()

    # ------------------------------------------------------------------ model
    def _new_model(self):
        self.model = Model(seed=self._seed_rng.random(), trace=True)
        self.model.echo_print = self.echo_print     # print() output always goes to the trace
        self.model.new_step = self._gated_new_step
        dm = self.model.dm
        for meth in ("retrieve", "partial_retrieve", "blended_retrieve"):
            setattr(dm, meth, self._capturing(getattr(dm, meth)))
        self.conflict_capture = None
        self._seen_results = 0
        self.activation_trace = {}
        self.steps_in_trial = 0

    @property
    def current_task(self) -> str | None:
        return self.model.current_task

    @property
    def time(self) -> float:
        return self.model.time

    @property
    def trial_active(self) -> bool:
        """True while a stepped trial is under way."""
        return self._thread is not None

    @property
    def task_names(self) -> list[str]:
        return [e.name for e in self.entries]

    def task_number(self, name: str) -> int:
        for i, e in enumerate(self.entries):
            if e.name == name:
                return i
        return -2

    def colour_index(self, model_task_number: int) -> int:
        """Session task number (the colour) of a task number used inside the model."""
        tasks = self.model.tasks
        if 0 <= model_task_number < len(tasks):
            return self.task_number(tasks[model_task_number].name)
        return -3

    def loaded_in_model(self, name: str) -> bool:
        return self.model.find_task(name) is not None

    # ------------------------------------------------------------------ loading
    def load(self, source, format: str | None = None, name: str | None = None):
        """Load a task into the model.

        ``source`` can be

        * a path to a ``.prims`` file or to a ``.py`` file that defines ``task``;
        * a :class:`pyprims.Task` built with the Python API;
        * model text, in PRIMs syntax or in Python.  The format is detected
          (``define task ...`` is PRIMs) unless ``format`` is ``"prims"`` or
          ``"python"``.

        Loading a task with a new name adds it to the model, so declarative
        memory and learned productions carry over (transfer).  Loading a task
        whose name is already loaded replaces its definition and resets the
        model, which is what you want after editing a model.
        """
        entry = self._make_entry(source, format)
        if name:
            entry.definition.name = name
            entry.name = name
        self._abort_trial()
        known = self.task_number(entry.name)
        if known >= 0:
            self.entries[known] = entry
            self.messages.append(f"Task {entry.name} redefined: the model has been reset.")
            self._new_model()
            self.model.load(entry.definition)
        else:
            self.entries.append(entry)
            self.model.load(entry.definition)
        self._new_row()
        return self

    def load_prims(self, text: str, **kw):
        """Load model text written in PRIMs syntax."""
        return self.load(text, format="prims", **kw)

    def load_python(self, code: str, **kw):
        """Load model text written with the Python Task API (it must define a Task)."""
        return self.load(code, format="python", **kw)

    def _make_entry(self, source, format):
        if isinstance(source, Task):
            return TaskEntry(source.name, source, "<Task>", "python")
        if isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source
                                              and source.strip().endswith((".prims", ".py"))):
            path = os.fspath(source)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            t = load_task(path)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            return TaskEntry(t.name, t, path, "python" if path.endswith(".py") else "prims", text)
        if not isinstance(source, str):
            raise TypeError("load() takes a path, a Task, or model text")
        if format is None:
            format = "prims" if _looks_like_prims(source) else "python"
        if format == "prims":
            t = parse_prims(source, source_dir=os.getcwd())
            return TaskEntry(t.name, t, "<text>", "prims", source)
        if format == "python":
            ns: dict = {"__name__": "pyprims_cell"}
            exec(compile(source, "<pyprims model>", "exec"), ns)
            t = ns.get("task")
            if not isinstance(t, Task):
                found = [v for v in ns.values() if isinstance(v, Task)]
                if len(found) != 1:
                    raise ParseError("Python model text must define one Task (preferably called 'task')")
                t = found[0]
            t.source_dir = t.source_dir or os.getcwd()
            return TaskEntry(t.name, t, "<text>", "python", source)
        raise ValueError("format must be 'prims' or 'python'")

    def select_task(self, name: str):
        """Make a task current (clicking a task in the Swift task list)."""
        self._abort_trial()
        if self.task_number(name) < 0:
            raise KeyError(f"task {name} is not loaded")
        if self.loaded_in_model(name):
            self.model.switch_task(name)
        else:
            self.model.load(self.entries[self.task_number(name)].definition)
        self._new_row()
        return self

    def reset(self):
        """Swift's Reset: a fresh model with the current task; results are kept."""
        self._abort_trial()
        name = self.current_task
        self._new_model()
        if name is not None and self.task_number(name) >= 0:
            self.model.load(self.entries[self.task_number(name)].definition)
            self._new_row()
        return self

    def clear(self):
        """Swift's Clear All: forget tasks, results and model."""
        self._abort_trial()
        self.entries = []
        self.rows = []
        self.messages = []
        self._seed_rng = random.Random(self.seed)
        self._new_model()
        return self

    # ------------------------------------------------------------------ results
    def _new_row(self):
        name = self.current_task or "?"
        num = self.task_number(name)
        if self.rows and not self.rows[-1].points:
            self.rows[-1].task, self.rows[-1].task_number = name, num
            return
        self.rows.append(ResultRow(name, num, len(self.rows)))

    def _collect_results(self):
        new = self.model.results[self._seen_results:]
        self._seen_results = len(self.model.results)
        if not new:
            return
        if not self.rows:
            self._new_row()
        row = self.rows[-1]
        for y in new:
            row.points.append((len(row.points) + 1, y))

    @property
    def results(self) -> list[float]:
        """Trial results (RT or plot-point values) of the current row."""
        return [y for _, y in self.rows[-1].points] if self.rows else []

    def results_table(self):
        """All results as a pandas DataFrame (task, run, trial, value), or a list of dicts."""
        recs = [dict(task=r.task, run=r.run, trial=x, value=y) for r in self.rows for x, y in r.points]
        try:
            from .frames import frame
            return frame(recs, ["task", "run", "trial", "value"])
        except ImportError:
            return recs

    # ------------------------------------------------------------------ the step gate
    def _gated_new_step(self):
        m = self.model
        stepping = threading.current_thread() is self._thread and not self._free_run
        if stepping:
            self._to_main.release()          # hand control back: paused before this step
            self._to_worker.acquire()
            if self._abort:
                raise _Abort()
        operator_half = not m.still_need_to_do_module_actions
        if stepping:
            self.steps_in_trial += 1
        if operator_half and self.track_operator_activations:
            self._record_operator_activations()
        Model.new_step(m)

    def _capturing(self, retrieve):
        """Wrap a retrieval so that, while stepping, its conflict set is explained at once.

        Right after the retrieval the buffers, time, noise and base levels are
        exactly those the model used, so the breakdown reproduces the
        activations.  (Later, a retrieval may have added a reference, and the
        buffers have changed.)
        """
        def wrapped(*args, **kw):
            result = retrieve(*args, **kw)
            if threading.current_thread() is self._thread and not self._free_run:
                from .views import explain_conflict_set
                self.conflict_capture = (self.model.dm.conflict_set, explain_conflict_set(self.model))
            return result
        wrapped.__wrapped__ = retrieve
        return wrapped

    def _record_operator_activations(self):
        m = self.model
        with frozen(m):
            t = m.time - m.start_time
            for ch in list(m.dm.chunks.values()):
                if ch.type == "operator":
                    pts = self.activation_trace.setdefault(ch.name, [])
                    if pts and t < pts[-1][0]:     # a script that runs several trials: keep the last
                        pts.clear()
                    pts.append((t, ch.activation()))

    def _check_ready(self):
        if self.current_task is None:
            raise RuntimeError("no task loaded: use load() first")

    def _start_trial(self):
        self._check_ready()
        self._finished = False
        self._error = None
        self._abort = False
        self._free_run = False
        self.steps_in_trial = 0
        self.activation_trace = {}
        self.model.tracing = True
        self._to_worker = threading.Semaphore(0)
        self._to_main = threading.Semaphore(0)

        def work():
            try:
                self.model.run_trial()
            except _Abort:
                pass
            except BaseException as e:      # reported in the main thread
                self._error = e
            finally:
                self._finished = True
                self._to_main.release()

        self._thread = threading.Thread(target=work, name="pyprims-trial", daemon=True)
        self._thread.start()
        self._to_main.acquire()             # paused before the first step, or finished
        return self._after_wait()

    def _after_wait(self) -> bool:
        """Clean up if the trial has finished; return True if it has."""
        if not self._finished:
            return False
        self._thread.join()
        self._thread = None
        self._free_run = False
        self.model.running = False
        self._collect_results()
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        return True

    def _resume(self) -> bool:
        self._to_worker.release()
        self._to_main.acquire()
        return self._after_wait()

    def _abort_trial(self):
        if self._thread is None:
            return
        self._abort = True
        self._to_worker.release()
        self._thread.join()
        self._thread = None
        self._abort = False
        self._finished = False
        self._error = None
        m = self.model
        m.running = False
        m.still_need_to_do_module_actions = False
        m.commit_to_trace(False)
        m.add_to_trace_field("[trial abandoned]")
        self._collect_results()

    # ------------------------------------------------------------------ running
    def step(self, n: int = 1):
        """Do n model steps (Swift's Step button; each is one half step).

        A step is either the selection and execution of one operator, or the
        module actions that follow it.  When a trial ends, the next step
        starts a new trial.  Returns self, so ``s.step().buffers()`` works.
        """
        with self._lock:
            for _ in range(n):
                if self._thread is None:
                    if self._start_trial():          # a script that never runs the model
                        continue
                self._resume()
        return self

    def step_operator(self, n: int = 1):
        """Do steps until n operators have been executed (module actions included)."""
        with self._lock:
            for _ in range(n):
                executed = False
                while True:
                    m = self.model
                    was_operator_half = self._thread is None or not m.still_need_to_do_module_actions
                    self.step()
                    executed = executed or was_operator_half
                    if self._thread is None:
                        break                         # the trial ended
                    if executed and not self.model.still_need_to_do_module_actions:
                        break                         # paused before the next operator
        return self

    def finish_trial(self):
        """Run the current stepped trial to its end."""
        with self._lock:
            if self._thread is not None:
                self._free_run = True
                self._resume()
        return self

    def run(self, n: int = 1, callback=None, every: int = 10):
        """Run n trials (Swift's Run, Run 10, Run 100).

        A stepped trial that is under way is finished first and counts as one
        of the n.  Only the last trial is traced.  ``callback(i, n)`` is
        called every ``every`` trials; :meth:`stop` (from another thread)
        ends the run after the current trial.
        """
        with self._lock:
            self._check_ready()
            self._stop_requested = False
            done = 0
            if self._thread is not None:
                self.finish_trial()
                done = 1
            m = self.model
            failed = timed_out = ran = 0
            try:
                for i in range(done, n):
                    if self._stop_requested:
                        break
                    m.tracing = (i == n - 1)
                    self.activation_trace = {}
                    try:
                        m.run_trial()
                    finally:
                        self._collect_results()
                    ran += 1
                    ends = [d.p1 for d in m.output_data if d.event_type == "trial-end"]
                    failed += "fail" in ends
                    timed_out += "timeout" in ends
                    if callback is not None and (i + 1) % every == 0:
                        callback(i + 1, n)
            finally:
                m.tracing = True
                self._stop_requested = False
            if self.warn_failed_trials and (failed or timed_out):
                _warn_failed(failed, timed_out, ran)
        return self

    def stop(self):
        """Ask a running ``run(n)`` to stop after the current trial."""
        self._stop_requested = True

    def run_time(self, seconds: float):
        """Run trials until the model has spent ``seconds`` of model time."""
        start = self.model.time
        while self.model.time - start < seconds:
            self.run(1)
        return self

    def batch(self, bprims: str, out_path: str | None = None, seed=None, progress: bool = False):
        """Run a .bprims batch script (independent of this session's model)."""
        from ..batch import run_batch
        return run_batch(bprims, out_path=out_path, seed=seed, progress=progress)

    # ------------------------------------------------------------------ inspection (views.py)
    def trace(self, level: int | None = None) -> str:
        """The trace of the current (or last) trial up to ``level``."""
        level = self.trace_level if level is None else level
        m = self.model
        lines = [s for lvl, s in m.trace if lvl <= level]
        lines += [s for lvl, s in m.trace_buffer if lvl <= level]     # not yet committed
        return "\n".join(lines)

    def status(self) -> dict:
        m = self.model
        return dict(task=self.current_task, time=round(m.time, 3),
                    trial_time=round(m.time - m.start_time, 3),
                    trial_active=self.trial_active, steps_in_trial=self.steps_in_trial,
                    next_half_step=("module actions" if m.still_need_to_do_module_actions else "operator"),
                    trials_in_row=len(self.rows[-1].points) if self.rows else 0,
                    operators=sum(1 for c in m.dm.chunks.values() if c.type == "operator"),
                    chunks=len(m.dm.chunks), productions=len(m.procedural.productions))

    def to_python(self, task: str | None = None) -> str:
        """The current (or named) task in Python syntax (converted if it was PRIMs)."""
        from ..convert import convert_text
        e = self.entries[self.task_number(task or self.current_task)]
        if e.format == "python":
            return e.text or "# defined as a Task object; no source text"
        return convert_text(e.text, (os.path.basename(e.source) if e.source and e.source[0] != "<" else "model.prims"))

    def source(self, task: str | None = None) -> str:
        e = self.entries[self.task_number(task or self.current_task)]
        return e.text or ""

    def __repr__(self):
        st = self.status()
        return (f"<pyprims Session task={st['task']} t={st['time']:.2f}s tasks={self.task_names} "
                f"rows={len(self.rows)} {'(trial under way)' if st['trial_active'] else ''}>")


# views and plots are attached as methods in views.py / plots.py
