"""Batch runs from .bprims scripts (port of BatchRun.swift).

The output file has the same format as the Swift version, one line per event:

    repeat task label trial time event p1 p2 p3 in0 in1 in2 in3 in4 firings

Supported commands: ``repeat n``, ``reset``, ``run task label n [params]``,
``run-time task label seconds [params]``, ``save-image name``,
``load-image name`` (images are Python pickles, not Swift archives), ``done``.
Task files are looked up as ``task.py`` first, then ``task.prims``.

:func:`run_trials` does the same from Python, and returns records for pandas.
"""
from __future__ import annotations

import os
import pickle
import random

from .model import Model
from .dsl import load_task


def _find_task_file(directory: str, name: str) -> str:
    for ext in (".py", ".prims"):
        p = os.path.join(directory, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Task {name} is not loaded nor can it be found in {directory}")


def format_line(rep: int, task: str, label: str, trial: int, dl) -> str:
    return (f"{rep} {task} {label} {trial} {repr(float(dl.time))} {dl.event_type} {dl.p1} {dl.p2} {dl.p3} "
            + "".join(x + " " for x in dl.inputs) + f"{dl.firings}\n")


def run_batch(bprims: str, out_path: str | None = None, seed=None, directory: str | None = None,
              progress: bool = False) -> list[str]:
    """Run a .bprims file; return the output lines (and write them if out_path)."""
    directory = directory or os.path.dirname(os.path.abspath(bprims))
    with open(bprims, encoding="utf-8") as f:
        lines = [ln.split("//")[0].strip() for ln in f.read().splitlines()]
    lines = [ln for ln in lines if ln]
    first = lines[0].split()
    if first[0] != "repeat":
        raise ValueError("a batch script starts with 'repeat n'")
    repeats = int(first[1])
    master = random.Random(seed)
    out: list[str] = []
    model = Model(seed=master.random(), trace=False, batch_mode=True)
    task_cache: dict[str, object] = {}
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for rep in range(repeats):
            if progress:
                print(f"Run #{rep + 1}", flush=True)
            for ln in lines[1:]:
                parts = ln.split()
                cmd = parts[0]
                if cmd in ("run", "run-time"):
                    task_name, label, crit = parts[1], parts[2], float(parts[3])
                    model.batch_parameters = parts[4:]
                    idx = model.find_task(task_name)
                    if idx is None:
                        if task_name not in task_cache:
                            task_cache[task_name] = load_task(_find_task_file(directory, task_name))
                        model.load(task_cache[task_name])
                        idx = model.find_task(task_name)
                    model.load_or_reload_task(idx)
                    model.arg = label
                    start = model.time
                    j = 0
                    while (cmd == "run" and j < int(crit)) or (cmd == "run-time" and model.time - start < crit):
                        j += 1
                        model.run_trial()
                        text = "".join(format_line(rep, task_name, label, j, dl) for dl in model.output_data)
                        out.append(text)
                        if fh:
                            fh.write(text)
                elif cmd == "reset":
                    model = Model(seed=master.random(), trace=False, batch_mode=True)
                elif cmd == "save-image":
                    with open(os.path.join(directory, parts[1] + ".pybrain"), "wb") as f:
                        pickle.dump(model, f)
                elif cmd == "load-image":
                    with open(os.path.join(directory, parts[1] + ".pybrain"), "rb") as f:
                        model = pickle.load(f)
                elif cmd in ("repeat", "done"):
                    pass
                else:
                    raise ValueError(f"unknown batch command {cmd}")
    finally:
        if fh:
            fh.close()
    return out


# ====================================================================== batch runs from Python
def run_trials(model: Model, n: int, task=None, events: bool = False, **fields) -> list[dict]:
    """Run ``n`` trials of a task and return the data as records (dicts), for pandas.

    This is the Python counterpart of a ``run`` line in a .bprims file::

        rows = []
        for rep in range(10):
            m = Model(seed=rep, trace=False, batch_mode=True)               # reset
            rows += run_trials(m, 100, "count.prims", repeat=rep, label="train")
            rows += run_trials(m, 100, "semantic.prims", repeat=rep, label="transfer")
        data = pandas.DataFrame(rows)

    ``task`` is a file name, a :class:`~pyprims.Task`, or ``None`` for the
    current task.  As in a batch file, a task that is already loaded is made
    current again rather than loaded a second time, so declarative memory and
    learned productions carry over.

    ``fields`` (``repeat=``, ``label=``, a condition, a parameter value ...)
    are copied into every record.

    With ``events=False`` (default) there is one record per trial end:
    ``trial`` (the number of the run of the script, from 1), ``task``,
    ``result`` (the trial time, or the value given to ``plot-point``) and
    ``outcome`` (``success``, ``fail`` when no operator matched, or
    ``timeout``), plus ``in0`` ... ``in4`` when the script set them with
    ``set-data-file-field``.  A script that ends several trials per run
    (``trial-end`` in a loop) gives several records with the same ``trial``.

    With ``events=True`` there is one record per event, with the columns of
    the batch output: ``trial task time event p1 p2 p3 in0 ... in4 firings``
    (``time`` is relative to the start of the trial).
    """
    if task is not None:
        _switch_or_load(model, task)
    name = model.current_task
    rows: list[dict] = []
    for j in range(1, n + 1):
        before = len(model.results)
        model.run_trial()
        if events:
            for dl in model.output_data:
                rec = dict(fields, trial=j, task=name, time=float(dl.time), event=dl.event_type,
                           p1=dl.p1, p2=dl.p2, p3=dl.p3)
                rec.update({f"in{k}": v for k, v in enumerate(dl.inputs)})
                rec["firings"] = dl.firings
                rows.append(rec)
            continue
        ends = [dl for dl in model.output_data if dl.event_type == "trial-end"]
        for k, value in enumerate(model.results[before:]):
            end = ends[k] if k < len(ends) else None
            rec = dict(fields, trial=j, task=name, result=float(value),
                       outcome=end.p1 if end is not None else "void")
            if end is not None:
                rec.update({f"in{i}": v for i, v in enumerate(end.inputs) if v != "void"})
            rows.append(rec)
    return rows


def _switch_or_load(model: Model, task):
    from .dsl import Task
    if isinstance(task, Task):
        definition = task
    else:
        definition = load_task(os.fspath(task))
    idx = model.find_task(definition.name)
    if idx is None:
        model.load(definition)
        idx = model.find_task(definition.name)
    model.load_or_reload_task(idx)
