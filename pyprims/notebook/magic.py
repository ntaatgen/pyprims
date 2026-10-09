"""IPython cell magics for writing models in notebook cells.

    %load_ext pyprims.notebook           # (importing pyprims.notebook also registers them)

    %%prims                              # a cell in standard PRIMs syntax
    define task count { ... }
    ...

    %%pyprims                            # a cell using the Python Task API
    task = Task("count", ...)
    ...

Both load the task into a :class:`Session` called ``session`` in the
notebook (created if needed); give another name as the argument, e.g.
``%%prims s2``.  Options: ``--reset`` starts that session afresh
(``clear()``) before loading; ``--python`` (``%%prims`` only) also prints
the model converted to Python.
"""
from __future__ import annotations

import os
import shlex

from ..dsl import Task
from .session import Session


def _session(shell, name: str, reset: bool) -> Session:
    s = shell.user_ns.get(name)
    if not isinstance(s, Session):
        s = Session()
        shell.user_ns[name] = s
    elif reset:
        s.clear()
    return s


def _parse(line: str):
    args = shlex.split(line or "")
    name = "session"
    flags = set()
    for a in args:
        if a.startswith("--"):
            flags.add(a[2:])
        else:
            name = a
    return name, flags


def prims(line, cell):
    """Load a cell written in PRIMs syntax: ``%%prims [session] [--reset] [--python]``."""
    from IPython import get_ipython
    shell = get_ipython()
    name, flags = _parse(line)
    s = _session(shell, name, "reset" in flags)
    s.load(cell, format="prims")
    print(f"Loaded task {s.current_task!r} into {name} (tasks: {', '.join(s.task_names)})")
    if "python" in flags:
        print()
        print(s.to_python())


def pyprims(line, cell):
    """Run a cell that defines a Task with the Python API and load it: ``%%pyprims [session] [--reset]``.

    The cell runs in the notebook's namespace, so it can use earlier
    variables, and its names (``task`` ...) stay available afterwards.
    """
    from IPython import get_ipython
    shell = get_ipython()
    name, flags = _parse(line)
    before = {id(v) for v in shell.user_ns.values() if isinstance(v, Task)}
    ns = shell.user_ns
    ns.setdefault("Task", Task)
    exec(compile(cell, "<%%pyprims cell>", "exec"), ns)
    new = [v for v in ns.values() if isinstance(v, Task) and id(v) not in before]
    if not new:
        t = ns.get("task")
        if not isinstance(t, Task):
            raise ValueError("the cell must define a Task (for example task = Task(...))")
        new = [t]
    s = _session(shell, name, "reset" in flags)
    for t in dict.fromkeys(new):          # keep order, no duplicates
        t.source_dir = t.source_dir or os.getcwd()
        s.load(t)
        e = s.entries[s.task_number(t.name)]
        e.text, e.source = cell, "<cell>"
        print(f"Loaded task {t.name!r} into {name} (tasks: {', '.join(s.task_names)})")


def register(shell=None):
    if shell is None:
        from IPython import get_ipython
        shell = get_ipython()
    if shell is None:
        return False
    shell.register_magic_function(prims, "cell", "prims")
    shell.register_magic_function(pyprims, "cell", "pyprims")
    return True
