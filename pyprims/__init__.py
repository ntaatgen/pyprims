"""pyprims: a Python implementation of the PRIMs cognitive architecture.

PRIMs (Taatgen, 2013, Psychological Review) represents procedural knowledge
as declarative operator chunks built from primitive information-processing
elements (PRIMs), and learns task-general productions by compiling PRIMs.
This package is a port of the Swift implementation; see PORTING.md for the
module map and for every place where behaviour knowingly differs.

Quick start::

    from pyprims import Model, load_task
    m = Model(seed=1)
    m.load("count.prims")        # or a .py file that defines `task`
    m.run(10)
    print(m.get_trace(3))
"""
from .model import Model, DataLine
from .dsl import Task, load_task, parse_prims, ParseError
from .batch import run_batch, run_trials

__all__ = ["Model", "Task", "load_task", "parse_prims", "run_batch", "run_trials", "ParseError", "DataLine"]
__version__ = "0.2.0"
