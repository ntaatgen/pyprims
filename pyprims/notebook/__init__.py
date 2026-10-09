"""Interactive use of pyprims in Jupyter: the functionality of the Swift PRIMs GUI.

    from pyprims.notebook import Session, Dashboard

    s = Session(seed=1)
    s.load("count.prims")            # .prims, .py, a Task, or model text
    s.step()                         # one model step
    s.buffers()                      # buffer panel        s.trace(3)   trace
    s.conflict_set()                 # conflict trace      s.dm()       declarative memory
    s.run(100)                       # Run 100             s.productions()
    s.plot_results()                 # results chart       s.plot_graph(2)
    Dashboard(s)                     # all of it as widgets (needs ipywidgets)

Models can be written in notebook cells, in either syntax:

    %%prims                          %%pyprims
    define task count { ... }        task = Task("count", ...)

Modules: ``session`` (stepping, running, results), ``views`` (buffers, DM,
conflict set, productions, associations), ``plots`` (results chart, operator activations,
PRIMs graph, batch output), ``widgets`` (the dashboard), ``magic``.
"""
from .session import Session, frozen, NoOperatorWarning
from . import views as _views          # noqa: F401  (adds the view methods to Session)
from .plots import (figure, show_figure, plot_results, plot_operator_activations, plot_graph, build_graph, layout,
                    draw_graph, read_batch, plot_batch, moving_average)
from .views import dm_table, production_table, association_table, BufferView, ConflictSetView, chunk_view


def Dashboard(*args, **kwargs):
    """The widget interface (see :class:`pyprims.notebook.widgets.Dashboard`)."""
    from .widgets import Dashboard as _D
    return _D(*args, **kwargs)


def load_ipython_extension(ipython):
    from .magic import register
    register(ipython)


try:                                   # register %%prims / %%pyprims when imported in IPython
    from IPython import get_ipython as _gi
    if _gi() is not None:
        from .magic import register as _register
        _register(_gi())
except ImportError:                    # pragma: no cover
    pass

__all__ = ["Session", "NoOperatorWarning", "Dashboard", "figure", "show_figure", "frozen", "plot_results", "plot_operator_activations", "plot_graph",
           "build_graph", "layout", "draw_graph", "read_batch", "plot_batch", "dm_table",
           "production_table", "association_table", "BufferView", "ConflictSetView", "chunk_view", "moving_average"]
