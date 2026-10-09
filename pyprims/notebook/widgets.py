"""An ipywidgets dashboard with the controls and panels of the Swift PRIMs window.

    from pyprims.notebook import Dashboard
    Dashboard(folder="examples")          # or Dashboard(session)

Top bar: Editor, file + Load, Step, Step operator, Run, Run 10, Run 100,
Stop, Reset, Clear All, and the task list.  Below: the trace (with the
Swift detail levels) next to the buffers, and tabs for the results chart,
the PRIMs graph, declarative memory, the conflict set, the productions and
the task list.  Run 10 and Run 100 run in the background and refresh every
ten trials, so Stop works while they run.
"""
from __future__ import annotations

import glob
import html
import io
import os
import threading
import traceback

from .session import Session
from . import views
from . import plots

TRACE_LEVELS = [("Operators only", 1), ("+ Modules", 2), ("+ Productions", 3),
                ("+ Compilation", 4), ("All", 5)]
GRAPH_LEVELS = [("Task – skill – operator", 1), ("Skill – operator – PRIMs", 2),
                ("Compiled productions", "productions"), ("Facts in DM", "dm")]


def _png(draw, size=(7.5, 4.2), dpi=110) -> bytes:
    """Render with a non-pyplot Figure (safe from a background thread)."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=dpi)
    ax = fig.add_subplot(111)
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


class Dashboard:
    """The Swift PRIMs interface as notebook widgets.  Display it as the last expression of a cell."""

    def __init__(self, session: Session | None = None, folder: str | None = None, seed=None,
                 trace_height: int = 360):
        import ipywidgets as W
        self.W = W
        self.s = session or Session(seed=seed)
        self.folder = folder or os.getcwd()
        self._graph = None
        self._graph_key = None
        self._busy = False
        self._worker = None

        btn = lambda d, icon, tip, style="": W.Button(description=d, icon=icon, tooltip=tip,   # noqa: E731
                                                       button_style=style, layout=W.Layout(width="auto"))
        self.b_editor = W.ToggleButton(description="Editor", icon="edit", layout=W.Layout(width="auto"))
        self.f_file = W.Combobox(placeholder=".prims or .py file", options=self._files(),
                                 ensure_option=False, layout=W.Layout(width="230px"))
        self.b_load = btn("Load", "folder-open", "Load the file (adds a task, or redefines one)")
        self.b_step = btn("Step", "step-forward", "One model step (operator, or module actions)")
        self.b_stepop = btn("Step op", "forward", "Steps until the next operator has been executed")
        self.b_run = btn("Run", "play", "Run (the rest of) one trial")
        self.b_run10 = btn("Run 10", "fast-forward", "Run 10 trials")
        self.b_run100 = btn("Run 100", "fast-forward", "Run 100 trials")
        self.b_stop = btn("Stop", "stop", "Stop a running Run 10/100", "danger")
        self.b_reset = btn("Reset", "undo", "Fresh model with the current task; results are kept")
        self.b_clear = btn("Clear All", "trash", "Forget all tasks and results", "danger")
        self.d_task = W.Dropdown(description="Task", options=[], layout=W.Layout(width="220px"),
                                 style={"description_width": "36px"})
        self.status = W.HTML()
        self.error = W.HTML()

        # editor
        self.e_format = W.ToggleButtons(options=[("PRIMs", "prims"), ("Python", "python")], value="prims",
                                        style={"button_width": "70px"})
        self.e_text = W.Textarea(layout=W.Layout(width="100%", height="320px"),
                                 placeholder="Model text (PRIMs syntax or Python Task API)")
        self.e_load = btn("Load editor text", "upload", "Load the text in the editor as a task", "primary")
        self.e_open = btn("Current task's source", "file-text", "Show the current task's text")
        self.e_convert = btn("PRIMs → Python", "exchange", "Convert the PRIMs text to Python")
        self.e_name = W.Text(placeholder="file name to save", layout=W.Layout(width="220px"))
        self.e_save = btn("Save", "save", "Save the editor text to this file")
        self.editor = W.VBox([W.HBox([self.e_format, self.e_load, self.e_open, self.e_convert,
                                      self.e_name, self.e_save]), self.e_text],
                             layout=W.Layout(display="none", border="1px solid rgba(128,128,128,.35)",
                                             padding="4px"))

        # trace and buffers
        self.trace_level = W.Dropdown(options=TRACE_LEVELS, value=min(max(self.s.trace_level, 1), 5),
                                      layout=W.Layout(width="160px"))
        self.trace = W.HTML()
        self.trace_height = trace_height
        self.buffers = W.HTML()
        left = W.VBox([W.HBox([W.HTML("<b>Trace</b>"), self.trace_level]), self.trace],
                      layout=W.Layout(width="55%", padding="0 8px 0 0"))
        right = W.VBox([W.HTML("<b>Buffers</b>"), self.buffers], layout=W.Layout(width="45%"))
        middle = W.HBox([left, right])

        # tabs
        self.c_kind = W.Dropdown(options=[("Model results", "results"), ("Operator activations", "ops")],
                                 layout=W.Layout(width="200px"))
        self.c_window = W.BoundedIntText(value=0, min=0, max=1000, description="avg window",
                                         layout=W.Layout(width="160px"), style={"description_width": "75px"},
                                         tooltip="0 = use set-average-window from the script")
        self.c_track = W.Checkbox(value=self.s.track_operator_activations, description="track operator activations",
                                  indent=False)
        self.c_img = W.Image(format="png", layout=W.Layout(max_width="760px"))
        chart = W.VBox([W.HBox([self.c_kind, self.c_window, self.c_track]), self.c_img])
        self.g_level = W.Dropdown(options=GRAPH_LEVELS, value=2, layout=W.Layout(width="230px"))
        self.g_refresh = btn("Refresh layout", "refresh", "Compute a new layout")
        self.g_labels = W.Checkbox(value=True, description="labels", indent=False)
        self.g_img = W.Image(format="png", layout=W.Layout(max_width="860px"))
        graph = W.VBox([W.HBox([self.g_level, self.g_refresh, self.g_labels]), self.g_img])
        self.m_type = W.Dropdown(options=[("all", None), ("fact", "fact"), ("operator", "operator"),
                                          ("skill", "goaltype")], layout=W.Layout(width="130px"))
        self.m_search = W.Text(placeholder="name contains", layout=W.Layout(width="180px"))
        self.m_list = W.Select(options=[], rows=18, layout=W.Layout(width="340px"))
        self.m_detail = W.HTML()
        dm = W.VBox([W.HBox([self.m_type, self.m_search]),
                     W.HBox([self.m_list, W.Box([self.m_detail], layout=W.Layout(padding="0 0 0 10px"))])])
        self.cs = W.HTML()
        self.prods = W.HTML()
        self.tasklist = W.HTML()
        self.tabs = W.Tab(children=[chart, graph, dm, self.cs, self.prods, self.tasklist])
        for i, t in enumerate(["Chart", "PRIMs graph", "Declarative memory", "Conflict set", "Productions", "Tasks"]):
            self.tabs.set_title(i, t)

        bar = W.HBox([self.b_editor, self.f_file, self.b_load, W.HTML("&nbsp;"), self.b_step, self.b_stepop,
                      self.b_run, self.b_run10, self.b_run100, self.b_stop, W.HTML("&nbsp;"), self.b_reset,
                      self.b_clear, self.d_task], layout=W.Layout(flex_flow="row wrap", align_items="center"))
        self.widget = W.VBox([bar, self.status, self.error, self.editor, middle, self.tabs])

        # wiring
        self.b_editor.observe(lambda c: self._toggle_editor(), "value")
        self.b_load.on_click(self._guard(self._on_load))
        self.b_step.on_click(self._guard(lambda _: self.s.step()))
        self.b_stepop.on_click(self._guard(lambda _: self.s.step_operator()))
        self.b_run.on_click(self._guard(lambda _: self.s.run(1)))
        self.b_run10.on_click(lambda _: self._run_background(10))
        self.b_run100.on_click(lambda _: self._run_background(100))
        self.b_stop.on_click(lambda _: self.s.stop())
        self.b_reset.on_click(self._guard(lambda _: self.s.reset(), relayout=True))
        self.b_clear.on_click(self._guard(lambda _: self.s.clear(), relayout=True))
        self.d_task.observe(self._on_task, "value")
        self.e_load.on_click(self._guard(self._on_editor_load, relayout=True))
        self.e_open.on_click(self._guard(self._on_editor_open, refresh=False))
        self.e_convert.on_click(self._guard(self._on_convert, refresh=False))
        self.e_save.on_click(self._guard(self._on_save, refresh=False))
        self.trace_level.observe(lambda c: self._refresh_trace(), "value")
        self.c_kind.observe(lambda c: self._refresh_tab(), "value")
        self.c_window.observe(lambda c: self._refresh_tab(), "value")
        self.c_track.observe(self._on_track, "value")
        self.g_level.observe(lambda c: self._refresh_tab(relayout=True), "value")
        self.g_refresh.on_click(lambda _: self._refresh_tab(relayout=True))
        self.g_labels.observe(lambda c: self._refresh_tab(), "value")
        self.m_type.observe(lambda c: self._refresh_tab(), "value")
        self.m_search.observe(lambda c: self._refresh_tab(), "value")
        self.m_list.observe(self._on_chunk, "value")
        self.tabs.observe(lambda c: self._refresh_tab(), "selected_index")
        self.refresh()

    # ------------------------------------------------------------------ display
    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def _files(self):
        out = []
        for ext in ("*.prims", "*.py"):
            out += [os.path.relpath(p, self.folder) for p in glob.glob(os.path.join(self.folder, ext))]
        return sorted(out)

    def _toggle_editor(self):
        self.editor.layout.display = "flex" if self.b_editor.value else "none"
        if self.b_editor.value and not self.e_text.value and self.s.current_task:
            self._on_editor_open(None)

    # ------------------------------------------------------------------ callbacks
    def _guard(self, fn, refresh=True, relayout=False):
        def handler(arg):
            if self._busy:
                return
            self.error.value = ""
            try:
                fn(arg)
            except Exception as e:                 # show errors in the dashboard, not in the log
                tb = traceback.format_exception_only(type(e), e)[-1]
                self.error.value = (f"<div style='color:#D32F2F;font-family:monospace;white-space:pre-wrap'>"
                                    f"{html.escape(tb)}</div>")
            if refresh:
                self.refresh(relayout=relayout)
        return handler

    def _on_load(self, _):
        name = self.f_file.value.strip()
        if not name:
            raise ValueError("type or choose a .prims or .py file")
        path = name if os.path.isabs(name) else os.path.join(self.folder, name)
        self.s.load(path)
        self._graph_key = None

    def _on_editor_load(self, _):
        self.s.load(self.e_text.value, format=self.e_format.value)
        self._graph_key = None

    def _on_editor_open(self, _):
        if not self.s.current_task:
            return
        e = self.s.entries[self.s.task_number(self.s.current_task)]
        self.e_format.value = e.format
        self.e_text.value = e.text or ""
        if e.source and not e.source.startswith("<"):
            self.e_name.value = e.source

    def _on_convert(self, _):
        from ..convert import convert_text
        if self.e_format.value != "prims":
            raise ValueError("the editor holds Python already")
        self.e_text.value = convert_text(self.e_text.value)
        self.e_format.value = "python"
        if self.e_name.value.endswith(".prims"):
            self.e_name.value = self.e_name.value[:-6] + ".py"

    def _on_save(self, _):
        name = self.e_name.value.strip()
        if not name:
            raise ValueError("give a file name")
        path = name if os.path.isabs(name) else os.path.join(self.folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.e_text.value)
        self.f_file.options = self._files()
        self.error.value = f"<span style='opacity:.7'>Saved {html.escape(path)}</span>"

    def _on_task(self, change):
        if self._busy or change["new"] is None or change["new"] == self.s.current_task:
            return
        self._guard(lambda _: self.s.select_task(change["new"]))(None)

    def _on_track(self, change):
        self.s.track_operator_activations = change["new"]
        if change["new"]:
            self.c_kind.value = "ops"

    def _on_chunk(self, change):
        if change["new"]:
            try:
                self.m_detail.value = views.chunk_view(self.s, change["new"])._repr_html_()
            except KeyError:
                self.m_detail.value = ""

    def _set_busy(self, busy):
        self._busy = busy
        for b in (self.b_load, self.b_step, self.b_stepop, self.b_run, self.b_run10, self.b_run100,
                  self.b_reset, self.b_clear, self.e_load):
            b.disabled = busy
        self.d_task.disabled = busy

    def _run_background(self, n):
        if self._busy:
            return
        self.error.value = ""
        self._set_busy(True)

        def progress(i, total):
            self.status.value = self._status_html(f"running … {i}/{total}")
            self._refresh_tab()

        def work():
            try:
                self.s.run(n, callback=progress, every=10)
            except Exception as e:
                tb = traceback.format_exception_only(type(e), e)[-1]
                self.error.value = (f"<div style='color:#D32F2F;font-family:monospace'>{html.escape(tb)}</div>")
            finally:
                self._set_busy(False)
                self.refresh()

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ refresh
    def _status_html(self, extra=""):
        st = self.s.status()
        parts = [f"time {st['time']:.3f} s", f"trial time {st['trial_time']:.3f} s",
                 f"{st['trials_in_row']} trials in this row",
                 f"{st['chunks']} chunks", f"{st['operators']} operators", f"{st['productions']} productions"]
        if st["trial_active"]:
            parts.insert(0, f"<b>trial under way</b>: {st['steps_in_trial']} steps, next: {st['next_half_step']}")
        if extra:
            parts.insert(0, f"<b>{extra}</b>")
        msg = ""
        if self.s.messages:
            msg = f" · <i>{html.escape(self.s.messages[-1])}</i>"
            self.s.messages.clear()
        return "<span style='opacity:.8;font-size:12px'>" + " · ".join(parts) + msg + "</span>"

    def _refresh_trace(self):
        text = html.escape(self.s.trace(self.trace_level.value))
        self.trace.value = (f"<div style='height:{self.trace_height}px;overflow:auto;display:flex;"
                            f"flex-direction:column-reverse;border:1px solid rgba(128,128,128,.35);padding:4px'>"
                            f"<pre style='margin:0;font-size:11.5px;line-height:1.3'>{text}</pre></div>")

    def refresh(self, relayout=False):
        s = self.s
        names = s.task_names
        if list(self.d_task.options) != names:
            self.d_task.options = names
        if s.current_task in names and self.d_task.value != s.current_task:
            self.d_task.unobserve(self._on_task, "value")
            self.d_task.value = s.current_task
            self.d_task.observe(self._on_task, "value")
        self.status.value = self._status_html()
        self._refresh_trace()
        self.buffers.value = (views.BufferView(s)._repr_html_() if s.current_task else
                              "<i>Load a model to start.</i>")
        self._refresh_tab(relayout=relayout)

    def _refresh_tab(self, relayout=False):
        s = self.s
        i = self.tabs.selected_index
        try:
            if i == 0:
                if self.c_kind.value == "results":
                    w = self.c_window.value or None
                    self.c_img.value = _png(lambda ax: plots.plot_results(s, ax=ax, average_window=w))
                else:
                    self.c_img.value = _png(lambda ax: plots.plot_operator_activations(s, ax=ax))
            elif i == 1:
                self._refresh_graph(relayout)
            elif i == 2:
                recs = views.dm_records(s, self.m_type.value, self.m_search.value or None)[:400]
                opts = [(f"{r['name']}   {r['activation']:.3f}", r["name"]) for r in recs]
                keep = self.m_list.value
                self.m_list.options = opts
                if keep in [o[1] for o in opts]:
                    self.m_list.value = keep
                    self._on_chunk({"new": keep})
            elif i == 3:
                self.cs.value = views.ConflictSetView(s)._repr_html_()
            elif i == 4:
                recs = views.production_records(s)
                rows = "".join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['utility']:.3f}</td>"
                               f"<td>{r['size']}</td><td>{html.escape(str(r['task']))}</td></tr>" for r in recs[:300])
                self.prods.value = views._wrap(
                    "<div style='max-height:420px;overflow:auto'><table><tr><th>production</th><th>utility</th>"
                    f"<th>PRIMs</th><th>learned in</th></tr>{rows}</table></div>"
                    f"<div class='pp-muted'>primU = {s.model.procedural.prim_u}; a compiled production is used "
                    "once its utility exceeds primU.</div>")
            elif i == 5:
                self.tasklist.value = views.TaskListView(s)._repr_html_()
        except Exception as e:
            self.error.value = (f"<div style='color:#D32F2F;font-family:monospace'>"
                                f"{html.escape(traceback.format_exception_only(type(e), e)[-1])}</div>")

    def _refresh_graph(self, relayout):
        s = self.s
        level = self.g_level.value
        g = plots.build_graph(s, level)
        key = (level, tuple(sorted(g.nodes)))
        if relayout or self._graph is None or key != self._graph_key:
            old = self._graph.nodes if (self._graph is not None and not relayout
                                        and self._graph_key and self._graph_key[0] == level) else {}
            for name, n in g.nodes.items():         # known nodes stay where they were
                if name in old:
                    n.x, n.y, n.fixed = old[name].x, old[name].y, True
            plots.layout(g)
            for n in g.nodes.values():
                n.fixed = False
        else:
            for name, n in g.nodes.items():
                n.x, n.y = self._graph.nodes[name].x, self._graph.nodes[name].y
        self._graph, self._graph_key = g, key
        self.g_img.value = _png(lambda ax: plots.plot_graph(s, level, ax=ax, graph=g, labels=self.g_labels.value),
                                size=(8.5, 7.0))
