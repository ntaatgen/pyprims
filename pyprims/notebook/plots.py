"""Plots: the Swift results chart, operator activations, the PRIMs graph, batch output.

Every function takes an optional matplotlib ``ax``.  Without one it draws a
new figure and shows it in the notebook as a PNG, independently of the
pyplot backend.  To combine plots, use :func:`figure`.  Each returns the axes.
"""
from __future__ import annotations

import contextlib
import functools
import gzip
import inspect
import math
import random
from dataclasses import dataclass, field

from ..values import describe
from .session import Session
from .views import task_color

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", ":", "-."]


def _in_kernel() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except ImportError:
        return False


def show_figure(fig, dpi=110):
    """Show a matplotlib Figure in the notebook as a PNG.

    This does not depend on the pyplot backend (``%matplotlib inline`` and
    friends), which on some Jupyter installations does not display figures.
    """
    import io
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    display(Image(data=buf.getvalue(), format="png"))


@contextlib.contextmanager
def figure(nrows=1, ncols=1, figsize=(7, 4), **kw):
    """Axes for several plots in one figure, shown when the block ends::

        with figure(1, 2, figsize=(14, 6)) as axes:
            s.plot_graph(1, ax=axes[0])
            s.plot_graph(2, ax=axes[1])

    Outside a notebook kernel it uses pyplot (call ``plt.show()`` yourself).
    """
    if _in_kernel():
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize)
        axes = fig.subplots(nrows, ncols, **kw)
        yield axes
        fig.tight_layout()
        show_figure(fig)
    else:
        import matplotlib.pyplot as plt
        _, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
        yield axes


def _plotting(size):
    """Decorator: when no ``ax`` is given, draw into a new figure and show it."""
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            if bound.arguments.get("ax") is not None:
                return fn(*args, **kwargs)
            with figure(1, 1, figsize=size) as ax:
                bound.arguments["ax"] = ax
                result = fn(*bound.args, **bound.kwargs)
            return result
        return wrapper
    return deco


def moving_average(ys, window):
    """The Swift chart's smoothing: each point is the mean of itself and the window-1 before it."""
    if window <= 1:
        return list(ys)
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        out.append(sum(ys[lo:i + 1]) / (i - lo + 1))
    return out


# ====================================================================== results chart
@_plotting((7, 4))
def plot_results(session: Session, ax=None, average_window: int | None = None, title: str | None = None,
                 tasks=None, legend: bool = True, labels=None):
    """Trial results, one line per run of a task (the 'Model results' chart).

    A new line starts when a task is loaded, selected or reset.  Lines are
    coloured by task; when a task has several lines (for example after a
    reset), the later ones are dashed or dotted and labelled "task, run 2",
    "task, run 3" ...  ``labels`` replaces these labels: a list with one
    label per plotted line, in order.  ``tasks`` restricts the plot to
    some tasks.  The window of the moving average is ``set-average-window``
    from the script unless given here.
    """
    m = session.model
    w = average_window if average_window is not None else max(1, int(getattr(m, "average_window", 1)))
    rows = [r for r in session.rows if r.points and (tasks is None or r.task in tasks)]
    per_task: dict = {}
    for r in rows:
        per_task[r.task] = per_task.get(r.task, 0) + 1
    seen: dict = {}
    any_label = False
    for i, row in enumerate(rows):
        seen[row.task] = k = seen.get(row.task, 0) + 1
        if labels is not None:
            label = labels[i] if i < len(labels) else None
        else:
            label = row.task if per_task[row.task] == 1 else f"{row.task}, run {k}"
        any_label = any_label or label is not None
        xs = [x for x, _ in row.points]
        ys = moving_average([y for _, y in row.points], w)
        ax.plot(xs, ys, color=task_color(row.task_number), marker=MARKERS[row.task_number % len(MARKERS)],
                linestyle=LINESTYLES[(k - 1) % len(LINESTYLES)],
                markersize=4, linewidth=1.3, label=label, markeredgecolor="black", markeredgewidth=0.3)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Value" if m.custom_points else "Time (sec)")
    t = title or m.graph_title
    if t:
        ax.set_title(t + (f"  (moving average over {w})" if w > 1 else ""))
    if legend and any_label:
        ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return ax


@_plotting((7, 4))
def plot_operator_activations(session: Session, ax=None, operators=None, legend: bool = True):
    """Operator activations at each operator step of the last trial (Swift 'Operator activations').

    Needs ``session.track_operator_activations = True`` before the trial.
    """
    tr = session.activation_trace
    if not tr:
        ax.text(0.5, 0.5, "No data: set session.track_operator_activations = True and run a trial",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return ax
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    for i, (name, pts) in enumerate(sorted(tr.items())):
        if operators is not None and name not in operators:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[i % len(MARKERS)], markersize=3,
                linewidth=1, label=name, color=cmap(i % 20))
    ax.set_xlabel("Time in trial (sec)")
    ax.set_ylabel("Activation")
    ax.set_title("Operator activations")
    if legend:
        ax.legend(frameon=False, fontsize=7, ncol=2 if len(tr) > 8 else 1)
    ax.grid(alpha=0.25)
    return ax


# ====================================================================== graphs
@dataclass
class Node:
    name: str
    label: str
    task: int = -2              # colour index (Swift taskNumber)
    kind: str = "prim"          # task | skill | operator | prim | production | fact
    halo: bool = False
    rank: float = 0.0
    defined_by: int | None = None
    x: float = 0.0
    y: float = 0.0
    fixed: bool = False


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)     # (from name, to name, learned)
    c: float = 0.3
    kind: str = ""


def build_graph(session: Session, level=2) -> Graph:
    """The node and edge structure of a Swift PRIMs graph.

    level 1  tasks, skills and operators (edges from a skill to the operators
             it is associated with; red edges are learned associations);
    level 2  skills, operators and their PRIMs.  Operators that share a
             sequence of PRIMs share nodes, so a yellow halo marks PRIMs
             used by operators of more than one task (the basis of transfer);
    "productions"  the compiled productions above primU, ranked by size;
    "dm"     facts and the facts they point to.
    """
    m = session.model
    g = Graph(kind=str(level))
    N = g.nodes
    tasks = session.entries
    col = session.colour_index

    def first(ch):           # colour of the task that defined a chunk
        return col(ch.defined_in[0]) if ch.defined_in else -3

    if level == "productions":
        g.c = 1.0
        for p in m.procedural.productions.values():
            if p.u > m.procedural.prim_u:
                N[p.full_name] = Node(p.full_name, p.name, col(p.task_id), "production",
                                      rank=len(p.conditions) + len(p.actions) - 1)
        for p in m.procedural.productions.values():
            if p.full_name in N:
                for par in (p.parent1, p.parent2):
                    if par is not None and par.full_name in N:
                        g.edges.append((p.full_name, par.full_name, False))
        return g

    if level == "dm":
        g.c = 1.0
        facts = [c for c in m.dm.chunks.values() if c.type == "fact"]
        for c in facts:
            N[c.name] = Node(c.name, c.name, first(c), "fact")
        for c in facts:
            for v in c.slotvals.values():
                if getattr(v, "type", None) == "fact" and v.name != c.name and v.name in N:
                    g.edges.append((c.name, v.name, False))
        return g

    if level not in (1, 2):
        raise ValueError("level must be 1, 2, 'productions' or 'dm'")
    if level == 1:
        for i, e in enumerate(tasks):
            if session.loaded_in_model(e.name):
                N[e.name + "T"] = Node(e.name + "T", e.name, i, "task", defined_by=i)
    for ch in list(m.dm.chunks.values()):
        if ch.type == "goaltype":
            N[ch.name] = Node(ch.name, ch.name, first(ch), "skill", defined_by=first(ch))
            if level == 1:
                for t in ch.defined_in:
                    tn = m.tasks[t].name + "T" if 0 <= t < len(m.tasks) else None
                    if tn in N:
                        g.edges.append((tn, ch.name, False))

    def chain(ch, key, colour):
        items = describe(ch.slotvals[key]).split(";") if key in ch.slotvals else []
        current, cur_name = None, ""
        while items:
            last = items.pop()
            cur_name = last if cur_name == "" else last + ";" + cur_name
            node = N.get(cur_name)
            if node is not None:
                if ch.defined_in and (node.defined_by != first(ch) or len(ch.defined_in) > 1):
                    node.halo = True
                current = node
            else:
                node = Node(cur_name, last, colour, "prim", halo=len(ch.defined_in) > 1, defined_by=first(ch))
                N[cur_name] = node
                if current is not None:
                    g.edges.append((cur_name, current.name, False))
                current = node
        return current

    for ch in list(m.dm.chunks.values()):
        if ch.type != "operator":
            continue
        cond_head = chain(ch, "condition", -2) if level > 1 else None
        op = Node(ch.name, ch.name, first(ch), "operator", halo=len(ch.defined_in) > 1)
        N[ch.name] = op
        for src, (strength, count) in ch.assocs.items():
            sname = src.split("%")[-1] if m.dm.extended_assocs else src
            sc = m.dm.chunks.get(sname)
            if sc is not None and sc.type == "goaltype" and strength > 0:
                if sname not in N:
                    N[sname] = Node(sname, sname, first(sc), "skill")
                g.edges.append((sname, ch.name, count > 0))
        if cond_head is not None:
            g.edges.append((ch.name, cond_head.name, False))
        if level > 1:
            act_head = chain(ch, "action", -1)
            if act_head is not None:
                g.edges.append((ch.name, act_head.name, False))
    return g


def layout(g: Graph, W=300.0, H=300.0, iterations=100, seed=None):
    """Fruchterman-Reingold layout, ported from FruchtermanReingold.swift."""
    import numpy as np
    names = list(g.nodes)
    n = len(names)
    if n == 0:
        return g
    rng = random.Random(seed)
    pos = np.array([[rng.uniform(0, W), rng.uniform(0, H)] if not g.nodes[k].fixed
                    else [g.nodes[k].x, g.nodes[k].y] for k in names])
    fixed = np.array([g.nodes[k].fixed for k in names])
    rank = np.array([g.nodes[k].rank for k in names])
    k = g.c * math.sqrt(W * H / n)
    idx = {name: i for i, name in enumerate(names)}
    E = np.array([(idx[a], idx[b]) for a, b, _ in g.edges if a in idx and b in idx], dtype=int).reshape(-1, 2)
    max_rank = rank.max() if n else 0
    rank_step = (H - 30) / (max_rank - 1) if max_rank > 1 else 0
    for it in range(iterations):
        temp = 0.1 * max(W, H) * (iterations - it) / iterations
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-3)
        rep = (delta / dist[..., None]) * (k * k / dist)[..., None]
        for i in range(n):
            rep[i, i] = 0
        disp = rep.sum(1)
        disp[:, 0] += 2.0 * k * k / (pos[:, 0] + 1) - 2.0 * k * k / (W + 1 - pos[:, 0])
        disp[:, 1] += 2.0 * k * k / (pos[:, 1] + 1) - 2.0 * k * k / (H + 1 - pos[:, 1])
        if len(E):
            d = pos[E[:, 0]] - pos[E[:, 1]]
            dl = np.maximum(np.sqrt((d ** 2).sum(-1)), 1e-3)
            f = (d / dl[:, None]) * (dl ** 2 / k)[:, None]
            np.add.at(disp, E[:, 0], -f)
            np.add.at(disp, E[:, 1], f)
        length = np.maximum(np.sqrt((disp ** 2).sum(-1)), 1e-9)
        step = np.stack([(disp[:, 0] / length) * np.minimum(np.abs(disp[:, 0]), temp),
                         (disp[:, 1] / length) * np.minimum(np.abs(disp[:, 1]), temp)], 1)
        step[fixed] = 0
        pos = np.clip(pos + step, [0, 0], [W, H])
        ranked = (rank > 0.1) & ~fixed
        pos[ranked, 1] = rank_step * (rank[ranked] - 1) + 15
    for i, name in enumerate(names):
        g.nodes[name].x, g.nodes[name].y = float(pos[i, 0]), float(pos[i, 1])
    return g


_SHAPE = {"task": "*", "skill": "p", "operator": "o", "prim": "o", "production": "o", "fact": "o"}
_SIZE = {"task": 260, "skill": 170, "operator": 90, "prim": 45, "production": 60, "fact": 60}


@_plotting((8, 7))
def draw_graph(g: Graph, ax=None, labels=True, title=None, legend_tasks=None):
    """Draw a laid-out graph in the style of the Swift PRIMs view."""
    from matplotlib.patches import FancyArrowPatch
    N = g.nodes
    for a, b, learned in g.edges:
        if a in N and b in N:
            ax.add_patch(FancyArrowPatch((N[a].x, N[a].y), (N[b].x, N[b].y), arrowstyle="-|>", mutation_scale=7,
                                         color="#D32F2F" if learned else "#555555",
                                         linewidth=1.1 if learned else 0.6, shrinkA=4, shrinkB=5, zorder=1))
    halo = [n for n in N.values() if n.halo]
    if halo:
        ax.scatter([n.x for n in halo], [n.y for n in halo], s=[_SIZE[n.kind] * 3.2 for n in halo],
                   color="#FFE082", zorder=2, linewidths=0)
    for kind, marker in _SHAPE.items():
        ns = [n for n in N.values() if n.kind == kind]
        if ns:
            ax.scatter([n.x for n in ns], [n.y for n in ns], s=_SIZE[kind], marker=marker,
                       c=[task_color(n.task) for n in ns], edgecolors="black",
                       linewidths=1.6 if kind in ("task", "skill") else 0.7, zorder=3)
    if labels:
        for n in N.values():
            big = n.kind in ("task", "skill")
            ax.annotate(n.label, (n.x, n.y), xytext=(0, -9 if not big else -12), textcoords="offset points",
                        ha="center", va="top", fontsize=9 if big else (5.5 if n.kind == "production" else 6.5),
                        fontweight="bold" if big else "normal", zorder=4)
    ax.set_xlim(-15, 315)
    ax.set_ylim(320, -15)               # y runs downward, as in the Swift view
    ax.set_axis_off()
    if legend_tasks:
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], marker="o", linestyle="", markerfacecolor=task_color(i), markeredgecolor="black",
                     label=name) for i, name in legend_tasks]
        ax.legend(handles=hs, frameon=False, loc="upper right", fontsize=8)
    if title:
        ax.set_title(title)
    return ax


_GRAPH_TITLES = {1: "Task – skill – operator", 2: "Skill – operator – PRIMs",
                 "productions": "Compiled productions (above primU)", "dm": "Declarative memory (facts)"}


@_plotting((8, 7))
def plot_graph(session: Session, level=2, ax=None, seed=None, labels=True, graph: Graph | None = None):
    """The Swift PRIMs graph (see :func:`build_graph` for the levels)."""
    g = graph or layout(build_graph(session, level), seed=seed)
    return draw_graph(g, ax, labels=labels, title=_GRAPH_TITLES.get(level, str(level)),
                      legend_tasks=[(i, e.name) for i, e in enumerate(session.entries)] if level in (1, 2) else None)


# ====================================================================== batch output
COLUMNS = ["repeat", "task", "label", "trial", "time", "event", "p1", "p2", "p3",
           "in0", "in1", "in2", "in3", "in4", "firings"]


def read_batch(source):
    """Batch output (a path, .gz path, or the list of lines run_batch returns) as a DataFrame."""
    import pandas as pd
    if isinstance(source, (list, tuple)):
        text = "".join(source)
    else:
        opener = gzip.open if str(source).endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8") as f:
            text = f.read()
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    from .frames import LeftFrame
    df = LeftFrame([r[:15] for r in rows], columns=COLUMNS[:max(len(r) for r in rows)] if rows else COLUMNS)
    for c in ("repeat", "trial", "firings"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


@_plotting((7, 4))
def plot_batch(source, ax=None, event: str = "trial-end", p1: str | None = "success", block: int = 1,
               by=("task", "label")):
    """Mean time of an event per trial (or per block of trials), one line per task and label,
    with ±1 SE over repeats."""
    import numpy as np
    df = source if hasattr(source, "columns") else read_batch(source)
    d = df[df.event == event]
    if p1 is not None:
        d = d[d.p1 == p1]
    d = d.assign(block=((d.trial - 1) // block) + 1)
    per_run = d.groupby(list(by) + ["repeat", "block"]).time.mean().reset_index()
    for i, (key, grp) in enumerate(per_run.groupby(list(by), sort=False)):
        stats = grp.groupby("block").time.agg(["mean", "std", "count"])
        se = stats["std"].fillna(0) / np.sqrt(stats["count"])
        x = stats.index.values
        line, = ax.plot(x, stats["mean"], marker=MARKERS[i % len(MARKERS)], markersize=4,
                        label=" / ".join(map(str, key if isinstance(key, tuple) else (key,))))
        ax.fill_between(x, stats["mean"] - se, stats["mean"] + se, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Trial" if block == 1 else f"Block of {block} trials")
    ax.set_ylabel(f"Mean {event} time (sec)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return ax


# ====================================================================== attach to Session
Session.plot_results = plot_results
Session.plot_operator_activations = plot_operator_activations
Session.plot_graph = plot_graph
