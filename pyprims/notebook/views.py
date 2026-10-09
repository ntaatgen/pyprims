"""Text and HTML views of a session: the panels of the Swift PRIMs window.

Each view is computed inside :func:`frozen`, so looking never changes what
the model does next.  The objects returned here print as text and render as
HTML in a notebook.
"""
from __future__ import annotations

import html
import math
import re

from ..values import describe, is_chunk
from .session import Session, frozen

PALETTE = {-3: "#8B5A2B", -2: "#9E9E9E", -1: "#FFFFFF", 0: "#E53935", 1: "#1E88E5", 2: "#43A047",
           3: "#8E24AA", 4: "#00ACC1", 5: "#3949AB", 6: "#FB8C00", 7: "#FDD835"}


def task_color(i: int) -> str:
    """The Swift ``numberToColor``: task 0 red, 1 blue, 2 green, ..."""
    return PALETTE.get(i, "#000000")


_CSS = """<style>
.pp-box{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.35}
.pp-box table{border-collapse:collapse;margin:2px 0}
.pp-box td,.pp-box th{border:1px solid rgba(128,128,128,.35);padding:3px 8px;vertical-align:top;text-align:left!important}
.pp-box th{background:rgba(128,128,128,.12);font-weight:600}
.pp-box pre{margin:0;font:inherit;white-space:pre-wrap;text-align:left!important;background:none!important;padding:0!important}
.pp-box .pp-new{background:rgba(255,193,7,.18)}
.pp-box .pp-muted{opacity:.6}
.pp-box details{margin:1px 0}
.pp-box summary{cursor:pointer}
</style>"""


def _wrap(inner: str) -> str:
    return _CSS + f'<div class="pp-box">{inner}</div>'


def _esc(s) -> str:
    return html.escape(str(s))


# ====================================================================== buffers
def format_buffer(chunk, abbrev: str, show_slot0: bool = True) -> str:
    """Swift ``formatBuffer``: 'G1 count' lines for slot1..slot9."""
    if chunk is None:
        return ""
    out = []
    if show_slot0:
        out.append(f"{abbrev}0 {describe(chunk.slot_value('slot0'))}")
    for n in range(1, 10):
        v = chunk.slotvals.get(f"slot{n}")
        if v is not None:
            out.append(f"{abbrev}{n} {describe(v)}")
    return "\n".join(out)


def format_operator(chunk) -> str:
    """Swift ``formatOperator``: name, conditions, actions, constants."""
    if chunk is None:
        return ""
    out = [chunk.name]
    cond = chunk.slotvals.get("condition")
    if cond is not None:
        out.append("Conditions: " + " ".join(describe(cond).split(";")))
    act = chunk.slotvals.get("action")
    if act is not None:
        out.append("Actions: " + " ".join(describe(act).split(";")))
    consts = format_buffer(chunk, "C", show_slot0=False)
    if consts:
        out.append("Constants:\n" + consts)
    return "\n".join(out)


def _changed(before, after, abbrev, slot0=True) -> str:
    """The 'New ...' cell: the buffer after the step, if what is shown changed."""
    a = format_buffer(after, abbrev, slot0)
    return "" if a == format_buffer(before, abbrev, slot0) else a


class BufferView:
    """The Swift buffer panel: buffers before the last step, and what changed.

    'Before' is the state at the start of the last step (``former_buffers``);
    'after' columns are filled only when the buffer changed.
    """

    def __init__(self, session: Session):
        m = session.model
        fb, b = m.former_buffers, m.buffers
        self.status = session.status()
        self.operator = format_operator(fb.get("operator"))
        sel = m.selected_operator
        if self.operator and sel is not None:     # show the DM name, not the name of the working copy
            self.operator = sel.name + self.operator[self.operator.index("\n"):] if "\n" in self.operator else sel.name
        self.panels = [
            ("Input", [("Input", format_buffer(fb.get("input"), "V", False)),
                       ("Action", format_buffer(fb.get("action"), "AC", False)),
                       ("New input", _changed(fb.get("input"), b.get("input"), "V", False))]),
            ("Goal", [("Goal", format_buffer(fb.get("goal"), "G")),
                      ("New goal", _changed(fb.get("goal"), b.get("goal"), "G"))]),
            ("Retrieval", [("Retrieval harvest", format_buffer(fb.get("retrievalH"), "RT")),
                           ("Retrieval request", format_buffer(fb.get("retrievalR"), "RT")),
                           ("New harvest", _changed(fb.get("retrievalH"), b.get("retrievalH"), "RT"))]),
            ("Imaginal", [("Imaginal", format_buffer(fb.get("imaginal"), "WM")),
                          ("New imaginal", _changed(fb.get("imaginal"), b.get("imaginal"), "WM"))]),
        ]
        extra = []
        bind = b.get("bindings")
        if bind is not None and any(k != "isa" for k in bind.slotvals):
            extra.append(("Bindings", "\n".join(f"{k} {describe(v)}" for k, v in bind.slotvals.items() if k != "isa")))
        if b.get("constants") is not None:
            extra.append(("Task constants", format_buffer(b.get("constants"), "T", False)))
        if extra:
            self.panels.append(("Other", extra))

    def as_dict(self) -> dict:
        d = {"Operator": self.operator}
        for _, cells in self.panels:
            d.update(dict(cells))
        return d

    def __str__(self):
        lines = ["Operator", "  " + self.operator.replace("\n", "\n  ")]
        for _, cells in self.panels:
            for title, text in cells:
                if text:
                    lines.append(title)
                    lines.append("  " + text.replace("\n", "\n  "))
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self):
        st = self.status
        head = (f"<div class='pp-muted'>task {_esc(st['task'])} · time {st['time']:.3f} s "
                f"(trial {st['trial_time']:.3f} s) · next: {st['next_half_step']}"
                f"{' · trial under way' if st['trial_active'] else ''}</div>")
        rows = [f"<tr><th colspan='3'>Operator</th></tr><tr><td colspan='3'><pre>{_esc(self.operator) or '—'}</pre></td></tr>"]
        for _, cells in self.panels:
            rows.append("<tr>" + "".join(f"<th>{_esc(t)}</th>" for t, _ in cells) + "</tr>")
            rows.append("<tr>" + "".join(
                f"<td class='{'pp-new' if t.startswith('New') and x else ''}'><pre>{_esc(x)}</pre></td>"
                for t, x in cells) + "</tr>")
        return _wrap(head + "<table>" + "".join(rows) + "</table>")


# ====================================================================== activation breakdown
def _spreading_lines(m, buffer_name, w, chunk, divide):
    """Port of ConflictSetTrace.spreadingFromBufferDescription."""
    if w == 0:
        return []
    bc = m.buffers.get(buffer_name)
    if bc is None:
        return []
    out = [f"Spreading from {buffer_name} buffer, spreading parameter = {w:.3f}"]
    n = sum(1 for v in bc.slotvals.values() if is_chunk(v)) if divide else 1
    for slot, v in list(bc.slotvals.items()):
        if is_chunk(v):
            sji = v.sji(chunk, buffer_name, slot)
            spreading = sji * w / max(n, 1)
            if spreading != 0:
                out.append(f"  spreading {spreading:.3f} from slot {slot} with value {v.name}, Sji = {sji:.3f}")
    return out


def format_operator_code(chunk) -> str:
    """The operator as PRIMs code with its constants filled in (as in the Swift conflict trace)."""
    consts = {k[4:]: describe(v) for k, v in chunk.slotvals.items() if k.startswith("slot")}

    def subst(s):
        # C3 -> its constant, but not the C in AC1
        return [re.sub(r"(?<![A-Za-z])C(\d+)", lambda mm: consts.get(mm.group(1), mm.group(0)), p)
                for p in s.split(";")]
    lines = [f"operator {chunk.name} {{"]
    cond = chunk.slotvals.get("condition")
    if cond is not None:
        lines += ["    " + p for p in subst(describe(cond))]
        lines.append("==>")
    act = chunk.slotvals.get("action")
    if act is not None:
        lines += ["    " + p for p in subst(describe(act))]
    lines.append("}")
    return "\n".join(lines)


def activation_breakdown(m, chunk, activation=None) -> str:
    """Base level, noise and every spreading source of a chunk (call inside ``frozen``)."""
    dm = m.dm
    s = format_operator_code(chunk) if chunk.type == "operator" else chunk.description()
    base = chunk.base_level_activation()
    noise = chunk.calculate_noise() if chunk.creation_time is not None else 0.0
    spread = chunk.spreading_activation()
    # embedded representation: partial-matching penalty of the last soft retrieval
    penalty = (getattr(dm, "last_mismatch", {}).get(chunk.name, 0.0)
               if getattr(m.representation, "soft_retrieval", False) else 0.0)
    total = base + spread + noise + penalty
    lines = [s.rstrip("\n"), ""]
    if activation is not None:
        lines.append(f"Activation in conflict set = {activation:.3f}")
    lines += [f"Total activation = {total:.3f}",
              f"Baselevel activation = {base:.3f}"
              + ("" if chunk.fixed_activation is None else f"  (fixed {chunk.fixed_activation:.3f}, refs {chunk.references})"),
              f"Current noise = {noise:.3f}",
              f"Total spreading activation = {spread:.3f}"]
    if penalty:
        lines.append(f"Partial-matching penalty = {penalty:.3f}  (mp x sum of (similarity - 1) over the cues)")
    lines += ["", "Spreading activation calculation"]
    goal = m.buffers.get("goal")
    if dm.goal_spreading_by_activation:
        lines.append(f"Spreading from the goal buffer, ga = {dm.goal_activation}")
        if goal is not None:
            for slot, v in goal.slotvals.items():
                if is_chunk(v):
                    wj = math.exp(v.base_level_activation())
                    sp = v.sji(chunk) * wj * dm.goal_activation
                    if sp > 0:
                        lines.append(f"  spreading {sp:.3f} from slot {slot} with value {v.name}, "
                                     f"Wj = {wj:.3f}, Sji = {v.sji(chunk):.3f}")
    else:
        lines += _spreading_lines(m, "goal", dm.goal_activation, chunk, divide=False)
    if goal is not None:
        for slot, v in goal.slotvals.items():
            if is_chunk(v) and v.type != "goaltype":
                nested = v.slotvals.get("slot1")
                if is_chunk(nested) and nested.type == "goaltype":
                    sp = nested.sji(chunk) * dm.goal_activation
                    if dm.goal_spreading_by_activation:
                        sp *= math.exp(v.base_level_activation())
                    if sp > 0:
                        lines.append(f"  spreading {sp:.3f} from slot {slot} via instantiated skill "
                                     f"{nested.name}, Sji = {nested.sji(chunk):.3f}")
    lines += _spreading_lines(m, "input", dm.input_activation, chunk, True)
    lines += _spreading_lines(m, "retrievalH", dm.retrieval_activation, chunk, True)
    lines += _spreading_lines(m, "imaginal", dm.imaginal_activation, chunk, True)
    return "\n".join(lines)


# ====================================================================== conflict set
def explain_conflict_set(m, limit: int = 30) -> list[dict]:
    """The chunks of the last retrieval, best first, each with its activation breakdown."""
    cs = sorted(m.dm.conflict_set, key=lambda x: x[1], reverse=True)[:limit]
    with frozen(m):
        return [dict(name=ch.name, type=ch.type, activation=act, text=activation_breakdown(m, ch, act))
                for ch, act in cs]


class ConflictSetView:
    """The Swift conflict trace: the chunks in the last retrieval, best first.

    While stepping, each retrieval is explained the moment it happens, so the
    breakdowns add up to the activations the model used.  After a Run the
    breakdown uses the current state, as the Swift GUI does, and can differ
    from the activation in the conflict set (e.g. a retrieval that added a
    reference, or buffers that changed since).
    """

    MAX = 30

    def __init__(self, session: Session):
        m = session.model
        cap = session.conflict_capture
        self.exact = cap is not None and cap[0] is m.dm.conflict_set
        self.items = cap[1] if self.exact else explain_conflict_set(m, self.MAX)
        sel = m.selected_operator
        self.selected = sel.name if sel is not None else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __str__(self):
        return "\n".join(f"{it['name']:<30} {it['activation']:8.3f}{'  <- selected' if it['name'] == self.selected else ''}"
                         for it in self.items) or "(empty)"

    __repr__ = __str__

    def _repr_html_(self):
        if not self.items:
            return _wrap("<i>Conflict set is empty (no retrieval yet).</i>")
        hi, lo = self.items[0]["activation"], self.items[-1]["activation"]
        rows = []
        for it in self.items:
            rel = 0.0 if hi == lo else (it["activation"] - lo) / (hi - lo)
            bar = f"background:linear-gradient(90deg,rgba(30,136,229,.25) {rel * 100:.0f}%,transparent 0)"
            chosen = " ← selected" if it["name"] == self.selected else ""
            rows.append(f"<details><summary style='{bar}'><b>{_esc(it['name'])}</b> "
                        f"<span class='pp-muted'>{_esc(it['type'])}</span> A = {it['activation']:.3f}"
                        f"<b>{chosen}</b></summary>"
                        f"<pre>{_esc(it['text'])}</pre></details>")
        note = ("<div class='pp-muted'>Operators are tried best first; the first whose conditions match is "
                "selected.</div>" if any(it["type"] == "operator" for it in self.items) else "")
        note += ("" if self.exact else "<div class='pp-muted'>Breakdown computed from the current state; "
                "step through a trial for breakdowns taken at the moment of retrieval.</div>")
        return _wrap(note + "".join(rows))


# ====================================================================== declarative memory
def dm_records(session: Session, type: str | None = None, contains: str | None = None) -> list[dict]:
    m = session.model
    recs = []
    with frozen(m):
        for ch in list(m.dm.chunks.values()):
            if type is not None and ch.type != type:
                continue
            if contains and contains not in ch.name:
                continue
            base = ch.base_level_activation()
            spread = ch.spreading_activation()
            slots = " ".join(f"{k}={describe(ch.slotvals[k])}" for k in ch.print_order
                             if k in ch.slotvals and k != "isa")
            recs.append(dict(name=ch.name, type=ch.type, activation=base + spread + ch.calculate_noise(),
                             base_level=base, spreading=spread, references=ch.references,
                             fan=ch.fan, slots=slots))
    recs.sort(key=lambda r: -r["activation"])
    return recs


def dm_table(session: Session, type: str | None = None, contains: str | None = None):
    """Declarative memory sorted by activation (a DataFrame when pandas is available)."""
    recs = dm_records(session, type, contains)
    try:
        from .frames import frame
        return frame(recs, ["name", "type", "activation", "base_level", "spreading",
                            "references", "fan", "slots"], index="name")
    except ImportError:
        return recs


class TextView:
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    __repr__ = __str__

    def _repr_html_(self):
        return _wrap(f"<pre>{_esc(self.text)}</pre>")


def chunk_view(session: Session, name: str) -> TextView:
    m = session.model
    ch = m.dm.chunks.get(name) or m.action.visicon.get(name) or m.buffers.get(name)
    if ch is None:
        raise KeyError(name)
    with frozen(m):
        text = activation_breakdown(m, ch)
        if ch.assocs:
            text += "\n\nAssociations (Sji) to this chunk\n" + "\n".join(
                f"  {src:<40} {v[0]:7.3f}  (updates {v[1]})" for src, v in sorted(ch.assocs.items()))
    return TextView(text)


# ====================================================================== associations
def association_records(session: Session, source: str | None = None, type: str | None = "operator",
                        learned: bool = False) -> list[dict]:
    m = session.model
    recs = []
    for ch in list(m.dm.chunks.values()):
        if type is not None and ch.type != type:
            continue
        for src, (value, updates) in ch.assocs.items():
            if source is not None and src != source:
                continue
            if learned and updates == 0:
                continue
            recs.append(dict(source=src, target=ch.name, Sji=value, updates=updates))
    recs.sort(key=lambda r: (r["source"], -r["Sji"]))
    return recs


def association_table(session: Session, source: str | None = None, type: str | None = "operator",
                      learned: bool = False):
    """Associations (Sji) from a source chunk to chunks of a type, strongest first.

    ``s.associations("count-learn")`` shows what the skill count-learn has
    learned to use: one row per operator with its Sji and the number of
    times the association was updated (0 means the value was set by the
    model, e.g. ``default-operator-assoc:``).  ``source=None`` gives all
    sources; ``type=None`` all chunk types; ``learned=True`` only
    associations that have been updated by learning.  Sources from
    ``context-operator-learning:`` are written ``buffer%slot%chunk``.
    """
    recs = association_records(session, source, type, learned)
    try:
        from .frames import frame
        return frame(recs, ["source", "target", "Sji", "updates"])
    except ImportError:
        return recs


# ====================================================================== productions
def production_records(session: Session) -> list[dict]:
    m = session.model
    out = []
    for key, p in m.procedural.productions.items():
        out.append(dict(name=p.name, utility=p.u, size=len(p.conditions) + len(p.actions),
                        above_primU=p.u > m.procedural.prim_u,
                        task=(m.tasks[p.task_id].name if 0 <= p.task_id < len(m.tasks) else p.task_id),
                        remaining=key[len(p.name):].lstrip("|"),
                        parents=" + ".join(x.name for x in (p.parent1, p.parent2) if x is not None)))
    out.sort(key=lambda r: -r["utility"])
    return out


def production_table(session: Session):
    """Learned (compiled) productions with their utilities, best first."""
    recs = production_records(session)
    try:
        from .frames import frame
        return frame(recs, ["name", "utility", "size", "above_primU", "task",
                                           "remaining", "parents"])
    except ImportError:
        return recs


# ====================================================================== tasks and trace
class TaskListView:
    def __init__(self, session: Session):
        self.rows = []
        for i, e in enumerate(session.entries):
            self.rows.append(dict(number=i, name=e.name, format=e.format, source=e.source,
                                  loaded=session.loaded_in_model(e.name),
                                  current=e.name == session.current_task))

    def __str__(self):
        return "\n".join(f"{'*' if r['current'] else ' '} {r['number']} {r['name']:<24} "
                         f"{'loaded' if r['loaded'] else 'not loaded':<11} {r['format']:<7} {r['source']}"
                         for r in self.rows) or "(no tasks)"

    __repr__ = __str__

    def _repr_html_(self):
        if not self.rows:
            return _wrap("<i>No tasks loaded.</i>")
        body = "".join(
            f"<tr><td><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{task_color(r['number'])}'></span></td><td><b>{_esc(r['name'])}</b>"
            f"{' ●' if r['current'] else ''}</td><td>{'loaded' if r['loaded'] else 'not loaded'}</td>"
            f"<td>{r['format']}</td><td class='pp-muted'>{_esc(r['source'])}</td></tr>" for r in self.rows)
        return _wrap("<table><tr><th></th><th>task</th><th>state</th><th>format</th><th>source</th></tr>"
                     + body + "</table>")


def trace_view(session: Session, level: int | None = None) -> TextView:
    return TextView(session.trace(level))


# ====================================================================== attach to Session
def _attach():
    Session.buffers = lambda self: BufferView(self)
    Session.buffers.__doc__ = BufferView.__doc__
    Session.conflict_set = lambda self: ConflictSetView(self)
    Session.conflict_set.__doc__ = ConflictSetView.__doc__
    Session.dm = lambda self, type=None, contains=None: dm_table(self, type, contains)
    Session.dm.__doc__ = dm_table.__doc__
    Session.chunk = lambda self, name: chunk_view(self, name)
    Session.chunk.__doc__ = "Slots, activation breakdown and associations of one chunk."
    Session.productions = lambda self: production_table(self)
    Session.associations = lambda self, source=None, type="operator", learned=False: \
        association_table(self, source, type, learned)
    Session.associations.__doc__ = association_table.__doc__
    Session.productions.__doc__ = production_table.__doc__
    Session.tasks = lambda self: TaskListView(self)
    Session.tasks.__doc__ = "The task list (Swift's task panel)."
    Session.show_trace = lambda self, level=None: trace_view(self, level)
    Session.show_trace.__doc__ = "The trace as a displayable object."


_attach()
