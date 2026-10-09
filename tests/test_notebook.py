"""Tests for pyprims.notebook: stepping and inspection must not change the simulation."""
import os

import pytest

from pyprims.notebook import Session

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
MODELS = ["count.prims", "stroop.py", "task-switching.prims", "semantic.prims"]


@pytest.mark.parametrize("name", MODELS)
def test_stepping_and_inspection_give_identical_runs(name):
    path = os.path.join(EX, name)
    plain = Session(seed=11)
    plain.load(path)
    plain.run(15)

    stepped = Session(seed=11)
    stepped.load(path)
    stepped.track_operator_activations = True       # computed, but must not disturb the run
    for i in range(120):                            # step through the first trials, inspecting all the way
        stepped.step()
        stepped.buffers()
        stepped.conflict_set()
        if i % 10 == 0:
            stepped.dm()
            stepped.trace(5)
    stepped.step_operator(3)
    stepped.run(15 - len(stepped.results))
    assert stepped.results[:15] == plain.results[:15]


def test_conflict_set_breakdown_reproduces_activations():
    s = Session(seed=3)
    s.load(os.path.join(EX, "count.prims"))
    for _ in range(6):
        s.step()
        for item in s.conflict_set():
            lines = dict(l.split(" = ") for l in item["text"].splitlines()
                         if l.startswith(("Activation in conflict set", "Total activation")))
            assert abs(float(lines["Activation in conflict set"]) - float(lines["Total activation"])) < 1e-9


def test_rows_tasks_reset_and_text_formats():
    s = Session(seed=1)
    s.load(os.path.join(EX, "count.prims"))
    s.run(5)
    with open(os.path.join(EX, "semantic.py")) as f:
        s.load(f.read())                             # Python text, detected
    s.run(3)
    assert [r.task for r in s.rows] == ["count", "semantic"]
    assert [len(r.points) for r in s.rows] == [5, 3]
    s.select_task("count")
    s.run(2)
    s.reset()
    s.run(1)
    assert [(r.task, len(r.points)) for r in s.rows] == [("count", 5), ("semantic", 3), ("count", 2), ("count", 1)]
    assert s.task_names == ["count", "semantic"] and not s.loaded_in_model("semantic")
    with open(os.path.join(EX, "count.prims")) as f:
        s.load(f.read())                             # same name: redefine and reset
    assert s.model.find_task("count") is not None and len(s.model.tasks) == 1
    assert "task = Task('count'" in s.to_python()


def test_abandoning_a_stepped_trial():
    s = Session(seed=2)
    s.load(os.path.join(EX, "count.prims"))
    s.step(3)
    assert s.trial_active
    s.reset()
    assert not s.trial_active
    s.run(2)
    assert len(s.results) == 2


def test_graphs_build():
    from pyprims.notebook import build_graph, layout
    s = Session(seed=1)
    s.load(os.path.join(EX, "count.prims"))
    s.run(30)
    s.load(os.path.join(EX, "semantic.prims"))
    s.run(5)
    g2 = layout(build_graph(s, 2), seed=1)
    assert any(n.halo for n in g2.nodes.values())     # count and semantic share PRIM sequences
    assert {"count", "semantic"} <= {n.name for n in g2.nodes.values() if n.kind == "skill"}
    for level in (1, "productions", "dm"):
        layout(build_graph(s, level), seed=1)
