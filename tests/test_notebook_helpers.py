"""Tests for the helpers added for the tutorial: associations, labelled results,
the no-operator warning, and run_trials."""
import os
import warnings

import pytest

from pyprims import Model, run_trials
from pyprims.notebook import Session, NoOperatorWarning

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")

NO_OPERATORS = """
define task empty {
    initial-skills: (empty)
}
define skill empty {
    operator never {
        V1 = something-that-never-appears
    ==>
        say -> AC1
    }
}
define script {
    screen("hello")
    run-until-action("say")
    trial-end()
}
"""


# ---------------------------------------------------------------- associations
def test_associations_shows_defaults_and_learning():
    s = Session(seed=1)
    s.load(os.path.join(EX, "task-switching.prims"))
    defaults = s.associations("choice-reaction")
    assert len(defaults) > 0
    assert set(defaults.columns) == {"source", "target", "Sji", "updates"}
    assert (defaults.updates == 0).all()                      # nothing learned yet
    s.run(5)
    learned = s.associations("choice-reaction", learned=True)
    assert len(learned) > 0 and (learned.updates > 0).all()
    assert list(learned.Sji) == sorted(learned.Sji, reverse=True)
    assert len(s.associations()) >= len(s.associations("choice-reaction"))


def test_associations_does_not_change_the_simulation():
    path = os.path.join(EX, "count.prims")
    a = Session(seed=4); a.load(path); a.run(10)
    b = Session(seed=4); b.load(path)
    for _ in range(10):
        b.run(1)
        b.associations()
    assert a.results == b.results


# ---------------------------------------------------------------- plot_results
def _legend(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def test_plot_results_labels_runs_of_the_same_task():
    from matplotlib.figure import Figure
    s = Session(seed=2)
    s.load(os.path.join(EX, "count.prims")); s.run(3)
    s.load(os.path.join(EX, "semantic.prims")); s.run(3)
    s.reset(); s.run(3)

    ax = Figure().subplots()
    s.plot_results(ax=ax)
    assert _legend(ax) == ["count", "semantic, run 1", "semantic, run 2"]
    styles = [line.get_linestyle() for line in ax.get_lines()]
    assert styles[1] != styles[2]

    ax = Figure().subplots()
    s.plot_results(ax=ax, labels=["train", "transfer", "control"])
    assert _legend(ax) == ["train", "transfer", "control"]

    ax = Figure().subplots()
    s.plot_results(ax=ax, tasks=["count"])
    assert _legend(ax) == ["count"]


# ---------------------------------------------------------------- warning
def test_run_warns_when_no_operator_matches():
    s = Session(seed=1)
    s.load(NO_OPERATORS)
    with pytest.warns(NoOperatorWarning, match="3 of 3 trials"):
        s.run(3)
    s.warn_failed_trials = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoOperatorWarning)
        s.run(2)


def test_no_warning_for_a_working_model():
    s = Session(seed=1)
    s.load(os.path.join(EX, "count.prims"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoOperatorWarning)
        s.run(5)


# ---------------------------------------------------------------- run_trials
def test_run_trials_gives_the_same_trials_as_model_run():
    path = os.path.join(EX, "count.prims")
    plain = Model(seed=7, trace=False, batch_mode=True)
    plain.load(path)
    plain.run(20)

    m = Model(seed=7, trace=False, batch_mode=True)
    rows = run_trials(m, 20, path, repeat=0, label="train")
    assert [r["result"] for r in rows] == plain.results
    assert [r["trial"] for r in rows] == list(range(1, 21))
    assert all(r["task"] == "count" and r["label"] == "train" and r["outcome"] == "success" for r in rows)


def test_run_trials_switches_to_a_loaded_task_and_carries_over():
    m = Model(seed=3, trace=False, batch_mode=True)
    run_trials(m, 30, os.path.join(EX, "count.prims"))
    run_trials(m, 5, os.path.join(EX, "semantic.prims"))
    rows = run_trials(m, 2, os.path.join(EX, "count.prims"))
    assert [t.name for t in m.tasks] == ["count", "semantic"]          # not loaded twice
    assert rows[0]["task"] == "count"


def test_run_trials_several_trial_ends_and_data_fields():
    m = Model(seed=1, trace=False, batch_mode=True)
    rows = run_trials(m, 2, os.path.join(EX, "stroop.prims"))
    assert len(rows) == 8                                              # four trial-ends per script run
    assert [r["trial"] for r in rows] == [1] * 4 + [2] * 4
    assert {r["in1"] for r in rows} == {"consistent", "conflict"}


def test_run_trials_events_match_batch_output():
    m = Model(seed=5, trace=False, batch_mode=True)
    events = run_trials(m, 3, os.path.join(EX, "count.prims"), events=True)
    assert {e["event"] for e in events} >= {"action", "trial-end"}
    assert {"time", "p1", "p2", "p3", "in0", "in4", "firings"} <= set(events[0])


def test_run_trials_reports_failed_trials():
    m = Model(seed=1, trace=False, batch_mode=True)
    from pyprims.dsl import parse_prims
    rows = run_trials(m, 2, parse_prims(NO_OPERATORS))
    assert [r["outcome"] for r in rows] == ["fail", "fail"]
