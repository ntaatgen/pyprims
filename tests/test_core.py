"""Tests for pyprims.  Run with:  python -m pytest tests"""
import os
import statistics

import pytest

from pyprims import Model, Task, load_task
from pyprims.prim import parse_name
from pyprims.dsl import compile_operator_body
from pyprims.script_translate import translate_script
from pyprims.values import string2double, describe, values_equal
from pyprims.convert import convert_text

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")


# ------------------------------------------------------------------ values
def test_string2double_prefix_quirk():
    assert string2double("3") == 3.0
    assert string2double("3abc") == 3.0          # Scanner.scanDouble reads a prefix
    assert string2double("one") is None
    assert string2double("-2.5e1x") == -25.0


def test_value_equality_is_by_description():
    assert describe(3.0) == "3.0"
    assert not values_equal(3.0, "3")            # Number(3.0) != Text("3"), as in Swift
    assert values_equal("red", "red")


# ------------------------------------------------------------------ PRIMs
def test_parse_name_canonical_order():
    assert parse_name("RT2=WM1")[5] == "WM1=RT2"
    assert parse_name("WM1=RT2")[5] is None
    assert parse_name("V1<>nil")[:4] == ("input", "slot1", "<>", None)
    assert parse_name("nil->G1")[:5] == (None, None, "->", "goal", "slot1")
    assert parse_name("V1->RT1")[3] == "retrievalR"      # RT on the right is the request
    assert parse_name("RT3->WM1")[0] == "retrievalH"     # RT on the left is the harvest
    assert parse_name("q->AC2")[2] == ""                 # one-letter constants do not parse


def test_operator_body_constants_and_bindings():
    cond, act, const = compile_operator_body("""
        V1 <> nil   // comment
        WM1 = nil
    ==>
        V1 -> WM1
        count-fact -> RT1
        say -> AC1
    """)
    assert cond == ["V1<>nil", "WM1=nil"]
    assert act == ["V1->WM1", "C1->RT1", "C2->AC1"]
    assert const == ["count-fact", "say"]
    cond, act, const = compile_operator_body("G1 = semantic ==> property -> *fact-type  iterate -> G1")
    # the binding target is numbered first, then the other constants (as in Parser.swift)
    assert cond == ["G1=C1"]
    assert act == ["C3->C2", "C4->G1"]
    assert const == ["semantic", "**fact-type", "property", "iterate"]


# ------------------------------------------------------------------ script translation
def run_script(src):
    func, code = translate_script(src)

    class Rec:
        arg = ""
        def __init__(self):
            self.printed = []
        def print(self, *a):
            self.printed.append(a)
    r = Rec()
    func(r)
    return r.printed, code


def test_script_semantics_match_swift():
    printed, code = run_script("""
        a = 10 - 3 - 2            // right-associative: 10 - (3 - 2) = 9
        b = 7 / 2                 // integer division
        c = -7 / 2                // truncates toward zero
        s = "n=" + 3
        arr = []
        for i in 0 to 2 { arr[i] = i * 2 }
        arr[5] = 1                // pads with zeros
        t = 0
        for x in arr { t = t + x }
        if ((a == 9) && (b == 3)) { ok = 1 } else { ok = 0 }   // PRIMs needs the inner parentheses
        print(a, b, c, s, arr, t, ok)
    """)
    assert printed == [(9, 3, -3, "n=3", [0, 2, 4, 0, 0, 1], 7, 1)]
    assert "10 - (3 - 2)" in code


def test_loop_semantics_match_swift():
    printed, code = run_script("""
        n = 3
        k = 0
        while (n) { n = n - 1  k = k + 1 }      // Swift: enter if n != 0, repeat only while n == 1
        j = 0
        for i in 0 to 5 { i = i + 1  j = j + 1 }  // the body's change to i is honoured
        arr = [1, 2]
        c = 0
        for x in arr { arr[2] = 9  c = c + 1 }   // iteration count is fixed at the start
        if ("") { e = 1 } else { e = 0 }        // any non-integer test is true
        f = (("b" < "a") || ("a" > "b"))       // strings are not ordered: both false
        print(k, i, j, c, e, f)
    """)
    assert printed == [(1, 6, 3, 2, 1, 0)]


# ------------------------------------------------------------------ model behaviour
def test_count_trace_matches_tutorial_timing():
    """The PRIM firing times of the first operator are those in the PRIMs tutorial."""
    m = Model(seed=0)
    m.load(os.path.join(EX, "count.prims"))
    m.run(1)
    lines = [l.split("  ", 1) for l in m.get_trace(3).splitlines() if "Firing" in l][:7]
    assert [(t, s) for t, s in lines] == [
        ("0.00", "Firing V1<>nil"), ("0.05", "Firing WM1=nil"), ("0.35", "Firing V1->WM1"),
        ("0.65", "Firing C1->RT1"), ("0.95", "Firing V1->RT2"), ("1.25", "Firing C2->AC1"),
        ("1.55", "Firing V1->AC2")]


def test_production_compilation_speeds_up_counting():
    m = Model(seed=1, trace=False, batch_mode=True)
    m.load(os.path.join(EX, "count.prims"))
    m.run(100)
    assert statistics.mean(m.results[:10]) > 2 * statistics.mean(m.results[-10:])
    assert any(p.u > m.procedural.prim_u for p in m.procedural.productions.values())


def test_transfer_from_count_to_semantic():
    """Training on count makes semantic faster (Taatgen, 2013): the core PRIMs result."""
    first = {"transfer": [], "control": []}
    for seed in range(5):
        m = Model(seed=seed, trace=False, batch_mode=True)
        m.load(os.path.join(EX, "count.prims"))
        m.run(50)
        m.load(os.path.join(EX, "semantic.prims"))
        m.run(10)
        first["transfer"].append(statistics.mean(m.results[-10:]))
        c = Model(seed=100 + seed, trace=False, batch_mode=True)
        c.load(os.path.join(EX, "semantic.prims"))
        c.run(10)
        first["control"].append(statistics.mean(c.results))
    assert statistics.mean(first["transfer"]) < 0.7 * statistics.mean(first["control"])


def test_python_task_api_equals_prims_file():
    a = Model(seed=3, trace=False, batch_mode=True)
    a.load(os.path.join(EX, "count.prims"))
    a.run(5)
    b = Model(seed=3, trace=False, batch_mode=True)
    b.load(os.path.join(EX, "count.py"))
    b.run(5)
    assert a.results == b.results


def test_converter_roundtrip(tmp_path):
    src = open(os.path.join(EX, "semantic.prims")).read()
    py = tmp_path / "semantic.py"
    py.write_text(convert_text(src, "semantic.prims"))
    r = []
    for path in (os.path.join(EX, "semantic.prims"), str(py)):
        m = Model(seed=9, trace=False, batch_mode=True)
        m.load(path)
        m.run(5)
        r.append(m.results)
    assert r[0] == r[1]


def test_handwritten_python_model():
    task = Task("tiny", initial_skills=["tiny"], default_activation=1.0, egs=0.0, ans=0.0)
    task.skill("tiny").operator("respond", """
        V1 <> nil
    ==>
        press -> AC1
        V1 -> AC2
    """)
    task.action("press", latency=0.2)

    @task.script
    def script(m):
        m.screen("x")
        m.run_until_action("press")
        m.issue_reward()
        m.trial_end()

    m = Model(seed=0, trace=False, batch_mode=True)
    m.load(task)
    m.run(3)
    acts = [(d.event_type, d.p1, d.p2) for d in m.output_data]
    assert ("action", "press", "x") in acts
    assert len(m.results) == 3


def test_stuck_model_raises():
    task = Task("stuck", initial_skills=["stuck"])

    @task.script
    def script(m):
        while True:
            m.run_until_action()

    m = Model(seed=0, trace=False, batch_mode=True)
    m.load(task)
    from pyprims.script_api import ModelStuck
    with pytest.raises(ModelStuck):
        m.run(1)


# ------------------------------------------------------------------ action functions
def _probe_task(name="probe", answer="yes", latency=None):
    """A task whose model asks the environment a question with an action."""
    task = Task(name, initial_skills=["probe"], default_activation=1.0)
    sk = task.skill("probe")
    sk.operator("ask", """
        V1 = question
        V2 = nil
    ==>
        ask -> AC1
        V1 -> AC2
        please -> AC3
    """)
    sk.operator("say-answer", """
        V2 <> nil
    ==>
        say -> AC1
        V2 -> AC2
    """)
    task.action("ask", latency=0.2)
    task.action("say", latency=0.3)

    @task.action_function
    def ask(m, *args):
        m.state.setdefault("calls", []).append(args)
        m.screen("question", answer)
        return latency

    @task.script
    def script(m):
        m.screen("question")
        m.run_until_action("say")
        m.state["trials"] = m.state.get("trials", 0) + 1
        m.trial_end()
    return task


def test_action_function_gets_arguments_and_changes_screen():
    m = Model(seed=1)
    m.load(_probe_task())
    m.run(2)
    said = [d for d in m.output_data if d.event_type == "action" and d.p1 == "say"]
    assert said and said[0].p2 == "yes"
    state = m.scenario.state
    assert state["calls"] == [("question", "please")] * 2
    assert state["trials"] == 2


def test_action_function_can_set_latency():
    def say_time(latency):
        m = Model(seed=1)
        m.load(_probe_task(latency=latency))
        m.run(1)
        ask = [d for d in m.output_data if d.event_type == "action" and d.p1 == "ask"][0]
        say = [d for d in m.output_data if d.event_type == "action" and d.p1 == "say"][0]
        return ask.time, say.time
    ask_default, say_default = say_time(None)
    ask_slow, say_slow = say_time(1.2)
    assert ask_default < ask_slow                   # the data line records the action's end time
    assert say_slow - say_default == pytest.approx(1.0, abs=0.01)     # operator latencies shift a little with time


def test_action_functions_belong_to_their_task():
    m = Model(seed=1)
    m.load(_probe_task("first", answer="one"))
    m.load(_probe_task("second", answer="two"))
    m.run(1)
    assert [d.p2 for d in m.output_data if d.p1 == "say"] == ["two"]
    m.switch_task("first")
    m.run(1)
    assert [d.p2 for d in m.output_data if d.p1 == "say"] == ["one"]
    assert "trials" in m.scenario.state and m.scenario.state["trials"] == 1


def test_action_function_name_matches_hyphenated_action():
    task = Task("hyphen", initial_skills=["k"], default_activation=1.0)
    task.skill("k").operator("press", """
        V1 = go
    ==>
        press-key -> AC1
        space -> AC2
    """)
    seen = []

    @task.action_function
    def press_key(m, key):
        seen.append(key)
        m.screen("done")

    @task.script
    def script(m):
        m.screen("go")
        m.run_until_action("press-key")
        m.trial_end()

    m = Model(seed=1)
    m.load(task)
    m.run(1)
    assert seen == ["space"]
