# pyprims

A Python implementation of the PRIMs cognitive architecture (Taatgen, 2013,
*Psychological Review*, 120, 439–471). It is a port of the Swift version and
runs on any platform with Python 3.10 or later. The symbolic core needs only
the standard library; the embedded representation needs numpy
(`pip install "pyprims[embedding]"`).

```bash
pip install "pyprims[notebook]"     # from PyPI
pip install -e ".[notebook,test]"   # or, in a clone of the repository
python -m pytest tests           # 53 tests
python -m pyprims run examples/count.prims -n 1 --trace 3
python -m pyprims batch examples/testbatch.bprims -o out.dat --seed 1
python -m pyprims convert mymodel.prims          # writes mymodel.py
```

## Tutorial

The folder `tutorial` has a PRIMs tutorial as Jupyter notebooks: Unit 1
(introduction: the structure of a model, running and inspecting it,
production compilation, transfer, batch runs), Unit 2 (variable binding and
tasks with several skills), Unit 3 (operator selection, learning skill–operator
associations, far transfer), and a reference notebook with all parameters and
script functions. Each unit folder contains the model files it uses and an
assignment. See `tutorial/README.md`.

## Models in Python

Existing `.prims` files load directly. Models can also be written in Python;
`convert` produces this form from a `.prims` file:

```python
from pyprims import Task, Model

task = Task("count", initial_skills=["count"], default_activation=1.0,
            ol=True, rt=-2.0, lf=0.2, default_operator_self_assoc=0.0,
            egs=0.05, retrieval_reinforces=True)

count = task.skill("count")
count.operator("start-count", """
    V1 <> nil        // there has to be a start number
    WM1 = nil
==>
    V1 -> WM1
    count-fact -> RT1
    V1 -> RT2
    say -> AC1
    V1 -> AC2
""")
# ... more operators ...

task.facts([("cf1", "count-fact", "one", "two"), ("cf2", "count-fact", "two", "three")])
task.action("say", latency=0.3, noise=0.1, distribution="uniform", output="Saying")

@task.script
def script(m):                       # m offers every PRIMs script function
    start = m.random(3)
    m.screen("one", "three")
    m.run_until_action("say", "stop")
    m.issue_reward()
    m.trial_end()

model = Model(seed=1)
model.load(task)                     # or model.load("count.prims") / ("count.py")
model.run(50)
print(model.results[:5])             # trial times
print(model.get_trace(3))            # trace of the last trial
```

The parts of a model map onto Python as follows:

- **Parameters** are keyword arguments with `-` written as `_`. So
  `default-operator-self-assoc:` becomes `default_operator_self_assoc=`,
  `t` becomes `True` and `nil` becomes `False`.
- **Operator bodies** keep the PRIM notation, because it is the theory's
  vocabulary. A body copied from a `.prims` file works unchanged.
- **Script functions** become methods of `m`, with `-` written as `_`. So
  `run-until-action` is `m.run_until_action`, `issue-reward` is
  `m.issue_reward`, and so on.
- **Transfer between tasks** works by loading several tasks into one model.
  Declarative memory and the learned productions carry over:
  `model.load("count.prims"); model.run(50); model.load("semantic.prims"); model.run(20)`.

## Jupyter notebooks

`pyprims.notebook` gives a notebook what the Swift GUI offers: stepping,
the trace with its five detail levels, buffers, declarative memory, the
conflict trace, productions, the results chart, the PRIMs graph, and a
widget dashboard with the Swift buttons. `examples/pyprims_in_jupyter.ipynb`
walks through all of it.

```bash
pip install "pyprims[notebook]"     # matplotlib, numpy, pandas, ipywidgets, jupyterlab
```

```python
from pyprims.notebook import Session, Dashboard
s = Session(seed=1)
s.load("count.prims")        # or a .py file, a Task, or model text in either syntax
s.step()                     # one model step, as the Swift Step button
s.buffers(); s.conflict_set(); print(s.trace(3))
s.run(100)
s.plot_results(); s.plot_graph(2)
s.associations("count")       # Sji between a skill and its operators (learned or set)
Dashboard(s)                 # the whole window as widgets
```

`plot_results` draws one line per load or reset of a task; repeated runs of a
task are dashed, and `labels=[...]` names the lines. After `run(n)`, a
`NoOperatorWarning` reports trials in which the model stopped because no
operator matched.

Models can be written in cells, in standard PRIMs syntax (`%%prims`) or
with the Python API (`%%pyprims`). Stepping and inspection never change
the simulation: a seeded model gives the same results whether it is stepped
and inspected or simply run (`tests/test_notebook.py`).

## Symbolic or embedded slot contents

By default a model is the original, symbolic PRIMs. An embedded
representation gives every slot value a vector. Similarity then drives the
comparison PRIMs (`=`, `<>`), retrieval (partial matching) and blending,
while operators, PRIMs and chunk identities stay symbolic. See
`EMBEDDINGS.md`; `tools/symbolic_limit.py` checks that the embedded model
reduces exactly to the symbolic one in its limit, and
`examples/fan_effect_embeddings.ipynb` explores it on the fan effect. `Holographic` goes one step further: regular chunks become vectors
(an order vector for retrieval and read-back, a bag vector for spreading),
while operators stay symbolic.

```python
from pyprims.embedding import Embedded, Embedding
m = Model(seed=1, representation=Embedded(Embedding(dim=512), compare="threshold",
                                          theta=0.6, retrieval="soft", mp=5.0))
```

## Batch runs from Python

`run_trials` is the Python counterpart of a `run` line in a `.bprims` file.
It loads a task (or switches back to it, so learning carries over), runs it,
and returns records for pandas:

```python
import pandas as pd
from pyprims import Model, run_trials

rows = []
for rep in range(10):
    m = Model(seed=rep, trace=False, batch_mode=True)
    rows += run_trials(m, 100, "count.prims", repeat=rep, label="train")
    rows += run_trials(m, 100, "semantic.prims", repeat=rep, label="transfer")
data = pd.DataFrame(rows)     # trial, task, result, outcome, + the fields given
```

`events=True` gives one record per event instead, with the columns of the
batch output.

## Inspecting a model

- `model.output_data`: the batch events (actions, trial ends, data lines).
- `model.results`: one point per trial; the trial time, or the value of
  `plot-point`.
- `model.get_trace(level)`: the trace of the last trial, at levels 0–5 as in
  the PRIMs GUI.
- `model.dm.chunks`, `model.chunk("cf1").activation()`,
  `model.operators_in_dm()`.
- `model.production_table()`: the learned productions and their utilities.

See `PORTING.md` for:

- the Swift-to-Python module map;
- every behaviour kept from Swift on purpose;
- every deliberate difference;
- the validation against Swift output.

## License

MIT; see `LICENSE`.
