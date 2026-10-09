# PRIMs tutorial (notebooks)

A hands-on introduction to the PRIMs cognitive architecture, as Jupyter
notebooks that use pyprims. Models are written in `.prims` files or in
notebook cells; you load, run and inspect them with function calls in the
notebook, or with the buttons of a dashboard.

## Setting up

You need Python 3.10 or later. Install pyprims with its notebook extras
(matplotlib, numpy, pandas, ipywidgets, jupyterlab):

```bash
pip install "pyprims[notebook] @ git+https://github.com/ntaatgen/pyprims"
```

or, in a clone of this repository, `pip install -e ".[notebook]"` from its
top folder.

Then start Jupyter (`jupyter lab`) and open the notebook of a unit **from its
own folder**: each notebook loads the model files that are next to it.

## Contents

| Notebook | Topics |
|---|---|
| `Unit 1/Unit 1 - Introduction.ipynb` | Declarative memory, the structure of a PRIMs model, the execution cycle, running and inspecting a model, the dashboard, production compilation, transfer between count and semantic, batch runs (batch files and Python). Assignment: addition by counting, with and without fingers |
| `Unit 2/Unit 2 - Binding and Skills.ipynb` | Variables in skills, bindings in declarative memory, tasks with several skills, the multilevel architecture. Assignment: semantic questions about attributes |
| `Unit 3/Unit 3 - Operator Selection.ipynb` | Competition between operators, learning skill–operator associations, far transfer from task switching to Stroop, batch files and analysis with pandas. Assignment: bottom-up learning of operators |
| `Reference.ipynb` | Parameters, script syntax and functions, batch files, models in Python, the notebook session |

The notebooks are saved with their output, so you can read them on GitHub
before running them. With the same seeds, running them gives the same
results.

Notebook versions of the later units of the PRIMs tutorial (extended vision
and composable goals) will follow.
