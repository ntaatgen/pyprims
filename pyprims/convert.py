"""Convert a .prims model file into an equivalent Python model file.

    python -m pyprims.convert count.prims            # writes count.py
    python -m pyprims.convert models/*.prims -o out/

The generated file defines ``task`` with the Python API in pyprims.dsl.  The
operator bodies are copied verbatim (comments included); the script is
translated into a Python function.  Converting and then loading the .py file
installs exactly the same model as loading the .prims file.
"""
from __future__ import annotations

import argparse
import os
import textwrap

from .dsl import (parse_prims, TaskSettings, SkillDef, FactsDef, VisualDef, ActionDef,
                  SjiDef, ScriptDef)
from .script_translate import translate_source


def _py_param(key: str) -> str:
    k = key.rstrip(":").replace("-", "_")
    return k if k.isidentifier() else None


def _py_value(v: str) -> str:
    if v == "t":
        return "True"
    if v == "nil":
        return "False"
    try:
        float(v)
        return v
    except ValueError:
        return repr(v)


_LIST_SETTINGS = {"initial-skills:": "initial_skills", "initial-goals:": "initial_skills",
                  "initial-skill:": "initial_skills", "skills:": "skills", "goals:": "skills",
                  "references:": "references", "task-constants:": "task_constants"}


def _settings_args(settings) -> list[str]:
    args = []
    for key, value in settings:
        if key in _LIST_SETTINGS:
            args.append(f"{_LIST_SETTINGS[key]}={list(value)!r}")
        else:
            k = _py_param(key)
            if k is None:
                raise ValueError(f"parameter {key} has no Python name")
            args.append(f"{k}={_py_value(value)}")
    return args


def _body(raw: str) -> str:
    lines = textwrap.dedent(raw.strip("\n")).rstrip().split("\n")
    return "\n".join("        " + ln if ln.strip() else "" for ln in lines)


def _tuple(items) -> str:
    return "(" + ", ".join(repr(x) for x in items) + ("," if len(items) == 1 else "") + ")"


def convert_text(text: str, source_name: str = "model.prims") -> str:
    task = parse_prims(text)
    out = [f'"""PRIMs model converted from {source_name} by pyprims.convert."""',
           "from pyprims import Task", "HELPERS", ""]
    first = True
    skill_vars = {}
    for d in task.defs:
        if isinstance(d, TaskSettings):
            args = _settings_args(d.settings)
            if first:
                inner = ",\n            ".join([repr(d.name)] + args)
                out.append(f"task = Task({inner})\n")
                first = False
            else:
                out.append("task.set(" + ", ".join(args) + ")\n")
        elif isinstance(d, SkillDef):
            if first:
                out.append(f"task = Task({task.name!r})\n")
                first = False
            var = "skill_" + d.name.replace("-", "_")
            if not var.isidentifier():
                var = f"skill_{len(skill_vars)}"
            skill_vars[d.name] = var
            out.append(f"{var} = task.skill({d.name!r})")
            for op in d.operators:
                act = "" if op.activation is None else f", activation={op.activation}"
                out.append(f"{var}.operator({op.name!r}, '''\n{_body(op.body)}\n    '''{act})")
            out.append("")
        elif isinstance(d, FactsDef):
            out.append("task.facts([")
            out += [f"    {_tuple(f)}," for f in d.facts]
            out.append("])\n")
        elif isinstance(d, VisualDef):
            out.append("task.visual([")
            out += [f"    {_tuple(f)}," for f in d.chunks]
            out.append("])\n")
        elif isinstance(d, ActionDef):
            args = [repr(d.name)]
            for k in ("latency", "noise", "distribution", "output"):
                v = getattr(d, k)
                if v is not None:
                    args.append(f"{k}={v!r}")
            out.append("task.action(" + ", ".join(args) + ")")
        elif isinstance(d, SjiDef):
            for j, i, v in d.entries:
                out.append(f"task.sji({j!r}, {i!r}, {v})")
        elif isinstance(d, ScriptDef):
            name = "init_script" if d.init else "script"
            from .dsl import strip_comments
            code = translate_source(strip_comments(d.source), name)
            out.append("")
            out.append(f"@task.{name}")
            out.append(code)
    text_out = "\n".join(out).rstrip() + "\n"
    from .script_translate import HELPER_NAMES
    used = [h for h in HELPER_NAMES if f"{h}(" in text_out]
    helper_line = f"from pyprims.script_api import {', '.join(used)}" if used else ""
    return text_out.replace("HELPERS\n", helper_line + "\n" if used else "", 1)


def convert_file(path: str, out_dir: str | None = None) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    py = convert_text(text, os.path.basename(path))
    base = os.path.splitext(os.path.basename(path))[0]
    target = os.path.join(out_dir or os.path.dirname(os.path.abspath(path)), base + ".py")
    with open(target, "w", encoding="utf-8") as f:
        f.write(py)
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert .prims models to Python")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out-dir")
    a = ap.parse_args(argv)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    for f in a.files:
        print(convert_file(f, a.out_dir))


if __name__ == "__main__":
    main()
