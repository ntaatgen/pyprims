"""Command line: ``python -m pyprims {run,batch,convert} ...``

    python -m pyprims run count.prims -n 10 --trace 3
    python -m pyprims batch testbatch.bprims -o out.dat --seed 1
    python -m pyprims convert count.prims semantic.prims -o models/
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyprims")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a model for n trials")
    r.add_argument("model", nargs="+", help=".prims or .py files; all are loaded, the last one is run")
    r.add_argument("-n", type=int, default=1)
    r.add_argument("--seed", type=int)
    r.add_argument("--trace", type=int, default=-1, help="print the last trial's trace up to this level (0-5)")

    b = sub.add_parser("batch", help="run a .bprims batch script")
    b.add_argument("bprims")
    b.add_argument("-o", "--out")
    b.add_argument("--seed", type=int)

    c = sub.add_parser("convert", help="convert .prims files to Python")
    c.add_argument("files", nargs="+")
    c.add_argument("-o", "--out-dir")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .model import Model
        m = Model(seed=a.seed, trace=a.trace >= 0, batch_mode=a.trace < 0)
        for f in a.model:
            m.load(f)
        for _ in range(a.n):
            m.run_trial()
        if a.trace >= 0:
            sys.stdout.write(m.get_trace(a.trace))
        for i, rt in enumerate(m.results, start=1):
            print(f"trial {i}: {rt:.3f}")
    elif a.cmd == "batch":
        from .batch import run_batch
        out = run_batch(a.bprims, out_path=a.out, seed=a.seed, progress=True)
        if not a.out:
            sys.stdout.write("".join(out))
    elif a.cmd == "convert":
        from .convert import main as convert_main
        convert_main(a.files + (["-o", a.out_dir] if a.out_dir else []))


if __name__ == "__main__":
    main()
