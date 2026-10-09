#!/usr/bin/env python3
"""Compare pyprims batch output with Swift PRIMs batch output.

    python tools/compare_with_swift.py SWIFT_OUTPUT PYTHON_OUTPUT [--block 20]
    python tools/compare_with_swift.py --run testbatch           # runs Python, compares to examples/swift_reference
    python tools/compare_with_swift.py --run taskswitchstrooptransfer --repeats 50

Both files are in the batch format: rep task label trial time event p1 p2 p3 in0..in4 firings.
For every label it reports, per block of trials, the mean trial duration
(trial-end time) with a standard error over *model runs* (the independent
unit; trials within a run are not independent), the z of the difference, and
the counts of each action.  Use it whenever the Python core changes, and
against a fresh Swift batch when Swift changes.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import math
import os
import re
import statistics as st
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load(path, block):
    dur = collections.defaultdict(lambda: collections.defaultdict(list))
    acts = collections.Counter()
    for ln in _open(path):
        f = ln.split()
        if len(f) < 15:
            continue
        rep, label, trial, t, ev = int(f[0]), f[2], int(f[3]), float(f[4]), f[5]
        if ev == "trial-end":
            dur[label][(rep, (trial - 1) // block)].append(t)
            acts[(label, "trial-end", f[6])] += 1
        elif ev == "action":
            acts[(label, "action", f[6])] += 1
    return dur, acts


def compare(swift, python, block=20, out=sys.stdout):
    sd, sa = load(swift, block)
    pd, pa = load(python, block)
    worst = 0.0
    for label in sorted(set(sd) & set(pd)):
        out.write(f"== {label}: mean trial duration per block of {block} trials (± SE over model runs)\n")
        blocks = sorted({b for (_, b) in sd[label]} & {b for (_, b) in pd[label]})
        for b in blocks:
            s = [st.mean(v) for (r, bb), v in sd[label].items() if bb == b]
            p = [st.mean(v) for (r, bb), v in pd[label].items() if bb == b]
            if len(s) < 2 or len(p) < 2:
                continue
            se_s, se_p = st.stdev(s) / math.sqrt(len(s)), st.stdev(p) / math.sqrt(len(p))
            z = (st.mean(p) - st.mean(s)) / math.hypot(se_s, se_p) if (se_s or se_p) else 0.0
            worst = max(worst, abs(z))
            out.write(f"  trials {b * block + 1:4d}-{(b + 1) * block:<4d} Swift {st.mean(s):8.3f} ± {se_s:.3f} (n={len(s)})"
                      f"   Python {st.mean(p):8.3f} ± {se_p:.3f} (n={len(p)})   z = {z:+.2f}\n")
    out.write("== event counts (Swift / Python; Python scaled to the Swift number of runs)\n")
    ns = len({k[0] for d in sd.values() for k in d}) or 1
    np_ = len({k[0] for d in pd.values() for k in d}) or 1
    for k in sorted(set(sa) | set(pa)):
        out.write(f"  {k[0]:20s} {k[1]:9s} {k[2]:15s} {sa.get(k, 0):8d} / {pa.get(k, 0) * ns / np_:10.1f}\n")
    out.write(f"largest |z| = {worst:.2f}\n")
    return worst


def run_reference(name, repeats=None, seed=1):
    sys.path.insert(0, os.path.join(HERE, ".."))
    from pyprims import run_batch
    src = os.path.join(EX, name + ".bprims")
    text = open(src).read()
    if repeats:
        text = re.sub(r"repeat\s+\d+", f"repeat {repeats}", text, count=1)
    tmp = tempfile.mkdtemp()
    bp = os.path.join(tmp, name + ".bprims")
    open(bp, "w").write(text)
    out = os.path.join(tmp, name + "_python.dat")
    run_batch(bp, out_path=out, seed=seed, directory=EX)
    ref = [f for f in os.listdir(os.path.join(EX, "swift_reference")) if f.startswith(name)][0]
    return os.path.join(EX, "swift_reference", ref), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--run", help="name of an examples/*.bprims with a Swift reference")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block", type=int, default=20)
    a = ap.parse_args()
    if a.run:
        swift, python = run_reference(a.run, a.repeats, a.seed)
    else:
        swift, python = a.files
    compare(swift, python, a.block)


if __name__ == "__main__":
    main()
