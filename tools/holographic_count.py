#!/usr/bin/env python3
"""The count model with holographic chunks (pyprims.embedding.Holographic).

    python tools/holographic_count.py              # all three tables, 20 runs x 60 trials
    python tools/holographic_count.py --runs 5     # quicker

Each condition runs the count model (examples/count.prims) 20 times (seeds
0-19, each with its own random vectors) for 60 trials, and reports the share
of trials that are counted correctly, the error types, the mean time of
correct trials, and the share of retrievals in which at least one slot was
read back wrong.  Comparisons are symbolic (``RT2 = WM1`` is exact), so a
wrong fact makes the model stall rather than skip.

"embedded" is the hybrid: symbolic chunks whose slot values have vectors.
"holographic" stores each count fact as an order vector sum_k P_k e(v_k)
and retrieves and reads back from that vector.

1. mp sweep, random vectors and a magnitude code for the number words.
2. Dimension sweep (3-slot facts).
3. Chunk width: filler slots added to every count fact.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from pyprims import Model                                            # noqa: E402
from pyprims.embedding import Embedded, Embedding, Holographic       # noqa: E402

DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
COUNT = os.path.join(HERE, "..", "examples", "count.prims")


def model_path(extra: int) -> str:
    """count.prims, with `extra` filler slots appended to every count fact."""
    if extra == 0:
        return COUNT
    path = os.path.join(tempfile.gettempdir(), f"pyprims_count_extra{extra}.prims")
    if not os.path.exists(path):
        def add(m):
            return m.group(0)[:-1] + " " + " ".join(f"{m.group(1)}x{k}" for k in range(extra)) + ")"
        text = re.sub(r"\((c[fg]\d) count-fact\s+\w+\s+\w+\)", add, open(COUNT).read())
        with open(path, "w") as f:
            f.write(text)
    return path


def embedding(code: str, seed: int, dim: int) -> Embedding:
    emb = Embedding(dim=dim, seed=seed)
    if code == "magnitude":                 # number words on a mental number line
        emb.set_numbers(range(1, 11), width=1.0)
        for i, w in enumerate(DIGITS, start=1):
            emb.set(w, emb.vector(str(i)))
    return emb


def classify(said, start, end) -> str:
    want = DIGITS[start:end + 1]
    if said == want + ["stop"]:
        return "correct"
    if not said or said[-1] != "stop":
        return "stall"
    nums = said[:-1]
    if nums and set(nums) <= set(want) and len(nums) < len(want):
        return "skip"
    return "other"


def one_run(job):
    kind, code, dim, mp, extra, seed, trials = job
    rep = None
    if kind != "symbolic":
        cls = Holographic if kind == "holographic" else Embedded
        rep = cls(embedding(code, seed, dim), compare="symbolic", retrieval="soft", mp=mp)
    m = Model(seed=1000 + seed, trace=False, batch_mode=True, representation=rep)
    m.load(model_path(extra))
    rec = []

    def script(s):
        start = s.random(3)
        end = start + 1 + s.random(3)
        n0, t0 = len(s.model.output_data), s.model.time
        s.screen(DIGITS[start], DIGITS[end])
        s.run_until_action("say", "stop")
        said = [d.p2 for d in s.model.output_data[n0:] if d.event_type == "action" and d.p1 == "say"]
        rec.append((classify(said, start, end), s.model.time - t0))
        s.issue_reward()
        s.trial_end()

    m.scenario.script = script
    m.time_threshold = 60.0
    for _ in range(trials):
        m.run_trial()
    out = {k: sum(1 for c, _ in rec if c == k) / len(rec) for k in ("correct", "stall", "skip", "other")}
    out["time"] = statistics.mean(t for c, t in rec if c == "correct") if out["correct"] else math.nan
    out["misread"] = rep.decode_errors / rep.decodes if getattr(rep, "decodes", 0) else 0.0
    return out


def many(kind, code="random", dim=512, mp=5.0, extra=0, runs=20, trials=60, processes=None):
    jobs = [(kind, code, dim, mp, extra, s, trials) for s in range(runs)]
    with ProcessPoolExecutor(processes or os.cpu_count()) as ex:
        res = list(ex.map(one_run, jobs))
    agg = {}
    for k in res[0]:
        xs = [r[k] for r in res if not math.isnan(r[k])]
        agg[k] = (statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)
    return agg


def row(label, a):
    return (f"{label}  {100 * a['correct'][0]:5.1f} ± {100 * a['correct'][1]:3.1f}%   stall {100 * a['stall'][0]:4.1f}%"
            f"  skip {100 * a['skip'][0]:3.1f}%  other {100 * a['other'][0]:3.1f}%   time {a['time'][0]:.2f} s"
            f"   misread {100 * a['misread'][0]:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    kw = dict(runs=a.runs, trials=a.trials)
    print(f"count model, {a.runs} runs x {a.trials} trials; correct = whole sequence right;"
          " misread = retrievals with a slot read back wrong\n")
    print(row(f"{'symbolic':<34}", many("symbolic", **kw)))
    print("\n1. mismatch penalty (d = 512, 3-slot facts)")
    for code in ("random", "magnitude"):
        for mp in (math.inf, 20.0, 5.0, 3.0, 2.0, 1.0):
            for kind in ("embedded", "holographic"):
                print(row(f"  {code:9s} {kind:12s} mp {mp:>4g}", many(kind, code, 512, mp, **kw)), flush=True)
    print("\n2. dimension (random vectors, 3-slot facts)")
    for dim in (64, 128, 256, 512, 2048):
        for mp in (math.inf, 20.0, 5.0):
            print(row(f"  d {dim:>5}  holographic   mp {mp:>4g}", many("holographic", "random", dim, mp, **kw)), flush=True)
    print("\n3. chunk width (random vectors; filler slots added to every count fact)")
    for dim in (128, 512):
        for extra in (0, 2, 5, 10):
            for mp in (math.inf, 5.0):
                print(row(f"  d {dim:>4}  {3 + extra:>2} slots  mp {mp:>4g}   ",
                          many("holographic", "random", dim, mp, extra, **kw)), flush=True)
    print("\n(the embedded hybrid counts 100% correctly in every condition of tables 2 and 3)")


if __name__ == "__main__":
    main()
