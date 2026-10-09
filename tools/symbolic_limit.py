#!/usr/bin/env python3
"""The symbolic limit of the embedded representation.

    python tools/symbolic_limit.py                 # all examples/*.prims
    python tools/symbolic_limit.py path/*.prims    # other models
    python tools/symbolic_limit.py --sweep         # also: how counting departs as mp falls

Part 1 (exact).  With random high-dimensional vectors, a similarity threshold
for comparisons and an infinite mismatch penalty, an embedded model must give
*identical* runs to the symbolic model: same trace, same events and times,
same chunk references, same production utilities.  Each model is run with
the same seed in four representations:

    symbolic                        the original PRIMs
    threshold                       comparisons by similarity >= theta
    soft, mp = inf                  retrieval by partial matching, symbolic limit
    threshold + soft, mp = inf      both

and the runs are fingerprinted.  The check also reports the largest
similarity between two distinct symbols the models used; it must stay below
theta, otherwise two different symbols would count as equal.

Part 2 (--sweep).  The count model with random vectors and a finite mismatch
penalty.  As mp falls, retrieval starts returning the wrong count fact and
counting goes wrong.  At large mp the model is statistically the symbolic one.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import statistics
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from pyprims import Model                                   # noqa: E402
from pyprims.embedding import Embedded, Embedding, Holographic   # noqa: E402

DIM, THETA = 512, 0.5


def representations(emb, spreading=False):
    reps = {
        "symbolic": None,
        "threshold": Embedded(emb, compare="threshold", theta=THETA, retrieval="symbolic"),
        "soft, mp=inf": Embedded(emb, compare="symbolic", retrieval="soft", mp=math.inf),
        "threshold + soft, mp=inf": Embedded(emb, compare="threshold", theta=THETA, retrieval="soft", mp=math.inf),
    }
    # regular chunks as order + bag vectors; retrieval and read-back from the vectors
    reps["holographic, mp=inf"] = Holographic(emb, compare="threshold", theta=THETA, mp=math.inf)
    if spreading:
        # similarity-based associations, beta = inf: counts the chunks that contain a
        # cue.  Differs from symbolic only where PRIMs' fan counter misses a reference
        # made before the chunk was defined (PORTING.md, kept quirk 3).
        reps["all + similarity spreading, beta=inf"] = Embedded(
            emb, compare="threshold", theta=THETA, retrieval="soft", mp=math.inf,
            spreading="similarity", beta=math.inf)
        reps["holographic + similarity spreading, beta=inf"] = Holographic(
            emb, compare="threshold", theta=THETA, mp=math.inf, spreading="similarity", beta=math.inf)
    return reps


def fingerprint(path, rep, seed=5, trials=8):
    m = Model(seed=seed, trace=True, batch_mode=True, representation=rep)
    try:
        m.load(path)
        for _ in range(trials):
            m.run_trial()
        sig = repr((m.time, m.results, [(d.event_type, d.p1, d.p2, d.time) for d in m.output_data],
                    m.get_trace(5),
                    sorted((c.name, c.references, c.creation_time) for c in m.dm.chunks.values()),
                    sorted((k, round(p.u, 12)) for k, p in m.procedural.productions.items())))
        symbols = set(m.dm.chunks) | {str(v) for c in m.dm.chunks.values() for v in c.slotvals.values()
                                      if isinstance(v, (str, float))}
    except Exception as e:                     # a model that fails must fail the same way
        sig, symbols = f"ERR {type(e).__name__}: {e}", set()
    return hashlib.sha256(sig.encode()).hexdigest()[:16], symbols


def exact_check(files, spreading=False):
    emb = Embedding(dim=DIM, seed=0)
    reps = representations(emb, spreading)
    all_symbols = set()
    bad = 0
    print(f"Part 1: exact identity (dim={DIM}, theta={THETA}), {len(files)} models\n")
    for f in files:
        prints = {}
        for name, rep in reps.items():
            prints[name], syms = fingerprint(f, rep)
            all_symbols |= syms
        same = len(set(prints.values())) == 1
        bad += not same
        print(f"  {'SAME' if same else 'DIFF'}  {os.path.relpath(f)}"
              + ("" if same else "   " + ", ".join(f"{k}={v}" for k, v in prints.items())))
    mx = emb.max_offdiagonal_similarity(all_symbols)
    print(f"\n  {len(files) - bad}/{len(files)} models identical in all {len(reps)} representations")
    print(f"  {len(all_symbols)} distinct symbols; largest similarity between two of them = {mx:.3f}"
          f" (must be < theta = {THETA})")
    return bad == 0 and mx < THETA


# ---------------------------------------------------------------------- part 2
DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def counting_script(record):
    def script(m):
        start = m.random(3)
        end = start + 1 + m.random(3)
        m.screen(DIGITS[start], DIGITS[end])
        m.run_until_action("say", "stop")
        said = [d.p2 for d in m.model.output_data if d.event_type == "action" and d.p1 == "say"]
        record.append(said == DIGITS[start:end + 1] + ["stop"])
        m.issue_reward()
        m.trial_end()
    return script


def _embedding(code: str, seed: int) -> Embedding:
    emb = Embedding(dim=DIM, seed=seed)
    if code == "magnitude":                      # number words on a mental number line
        emb.set_numbers(range(1, 11), width=1.0)
        for i, w in enumerate(DIGITS, start=1):
            emb.set(w, emb.vector(str(i)))
    return emb


def sweep(model_path, runs=20, trials=60):
    for code in ("random", "magnitude"):
        print(f"\nPart 2: count model, {code} vectors (dim={DIM}), soft retrieval, {runs} runs x {trials} trials")
        if code == "magnitude":
            e = _embedding(code, 0)
            print("  similarity of 'three' to: " + ", ".join(f"{w} {e.sim('three', w):.2f}" for w in DIGITS[:6]))
        print(f"  {'mp':>6}  {'correct sequences':>18}  {'mean trial time (s)':>20}")
        for mp in (None, math.inf, 20.0, 5.0, 3.0, 2.0, 1.0, 0.5):
            if code == "magnitude" and mp is None:
                continue
            acc, times = [], []
            for r in range(runs):
                rep = None if mp is None else Embedded(_embedding(code, r), compare="symbolic",
                                                        retrieval="soft", mp=mp)
                m = Model(seed=1000 + r, trace=False, batch_mode=True, representation=rep)
                m.load(model_path)
                rec = []
                m.scenario.script = counting_script(rec)
                m.time_threshold = 60.0
                for _ in range(trials):
                    m.run_trial()
                acc.append(statistics.mean(rec))
                times.append(statistics.mean(m.results))
            label = "symb." if mp is None else ("inf" if mp == math.inf else f"{mp:g}")
            se = statistics.stdev(times) / math.sqrt(runs)
            print(f"  {label:>6}  {100 * statistics.mean(acc):17.1f}%  {statistics.mean(times):14.3f} ± {se:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--spreading", action="store_true", help="also check similarity spreading at beta = inf")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    files = a.files or sorted(glob.glob(os.path.join(HERE, "..", "examples", "*.prims")))
    ok = exact_check(files, a.spreading)
    if a.sweep:
        sweep(os.path.join(HERE, "..", "examples", "count.prims"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
