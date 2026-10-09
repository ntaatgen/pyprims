"""The fan effect (Anderson, 1974) as a PRIMs model.

Participants study sentences such as "the hippie is in the park" and then
verify probes: targets (studied pairs) and foils (a studied person with a
studied location they were never paired with).  Verification time grows with
the number of facts about the person and about the location (the "fan").

The PRIMs model:

    retrieve-fact          V1 <> nil, WM1 = nil  ==>  V1 -> WM1, in -> RT1
                           (a retrieval cued only by type; the probe's person
                            and location in V1/V2 spread activation to the facts)
    respond-yes            V1 = RT2, V2 = RT3    ==>  press yes
    respond-no-person      V1 <> RT2             ==>  press no
    respond-no-location    V2 <> RT3             ==>  press no
    respond-no-failure     RT1 = error           ==>  press no

In the symbolic representation the fan effect comes from S_ji = mas - ln(fan_j).
With the embedded representation (pyprims.embedding) the same model can be
run with similarity-based spreading, partial matching, or both; see
fan_effect_embeddings.ipynb.

``DATA`` are the mean verification times (s) and ``ERRORS`` the error rates of
Anderson (1974), Table 5: the means over Expts 1 and 2 and the no-precue
condition of Expt 3, 63 subjects and 432 observations per cell.  The ACT-R
tutorial (Unit 5) gives the same times rounded to 10 ms.

Anderson, J. R. (1974). Retrieval of propositional information from long-term
memory. Cognitive Psychology, 6, 451-474.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from pyprims import Model, Task

PERSONS = ["hippie", "captain", "debutante", "fireman", "giant", "earl", "lawyer", "doctor",
           "sailor", "teacher", "banker", "farmer"]
LOCATIONS = ["park", "church", "bank", "cave", "beach", "castle", "dungeon", "forest",
             "store", "garden", "harbor", "museum"]

# 24 studied facts; every (person fan, location fan) cell from 1x1 to 3x3 occurs
FACTS = [("hippie", "dungeon"), ("captain", "garden"), ("debutante", "park"), ("fireman", "museum"),
         ("giant", "harbor"), ("giant", "store"), ("earl", "garden"), ("earl", "museum"),
         ("lawyer", "castle"), ("lawyer", "church"), ("doctor", "harbor"), ("doctor", "beach"),
         ("sailor", "museum"), ("sailor", "dungeon"), ("sailor", "forest"),
         ("teacher", "cave"), ("teacher", "forest"), ("teacher", "garden"),
         ("banker", "store"), ("banker", "beach"), ("banker", "castle"),
         ("farmer", "harbor"), ("farmer", "bank"), ("farmer", "store")]

# Anderson (1974), Table 5.  Rows: location fan 1..3; columns: person fan 1..3.
_T = [[1.111, 1.174, 1.222], [1.167, 1.198, 1.222], [1.153, 1.233, 1.357]]    # trues, s
_F = [[1.197, 1.221, 1.264], [1.250, 1.356, 1.291], [1.262, 1.471, 1.465]]    # falses, s
_TE = [[.051, .042, .046], [.065, .056, .060], [.063, .044, .054]]            # error rates
_FE = [[.019, .042, .030], [.014, .037, .044], [.042, .079, .051]]


def _cells(rows):
    return {(p + 1, l + 1): rows[l][p] for p in range(3) for l in range(3)}


# Key: (person fan, location fan).
DATA = {"target": _cells(_T), "foil": _cells(_F)}
ERRORS = {"target": _cells(_TE), "foil": _cells(_FE)}


def fans(facts=FACTS):
    pf, lf = defaultdict(int), defaultdict(int)
    for p, l in facts:
        pf[p] += 1
        lf[l] += 1
    return dict(pf), dict(lf)


def probes(facts=FACTS, foils_per_cell=2, seed=0):
    """Targets: every studied fact.  Foils: up to n unstudied pairs per fan cell."""
    pf, lf = fans(facts)
    studied = set(facts)
    out = [(p, l, "target", pf[p], lf[l]) for p, l in facts]
    cells = defaultdict(list)
    for p in pf:
        for l in lf:
            if (p, l) not in studied:
                cells[(pf[p], lf[l])].append((p, l))
    rng = random.Random(seed)
    for cell in sorted(cells):
        pairs = cells[cell]
        rng.shuffle(pairs)
        out += [(p, l, "foil", cell[0], cell[1]) for p, l in pairs[:foils_per_cell]]
    return out


def make_task(facts=FACTS, *, lf=0.8, mas=2.0, ans=0.02, w=1.0, rt=-2.0, extra_facts=(), cue="spreading"):
    """The fan task.

    cue="spreading": the retrieval request only asks for a fact; the probe's
    person and location reach the facts through spreading activation (w is
    input-activation:).  cue="slots": the person and location are cues in the
    request itself (exact match symbolically, partial matching when embedded)
    and there is no spreading from the probe.
    """
    if cue not in ("spreading", "slots"):
        raise ValueError("cue must be 'spreading' or 'slots'")
    task = Task("fan", initial_skills=["fan"], default_activation=0.0, ol=True, bll=0.5,
                lf=lf, mas=mas, ans=ans, rt=rt, input_activation=(w if cue == "spreading" else 0.0),
                egs=0.05)
    sk = task.skill("fan")
    if cue == "spreading":
        sk.operator("retrieve-fact", """
            V1 <> nil
            WM1 = nil
        ==>
            V1 -> WM1
            in -> RT1
        """)
    else:
        sk.operator("retrieve-fact", """
            V1 <> nil
            WM1 = nil
        ==>
            V1 -> WM1
            in -> RT1
            V1 -> RT2
            V2 -> RT3
        """)
    sk.operator("respond-yes", """
        RT2 = V1
        RT3 = V2
    ==>
        press -> AC1
        yes -> AC2
    """)
    sk.operator("respond-no-person", """
        RT2 <> V1
    ==>
        press -> AC1
        no -> AC2
    """)
    sk.operator("respond-no-location", """
        RT3 <> V2
    ==>
        press -> AC1
        no -> AC2
    """)
    sk.operator("respond-no-failure", """
        RT1 = error
    ==>
        press -> AC1
        no -> AC2
    """)
    task.facts([(f"f{k}", "in", p, l) for k, (p, l) in enumerate(facts, start=1)] + list(extra_facts))
    task.action("press", latency=0.2)

    @task.script
    def script(m):                       # the probe is set by run() before each trial
        p, l, kind = m.model.current_probe[:3]
        m.screen(p, l)
        m.run_until_action("press")
        ac = m.last_action()
        m.model.probe_response = ac[1] if len(ac) > 1 else None
        m.trial_end()
        m.sleep(1.0)
    return task


def run(task=None, representation=None, *, blocks=8, warmup_blocks=2, seed=1, probe_list=None,
        model_hook=None):
    """Run the experiment; return one record per test trial (after warm-up blocks).

    Each block presents every probe once, in random order.  Before the first
    block the model waits 300 s so that study-phase base levels have settled.
    """
    task = task or make_task()
    plist = probe_list or probes()
    m = Model(seed=seed, trace=False, batch_mode=True, representation=representation)
    m.load(task)
    if model_hook:
        model_hook(m)
    m.time += 300.0
    rng = random.Random(seed + 17)
    recs = []
    for b in range(blocks):
        order = list(plist)
        rng.shuffle(order)
        for pr in order:
            m.current_probe = pr
            m.probe_response = None
            m.run_trial()
            if b >= warmup_blocks:
                p, l, kind, pf, lf = pr
                want = "yes" if kind == "target" else "no"
                recs.append(dict(person=p, location=l, kind=kind, pf=pf, lf=lf,
                                 rt=m.results[-1], correct=m.probe_response == want))
    return recs


def summarize(recs, by_probe=False):
    """Mean RT of correct trials and error rate per (kind, person fan, location fan),
    or per (kind, person, location) with by_probe=True."""
    by = defaultdict(list)
    for r in recs:
        key = (r["kind"], r["person"], r["location"]) if by_probe else (r["kind"], r["pf"], r["lf"])
        by[key].append(r)
    out = {}
    for k, rs in sorted(by.items()):
        ok = [r["rt"] for r in rs if r["correct"]]
        out[k] = dict(rt=statistics.mean(ok) if ok else math.nan, err=1 - len(ok) / len(rs), n=len(rs))
    return out


def fit(summary, data=DATA):
    """Best intercept shift, RMSD and correlation of model means against the data."""
    pairs = [(summary[(kind, pf, lf)]["rt"], data[kind][(pf, lf)])
             for kind in data for (pf, lf) in data[kind] if (kind, pf, lf) in summary]
    pairs = [(m, d) for m, d in pairs if not math.isnan(m)]
    shift = statistics.mean(d - m for m, d in pairs)
    rmsd = math.sqrt(statistics.mean((m + shift - d) ** 2 for m, d in pairs))
    r = statistics.correlation([m for m, _ in pairs], [d for _, d in pairs])
    return dict(shift=shift, rmsd=rmsd, r=r)


# ====================================================================== many runs
# ACT-R practice: a prediction is the average over many model runs (simulated
# participants), because a single run is noisy.  run_many() does that, in
# parallel over the available cores.  Representations are passed as a spec
# (a dict) and built inside each run, so that every run can get its own
# random vectors and the work can be sent to other processes.

def make_similar(emb, x, y, c):
    """Give symbol x a vector with cosine similarity c to symbol y."""
    import numpy as np
    u = emb.vector(y)
    w = emb._random(x + "\x00own")
    w = w - (w @ u) * u
    w /= np.linalg.norm(w)
    emb.set(x, c * u + math.sqrt(1 - c * c) * w)


def make_category(emb, members, s, name):
    """Give the members a shared component, so that any two have cosine similarity ~s.

    Each member becomes sqrt(s) * u + sqrt(1 - s) * w, with u a random vector for the
    category and w the member's own random vector, made orthogonal to u.
    """
    import numpy as np
    u = emb.vector("\x00category " + name)
    for x in members:
        w = emb._random(x + "\x00own")
        w = w - (w @ u) * u
        w /= np.linalg.norm(w)
        emb.set(x, math.sqrt(s) * u + math.sqrt(1 - s) * w)


def build_representation(spec, seed):
    """None -> symbolic.  A dict -> Embedded with its own vectors for this run.

    Keys: dim (512); categories {name: (members, s), ...}, applied first;
    similar [(x, y, c), ...]; and any Embedded option (compare, theta,
    retrieval, mp, spreading, beta, ...).
    """
    if spec is None:
        return None
    from pyprims.embedding import Embedded, Embedding
    spec = dict(spec)
    emb = Embedding(dim=spec.pop("dim", 512), seed=seed)
    for name, (members, s) in spec.pop("categories", {}).items():
        if s:
            make_category(emb, members, s, name)
    for x, y, c in spec.pop("similar", ()):
        if c:
            make_similar(emb, x, y, c)
    return Embedded(emb, **spec)


def _one_run(job):
    task_kw, rep_spec, seed, blocks, warmup, probe_list = job
    recs = run(make_task(**task_kw), build_representation(rep_spec, seed), blocks=blocks,
               warmup_blocks=warmup, seed=seed, probe_list=probe_list)
    return summarize(recs), summarize(recs, by_probe=True)


class Result:
    """Averages over runs.  cells / probes: key -> dict(rt, se, err, err_se, runs)."""

    def __init__(self, per_run):
        self.runs = [c for c, _ in per_run]            # per-run cell summaries
        self.probe_runs = [p for _, p in per_run]
        self.cells = self._average(self.runs)
        self.probes = self._average(self.probe_runs)

    @staticmethod
    def _average(summaries):
        out = {}
        for k in summaries[0]:
            rts = [s[k]["rt"] for s in summaries if k in s and not math.isnan(s[k]["rt"])]
            errs = [s[k]["err"] for s in summaries if k in s]
            n = len(rts)
            out[k] = dict(rt=statistics.mean(rts) if rts else math.nan,
                          se=statistics.stdev(rts) / math.sqrt(n) if n > 1 else math.nan,
                          err=statistics.mean(errs),
                          err_se=statistics.stdev(errs) / math.sqrt(len(errs)) if len(errs) > 1 else math.nan,
                          runs=n)
        return out

    def fit(self, data=None):
        return fit(self.cells, data or DATA)

    def run_rmsds(self, data=None):
        """RMSD of each single run (each with its own best intercept)."""
        return [fit(s, data or DATA)["rmsd"] for s in self.runs]


_POOL = None


def _pool(processes):
    global _POOL
    if _POOL is None or _POOL._max_workers != processes:
        from concurrent.futures import ProcessPoolExecutor
        if _POOL is not None:
            _POOL.shutdown()
        _POOL = ProcessPoolExecutor(max_workers=processes)
    return _POOL


def run_many(task_kw=None, rep=None, *, runs=50, blocks=12, warmup_blocks=4, probe_list=None,
             processes=None, seed0=0):
    """Run the experiment `runs` times (seeds seed0, seed0+1, ...) and average.

    rep is a representation spec (see build_representation); each run gets its
    own random vectors (Embedding seed = run seed).  processes=None uses every
    core; processes=1 runs serially in this process.
    """
    import os
    jobs = [(dict(task_kw or {}), rep, seed0 + i, blocks, warmup_blocks, probe_list) for i in range(runs)]
    processes = processes or os.cpu_count() or 1
    if processes == 1:
        per_run = [_one_run(j) for j in jobs]
    else:
        per_run = list(_pool(processes).map(_one_run, jobs, chunksize=max(1, runs // (4 * processes))))
    return Result(per_run)
