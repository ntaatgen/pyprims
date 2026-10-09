"""Tests for the embedded representation (pyprims.embedding)."""
import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")

from pyprims import Model, Task
from pyprims.chunk import Chunk
from pyprims.embedding import Embedded, Embedding, Symbolic
from pyprims.values import values_equal

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


# ------------------------------------------------------------------ the switch
def test_symbolic_is_the_default_and_uses_exact_equality():
    m = Model(seed=0)
    assert isinstance(m.representation, Symbolic)
    assert m.compare is values_equal
    assert not m.representation.soft_retrieval and not m.representation.vector_blending


def test_switching_back_to_symbolic():
    m = Model(seed=0, representation=Embedded(dim=64))
    assert m.compare is not values_equal
    m.set_representation(None)
    assert m.compare is values_equal


# ------------------------------------------------------------------ vectors
def test_random_vectors_are_order_independent_and_near_orthogonal():
    a, b = Embedding(dim=512, seed=3), Embedding(dim=512, seed=3)
    va = a.vector("lion")
    b.vector("tulip")
    assert np.allclose(va, b.vector("lion"))
    assert a.sim("lion", "lion") == 1.0
    assert abs(a.sim("lion", "tulip")) < 0.2
    assert Embedding(dim=512, seed=4).sim("lion", "tulip") != a.sim("lion", "tulip")


def test_magnitude_code_is_graded_and_aliases_text_and_numbers():
    e = Embedding(dim=1024, seed=0)
    e.set_numbers(range(0, 21), width=2.0)
    s = [e.sim("10", str(k)) for k in (10, 11, 12, 14, 18)]
    assert s == sorted(s, reverse=True) and s[0] == 1.0 and s[-1] < 0.2
    assert e.sim("10", 10.0) == pytest.approx(1.0)       # "10" in a model, 10.0 in a slot


# ------------------------------------------------------------------ the symbolic limit
@pytest.mark.parametrize("name", ["count.prims", "semantic.prims", "stroop.prims"])
def test_symbolic_limit_is_exact(name):
    import symbolic_limit as sl
    reps = sl.representations(Embedding(dim=512, seed=0))
    prints = {k: sl.fingerprint(os.path.join(EX, name), r)[0] for k, r in reps.items()}
    assert len(set(prints.values())) == 1, prints


# ------------------------------------------------------------------ soft retrieval and blending
def _facts_model(rep, facts, ans=0.0):
    task = Task("facts", initial_skills=["facts"], default_activation=1.0, ans=ans, rt=-2.0)
    task.facts(facts)

    @task.script
    def script(m):
        pass
    m = Model(seed=0, trace=False, batch_mode=True, representation=rep)
    m.load(task)
    return m


def _request(m, **slots):
    r = Chunk("request", m)
    r.set_slot("isa", "fact")
    for k, v in slots.items():
        r.set_slot(k, v)
    return r


def test_soft_retrieval_finds_the_most_similar_fact():
    emb = Embedding(dim=256, seed=1)
    base = emb.vector("lion")
    emb.set("tiger", base + 0.3 * emb.vector("tiger-own"))     # similar to lion
    facts = [("f1", "isa-mammal", "lion", "mammal"), ("f2", "isa-bird", "robin", "bird")]
    sym = _facts_model(None, facts)
    assert sym.dm.retrieve(_request(sym, slot2="tiger"))[1] is None          # no exact match
    soft = _facts_model(Embedded(emb, compare="symbolic", retrieval="soft", mp=1.0), facts)
    _, got = soft.dm.retrieve(_request(soft, slot2="tiger"))
    assert got is not None and got.name == "f1"
    assert soft.dm.last_mismatch["f1"] > soft.dm.last_mismatch["f2"]         # smaller penalty
    inf = _facts_model(Embedded(emb, compare="symbolic", retrieval="soft", mp=math.inf), facts)
    assert inf.dm.retrieve(_request(inf, slot2="tiger"))[1] is None          # symbolic limit


def _blend(width, cleanup):
    emb = Embedding(dim=2048, seed=0)
    emb.set_numbers(range(0, 101), width=width)
    facts = [("f1", "times-ten", "4", "40"), ("f2", "times-ten", "6", "60")]
    rep = Embedded(emb, compare="symbolic", retrieval="soft", mp=1.0, blending="vector",
                   cleanup=cleanup, temperature=1.0)
    m = _facts_model(rep, facts)
    # the cue 5 matches 4 and 6 equally well, so 40 and 60 get equal weights
    _, got = m.dm.blended_retrieve(_request(m, slot1="times-ten", slot2="5"))
    w = m.dm.last_blend["slot3"]["weights"]
    assert w["f1"] == pytest.approx(w["f2"], abs=0.05)
    return got.slotvals["slot3"]


def test_vector_blending_broad_code_gives_an_intermediate_answer():
    # a code broad relative to the distance 40-60: the mean vector is closest to ~50
    assert abs(_blend(width=15.0, cleanup="vocabulary") - 50.0) <= 4.0


def test_vector_blending_narrow_code_snaps_to_a_stored_value():
    # a narrow code makes 40 and 60 nearly orthogonal: their mean is closer to
    # each of them than to 50, so clean-up returns a stored value
    assert _blend(width=4.0, cleanup="vocabulary") in (40.0, 60.0)


def test_vector_blending_candidate_cleanup_returns_a_stored_value():
    assert _blend(width=15.0, cleanup="candidates") in (40.0, 60.0)


def test_threshold_comparison_in_prims():
    emb = Embedding(dim=256, seed=2)
    emb.set("crimson", emb.vector("red") + 0.2 * emb.vector("crimson-own"))
    m = Model(seed=0, representation=Embedded(emb, compare="threshold", theta=0.8, retrieval="symbolic"))
    assert m.compare("red", "crimson") and not m.compare("red", "blue")


# ------------------------------------------------------------------ similarity-based spreading
def _fan_model(beta, seed=0):
    sys.path.insert(0, EX)
    import fan
    rep = Embedded(Embedding(512, seed), compare="symbolic", retrieval="symbolic",
                   spreading="similarity", beta=beta)
    m = Model(seed=0, trace=False, batch_mode=True, representation=rep)
    m.load(fan.make_task(mas=2.0))
    return m, rep


def _person_fact(m, person):
    j = m.dm.chunks[person]
    return j, next(c for c in m.dm.chunks.values() if c.type == "fact" and c.slotvals.get("slot2") is j)


def test_similarity_spreading_beta_inf_is_mas_minus_ln_fan():
    m, rep = _fan_model(math.inf)
    for person, f in (("hippie", 1), ("giant", 2), ("sailor", 3)):
        j, i = _person_fact(m, person)
        assert rep.sji(j, i) == pytest.approx(2.0 - math.log(f), abs=1e-12)
        assert rep.sji(j, i) == pytest.approx(j.fan and max(0, 2.0 - math.log(j.fan)), abs=1e-12)


def test_similarity_spreading_finite_beta_follows_closed_form():
    m, rep = _fan_model(6.0)
    n = len(m.dm.chunks) - 1
    for person, f in (("hippie", 1), ("sailor", 3)):
        j, i = _person_fact(m, person)
        closed = 2.0 - math.log(f + (n - f) * math.exp(-6.0))
        assert rep.sji(j, i) == pytest.approx(closed, abs=0.05)


def test_similarity_spreading_limit_is_exact_on_the_fan_task():
    sys.path.insert(0, EX)
    import fan
    rep = Embedded(Embedding(512, 0), compare="threshold", theta=0.5, retrieval="soft", mp=math.inf,
                   spreading="similarity", beta=math.inf)
    assert fan.run(blocks=4, seed=3) == fan.run(blocks=4, seed=3, representation=rep)


def test_similar_cue_shares_the_association():
    m, rep = _fan_model(6.0)
    j, i = _person_fact(m, "hippie")
    before = rep.sji(j, i)
    emb = rep.embedding
    u = emb.vector("teacher")
    w = emb._random("hippie-own")
    w = w - (w @ u) * u
    emb.set("hippie", 0.9 * u + math.sqrt(1 - 0.81) * w / np.linalg.norm(w))
    rep._assoc_cache.clear()
    assert rep.sji(j, i) < before - 0.3


def test_association_cache_follows_in_place_edits():
    """instantiate-skill / set-goal rewrite DM chunks in place; associations must follow."""
    task = Task("edit", initial_skills=["edit"], default_activation=1.0)
    task.facts([("g1", "goalish", "alpha"), ("other", "thing", "beta")])

    @task.script
    def script(m):
        pass
    rep = Embedded(Embedding(128, 0), compare="symbolic", retrieval="symbolic",
                   spreading="similarity", beta=math.inf)
    m = Model(seed=0, trace=False, batch_mode=True, representation=rep)
    m.load(task)
    from pyprims.script_api import ScriptAPI
    api = ScriptAPI(m)
    m.dm.chunks["g1"].set_slot("isa", "goaltype")
    alpha, beta = m.dm.chunks["alpha"], m.dm.chunks["beta"]
    assert rep.sji(alpha, m.dm.chunks["g1"]) > 0
    api.set_goal("g1", ["slot1", "beta"])            # g1 no longer contains alpha
    assert rep.sji(alpha, m.dm.chunks["g1"]) == 0.0
    assert rep.sji(beta, m.dm.chunks["g1"]) > 0


def test_run_many_parallel_equals_serial():
    sys.path.insert(0, EX)
    import fan
    spec = dict(compare="symbolic", retrieval="symbolic", spreading="similarity", beta=6.0)
    a = fan.run_many(dict(mas=1.5, lf=0.6), spec, runs=3, blocks=4, warmup_blocks=1, processes=1)
    b = fan.run_many(dict(mas=1.5, lf=0.6), spec, runs=3, blocks=4, warmup_blocks=1, processes=2)
    assert a.cells == b.cells and len(a.runs) == 3
    assert all(v["runs"] == 3 for v in a.cells.values() if not math.isnan(v["rt"]))


def test_category_similarity():
    sys.path.insert(0, EX)
    import fan
    emb = Embedding(512, 0)
    fan.make_category(emb, fan.PERSONS, 0.5, "person")
    within = [emb.sim(a, b) for a in fan.PERSONS for b in fan.PERSONS if a < b]
    assert abs(sum(within) / len(within) - 0.5) < 0.03
    assert abs(emb.sim("hippie", "park")) < 0.2


# ------------------------------------------------------------------ holographic chunks
def _holo_count(mp, dim=512, seed=0, **kw):
    from pyprims.embedding import Holographic
    rep = Holographic(Embedding(dim, seed), compare="symbolic", mp=mp, **kw)
    m = Model(seed=0, trace=False, batch_mode=True, representation=rep)
    m.load(os.path.join(EX, "count.prims"))
    return m, rep


def test_holographic_order_vector_reads_back_and_keeps_order():
    m, rep = _holo_count(5.0)
    cf2 = m.dm.chunks["cf2"]                              # count-fact two three
    dec = rep.decode(cf2)
    assert [dec[s][0].name for s in ("slot1", "slot2", "slot3")] == ["count-fact", "two", "three"]
    assert rep.decoded_chunk(cf2)[0] is cf2               # exact read-back: the DM chunk itself
    # "two" in slot 2 (cf2) and in slot 3 (cf1): only the right position matches
    req = _request(m, slot1=m.dm.chunks["count-fact"], slot2=m.dm.chunks["two"])
    req.slotvals.pop("isa")
    req.set_slot("isa", "fact")
    assert rep.cue_penalty(cf2, req, 5.0) > -1.0
    assert rep.cue_penalty(m.dm.chunks["cf1"], req, 5.0) < -4.0
    assert rep.cue_penalty(m.dm.chunks["cf1"], req, math.inf) is None


def test_holographic_bag_containment_ignores_position():
    m, rep = _holo_count(5.0)
    two = m.dm.chunks["two"]
    for name in ("cf1", "cf2"):                           # "two" in slot 3 and in slot 2
        assert rep.containment(m.dm.chunks[name], two) == pytest.approx(1.0, abs=0.25)
    assert abs(rep.containment(m.dm.chunks["cf5"], two)) < 0.25
    assert rep.contains(m.dm.chunks["cf1"], two) and not rep.contains(m.dm.chunks["cf5"], two)


def test_holographic_crosstalk_grows_with_width_and_shrinks_with_dim():
    def spread(dim, n):
        from pyprims.embedding import Holographic
        rep = Holographic(Embedding(dim, 1), mp=5.0)
        rep.model = None
        errs = []
        for k in range(60):
            vals = [f"v{k}_{i}" for i in range(n)]
            o = sum(rep.bind(f"slot{i + 1}", rep.embedding.vector(v)) for i, v in enumerate(vals))
            errs.append(float(rep.bind("slot1", rep.embedding.vector(vals[0])) @ o) - 1.0)
        return float(np.std(errs))
    assert spread(512, 13) > 2 * spread(512, 3)
    assert spread(2048, 3) < spread(128, 3) / 2
