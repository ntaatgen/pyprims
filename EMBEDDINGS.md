# Embedded slot contents (pyprims.embedding)

Status, 6 October 2026: steps 1–3 of the plan are built.

1. Comparison is separated from storage.
2. Soft retrieval and vector blending work.
3. The symbolic limit is checked.

Similarity-based spreading has been added since, and a first holographic
representation of regular chunks (`Holographic`, see below). Graded
*stochastic* comparison (step 4) is not built yet.
`examples/fan_effect_embeddings.ipynb` explores the representation on the fan
effect; `tools/holographic_count.py` runs the holographic chunks on the count
model.

## The switch

The representation is a property of the model. **Symbolic is the default**,
and the symbolic code paths are unchanged:

- `model.compare` *is* the original `values_equal` function;
- retrieval takes the original code path;
- all 36 tutorial models give bit-identical runs before and after this
  change (trace, events, chunk references, production utilities).

```python
from pyprims import Model
from pyprims.embedding import Embedded, Embedding

m = Model(seed=1)                                   # symbolic: the original PRIMs

emb = Embedding(dim=512, seed=0)                    # random, near-orthogonal vectors
emb.set_numbers(range(0, 101), width=3.0)           # optional magnitude code for numbers
emb.update({"lion": vec_lion, "tiger": vec_tiger})  # optional: any vectors, e.g. pretrained
m = Model(seed=1, representation=Embedded(emb, compare="threshold", theta=0.6,
                                          retrieval="soft", mp=5.0, blending="vector"))
m.set_representation(None)                          # back to symbolic
```

## What stays symbolic, and what changes

In both modes these stay symbolic:

- the PRIMs, meaning which slot is compared or copied to which;
- operators and compiled productions;
- chunk identity: names, duplicate detection when chunks enter memory,
  bindings;
- traces, which show the symbols, not the vectors.

The vectors are an overlay that says how similar two values are. They act
in three places.

| place | symbolic | embedded |
|---|---|---|
| comparison PRIMs `=`, `<>` and retrieval cues | identical values match | `compare="threshold"`: identical, or similarity ≥ θ |
| retrieval | every cue slot must match | `retrieval="soft"`: partial matching, A_i = base + spreading + noise + mp · Σ_k (sim(cue_k, value_ik) − 1); `isa` stays an exact filter (`hard_slots`) |
| blending | numbers averaged (Swift) | `blending="vector"`: each non-cue slot is the clean-up of Σ_i P_i e(value_i), P = softmax(A/t), t = √2·ans; clean-up over the candidates' values or over the whole vocabulary |
| spreading (associations) | S_ji = mas − ln(fan_j) | `spreading="similarity"`: S_ji = max(0, mas + ln P(i \| j)), P(i \| j) = softmax over memory of β · c_ij, where c_ij is the largest similarity of cue j to a slot value of chunk i; β = ∞ counts the chunks that contain j |

**Vectors.** Random vectors come from a hash of the symbol and the seed:

- they do not depend on the order in which symbols are first seen;
- creating them never touches the model's random stream;
- in 512 dimensions, two random vectors have similarity of about ±0.04.

`set_numbers` gives a Gaussian-tuning (random Fourier feature) code, so that
sim(n, m) ≈ exp(−(n − m)² / 4w²). Each integer is registered both as text
("6", as typed in a model) and as a number ("6.0", as stored in a slot).

**Inspection.**

- `m.dm.last_mismatch`: the partial-matching penalty of each chunk in the
  last soft retrieval.
- `m.dm.last_blend`: per slot, the clean-up result, its similarity and the
  weights.
- The notebook's conflict trace lists the penalty in each activation
  breakdown, and the breakdowns still add up to the activations used.

## The symbolic limit (`tools/symbolic_limit.py`)

**Exact.** Each model is run with the same seed in four representations:

- symbolic;
- threshold comparison (θ = 0.5);
- soft retrieval with mp = ∞;
- both together.

All 36 tutorial models give identical fingerprints in all four. The 671
distinct symbols they use have a largest pairwise similarity of 0.215, well
below θ.

The check is not vacuous: in the semantic model the embedded run made 96
soft retrievals and 103 similarity comparisons, and setting θ = 0 or a finite
mp is detected as different.

**Why the limit is mp = ∞ exactly, and finite mp only statistically.** A
soft retrieval evaluates the activation of every chunk that passes `isa`,
not only of exact matches. Activation noise is drawn when an activation is
computed, so the extra draws shift the random stream. Behaviour is then the
same in distribution, not run for run. mp = ∞ excludes non-matching chunks
before their activation is computed, which is what makes it exact.

**Departure from the limit**: the count model, 20 runs × 60 trials, with soft
retrieval and symbolic comparison.

| mp | random vectors: correct sequences | magnitude code (w = 1): correct sequences |
|---:|---:|---:|
| ∞ | 100% | 100% |
| 20 | 100% | 100% |
| 5 (ACT-R default) | 100% | 92.4% |
| 3 | 100% | 66.5% |
| 2 | 99.4% | 42.2% |
| 1 | 77.4% | 21.3% |

With random vectors, unrelated symbols are almost orthogonal, so any
reasonable penalty excludes the wrong facts. With a magnitude code,
"three" has similarity 0.75 to "two" and "four", and retrieval of the
neighbouring count fact starts at the default mp.

## Two findings that matter for step 4

1. **Soft retrieval alone does not produce human-like counting errors.**
   - When the neighbouring fact is retrieved, the exact comparison
     `RT2 = WM1` in the next operator catches it, no operator matches, and
     the model stalls (all 7.6% of errors at mp = 5).
   - With graded comparison (θ = 0.8) the same model skips and stops early
     instead, e.g. "one two three" → "one three" (11.9% of trials).
   - So partial matching in retrieval and graded comparison in PRIMs have to
     be designed together.
2. **Vector blending interpolates only when the code is broad** relative to
   the values blended.
   - With w = 4, 40 and 60 are nearly orthogonal. Their mean vector is closer
     to each of them than to 50, so the blend snaps to a stored value.
   - With w = 15 it cleans up near 50.
   - Near-miss answers therefore require a broad code (or a scalar blend, as
     ACT-R uses for numbers). How broad becomes an empirical question that
     error-content data can answer.

## Similarity-based spreading and the fan effect

ACT-R's S_ji = mas − ln(fan_j) is the case S_ji = mas + ln P(i | j) in which cue
j predicts each of the fan_j facts that contain it equally well. With
embeddings, P(i | j) comes from similarity, with one sharpness parameter β.

- **β = ∞** counts the chunks that contain j and reproduces mas − ln(fan).
  - The fan model gives run-for-run identical records.
  - So do 35 of the 36 tutorial models (`tools/symbolic_limit.py --spreading`).
  - The exception, `count-rev2`, is PRIMs' own fan counter: it misses a
    reference made before the referenced chunk was defined (PORTING.md, kept
    quirk 3). The similarity count is one higher there.
- **Finite β** lets every chunk in memory take a share. For random vectors
  S ≈ mas − ln(fan + N e^(−β)), where N is the number of chunks in memory.

Results of the fan notebook (each prediction is the mean of 20 model runs, each
run with its own random vectors):

- **Baseline.** The symbolic model fits the 18 means of Anderson (1974,
  Table 5; Expts 1, 2 and the no-precue condition of Expt 3) with RMSD 43 ms,
  r = 0.90 (mas 1.5, lf 0.6, one free intercept); single runs range from 38 to
  52 ms. It reproduces the over-additive interaction of the two fans (model 48,
  116, 168 ms against 71, 101, 186 ms) but makes almost none of the 4–5% errors.
- **Partial matching** with the probe in the retrieval request gives **no fan
  effect** for targets. For foils it is flat, or even reversed at small mp
  (−174 ± 36 ms from fan 1 to 3 at mp = 2), because more partial matches make
  the best of them come sooner. The fan effect needs the competition that lives
  in the associations.
- **Similarity spreading** is indistinguishable from symbolic at β = 8, slightly
  worse at β = 6 (+2.8 ± 0.9 ms RMSD per run), and clearly worse below that,
  where the fan effect compresses: S ≈ mas − ln(fan + N e^(−β)).
- **Similar cues** slow verification: at β = 6, a fan-1 person made similar
  (c = 0.9) to a fan-3 person is 143 ± 7 ms slower; at β = ∞ nothing changes.
  This is a similarity-based fan effect.
- **Related facts** about one person (similar locations) give *more*
  interference: false rejections rise from 1% to 64%. That is the opposite of
  the integration effect (Smith, Adams & Schorr, 1978; Reder & Anderson, 1980),
  so integration needs a further mechanism.
- **Memory size.** At β = 6, 100 unrelated facts add 113 ± 2 ms and raise
  target errors from 0.1% to 6.9%. The symbolic model predicts no effect.
- **Categories.** Giving all persons, or all locations, a shared vector
  component (`categories` in `fan.build_representation`; pairwise similarity s)
  adds K = n_other e^(−β(1−s)) to every fan of that category. The shares to
  other facts stay below e^(−mas) and are clipped, so K only compresses the
  fan effect.
  - Similarity within both categories never improves the fit.
  - Similarity among persons only, at a small K (β = 8, s = 0.3 or β = 12,
    s = 0.5), lowers the RMSD from 43 to 38 ms (−5.0 ± 0.8 ms per run,
    paired). It does so through the foils: the location fan now matters
    for foils with a fan-1 person, as in the data.
  - The gain reflects an asymmetry in the data (for foils, location fan
    172 ms against person fan 104 ms) that a person-specific symbolic
    parameter could capture as well.

## Holographic chunks (`Holographic`)

In `Embedded`, constants are vectors but chunks are still symbolic records.
`Holographic` takes the next step for **regular chunks**; operators (and any
type listed in `symbolic_types`) stay symbolic. A regular chunk is a list of
pointers (slot1, slot2, ...), so the only roles are positions, and binding
needs no circular convolution: each position k has a fixed random
permutation P_k, which is exactly invertible.

```python
from pyprims.embedding import Holographic
m = Model(seed=1, representation=Holographic(Embedding(512), compare="symbolic", mp=5.0))
```

Every regular DM chunk has two vectors (`isa` stays a symbolic filter):

| vector | definition | used for |
|---|---|---|
| order | o = Σ_k P_k e(v_k) | retrieval matching, read-back into the buffer, blending |
| bag | b = Σ_k e(v_k) | similarity-based spreading: chunk i contains source j to the degree ⟨e(j), b_i⟩, whatever the position |

- **Retrieval.** For each cue slot k, d_k = ⟨P_k e(cue_k), o⟩ = sim(cue_k, v_k)
  + crosstalk, and the activation gets mp · Σ_k (d_k − 1). With mp = ∞ a
  chunk passes if every d_k ≥ `decode_theta` (0.5) and then gets no penalty.
- **Read-back.** The retrieved chunk is decoded: each slot is the clean-up of
  P_k⁻¹ o against the values in DM. If every slot decodes to its stored value
  the buffer holds the DM chunk itself, otherwise a copy with the decoded
  values (counted in `decode_errors` / `decodes`; `dm.last_decode` has the
  details).
- **Crosstalk.** With n slots, d_k carries noise of standard deviation
  √((n − 1)/dim), 0.06 for a 3-slot fact in 512 dimensions. It is fixed for a
  given set of vectors, so it acts like an item-specific bias, not trial noise.
- Duplicate detection when chunks enter memory stays symbolic, and a slot
  value that is a chunk is represented by its name vector (a pointer), not
  yet by its own content.

**The symbolic limit holds exactly.** With random vectors and mp = ∞,
holographic chunks give runs identical to the symbolic model on all 36
tutorial models; with similarity spreading at β = ∞ they are identical to the
`Embedded` similarity spreading (`tools/symbolic_limit.py`). Symbolic runs are
unchanged.

**The count model** (`tools/holographic_count.py`; 20 runs × 60 trials per
condition, each run with its own vectors; comparisons symbolic, so a wrong
fact makes the model stall):

1. **The mismatch penalty is bounded by the dimension.** With random vectors
   at d = 512 the holographic model counts as well as the hybrid for mp
   between 2 and 5, but at mp = 20 it stalls on 11% of trials, where the hybrid
   is perfect. The penalty multiplies the crosstalk on the *correct* fact
   (mp · Σε, sd ≈ 1.8 at mp = 20), and a fact with unlucky crosstalk falls
   below the retrieval threshold for that whole simulated participant. A
   perfect match cannot be represented exactly, so mp cannot be arbitrarily
   large: roughly mp · √((n − 1)/dim) must stay well below 1. At d = 2048,
   mp = 20 is perfect again; at d = 64 it fails on half the trials.
2. **Identity is not available in vectors.** With the magnitude code for
   number words (sim(three, four) = 0.75), mp = ∞ means "all cues above
   θ = 0.5", so neighbouring count facts pass as matches and the model stalls
   on 83% of trials (the hybrid, which tests identity, is perfect). For
   finite mp the two behave alike (mp = 5: 89% against 92% correct).
3. **Chunk width costs capacity.** Adding filler slots to the count facts
   leaves the hybrid at 100%, but the holographic model degrades as
   (n − 1)/dim grows. At d = 128 and mp = ∞: 3 slots 100% correct, 5 slots
   91%, 8 slots 66% with 28% of retrievals misread, 13 slots 27% (mp = 5 is
   similar). At d = 512 up to 8 slots are
   perfect and 13 slots give the first misreadings (1%). This is a capacity
   limit on chunk size that the symbolic model would have to impose by hand.
4. As in the hybrid, all errors are stalls, because `RT2 = WM1` compares
   exactly; graded comparison would turn them into skips.

## Files

- `pyprims/embedding.py`: `Embedding`, `Symbolic`, `Embedded`, `Holographic`.
- Changed:
  - `model.py`: the `representation` argument and `set_representation`;
  - `prim.py`: `=` and `<>` call `model.compare`;
  - `declarative.py`: cue matching, `_soft_retrieve`, `_vector_blend`;
  - `notebook/views.py`: the penalty line in the breakdown.
- `tests/test_embedding.py`: 22 tests.
- `tools/holographic_count.py`: the count-model experiments with holographic
  chunks.
- `tools/symbolic_limit.py`: `--sweep` adds the table above; `--spreading`
  adds similarity spreading at β = ∞.
- `examples/fan.py`: the fan task (materials, data, operators, run/score
  helpers, and `run_many` for averaging runs in parallel).
- `examples/fan_effect_embeddings.ipynb`: the exploration; 20 runs per setting,
  about 12 minutes on 2 cores.
- Similarity spreading also changed `chunk.py` (the hook in `sji`),
  `declarative.py` (a version counter for the association cache), and
  `script_api.py` / `modules.py`: `set-goal`, `instantiate-skill` and
  `build-goal` rewrite DM chunks in place, so they bump that counter. After
  editing DM chunks by hand, call `model.representation.invalidate()`.
