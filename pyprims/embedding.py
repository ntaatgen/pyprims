"""Symbolic or embedded representation of slot contents.

PRIMs' control structure stays symbolic in both modes: which slot is compared
or copied to which, the operators, and the compiled productions.  What
changes is how two slot *values* are compared:

* :class:`Symbolic` (the default): values match when they are identical.
  This is the original PRIMs, and with it the model takes exactly the same
  code paths as before this module existed.
* :class:`Embedded`: every value has a vector, and similarity between vectors
  drives comparison PRIMs (``=``, ``<>``), retrieval (partial matching) and
  blending.  Chunks keep their symbolic identity (names, duplicate detection,
  traces), so an embedded model is still readable; the vectors are an
  overlay that says how similar two symbols are.

Usage::

    from pyprims import Model
    from pyprims.embedding import Embedded, Embedding

    emb = Embedding(dim=512, seed=0)              # random vectors: near-orthogonal
    emb.set_numbers(range(0, 101), width=3.0)     # optional: a magnitude code for numbers
    m = Model(seed=1, representation=Embedded(emb, compare="threshold", theta=0.6,
                                              retrieval="soft", mp=5.0))

The symbolic model is the limit of the embedded one: with random
high-dimensional vectors, ``compare="threshold"`` and ``mp=inf``, an embedded
model gives exactly the same runs as the symbolic one
(``tools/symbolic_limit.py`` checks this on every tutorial model).
"""
from __future__ import annotations

import hashlib
import math

try:                                   # numpy is needed only for the embedded representation
    import numpy as np
except ImportError:                    # pragma: no cover
    np = None

from .values import describe, values_equal


# ====================================================================== vectors
class Embedding:
    """A map from symbols (slot-value descriptions) to unit vectors.

    Vectors not set explicitly are random, derived from a hash of the symbol
    and the seed, so they do not depend on the order in which symbols are
    first seen and do not touch the model's random stream.  In ``dim``
    dimensions two random vectors have similarity ~ N(0, 1/sqrt(dim)).
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        if np is None:
            raise ImportError("the embedded representation needs numpy: pip install numpy")
        self.dim = int(dim)
        self.seed = int(seed)
        self.vectors: dict[str, np.ndarray] = {}
        self._fixed: set[str] = set()       # symbols whose vectors were set explicitly
        self._sim_cache: dict[tuple[str, str], float] = {}

    # ---------------------------------------------------------------- lookup
    @staticmethod
    def key(value) -> str:
        return value if isinstance(value, str) else describe(value)

    def _random(self, key: str) -> np.ndarray:
        h = hashlib.sha256(f"{self.seed}\x00{key}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def vector(self, value) -> np.ndarray:
        k = self.key(value)
        v = self.vectors.get(k)
        if v is None:
            v = self._random(k)
            self.vectors[k] = v
        return v

    def sim(self, a, b) -> float:
        """Cosine similarity; identical symbols have similarity 1."""
        ka, kb = self.key(a), self.key(b)
        if ka == kb:
            return 1.0
        pair = (ka, kb) if ka < kb else (kb, ka)
        s = self._sim_cache.get(pair)
        if s is None:
            s = float(self.vector(ka) @ self.vector(kb))
            self._sim_cache[pair] = s
        return s

    # ---------------------------------------------------------------- setting vectors
    def set(self, symbol, vector):
        """Give a symbol an explicit vector (normalised to unit length)."""
        v = np.asarray(vector, dtype=float)
        if v.shape != (self.dim,):
            raise ValueError(f"vector for {symbol!r} has shape {v.shape}, expected ({self.dim},)")
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError(f"zero vector for {symbol!r}")
        k = self.key(symbol)
        self.vectors[k] = v / n
        self._fixed.add(k)
        self._sim_cache.clear()

    def update(self, mapping: dict):
        """Set many vectors, e.g. pretrained word or sentence embeddings."""
        for s, v in mapping.items():
            self.set(s, v)

    def set_numbers(self, values, width: float = 1.0, mix: float = 1.0, aliases: bool = True):
        """A magnitude code for numbers: similar numbers get similar vectors.

        Each number n gets a vector whose similarity to m is
        ``mix * exp(-(n - m)^2 / (4 width^2)) + (1 - mix) * noise``: a Gaussian
        tuning curve (random Fourier features of an RBF kernel), optionally
        mixed with an independent random part.  With ``aliases`` the integer
        is registered both as text ("6", as it is typed in a model) and as a
        number ("6.0", as it is stored in a slot).
        """
        rng = np.random.default_rng(self.seed + 7919)
        freqs = rng.standard_normal(self.dim) / (width * math.sqrt(2))
        phases = rng.uniform(0, 2 * math.pi, self.dim)
        for n in values:
            x = float(n)
            rbf = np.sqrt(2.0 / self.dim) * np.cos(freqs * x + phases)
            keys = {describe(x)}
            if aliases and x == int(x):
                keys.add(str(int(x)))
            own = self._random(describe(x) + "\x00num")     # shared by the aliases
            for k in keys:
                v = math.sqrt(mix) * rbf / max(np.linalg.norm(rbf), 1e-12) + math.sqrt(1 - mix) * own
                self.set(k, v)

    # ---------------------------------------------------------------- clean-up
    def cleanup(self, vector: np.ndarray, candidates):
        """The candidate whose vector is most similar to ``vector``."""
        best, best_s = None, -math.inf
        for c in candidates:
            s = float(self.vector(c) @ vector)
            if s > best_s:
                best, best_s = c, s
        return best, best_s

    def vocabulary(self):
        return list(self.vectors.keys())

    def max_offdiagonal_similarity(self, symbols) -> float:
        """Largest similarity between distinct symbols (for the symbolic-limit check)."""
        keys = sorted({self.key(s) for s in symbols})
        if len(keys) < 2:
            return -1.0
        M = np.stack([self.vector(k) for k in keys])
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        return float(S.max())


# ====================================================================== representations
class Symbolic:
    """The original PRIMs: values match only when identical."""

    name = "symbolic"
    holographic = False
    soft_retrieval = False
    vector_blending = False
    similarity_spreading = False
    embedding = None

    def attach(self, model):
        model.compare = values_equal          # the exact function used before this module

    def describe(self) -> str:
        return "symbolic"


class Embedded:
    """Embedded slot contents.

    Parameters
    ----------
    embedding : Embedding, optional
        The vectors (default: random, ``dim`` dimensions).
    compare : "symbolic" | "threshold"
        How comparison PRIMs (``=``, ``<>``) and retrieval cues decide a match.
        ``threshold``: two values match if their similarity is at least ``theta``.
    theta : float
        Similarity threshold for ``compare="threshold"``.
    retrieval : "symbolic" | "soft"
        ``soft``: retrieval is partial matching on similarity: a chunk's
        activation gets ``mp * sum_k (sim(cue_k, value_k) - 1)`` over the cue
        slots, so a non-identical chunk can be retrieved.  ``mp=inf`` excludes
        every chunk that is not identical on all cue slots (the symbolic limit).
    mp : float or None
        Mismatch penalty; None uses the model parameter ``mp:`` (default 5).
    hard_slots : tuple of slot names
        Cue slots that must match exactly in soft retrieval (default: ``isa``,
        so a fact request never retrieves an operator).
    blending : "symbolic" | "vector"
        ``vector``: a retrieval returns, for every non-cue slot, the clean-up of
        the activation-weighted average of the candidates' vectors.
    cleanup : "candidates" | "vocabulary"
        Clean-up targets: the values present in the blended chunks, or every
        symbol the embedding knows (which allows answers no chunk holds).
    temperature : float or None
        Blending temperature; None uses sqrt(2) * ans, as in ACT-R.
    spreading : "symbolic" | "similarity"
        ``similarity``: the structural association S_ji = mas - ln(fan_j) is
        replaced by S_ji = max(0, mas + ln P(i | j)), where P(i | j) is a
        softmax over all DM chunks i of beta * c_ij, and c_ij is how strongly
        chunk i contains source j: the largest similarity between j and one
        of i's slot values.  For beta = inf, P(i | j) = 1 / n_j over the n_j
        chunks that contain j itself, which is ACT-R's mas - ln(fan).  For
        finite beta every chunk in memory takes a share, and similar sources
        share associations.  Declared or learned Sji's still take precedence.
    beta : float
        Sharpness of the association softmax (default inf).
    """

    name = "embedded"
    holographic = False

    def __init__(self, embedding: Embedding | None = None, *, dim: int = 512, seed: int = 0,
                 compare: str = "threshold", theta: float = 0.5, retrieval: str = "soft",
                 mp: float | None = None, hard_slots=("isa",), blending: str = "symbolic",
                 cleanup: str = "candidates", temperature: float | None = None,
                 spreading: str = "symbolic", beta: float = math.inf):
        if compare not in ("symbolic", "threshold"):
            raise ValueError("compare must be 'symbolic' or 'threshold'")
        if retrieval not in ("symbolic", "soft"):
            raise ValueError("retrieval must be 'symbolic' or 'soft'")
        if blending not in ("symbolic", "vector"):
            raise ValueError("blending must be 'symbolic' or 'vector'")
        if cleanup not in ("candidates", "vocabulary"):
            raise ValueError("cleanup must be 'candidates' or 'vocabulary'")
        if spreading not in ("symbolic", "similarity"):
            raise ValueError("spreading must be 'symbolic' or 'similarity'")
        self.embedding = embedding if embedding is not None else Embedding(dim=dim, seed=seed)
        self.compare_mode = compare
        self.theta = float(theta)
        self.retrieval = retrieval
        self.mp = mp
        self.hard_slots = tuple(hard_slots)
        self.blending = blending
        self.cleanup = cleanup
        self.temperature = temperature
        self.soft_retrieval = retrieval == "soft"
        self.vector_blending = blending == "vector"
        self.spreading = spreading
        self.similarity_spreading = spreading == "similarity"
        self.beta = float(beta)
        self.model = None
        self._assoc_cache: dict = {}

    def attach(self, model):
        self.model = model
        self._assoc_cache = {}
        model.compare = values_equal if self.compare_mode == "symbolic" else self.match

    def describe(self) -> str:
        return (f"embedded(dim={self.embedding.dim}, compare={self.compare_mode}"
                + (f", theta={self.theta}" if self.compare_mode == "threshold" else "")
                + f", retrieval={self.retrieval}"
                + (f", mp={self.mp}" if self.soft_retrieval else "")
                + f", blending={self.blending}"
                + (f", spreading=similarity, beta={self.beta:g}" if self.similarity_spreading else "")
                + ")")

    # ---------------------------------------------------------------- comparison
    def sim(self, a, b) -> float:
        return self.embedding.sim(a, b)

    def match(self, a, b) -> bool:
        """Graded equality for comparison PRIMs and retrieval cues."""
        if values_equal(a, b):
            return True
        return self.embedding.sim(a, b) >= self.theta

    def penalty(self, model) -> float:
        return model.dm.mismatch_penalty if self.mp is None else float(self.mp)

    # ---------------------------------------------------------------- spreading
    def containment(self, i, j) -> float:
        """How strongly chunk i contains source j: max similarity of j to i's slot values."""
        best = -math.inf
        for slot, v in i.slotvals.items():
            if slot == "isa":
                continue
            s = 1.0 if values_equal(v, j) else self.embedding.sim(v, j)
            if s > best:
                best = s
        return best

    def contains(self, i, j) -> bool:
        """Whether chunk i contains source j (the beta = inf case): an identical slot value."""
        return any(values_equal(v, j) for s, v in i.slotvals.items() if s != "isa")

    def association_distribution(self, j) -> dict:
        """P(i | j) over the DM chunks i other than j itself (cached per DM version)."""
        dm = self.model.dm
        key = (id(dm), dm.version, j.name)
        dist = self._assoc_cache.get(key)
        if dist is not None:
            return dist
        if len(self._assoc_cache) > 50000:
            self._assoc_cache.clear()
        if self.beta == math.inf:
            holders = [i.name for i in dm.chunks.values() if i is not j and self.contains(i, j)]
            dist = {n: 1.0 / len(holders) for n in holders}
        else:
            scores = [(i.name, self.containment(i, j)) for i in dm.chunks.values() if i is not j]
            scores = [(n, s) for n, s in scores if s > -math.inf]
            if not scores:
                dist = {}
            else:
                top = max(s for _, s in scores)
                w = [(n, math.exp(self.beta * (s - top))) for n, s in scores]
                z = sum(x for _, x in w)
                dist = {n: x / z for n, x in w}
        self._assoc_cache[key] = dist
        return dist

    def invalidate(self):
        """Forget cached associations (call after editing DM chunks by hand)."""
        self._assoc_cache = {}

    def sji(self, j, i) -> float:
        """S_ji = max(0, mas + ln P(i | j)); beta = inf gives mas - ln(fan)."""
        p = self.association_distribution(j).get(i.name, 0.0)
        if p <= 0.0:
            return 0.0
        return max(0.0, self.model.dm.maximum_associative_strength + math.log(p))


# ====================================================================== holographic chunks
class Holographic(Embedded):
    """Regular chunks are vectors; operators stay symbolic.

    Every regular DM chunk (any type not in ``symbolic_types``) is represented
    by two vectors built from the vectors of its slot values (``isa`` and the
    other ``hard_slots`` stay symbolic filters):

    * the **order vector** o = sum_k P_k e(v_k), where P_k is a fixed random
      permutation for slot k (a position: slot1, slot2, ...).  Binding by
      permutation is exactly invertible, so the value in slot k is read back
      as P_k^-1 o = e(v_k) + crosstalk from the other slots, with crosstalk of
      order sqrt((n - 1) / dim).
    * the **bag vector** b = sum_k e(v_k), which ignores position.

    The order vector serves retrieval and decoding:

    * A retrieval request is matched by dot products: for each cue slot k,
      d_k = <P_k e(cue_k), o>, which is sim(cue_k, v_k) plus crosstalk.  The
      activation gets mp * sum_k (d_k - 1).  With mp = inf a chunk is excluded
      unless every d_k >= ``decode_theta`` (and then gets no penalty), which
      is the symbolic limit for random vectors.
    * The retrieved chunk is *decoded* into the retrieval buffer: each slot is
      the clean-up of P_k^-1 o against the values in DM.  If every slot
      decodes to its stored value the buffer holds the DM chunk itself;
      otherwise it holds a copy with the decoded values (a decoding error,
      counted in ``decode_errors``).
    * Vector blending averages order vectors and decodes the average.

    The bag vector serves similarity-based spreading: chunk i contains source
    j to the degree <e(j), b_i>, whatever the position (``spreading="similarity"``).

    Operators (and other ``symbolic_types``) keep the slot-by-slot treatment
    of :class:`Embedded`.  Duplicate detection when chunks enter memory stays
    symbolic.  All other options are those of :class:`Embedded`; retrieval is
    always soft.
    """

    name = "holographic"
    holographic = True

    def __init__(self, embedding: Embedding | None = None, *, symbolic_types=("operator",),
                 decode_theta: float = 0.5, **kw):
        kw.setdefault("retrieval", "soft")
        if kw["retrieval"] != "soft":
            raise ValueError("holographic chunks are always retrieved by (soft) vector matching")
        super().__init__(embedding, **kw)
        self.symbolic_types = tuple(symbolic_types)
        self.decode_theta = float(decode_theta)
        self._roles: dict = {}
        self._vectors: dict = {}
        self._vectors_version = None
        self._cleanup_set = None
        self.decodes = 0
        self.decode_errors = 0

    def attach(self, model):
        super().attach(model)
        self._vectors, self._vectors_version, self._cleanup_set = {}, None, None
        self.decodes = self.decode_errors = 0

    def describe(self) -> str:
        return "holographic " + super().describe()[len("embedded"):]

    def invalidate(self):
        super().invalidate()
        self._vectors, self._vectors_version, self._cleanup_set = {}, None, None

    # ---------------------------------------------------------------- encoding
    def is_regular(self, chunk) -> bool:
        return chunk.type not in self.symbolic_types

    def role(self, slot: str):
        """The fixed random permutation for a slot (and its inverse)."""
        r = self._roles.get(slot)
        if r is None:
            h = hashlib.sha256(f"{self.embedding.seed}\x00role\x00{slot}".encode()).digest()
            p = np.random.default_rng(int.from_bytes(h[:8], "little")).permutation(self.embedding.dim)
            r = (p, np.argsort(p))
            self._roles[slot] = r
        return r

    def bind(self, slot: str, vec):
        return vec[self.role(slot)[0]]

    def unbind_vector(self, slot: str, vec):
        return vec[self.role(slot)[1]]

    def encoded_slots(self, chunk):
        return [(s, v) for s, v in chunk.slotvals.items() if s not in self.hard_slots]

    def encode(self, chunk):
        """(order vector, bag vector) of a chunk."""
        d = self.embedding.dim
        order, bag = np.zeros(d), np.zeros(d)
        for s, v in self.encoded_slots(chunk):
            e = self.embedding.vector(v)
            order += self.bind(s, e)
            bag += e
        return order, bag

    def vectors(self, chunk):
        """Cached (order, bag) for DM chunks; computed afresh for other chunks."""
        dm = self.model.dm
        if dm.chunks.get(chunk.name) is not chunk:
            return self.encode(chunk)
        if self._vectors_version != dm.version:
            self._vectors, self._vectors_version, self._cleanup_set = {}, dm.version, None
        v = self._vectors.get(chunk.name)
        if v is None:
            v = self.encode(chunk)
            self._vectors[chunk.name] = v
        return v

    def unbind(self, chunk, slot: str):
        """P_k^-1 o: the (noisy) vector of the value in slot k."""
        return self.unbind_vector(slot, self.vectors(chunk)[0])

    # ---------------------------------------------------------------- retrieval
    def cue_penalty(self, chunk, request, mp: float):
        """Partial-matching penalty of a regular chunk, or None if excluded."""
        order = self.vectors(chunk)[0]
        total = 0.0
        for slot, value in request.slotvals.items():
            if slot in self.hard_slots:
                v1 = chunk.slot_value(slot)
                if v1 is None or not values_equal(v1, value):
                    return None
                continue
            d = float(self.bind(slot, self.embedding.vector(value)) @ order)
            if mp == math.inf:
                if d < self.decode_theta:
                    return None
            else:
                total += d - 1.0
        return mp * total if total != 0.0 else 0.0

    def _cleanup_values(self):
        dm = self.model.dm
        if self._vectors_version != dm.version:
            self._vectors, self._vectors_version, self._cleanup_set = {}, dm.version, None
        if self._cleanup_set is None:
            values = {}
            for ch in dm.chunks.values():
                if self.is_regular(ch):
                    for _, v in self.encoded_slots(ch):
                        values.setdefault(describe(v), v)
            vals = list(values.values())
            M = np.stack([self.embedding.vector(v) for v in vals]) if vals else np.zeros((0, self.embedding.dim))
            self._cleanup_set = (vals, M)
        return self._cleanup_set

    def decode(self, chunk) -> dict:
        """slot -> (clean-up of P_k^-1 o over the values in DM, its similarity)."""
        vals, M = self._cleanup_values()
        out = {}
        for s, _ in self.encoded_slots(chunk):
            sims = M @ self.unbind(chunk, s)
            k = int(np.argmax(sims))
            out[s] = (vals[k], float(sims[k]))
        return out

    def decoded_chunk(self, chunk):
        """The chunk as read back from its vector (the chunk itself if decoding is exact)."""
        dec = self.decode(chunk)
        self.decodes += 1
        if all(values_equal(v, chunk.slotvals[s]) for s, (v, _) in dec.items()):
            return chunk, dec
        self.decode_errors += 1
        c = chunk.copy_chunk()
        for s, (v, _) in dec.items():
            c.set_slot_value(s, v)
        return c, dec

    # ---------------------------------------------------------------- spreading
    def containment(self, i, j) -> float:
        if not self.is_regular(i):
            return super().containment(i, j)
        return float(self.embedding.vector(j) @ self.vectors(i)[1])

    def contains(self, i, j) -> bool:
        if not self.is_regular(i):
            return super().contains(i, j)
        return self.containment(i, j) >= self.decode_theta
