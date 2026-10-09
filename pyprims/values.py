"""Low-level helpers shared by all modules.

Slot values in PRIMs are one of three kinds (Support.swift, ``enum Value``):

* a :class:`~pyprims.chunk.Chunk` (Swift ``.symbol``),
* a number, always a float (Swift ``.Number(Double)``),
* a string (Swift ``.Text``).

Swift compares slot values by their *description* (``Value.isEqual``), so the
number 3.0 and the text "3" are different values ("3.0" != "3").  The helpers
below reproduce that behaviour exactly.
"""
from __future__ import annotations

import math
import re

# ---------------------------------------------------------------- numbers

_DOUBLE_PREFIX = re.compile(
    r"\s*[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")


def string2double(s: str):
    """Port of ``string2Double``: Foundation's ``Scanner.scanDouble``.

    Like the Swift original this accepts a numeric *prefix*: "3abc" -> 3.0.
    That quirk matters, because ``Chunk.setSlot(String)`` uses it to decide
    whether a value is a number, so a chunk name such as "2nd" becomes 2.0.
    """
    if not isinstance(s, str):
        return None
    m = _DOUBLE_PREFIX.match(s)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:            # pragma: no cover - regex guarantees a float
        return None


def describe(v) -> str:
    """Swift ``description`` of a slot value or script value."""
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)            # Swift and Python both print shortest repr: 3.0, 1e-05
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "[" + ", ".join(describe(x) for x in v) + "]"
    name = getattr(v, "name", None)   # Chunk
    if name is not None:
        return name
    return str(v)


def values_equal(a, b) -> bool:
    """``Value.isEqual``: equality of descriptions."""
    return describe(a) == describe(b)


def is_chunk(v) -> bool:
    return hasattr(v, "slotvals")


def safe_log(x: float) -> float:
    if x > 0:
        return math.log(x)
    if x == 0:
        return -math.inf
    return math.nan


def safe_pow(base: float, exponent: float) -> float:
    if base > 0:
        return base ** exponent
    if base == 0:
        return math.inf if exponent < 0 else 0.0
    return math.nan


# ---------------------------------------------------------------- noise

def actr_noise(rng, s: float) -> float:
    """``actrNoise``: logistic noise with scale s.

    Swift draws ``arc4random_uniform(99998) + 1`` and divides by 100000, so the
    uniform deviate lies in {0.00001, ..., 0.99998}; we use the same grid.
    """
    if s == 0:
        return 0.0
    rand = rng.randint(1, 99998) / 100000.0
    return s * math.log((1.0 - rand) / rand)


def uniform01(rng) -> float:
    """The uniform deviate used for uniform action noise."""
    return rng.randint(1, 99998) / 100000.0


# ---------------------------------------------------------------- PRIM lists

def chop_prims(s: str, n: int):
    """``chopPrims``: split a ';'-separated PRIM list after the first n PRIMs."""
    x = s.split(";")
    if len(x) == n:
        return s, ""
    return ";".join(x[:n]), ";".join(x[n:])
