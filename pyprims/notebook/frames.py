"""DataFrames that show text columns left-aligned in a notebook.

Jupyter right-aligns every table cell.  ``LeftFrame`` is an ordinary pandas
DataFrame (``head()``, filtering etc. keep the class) whose HTML display
left-aligns the index and every non-numeric column; numbers stay right-aligned.
"""
from __future__ import annotations

import itertools

import pandas as pd

_ids = itertools.count()


class LeftFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return LeftFrame

    def _repr_html_(self):
        html = super()._repr_html_()
        if html is None:
            return None
        cls = f"pp-df{next(_ids)}"
        n_index = self.index.nlevels
        rules = [f".{cls} tbody th, .{cls} thead th:nth-child(-n+{n_index})"
                 "{text-align:left!important}"]
        for j, col in enumerate(self.columns):
            if not pd.api.types.is_numeric_dtype(self[col]) or pd.api.types.is_bool_dtype(self[col]):
                k = n_index + j + 1
                rules.append(f".{cls} td:nth-child({k}), .{cls} thead th:nth-child({k})"
                             "{text-align:left!important}")
        return f"<style>{''.join(rules)}</style><div class='{cls}'>{html}</div>"


def frame(records, columns, index=None):
    """A LeftFrame from a list of dicts (or the list itself when pandas is missing)."""
    df = LeftFrame(records, columns=columns)
    return df.set_index(index) if index else df
