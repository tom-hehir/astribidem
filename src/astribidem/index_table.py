"""The layout of the index tables that every matcher returns.

An index holds an int64 ``<survey>/row_index`` column per survey: row ``i``
of a survey is its ``i``-th input row, and null means the survey is absent
from that entity. Survey names therefore become column prefixes.
"""

from __future__ import annotations

from collections.abc import Iterable


def check_survey_names(names: Iterable[str]) -> None:
    """Raise unless every name can prefix an index column: non-empty, no ``/``."""
    for name in names:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"survey names must be non-empty and must not contain '/': {name!r}"
            )
