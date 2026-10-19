"""The layout of the index tables that every matcher returns.

An index holds an int64 ``<survey>/row_index`` column per survey: row ``i``
of a survey is its ``i``-th input row, and null means the survey is absent
from that entity. A pair of surveys is written ``<a>+<b>``, as in the
``<a>+<b>/separation_arcsec`` columns of a subset join. Survey names
therefore become column prefixes. Settings,
dedupe outcomes and a summary are recorded as JSON in the schema metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

RESOLVED_CONFIG_KEY = b"astribidem.resolved_config"
DEDUPE_KEY = b"astribidem.dedupe"
SUMMARY_KEY = b"astribidem.summary"


def row_column(survey: str) -> str:
    """The column holding a survey's row positions."""
    return f"{survey}/row_index"


def pair_name(survey_a: str, survey_b: str) -> str:
    """How a pair of surveys is written in column and file names."""
    return f"{survey_a}+{survey_b}"


def separation_column(survey_a: str, survey_b: str) -> str:
    """The column holding a pair's separations in a subset join."""
    return f"{pair_name(survey_a, survey_b)}/separation_arcsec"


def check_survey_names(names: Iterable[str]) -> None:
    """Raise unless every name can prefix a column: non-empty, no ``/`` or ``+``.

    Without ``+`` in names, two different pairs never share a pair name.
    """
    for name in names:
        if not isinstance(name, str) or not name or "/" in name or "+" in name:
            raise ValueError(
                f"survey names must be non-empty and must not contain '/' or '+': "
                f"{name!r}"
            )


def encode_metadata(value: Any) -> bytes:
    """Compact, deterministic JSON for one schema metadata value."""
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def index_summary(index: pa.Table) -> dict[str, Any]:
    """The summary a spatial index records, such as its entity count."""
    return json.loads(index.schema.metadata[SUMMARY_KEY])
