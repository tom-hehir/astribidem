"""The layout of the index tables that every matcher returns.

An index holds an int64 ``<survey>/row_index`` column per survey: row ``i``
of a survey is its ``i``-th input row, and null means the survey is absent
from that entity. Survey names therefore become column prefixes. Settings,
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


def check_survey_names(names: Iterable[str]) -> None:
    """Raise unless every name can prefix an index column: non-empty, no ``/``."""
    for name in names:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"survey names must be non-empty and must not contain '/': {name!r}"
            )


def encode_metadata(value: Any) -> bytes:
    """Compact, deterministic JSON for one schema metadata value."""
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def index_summary(index: pa.Table) -> dict[str, Any]:
    """The summary a spatial index records, such as its entity count."""
    return json.loads(index.schema.metadata[SUMMARY_KEY])
