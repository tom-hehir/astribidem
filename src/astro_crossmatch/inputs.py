"""The seam between the edge substrate and the mode builders.

``ResolveInputs`` carries everything a mode consumes, expressed in row
positions (0..n-1 per survey, in input order). Mode builders emit those rows
as ``<survey>/row_index`` columns; ``astro_crossmatch.rows_to_ids`` maps them
to caller IDs afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairEdges:
    """All candidate edges between two surveys' active rows (positions)."""

    survey_a: str
    survey_b: str
    radius_arcsec: float
    row_a: np.ndarray
    row_b: np.ndarray
    sep_arcsec: np.ndarray


@dataclass(frozen=True)
class ResolveInputs:
    """Everything a mode builder consumes.

    ``active_row_index`` / ``dedupe_disputed`` / pair rows are row positions
    (ascending, per survey input order). ``delimiter`` joins survey names with
    column names in the output schema.
    """

    survey_names: tuple[str, ...]
    active_row_index: dict[str, np.ndarray]
    dedupe_disputed: dict[str, np.ndarray]
    pairs: dict[frozenset[str], PairEdges]
    pair_radius_arcsec: dict[frozenset[str], float]
    delimiter: str = "/"

    def oriented_edges(
        self, first: str, second: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A pair's edges as (row_first, row_second, sep), swapping if needed."""
        pair = self.pairs[frozenset((first, second))]
        if pair.survey_a == first:
            return pair.row_a, pair.row_b, pair.sep_arcsec
        return pair.row_b, pair.row_a, pair.sep_arcsec

    def row_column_name(self, survey: str) -> str:
        """The output column carrying this survey's row positions."""
        return f"{survey}{self.delimiter}row_index"

    def radius_metadata(self, pair_names) -> list[dict[str, Any]]:
        entries = []
        for a, b in pair_names:
            radius = self.pair_radius_arcsec[frozenset((a, b))]
            entries.append({"surveys": [a, b], "radius_arcsec": radius})
        return entries
