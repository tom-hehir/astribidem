"""Candidate edges: everything the edge build hands to resolution.

``build_edges`` produces a ``CandidateEdges`` object in memory, ``resolve``
consumes one, and ``write_edges`` / ``read_edges`` save and reload it. All
values are row positions: row ``i`` of a survey is the ``i``-th coordinate
passed to the edge build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DedupeOutcome:
    """One survey's row count and its dedupe results.

    ``dropped_rows[k]`` was a duplicate of ``kept_rows[k]``, the lowest row of
    its duplicate group. ``disputed_rows`` belong to groups that are not
    cliques and therefore have no kept row. Every other row is active.
    """

    name: str
    n_rows: int
    dedupe_radius_arcsec: float
    dropped_rows: np.ndarray
    kept_rows: np.ndarray
    disputed_rows: np.ndarray

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_rows)

    @property
    def n_disputed(self) -> int:
        return len(self.disputed_rows)

    def active_mask(self) -> np.ndarray:
        """``True`` for rows that take part in cross-survey matching."""
        mask = np.ones(self.n_rows, dtype=bool)
        mask[self.dropped_rows] = False
        mask[self.disputed_rows] = False
        return mask

    def active_rows(self) -> np.ndarray:
        """Active rows in ascending order."""
        return np.flatnonzero(self.active_mask())


@dataclass(frozen=True)
class PairEdges:
    """Every in-radius pair of active rows between two surveys.

    Edges are sorted by ``(row_a, row_b)``.
    """

    survey_a: str
    survey_b: str
    radius_arcsec: float
    row_a: np.ndarray
    row_b: np.ndarray
    separation_arcsec: np.ndarray


@dataclass(frozen=True)
class CandidateEdges:
    """Dedupe outcomes for each survey plus the edges of each built pair.

    ``pairs`` holds every survey pair unless the build was restricted to some
    pairs, in which case only degenerate modes over those pairs can use it.
    """

    surveys: tuple[DedupeOutcome, ...]
    pairs: dict[frozenset[str], PairEdges]

    @property
    def survey_names(self) -> tuple[str, ...]:
        return tuple(survey.name for survey in self.surveys)

    def survey(self, name: str) -> DedupeOutcome:
        for survey in self.surveys:
            if survey.name == name:
                return survey
        raise KeyError(name)

    def oriented_edges(
        self, first: str, second: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A pair's edges as (row_first, row_second, separation), swapping if needed."""
        pair = self.pairs[frozenset((first, second))]
        if pair.survey_a == first:
            return pair.row_a, pair.row_b, pair.separation_arcsec
        return pair.row_b, pair.row_a, pair.separation_arcsec

    def radius_metadata(self, pair_names) -> list[dict[str, Any]]:
        return [
            {
                "surveys": [a, b],
                "radius_arcsec": self.pairs[frozenset((a, b))].radius_arcsec,
            }
            for a, b in pair_names
        ]
