"""Candidate edges: everything the edge build hands to resolution.

``build_edges`` produces a ``CandidateEdges`` object in memory, ``resolve``
consumes one, and ``write_edges`` / ``read_edges`` save and reload it. All
values are row positions: row ``i`` of a survey is the ``i``-th coordinate
passed to the edge build.

A ``CandidateEdges`` object is one segment: a self-contained slice of the
answer in which every edge joins two of the segment's rows and no connected
group of rows is split with another segment. An in-memory build is a single
segment covering every row; a banded build produces many segments, each
covering some rows. Row numbers are global in every segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DedupeOutcome:
    """One survey's rows in a segment and their dedupe results.

    ``n_rows`` is the survey's total row count. ``rows`` lists, in ascending
    order, the rows this segment covers; ``None`` means every row.
    ``dropped_rows[k]`` was a duplicate of ``kept_rows[k]``, the lowest row of
    its duplicate group. ``disputed_rows`` belong to groups that are not
    cliques and therefore have no kept row. Every other covered row is active.
    """

    name: str
    n_rows: int
    dedupe_radius_arcsec: float
    dropped_rows: np.ndarray
    kept_rows: np.ndarray
    disputed_rows: np.ndarray
    rows: np.ndarray | None = None

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_rows)

    @property
    def n_disputed(self) -> int:
        return len(self.disputed_rows)

    @property
    def n_covered(self) -> int:
        """How many rows this segment covers."""
        return self.n_rows if self.rows is None else len(self.rows)

    @property
    def n_active(self) -> int:
        return self.n_covered - self.n_dropped - self.n_disputed

    def active_rows(self) -> np.ndarray:
        """Covered rows that take part in cross-survey matching, ascending."""
        inactive = np.concatenate([self.dropped_rows, self.disputed_rows])
        if self.rows is None:
            mask = np.ones(self.n_rows, dtype=bool)
            mask[inactive] = False
            return np.flatnonzero(mask)
        return np.setdiff1d(self.rows, inactive, assume_unique=True)


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
    """One segment: dedupe outcomes for each survey plus each built pair's edges.

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


def _sorted_union(parts: list[np.ndarray]) -> np.ndarray:
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def combine_segments(segments: list[CandidateEdges]) -> CandidateEdges:
    """Concatenate segments into one segment covering all of their rows.

    Segments must cover disjoint rows of the same surveys and pairs. Dedupe
    results and edges are merged in row order, so the result equals the
    single segment an in-memory build of the same rows would produce.
    """
    if not segments:
        raise ValueError("at least one segment is required")
    if len(segments) == 1:
        return segments[0]
    first = segments[0]
    surveys = []
    for index, template in enumerate(first.surveys):
        parts = [segment.surveys[index] for segment in segments]
        dropped = np.concatenate([part.dropped_rows for part in parts])
        kept = np.concatenate([part.kept_rows for part in parts])
        order = np.argsort(dropped, kind="stable")
        rows = _sorted_union(
            [
                np.arange(part.n_rows, dtype=np.int64)
                if part.rows is None
                else part.rows
                for part in parts
            ]
        )
        surveys.append(
            DedupeOutcome(
                name=template.name,
                n_rows=template.n_rows,
                dedupe_radius_arcsec=template.dedupe_radius_arcsec,
                dropped_rows=dropped[order],
                kept_rows=kept[order],
                disputed_rows=_sorted_union([part.disputed_rows for part in parts]),
                rows=None if len(rows) == template.n_rows else rows,
            )
        )
    pairs = {}
    for key, template in first.pairs.items():
        parts = [segment.pairs[key] for segment in segments]
        row_a = np.concatenate([part.row_a for part in parts])
        row_b = np.concatenate([part.row_b for part in parts])
        separation = np.concatenate([part.separation_arcsec for part in parts])
        order = np.lexsort((row_b, row_a))
        pairs[key] = PairEdges(
            survey_a=template.survey_a,
            survey_b=template.survey_b,
            radius_arcsec=template.radius_arcsec,
            row_a=row_a[order],
            row_b=row_b[order],
            separation_arcsec=separation[order],
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)
