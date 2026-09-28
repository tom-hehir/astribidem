"""Coordinates, deduplication and the candidate-edge build.

Adapted from Astral Projections. Everything here works in row positions: row
``i`` of a survey is the ``i``-th coordinate passed in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite

import numpy as np

from astribidem.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astribidem.graph import (
    clique_mask,
    component_sizes_and_edge_counts,
    connected_components,
)
from astribidem.kernel import (
    CatalogKernel,
    float64_coordinates,
    radec_to_xyz,
    resolve_workers,
)


@dataclass(frozen=True)
class SurveyCoords:
    """One survey's unit sky vectors; row ``i`` is ``xyz[i]``."""

    name: str
    xyz: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)


def survey_coords_from_arrays(
    name: str,
    ra: np.ndarray,
    dec: np.ndarray,
) -> SurveyCoords:
    """Build coordinates from RA/Dec degree arrays.

    Coordinates become float64 before anything else; see
    ``kernel.float64_coordinates``.
    """
    ra, dec = float64_coordinates(ra, dec, name=name)
    if ra.ndim != 1 or dec.ndim != 1:
        raise ValueError(f"survey {name!r}: ra/dec must be 1-D")
    if len(ra) != len(dec):
        raise ValueError(
            f"survey {name!r}: ra/dec shapes differ: {ra.shape} vs {dec.shape}"
        )
    return SurveyCoords(name=name, xyz=radec_to_xyz(ra, dec))


def dedupe_survey(
    coords: SurveyCoords,
    radius_arcsec: float,
    kernel: CatalogKernel | None = None,
) -> tuple[DedupeOutcome, np.ndarray]:
    """De-duplicate one survey; return the outcome and the active-row mask.

    Within-radius components that are cliques are duplicates by the physical
    argument (below the survey's resolution floor, two rows cannot be two
    resolved objects): keep the **lowest row**, drop the rest. The kept row
    therefore depends on input order; sort the inputs first when that order
    is not meaningful. Non-clique components (chains from shredding, flanking
    geometries) are disputed: every member is flagged and excluded from
    cross-survey matching.

    ``kernel`` is the survey's already-built ``CatalogKernel`` (``crossmatch``
    shares one per survey with pair matching); omitted, one is built here.
    """
    n = len(coords)
    active = np.ones(n, dtype=bool)
    empty = np.empty(0, dtype=np.int64)
    if radius_arcsec <= 0 or n == 0:
        outcome = DedupeOutcome(
            name=coords.name,
            n_rows=n,
            dedupe_radius_arcsec=radius_arcsec,
            dropped_rows=empty,
            kept_rows=empty,
            disputed_rows=empty,
        )
        return outcome, active

    if kernel is None:
        kernel = CatalogKernel.from_xyz(coords.xyz)
    # The complete within-radius pair set: nearest-only truncation would hide
    # exactly the extra candidates that make components disputed.
    first, second, _ = kernel.self_pairs(radius_arcsec)
    dropped, kept, disputed = dedupe_from_pairs(
        np.arange(n, dtype=np.int64), first, second
    )
    active[dropped] = False
    active[disputed] = False
    outcome = DedupeOutcome(
        name=coords.name,
        n_rows=n,
        dedupe_radius_arcsec=radius_arcsec,
        dropped_rows=dropped,
        kept_rows=kept,
        disputed_rows=disputed,
    )
    return outcome, active


def dedupe_from_pairs(
    rows: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dedupe verdicts from one survey's within-radius pairs, without coordinates.

    ``rows`` holds the global row number of each of ``n`` nodes, in any order;
    ``first`` / ``second`` index those nodes, one entry per pair. Groups of
    rows joined by pairs that form cliques keep their lowest row and drop the
    rest; other groups of two or more rows are disputed. Returns global rows:
    ``dropped`` (ascending), the ``kept`` row of each dropped row, and
    ``disputed`` (ascending).
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n == 0 or len(first) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    labels = connected_components(n, first, second)
    sizes, edge_counts = component_sizes_and_edge_counts(labels, first)
    is_clique = clique_mask(sizes, edge_counts)
    multi = sizes[labels] > 1
    disputed = ~is_clique[labels] & multi
    keeper = np.full(len(sizes), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(keeper, labels, rows)
    dropped = is_clique[labels] & multi & (rows != keeper[labels])
    order = np.argsort(rows[dropped], kind="stable")
    return (
        rows[dropped][order],
        keeper[labels[dropped]][order],
        np.sort(rows[disputed]),
    )


def pair_edge_chunks(
    kernel_a: CatalogKernel,
    kernel_b: CatalogKernel,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    *,
    workers: int = 1,
    chunk_rows: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every in-radius pair of active rows, as ordered ``(row_a, row_b, sep)`` chunks.

    The kernels always emit the *complete* in-radius pair set: resolution
    assumes completeness (the entitywise mode needs all edges to detect
    disputes, ``mutual_unique`` needs candidate counts), so it must not be
    truncated. Kernels cover each survey's full row set (shared with dedupe);
    pairs touching dedupe-inactive rows are dropped here. ``chunk_rows`` sets
    how many of ``a``'s rows each chunk covers.
    """
    for i, j, separation in kernel_a.all_pairs_chunks(
        kernel_b, radius_arcsec, workers=workers, chunk_rows=chunk_rows
    ):
        keep = active_a[i] & active_b[j]
        yield i[keep], j[keep], separation[keep]


def build_pair_edges(
    a: SurveyCoords,
    b: SurveyCoords,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    kernel_a: CatalogKernel | None = None,
    kernel_b: CatalogKernel | None = None,
    *,
    workers: int = 1,
) -> PairEdges:
    """All candidate pairs between two surveys' active rows, held in memory."""
    if kernel_a is None:
        kernel_a = CatalogKernel.from_xyz(a.xyz)
    if kernel_b is None:
        kernel_b = CatalogKernel.from_xyz(b.xyz)
    chunks = list(
        pair_edge_chunks(
            kernel_a, kernel_b, active_a, active_b, radius_arcsec, workers=workers
        )
    )
    row_a, row_b, separation = (
        chunks[0]
        if len(chunks) == 1
        else tuple(np.concatenate([chunk[k] for chunk in chunks]) for k in range(3))
    )
    return PairEdges(
        survey_a=a.name,
        survey_b=b.name,
        radius_arcsec=radius_arcsec,
        row_a=row_a,
        row_b=row_b,
        separation_arcsec=separation,
    )


# --- the edge build ------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSettings:
    """Validated matching settings for an edge build, independent of coordinates.

    ``pair_radius_arcsec`` maps each built pair, in survey order, to its radius.
    """

    survey_names: tuple[str, ...]
    n_rows: dict[str, int]
    dedupe_radius_arcsec: dict[str, float]
    pair_radius_arcsec: dict[tuple[str, str], float]

    @property
    def max_radius_arcsec(self) -> float:
        """The largest pair or dedupe radius: the margin a region needs."""
        radii = [*self.pair_radius_arcsec.values(), *self.dedupe_radius_arcsec.values()]
        return max(radii, default=0.0)


@dataclass(frozen=True)
class _EdgeBuildPlan:
    """Validated edge-build settings, shared by in-memory and streamed builds."""

    surveys: tuple[SurveyCoords, ...]
    dedupe_radius_arcsec: dict[str, float]
    pair_radius_arcsec: dict[tuple[str, str], float]  # built pairs, survey order


def _validate_survey(survey: SurveyCoords) -> None:
    if not isinstance(survey.name, str) or not survey.name or "/" in survey.name:
        raise ValueError("survey names must be non-empty and must not contain '/'")
    xyz = np.asarray(survey.xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"survey {survey.name!r}: xyz must have shape (n, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError(f"survey {survey.name!r}: coordinates must be finite")


def edge_settings(
    n_rows: Mapping[str, int],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> EdgeSettings:
    """Validate matching settings for surveys with the given row counts.

    ``n_rows`` maps each survey name, in survey order, to its row count. The
    remaining arguments are those of ``build_edges``.
    """
    names = tuple(n_rows)
    if not names:
        raise ValueError("at least one survey is required")
    for name in names:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("survey names must be non-empty and must not contain '/'")
    known = set(names)
    if set(dedupe_radius_arcsec) != known:
        raise ValueError("dedupe_radius_arcsec must name every survey exactly once")
    dedupe_radii = {name: float(dedupe_radius_arcsec[name]) for name in names}
    if any(not isfinite(value) or value < 0 for value in dedupe_radii.values()):
        raise ValueError("dedupe radii must be finite and >= 0")
    radius_arcsec = float(radius_arcsec)
    if not isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be finite and > 0")

    radii = {frozenset(pair): radius_arcsec for pair in combinations(names, 2)}
    overridden: set[frozenset[str]] = set()
    for pair, radius in (pair_radius_overrides or {}).items():
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("each radius override must name exactly two surveys")
        key = frozenset(pair)
        if len(key) != 2 or not key <= known:
            raise ValueError(f"radius override names invalid survey pair: {pair!r}")
        if key in overridden:
            raise ValueError(f"duplicate radius override for pair: {sorted(key)}")
        radius = float(radius)
        if not isfinite(radius) or radius <= 0:
            raise ValueError("override radii must be finite and > 0")
        overridden.add(key)
        radii[key] = radius

    selected = None
    if pairs is not None:
        selected = set()
        for pair in pairs:
            key = frozenset(pair)
            if (
                isinstance(pair, str)
                or len(pair) != 2
                or len(key) != 2
                or not key <= known
            ):
                raise ValueError(f"pairs names an invalid survey pair: {pair!r}")
            selected.add(key)
    return EdgeSettings(
        survey_names=names,
        n_rows={name: int(n_rows[name]) for name in names},
        dedupe_radius_arcsec=dedupe_radii,
        pair_radius_arcsec={
            (a, b): radii[frozenset((a, b))]
            for a, b in combinations(names, 2)
            if selected is None or frozenset((a, b)) in selected
        },
    )


def _plan_edge_build(
    surveys: Sequence[SurveyCoords],
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float] | None,
    pairs: Iterable[tuple[str, str]] | None,
) -> _EdgeBuildPlan:
    surveys = tuple(surveys)
    if not surveys:
        raise ValueError("at least one survey is required")
    for survey in surveys:
        _validate_survey(survey)
    names = [survey.name for survey in surveys]
    if len(set(names)) != len(names):
        raise ValueError("survey names must be unique")
    settings = edge_settings(
        {survey.name: len(survey) for survey in surveys},
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        pair_radius_overrides=pair_radius_overrides,
        pairs=pairs,
    )
    return _EdgeBuildPlan(
        surveys=surveys,
        dedupe_radius_arcsec=settings.dedupe_radius_arcsec,
        pair_radius_arcsec=settings.pair_radius_arcsec,
    )


def _kernels_and_dedupe(
    plan: _EdgeBuildPlan,
) -> tuple[dict[str, CatalogKernel], tuple[DedupeOutcome, ...], dict[str, np.ndarray]]:
    """One kernel per survey, shared by its dedupe and every pair it joins."""
    kernels = {
        survey.name: CatalogKernel.from_xyz(survey.xyz) for survey in plan.surveys
    }
    outcomes = []
    active = {}
    for survey in plan.surveys:
        outcome, active[survey.name] = dedupe_survey(
            survey, plan.dedupe_radius_arcsec[survey.name], kernel=kernels[survey.name]
        )
        outcomes.append(outcome)
    return kernels, tuple(outcomes), active


def build_edges(
    surveys: Sequence[SurveyCoords],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
) -> CandidateEdges:
    """Dedupe every survey and find every in-radius pair between surveys.

    Every survey requires an explicit dedupe radius; ``0.0`` opts out.
    ``radius_arcsec`` is the default cross-survey radius. Optional pair
    overrides name exactly two surveys; the same unordered pair cannot be
    specified twice, even with equal radii. ``pairs`` restricts which survey
    pairs are joined (default: all); the result then serves only degenerate
    modes over those pairs. Coordinates must already be cleaned.
    """
    plan = _plan_edge_build(
        surveys, radius_arcsec, dedupe_radius_arcsec, pair_radius_overrides, pairs
    )
    workers = resolve_workers(workers)
    kernels, outcomes, active = _kernels_and_dedupe(plan)
    coords = {survey.name: survey for survey in plan.surveys}
    edges = {
        frozenset((a, b)): build_pair_edges(
            coords[a],
            coords[b],
            active[a],
            active[b],
            radius,
            kernel_a=kernels[a],
            kernel_b=kernels[b],
            workers=workers,
        )
        for (a, b), radius in plan.pair_radius_arcsec.items()
    }
    return CandidateEdges(surveys=outcomes, pairs=edges)
